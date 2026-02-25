# --- DISABLE PYCACHE GENERATION ---
import sys
sys.dont_write_bytecode = True
# ----------------------------------

import argparse
import pandas as pd
import xarray as xr
import shutil
from pathlib import Path
from dask.diagnostics import ProgressBar

from lab_fetcher.hrrr_fetcher import HRRRFetcher
from lab_fetcher.rave_fetcher import RAVEFetcher
from lab_fetcher.grid import regrid_rave_to_hrrr 

def parse_bbox(text):
    lat_min, lat_max, lon_min, lon_max = map(float, text.split(","))
    return dict(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    args = parser.parse_args()

    # 1. Parse BBox
    bbox_dict = parse_bbox(args.bbox)
    pad = 0.5
    
    bbox_tuple = (
        bbox_dict["lon_min"] - pad,
        bbox_dict["lon_max"] + pad,
        bbox_dict["lat_min"] - pad,
        bbox_dict["lat_max"] + pad
    )

    times = pd.date_range(args.start, args.end, freq="1h")
    
    print("\nInitializing Pipeline (Targeting: RAVE FRP + PM25)...")
    
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()

    # Create temp dir
    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # --- DEFINE WEIGHTS FILE INSIDE TEMP DIR ---
    # This ensures it is reused (fast) but deleted at the end (clean)
    weights_file = temp_dir / "weights_temp.nc"
    # -------------------------------------------

    saved_files = []

    for t in times:
        print(f"\n--- Processing {t} ---")
        
        try:
            # Step 1: HRRR
            print("Fetching & clipping HRRR...")
            hrrr_ds = hrrr_fetcher.fetch_range(t, t, bbox=bbox_tuple)
            
            if hrrr_ds is None: 
                print("Skipping: No HRRR data found.")
                continue

            if "latitude" in hrrr_ds:
                hrrr_ds = hrrr_ds.rename({"latitude": "lat", "longitude": "lon"})

            # Step 2: RAVE
            print("Fetching & clipping RAVE...")
            rave_ds = rave_fetcher.fetch_range(t, t, bbox=bbox_tuple)
            
            if "grid_latt" in rave_ds:
                rave_ds = rave_ds.rename({"grid_latt": "lat", "grid_lont": "lon"})

            # --- SELECT & RENAME RAVE VARIABLES ---
            rave_subset = xr.Dataset()

            if "FRP_MEAN" in rave_ds:
                rave_subset["rave_frp"] = rave_ds["FRP_MEAN"].fillna(0.0)
            
            if "PM25" in rave_ds:
                rave_subset["rave_pm25"] = rave_ds["PM25"].fillna(0.0)
                
            if "PM25_scaled" in rave_ds:
                rave_subset["rave_pm25_scaled"] = rave_ds["PM25_scaled"].fillna(0.0)

            if len(rave_subset.data_vars) == 0:
                print(f"Warning: No valid RAVE variables found for {t}. Skipping.")
                continue

            rave_subset = rave_subset.assign_coords({
                "lat": rave_ds.lat,
                "lon": rave_ds.lon
            })
            
            # Step 4: Regrid RAVE -> HRRR
            # PASS THE TEMP WEIGHTS PATH HERE
            rave_rg = regrid_rave_to_hrrr(
                rave_subset, 
                hrrr_ds, 
                weights_path=weights_file, # <--- Fix applied here
                method="bilinear"
            )

            # Step 5: Merge
            merged = xr.merge([hrrr_ds, rave_rg], compat="override")

            # Step 6: Save temp file
            out_path = temp_dir / f"merged_{t.strftime('%Y%m%d%H')}.nc"
            merged.to_netcdf(out_path)
            saved_files.append(out_path)
            
            hrrr_ds.close()
            rave_ds.close()
            merged.close()

        except Exception as e:
            print(f"Skipping {t} due to error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not saved_files:
        print("\nNo files were processed successfully. Exiting.")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return

    # Step 7: Combine (Memory Safe Method)
    print("\nCombining all processed hours into final dataset...")
    
    datasets = []
    for p in saved_files:
        try:
            with xr.open_dataset(p) as ds:
                datasets.append(ds.load())
        except Exception as e:
            print(f"Error loading temp file {p}: {e}")

    if datasets:
        combined = xr.concat(datasets, dim="time")
        
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        final_out_path = data_dir / "combined_final.nc"
        
        print(f"Writing final combined NetCDF to {final_out_path}...")
        combined.to_netcdf(final_out_path)
        print("Done! Data saved to combined_final.nc")
    
    # Clean up (This deletes the weights file too)
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()