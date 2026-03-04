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
from lab_fetcher.landfire_fetcher import LandfireFetcher # <--- NEW
from lab_fetcher.grid import regrid_rave_to_hrrr, regrid_static_to_hrrr # <--- Updated Import

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
    # Removed --landfire argument (now automatic)
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
    
    # Landfire fetcher expects (min_lon, max_lon, min_lat, max_lat)
    # We construct a slightly different tuple for it just to be safe with ordering
    lf_bbox = (
        bbox_dict["lon_min"] - pad,
        bbox_dict["lon_max"] + pad,
        bbox_dict["lat_min"] - pad,
        bbox_dict["lat_max"] + pad
    )

    times = pd.date_range(args.start, args.end, freq="1h")
    
    print("\nInitializing Pipeline (RAVE + LANDFIRE)...")
    
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()
    lf_fetcher = LandfireFetcher() # <--- NEW

    # --- PRE-FETCHING ---
    print("\n--- 1. Pre-fetching RAVE Data ---")
    rave_file_map = rave_fetcher.prefetch(args.start, args.end)

    print("\n--- 2. Fetching LANDFIRE Data ---")
    # Fetch once, load into memory
    landfire_ds = lf_fetcher.fetch_bbox(lf_bbox)
    # --------------------

    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)
    weights_file = temp_dir / "weights_temp.nc"
    
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

            # Step 2: RAVE (Local Load)
            if t in rave_file_map:
                print(f"Loading cached RAVE file...")
                rave_ds = rave_fetcher.open_local(rave_file_map[t], bbox=bbox_tuple)
            else:
                print(f"Warning: No RAVE file found for {t}. Skipping.")
                continue

            if rave_ds is None: continue

            if "grid_latt" in rave_ds:
                rave_ds = rave_ds.rename({"grid_latt": "lat", "grid_lont": "lon"})

            # --- SELECT RAVE VARIABLES ---
            rave_subset = xr.Dataset()
            if "FRP_MEAN" in rave_ds:
                rave_subset["rave_frp"] = rave_ds["FRP_MEAN"].fillna(0.0)

            if len(rave_subset.data_vars) == 0:
                print("Warning: FRP_MEAN missing in RAVE.")
                continue

            rave_subset = rave_subset.assign_coords({
                "lat": rave_ds.lat,
                "lon": rave_ds.lon
            })
            
            # Step 3: Regrid RAVE -> HRRR
            rave_rg = regrid_rave_to_hrrr(
                rave_subset, 
                hrrr_ds, 
                weights_path=weights_file, 
                method="bilinear"
            )

            # Step 4: Regrid LANDFIRE -> HRRR
            # We do this every loop to ensure it aligns with the current HRRR grid slice
            # (Nearest Neighbor method for Fuel Models)
            landfire_rg = None
            if landfire_ds:
                landfire_rg = regrid_static_to_hrrr(
                    landfire_ds,
                    hrrr_ds,
                    weights_path=weights_file
                )

            # Step 5: Merge All
            merge_list = [hrrr_ds, rave_rg]
            if landfire_rg: merge_list.append(landfire_rg)
            
            merged = xr.merge(merge_list, compat="override")

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
        print("\nNo files processed. Exiting.")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return

    # Step 7: Combine
    print("\nCombining all processed hours...")
    datasets = []
    for p in saved_files:
        try:
            with xr.open_dataset(p) as ds:
                datasets.append(ds.load())
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if datasets:
        combined = xr.concat(datasets, dim="time")
        
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        final_out_path = data_dir / "combined_final.nc"
        
        print(f"Writing final NetCDF to {final_out_path}...")
        combined.to_netcdf(final_out_path)
        print("Done!")
    
    print("Cleaning up...")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()