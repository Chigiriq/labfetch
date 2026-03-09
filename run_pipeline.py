# --- DISABLE PYCACHE GENERATION ---
import sys
import os
sys.dont_write_bytecode = True
# ----------------------------------

import argparse
import pandas as pd
import xarray as xr
import shutil
from pathlib import Path

from fetchers.hrrr_fetcher import HRRRFetcher
from fetchers.rave_fetcher import RAVEFetcher
from processors.grid import regrid_rave_to_hrrr

def parse_bbox(text):
    lat_min, lat_max, lon_min, lon_max = map(float, text.split(","))
    return dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    args = parser.parse_args()

    bbox_dict = parse_bbox(args.bbox)
    pad = 0.5
    bbox_tuple = (
        bbox_dict["lon_min"] - pad, bbox_dict["lon_max"] + pad,
        bbox_dict["lat_min"] - pad, bbox_dict["lat_max"] + pad
    )

    times = pd.date_range(args.start, args.end, freq="1h")
    
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()

    print("\n--- Pre-fetching RAVE Data ---")
    rave_fetcher.prefetch(args.start, args.end) # Internalizes the file map

    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)
    weights_file = temp_dir / "weights_temp.nc"
    saved_files = []

    for t in times:
        print(f"\n--- Processing {t} ---")
        
        try:
            # 1. Fetch HRRR using Base interface
            hrrr_ds = hrrr_fetcher.process(t, t, bbox=bbox_tuple)
            if hrrr_ds is None: continue
            if "latitude" in hrrr_ds:
                hrrr_ds = hrrr_ds.rename({"latitude": "lat", "longitude": "lon"})

            # 2. Fetch RAVE using Base interface
            rave_ds = rave_fetcher.process(t, t, bbox=bbox_tuple)
            if rave_ds is None: continue
            if "grid_latt" in rave_ds:
                rave_ds = rave_ds.rename({"grid_latt": "lat", "grid_lont": "lon"})

            # 3. Clean up RAVE vars
            rave_subset = xr.Dataset()
            if "FRP_MEAN" in rave_ds:
                rave_subset["rave_frp"] = rave_ds["FRP_MEAN"].fillna(0.0)
            
            rave_subset = rave_subset.assign_coords({"lat": rave_ds.lat, "lon": rave_ds.lon})
            
            # 4. Regrid
            rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_ds, weights_path=weights_file, method="bilinear")

            # 5. Merge and Save
            merged = xr.merge([hrrr_ds, rave_rg], compat="override")
            out_path = temp_dir / f"merged_{t.strftime('%Y%m%d%H')}.nc"
            merged.to_netcdf(out_path)
            saved_files.append(out_path)
            
            hrrr_ds.close()
            rave_ds.close()
            merged.close()

        except Exception as e:
            print(f"Skipping {t} due to error: {e}")
            continue

    if not saved_files:
        print("\nNo files were processed successfully. Exiting.")
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return

    print("\nCombining all processed hours into final dataset...")
    datasets = [xr.open_dataset(p).load() for p in saved_files]

    if datasets:
        combined = xr.concat(datasets, dim="time")
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        final_out_path = data_dir / "combined_final.nc"
        combined.to_netcdf(final_out_path)
        print("Done! Data saved to combined_final.nc")
    
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()