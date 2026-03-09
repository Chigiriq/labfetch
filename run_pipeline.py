# --- DISABLE PYCACHE GENERATION ---
import sys
import os
sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------

import argparse
import pandas as pd
import xarray as xr
import shutil
from pathlib import Path

from fetchers.hrrr_fetcher import HRRRFetcher
from fetchers.rave_fetcher import RAVEFetcher
from fetchers.wfigs_fetcher import WFIGSFetcher 
from processors.grid import regrid_rave_to_hrrr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    # 1. Initialize Fetchers
    wfigs_fetcher = WFIGSFetcher()
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()

    print("\n=== STEP 1: Identifying Fires via WFIGS ===")
    wfigs_gdf = wfigs_fetcher.process(args.start, args.end)
    
    if wfigs_gdf is None:
        print("No fires found matching criteria. Exiting.")
        return

    fire_tasks = wfigs_fetcher.generate_fire_tasks(wfigs_gdf, pad=0.5)
    print(f"Found {len(fire_tasks)} fires to process.")

    print("\n=== STEP 2: Pre-fetching RAVE Data ===")
    rave_fetcher.prefetch(args.start, args.end)

    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    times = pd.date_range(args.start, args.end, freq="1h")

    # === STEP 3: Time on the OUTSIDE, Fires on the INSIDE ===
    for t in times:
        print(f"\n=== Fetching Global/CONUS Data for {t} ===")
        
        # Fetch the entire map exactly ONCE per hour
        try:
            hrrr_conus = hrrr_fetcher.process(t, t, bbox=None)
            rave_conus = rave_fetcher.process(t, t, bbox=None)
        except Exception as e:
            print(f"Failed to fetch global data for {t}: {e}")
            continue
            
        if hrrr_conus is None or rave_conus is None:
            print(f"Skipping {t}: Missing HRRR or RAVE global data.")
            continue

        for task in fire_tasks:
            fire_id = task["fire_id"]
            bbox = task["bbox"]
            
            try:
                # Clip the specific fire out of the global memory instantly
                hrrr_clip = hrrr_fetcher._spatial_subset(hrrr_conus, *bbox)
                rave_clip = rave_fetcher._spatial_subset(rave_conus, *bbox)
                
                if hrrr_clip is None or rave_clip is None:
                    # Skips if fire geometry completely misses the datasets
                    continue

                # Standardize Coordinates
                hrrr_clip = hrrr_clip.rename({"latitude": "lat", "longitude": "lon"})
                rave_clip = rave_clip.rename({"grid_latt": "lat", "grid_lont": "lon"})

                rave_subset = xr.Dataset()
                if "FRP_MEAN" in rave_clip:
                    rave_subset["rave_frp"] = rave_clip["FRP_MEAN"].fillna(0.0)
                rave_subset = rave_subset.assign_coords({"lat": rave_clip.lat, "lon": rave_clip.lon})

                # Regrid & Merge
                weights_file = temp_dir / f"weights_{fire_id}.nc"
                rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_clip, weights_path=weights_file, method="bilinear")

                merged = xr.merge([hrrr_clip, rave_rg], compat="override")
                
                # Ensure time dim exists for later concatenation
                if "time" not in merged.dims:
                    merged = merged.expand_dims("time")
                    
                out_path = temp_dir / f"{fire_id}_{t.strftime('%Y%m%d%H')}.nc"
                merged.to_netcdf(out_path)
                
                hrrr_clip.close()
                rave_clip.close()
                merged.close()

            except Exception as e:
                print(f"  -> Error processing {task['name']} at {t}: {e}")
                continue

    # === STEP 4: Assembling the Final Zarr Store ===
    print("\n=== STEP 4: Assembling Final Zarr Store ===")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = data_dir / "all_fires_combined.zarr"

    processed_fires = 0
    for task in fire_tasks:
        fire_id = task["fire_id"]
        # Grab all temporary hourly files for this specific fire
        files = sorted(temp_dir.glob(f"{fire_id}_*.nc"))
        
        if not files:
            continue
            
        print(f"Writing {task['name']} to Zarr group...")
        datasets = [xr.open_dataset(p).load() for p in files]
        combined = xr.concat(datasets, dim="time")
        
        # Attach fire metadata 
        combined.attrs["Fire_Name"] = task["name"]
        combined.attrs["Fire_Acres"] = task["acres"]
        
        # Save to a specific hierarchical group inside the Zarr store
        combined.to_zarr(zarr_path, group=fire_id, mode="a")
        processed_fires += 1

    print(f"\nPipeline Complete! {processed_fires} fires safely stored in {zarr_path}")
    if temp_dir.exists(): shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()