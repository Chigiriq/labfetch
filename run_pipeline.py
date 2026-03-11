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
import numpy as np
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

    # --- Initialize Storage & Logging ---
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "download_log.txt"
    zarr_path = data_dir / "all_fires_combined.zarr"
    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n--- Pipeline Run: {args.start} to {args.end} ---\n")

    # --- Fetchers ---
    wfigs_fetcher = WFIGSFetcher()
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()

    print("\n=== STEP 1: Identifying Fires via WFIGS ===")
    wfigs_gdf = wfigs_fetcher.process(args.start, args.end)
    if wfigs_gdf is None: return

    fire_tasks = wfigs_fetcher.generate_fire_tasks(wfigs_gdf, pad=0.5)
    print(f"Found {len(fire_tasks)} fires to process.")

    print("\n=== STEP 2: Pre-fetching RAVE Data ===")
    rave_fetcher.prefetch(args.start, args.end)
    
    fire_initialized = {task["fire_id"]: False for task in fire_tasks}
    times = pd.date_range(args.start, args.end, freq="1h")

    # --- Processing Loop ---
    for t in times:
        print(f"\n=== Processing {t} ===")
        t_pd = pd.to_datetime(t)
        
        try:
            hrrr_conus = hrrr_fetcher.process(t, t, bbox=None)
            rave_conus = rave_fetcher.process(t, t, bbox=None)
            
            # Log HRRR Fetch
            with open(log_path, "a") as f:
                f.write(f"HRRR FETCH: Searched AWS index for {t}\n")
                
        except Exception as e:
            print(f"Failed to fetch global data: {e}")
            continue
            
        if hrrr_conus is None or rave_conus is None: continue

        for task in fire_tasks:
            fire_id = task["fire_id"]
            
            # --- Time Buffering ---
            # Create a 24-hour buffer around the fire's specific start/end dates
            fire_start = pd.to_datetime(task["start"]) - pd.Timedelta(hours=24)
            
            if pd.notna(task["end"]): #required as many fires dont have a recorded end
                fire_end = pd.to_datetime(task["end"]) + pd.Timedelta(hours=24)
            else:
                fire_end = pd.to_datetime(args.end)
            
            if not (fire_start <= t_pd <= fire_end):
                continue
            
            try:
                hrrr_clip = hrrr_fetcher._spatial_subset(hrrr_conus, *task["bbox"])
                rave_clip = rave_fetcher._spatial_subset(rave_conus, *task["bbox"])
                if hrrr_clip is None or rave_clip is None: continue

                hrrr_clip = hrrr_clip.rename({"latitude": "lat", "longitude": "lon"})
                rave_clip = rave_clip.rename({"grid_latt": "lat", "grid_lont": "lon"})

                rave_subset = xr.Dataset()
                if "FRP_MEAN" in rave_clip:
                    rave_subset["rave_frp"] = rave_clip["FRP_MEAN"].fillna(0.0)
                else:
                    rave_subset["rave_frp"] = xr.DataArray(
                        np.zeros_like(rave_clip.lon.values),
                        coords={"lat": rave_clip.lat, "lon": rave_clip.lon},
                        dims=rave_clip.lon.dims
                    )
                rave_subset = rave_subset.assign_coords({"lat": rave_clip.lat, "lon": rave_clip.lon})

                weights_file = temp_dir / f"weights_{fire_id}.nc"
                rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_clip, weights_path=weights_file, method="bilinear")

                merged = xr.merge([hrrr_clip, rave_rg], compat="override")
                if "time" not in merged.dims: merged = merged.expand_dims("time")
                
                merged.attrs["Fire_Name"] = task["name"]
                merged.attrs["Fire_Acres"] = task["acres"]

                # --- Adding to Zarr ---
                if not fire_initialized[fire_id]:
                    merged.to_zarr(
                        zarr_path, 
                        group=fire_id, 
                        mode="a", 
                        consolidated=False,
                        zarr_format=2  # V3 apparently blows up with other geo packages so staying on v2 for now
                    )
                    fire_initialized[fire_id] = True
                else:
                    merged.to_zarr(
                        zarr_path, 
                        group=fire_id, 
                        append_dim="time", 
                        consolidated=False,
                        zarr_format=2  # see above
                    )

                hrrr_clip.close()
                rave_clip.close()
                merged.close()

            except Exception as e:
                print(f"  -> Error processing {task['name']}: {e}")
                continue

    # --- Remove installed local files ---
    print("\n=== STEP 4: Cleaning up local raw data ===")
    hrrr_dir = data_dir / "hrrr"
    rave_dir = data_dir / "rave"
    
    if hrrr_dir.exists(): shutil.rmtree(hrrr_dir)
    if rave_dir.exists(): shutil.rmtree(rave_dir)
    if temp_dir.exists(): shutil.rmtree(temp_dir)
        
    print(f"\nPipeline Complete! Data safely streamed to {zarr_path}")
    print(f"Raw data deleted to save space. Log saved to {log_path}")

if __name__ == "__main__":
    main()