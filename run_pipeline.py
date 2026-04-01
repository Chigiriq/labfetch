import argparse
import pandas as pd
import numpy as np
import xarray as xr
import shutil
import sys
import os
from pathlib import Path

# --- DISABLE PYCACHE GENERATION ---
sys.dont_write_bytecode = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fetchers.hrrr_fetcher import HRRRFetcher
from fetchers.rave_fetcher import RAVEFetcher
from fetchers.wfigs_fetcher import WFIGSFetcher 
from processors.grid import regrid_rave_to_hrrr

def main():
    parser = argparse.ArgumentParser(description="LabFetch Wildfire Data Pipeline")
    
    # Required Time Range
    parser.add_argument("--start", required=True, help="Start time: YYYY-MM-DD HH:MM")
    parser.add_argument("--end", required=True, help="End time: YYYY-MM-DD HH:MM")
    
    # Padding Controls
    parser.add_argument("--spatial_pad", type=float, default=0.5, 
                        help="Base spatial padding in degrees (default: 0.5)")
    parser.add_argument("--time_pad", type=int, default=24, 
                        help="Temporal padding in hours around fire discovery/containment (default: 24)")
    
    # Manual Override
    parser.add_argument("--bbox", type=str, default=None, 
                        help="Manual BBox override: min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--fire_id", type=str, default="manual_fetch", 
                        help="ID/Group name for manual bbox fetch")

    args = parser.parse_args()

    # --- Initialize Storage ---
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "download_log.txt"
    zarr_path = data_dir / "all_fires_combined.zarr"
    
    # PERSISTENT WEIGHTS: Stored in data/weights so they survive the hourly loop
    weights_dir = data_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # TEMP PROCESSING: For files that truly only need to exist for one hour
    temp_dir = Path("temp_processing")
    temp_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n--- Pipeline Run: {args.start} to {args.end} ---\n")

    # --- Initialize Fetchers ---
    wfigs_fetcher = WFIGSFetcher()
    hrrr_fetcher = HRRRFetcher()
    rave_fetcher = RAVEFetcher()

    fire_tasks = []

    # --- STEP 1: Determine Mode (Manual BBox vs WFIGS Discovery) ---
    if args.bbox:
        print(f"\n=== MANUAL BBOX MODE: {args.fire_id} ===")
        try:
            coords = [float(x.strip()) for x in args.bbox.split(",")]
            if len(coords) != 4:
                raise ValueError("BBox must have 4 coordinates.")
            
            lon_min, lat_min, lon_max, lat_max = coords
            bbox_tuple = (
                lon_min - args.spatial_pad, 
                lon_max + args.spatial_pad, 
                lat_min - args.spatial_pad, 
                lat_max + args.spatial_pad
            )
            
            fire_tasks.append({
                "fire_id": args.fire_id,
                "name": "Manual Override",
                "start": args.start,
                "end": args.end,
                "acres": 0,
                "bbox": bbox_tuple
            })
        except Exception as e:
            print(f"Error parsing --bbox: {e}")
            return
    else:
        print("\n=== STEP 1: Identifying Fires via WFIGS ===")
        wfigs_gdf = wfigs_fetcher.process(args.start, args.end)
        if wfigs_gdf is None or wfigs_gdf.empty:
            print("No fires found in WFIGS for this range.")
            return

        fire_tasks = wfigs_fetcher.generate_fire_tasks(wfigs_gdf, base_pad=args.spatial_pad)
        print(f"Found {len(fire_tasks)} fires to process.")

    # --- STEP 2: Pre-fetching RAVE ---
    print("\n=== STEP 2: Pre-fetching RAVE Data ===")
    rave_fetcher.prefetch(args.start, args.end)
    
    fire_initialized = {task["fire_id"]: False for task in fire_tasks}
    times = pd.date_range(args.start, args.end, freq="1h")

    # --- STEP 3: Processing Loop ---
    for t in times:
        print(f"\n--- Processing Timestep: {t} ---")
        t_pd = pd.to_datetime(t)
        
        hrrr_conus = hrrr_fetcher.process(t, t, bbox=None)
        rave_conus = rave_fetcher.process(t, t, bbox=None)
            
        if hrrr_conus is None or rave_conus is None:
            print(f"  -> Skipping {t}: Missing HRRR or RAVE data.")
            continue

        for task in fire_tasks:
            fire_id = task["fire_id"]
            fire_start = pd.to_datetime(task["start"]) - pd.Timedelta(hours=args.time_pad)
            
            if pd.notna(task["end"]):
                fire_end = pd.to_datetime(task["end"]) + pd.Timedelta(hours=args.time_pad)
            else:
                fire_end = pd.to_datetime(args.end)
            
            if not (fire_start <= t_pd <= fire_end):
                continue
            
            try:
                # Spatial Subset
                hrrr_clip = hrrr_fetcher._spatial_subset(hrrr_conus, *task["bbox"])
                rave_clip = rave_fetcher._spatial_subset(rave_conus, *task["bbox"])
                
                if hrrr_clip is None or rave_clip is None: continue

                # Standardization
                hrrr_clip = hrrr_clip.rename({"latitude": "lat", "longitude": "lon"})
                rave_clip = rave_clip.rename({"grid_latt": "lat", "grid_lont": "lon"})

                # Extract FRP
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

                # --- WEIGHT REUSE IMPLEMENTATION ---
                # Unique weight file per fire task. 
                # grid.py will check if this exists and reuse it automatically.
                weights_file = weights_dir / f"weights_{fire_id}.nc"
                
                rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_clip, weights_path=weights_file)

                # Merge and add metadata
                merged = xr.merge([hrrr_clip, rave_rg], compat="override")
                if "time" not in merged.dims: 
                    merged = merged.expand_dims("time")
                
                merged.attrs["Fire_Name"] = task["name"]
                merged.attrs["Fire_Acres"] = task["acres"]

                # Write to Zarr
                if not fire_initialized[fire_id]:
                    merged.to_zarr(zarr_path, group=fire_id, mode="a", consolidated=False)
                    fire_initialized[fire_id] = True
                else:
                    merged.to_zarr(zarr_path, group=fire_id, append_dim="time", consolidated=False)

                # Cleanup per-task datasets
                hrrr_clip.close(); rave_clip.close(); merged.close()

            except Exception as e:
                print(f"  -> Error processing {task['name']} at {t}: {e}")

    # --- STEP 4: Cleanup ---
    print("\n=== STEP 4: Cleaning up local raw data ===")
    # Note: We keep weights_dir intact in case you run the script again for overlapping ranges, 
    # but you can add it to this list if you want a totally fresh start each run.
    for d in [data_dir / "hrrr", data_dir / "rave", temp_dir]:
        if d.exists(): shutil.rmtree(d)
        
    print(f"\nPipeline Complete! Data saved to: {zarr_path}")

if __name__ == "__main__":
    main()