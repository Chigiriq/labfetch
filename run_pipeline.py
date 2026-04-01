import sys
import os
# --- SILENCE HDF5 C-LIBRARY WARNINGS ---
# has to go here before other packages load
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_PRINT_ERRORS"] = "FALSE"

import argparse
import pandas as pd
import numpy as np
import xarray as xr
import shutil

import yaml
import logging
import concurrent.futures
import time
from pathlib import Path



# --- PROJECT SETUP ---
sys.dont_write_bytecode = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fetchers.hrrr_fetcher import HRRRFetcher
from fetchers.rave_fetcher import RAVEFetcher
from fetchers.wfigs_fetcher import WFIGSFetcher 
from processors.grid import regrid_rave_to_hrrr

def setup_logging(log_path):
    """Industry standard logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("LabFetch")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_data_task(hrrr_fetcher, rave_fetcher, timestamp):
    """Worker function to download data in the background."""
    try:
        # These calls trigger the internal download/caching logic for one specific hour
        h = hrrr_fetcher.process(timestamp, timestamp, bbox=None)
        r = rave_fetcher.process(timestamp, timestamp, bbox=None)
        return h, r
    except Exception:
        return None, None

def main():
    config = load_config()
    conf_paths = config['paths']
    conf_defaults = config['pipeline_defaults']

    parser = argparse.ArgumentParser(description="LabFetch Wildfire Data Pipeline")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data_root", default=conf_paths['data_root'])
    parser.add_argument("--spatial_pad", type=float, default=conf_defaults['spatial_pad'])
    parser.add_argument("--time_pad", type=int, default=conf_defaults['time_pad'])
    parser.add_argument("--bbox", type=str, default=None)
    parser.add_argument("--fire_id", type=str, default="manual_fetch")
    args = parser.parse_args()

    # Dynamic Filenaming
    fmt = "%Y%m%d_%H%M"
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    zarr_name = f"{start_dt.strftime(fmt)}-{end_dt.strftime(fmt)}_WF.zarr"

    # Path Init
    root = Path(args.data_root)
    hrrr_dir = root / "raw_hrrr"
    rave_dir = root / "raw_rave"
    weights_dir = root / "temp_weights"
    zarr_path = root / zarr_name
    log_path = root / "pipeline.log"
    
    for d in [root, hrrr_dir, rave_dir, weights_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path)
    
    wfigs_fetcher = WFIGSFetcher()
    hrrr_fetcher = HRRRFetcher(save_dir=hrrr_dir)
    rave_fetcher = RAVEFetcher(save_dir=rave_dir)

    # --- STEP 1: DISCOVERY & VALIDATION ---
    valid_tasks = []
    invalid_summary = []

    if args.bbox:
        logger.info(f"Manual BBox Mode: {args.fire_id}")
        coords = [float(x.strip()) for x in args.bbox.split(",")]
        bbox_tuple = (coords[0]-args.spatial_pad, coords[2]+args.spatial_pad, 
                      coords[1]-args.spatial_pad, coords[3]+args.spatial_pad)
        valid_tasks = [{"fire_id": args.fire_id, "name": "Manual", "start": args.start, "end": args.end, "acres": 0, "bbox": bbox_tuple}]
    else:
        logger.info("Querying WFIGS for fire incidents...")
        wfigs_gdf = wfigs_fetcher.process(args.start, args.end)
        if wfigs_gdf is None or wfigs_gdf.empty: 
            logger.warning("No fires found in WFIGS for this range. Exiting.")
            return
            
        raw_tasks = wfigs_fetcher.generate_fire_tasks(wfigs_gdf, base_pad=args.spatial_pad)
        
        for task in raw_tasks:
            f_start = pd.to_datetime(task["start"]) - pd.Timedelta(hours=args.time_pad)
            
            # Handle open-ended fires safely
            if task.get("end") is None or pd.isna(task.get("end")):
                f_end = end_dt + pd.Timedelta(hours=args.time_pad)
            else:
                f_end = pd.to_datetime(task["end"]) + pd.Timedelta(hours=args.time_pad)

            # Validate overlap
            if (f_start <= end_dt) and (f_end >= start_dt):
                valid_tasks.append(task)
            else:
                invalid_summary.append(f"{task['fire_id']} (Outside temporal window)")

        # --- The Restored Logging Block ---
        logger.info(f"{len(raw_tasks)} fires found - {len(valid_tasks)} valid")

        if valid_tasks:
            logger.info("--- VALID FIRE IDs ---")
            for vt in valid_tasks:
                logger.info(f"   + {vt['fire_id']} ({vt['name']})")
        
        if invalid_summary:
            logger.info("--- INVALID/SKIPPED FIRE IDs ---")
            for inv in invalid_summary:
                logger.info(f"   - {inv}")

    if not valid_tasks: 
        logger.warning("No valid tasks remain after filtering. Exiting.")
        return
        
    fire_tasks = valid_tasks

    # --- STEP 2: RAVE PREFETCH ---
    rave_fetcher.prefetch(args.start, args.end)
    
    fire_initialized = {task["fire_id"]: False for task in fire_tasks}
    times = pd.date_range(args.start, args.end, freq="1h")

    # --- STEP 3: MULTITHREADED PROCESSING LOOP ---
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    # Pre-fetch the FIRST hour
    future_fetch = executor.submit(fetch_data_task, hrrr_fetcher, rave_fetcher, times[0])

    for i, t in enumerate(times):
        logger.info(f"Processing hour {i+1}/{len(times)}: {t}")
        
        # Get data from background thread (blocks until download finishes)
        hrrr_conus, rave_conus = future_fetch.result()
        
        # Immediately start fetching the NEXT hour
        if i + 1 < len(times):
            future_fetch = executor.submit(fetch_data_task, hrrr_fetcher, rave_fetcher, times[i+1])

        if hrrr_conus is None or rave_conus is None:
            continue

        for task in fire_tasks:
            # --- FIXED: Use the safe temporal check for open-ended fires ---
            f_start = pd.to_datetime(task["start"]) - pd.Timedelta(hours=args.time_pad)
            
            if task.get("end") is None or pd.isna(task.get("end")):
                f_end = end_dt + pd.Timedelta(hours=args.time_pad)
            else:
                f_end = pd.to_datetime(task["end"]) + pd.Timedelta(hours=args.time_pad)

            if not (f_start <= t <= f_end): 
                continue
            # ---------------------------------------------------------------
            
            try:
                hrrr_clip = hrrr_fetcher._spatial_subset(hrrr_conus, *task["bbox"])
                rave_clip = rave_fetcher._spatial_subset(rave_conus, *task["bbox"])
                
                # --- FIXED: Safety check in case the fire is outside the satellite grid ---
                if hrrr_clip is None or rave_clip is None: 
                    continue
                
                weights_file = weights_dir / f"weights_{task['fire_id']}.nc"
                hrrr_clip = hrrr_clip.rename({"latitude": "lat", "longitude": "lon"})
                rave_clip = rave_clip.rename({"grid_latt": "lat", "grid_lont": "lon"})
                
                rave_subset = xr.Dataset({"rave_frp": rave_clip["FRP_MEAN"].fillna(0.0)})
                rave_subset = rave_subset.assign_coords({"lat": rave_clip.lat, "lon": rave_clip.lon})

                rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_clip, weights_path=weights_file)
                merged = xr.merge([hrrr_clip, rave_rg], compat="override")
                if "time" not in merged.dims: merged = merged.expand_dims("time")
                
                mode = "a" if fire_initialized[task["fire_id"]] else "w"
                merged.to_zarr(zarr_path, group=task["fire_id"], mode=mode if mode == "w" else None, 
                               append_dim="time" if mode == "a" else None, consolidated=False)
                fire_initialized[task["fire_id"]] = True
                merged.close()
            except Exception as e:
                logger.error(f"Error on {task['name']} at {t}: {e}")

        # CLEANUP CURRENT HOUR FROM DISK
        hrrr_conus.close(); rave_conus.close()
        hrrr_fetcher.cleanup_timestamp(t)
        rave_fetcher.cleanup_timestamp(t)

    executor.shutdown()
    
    # Final Cleanup
    for d in [hrrr_dir, rave_dir, weights_dir]:
        if d.exists(): shutil.rmtree(d)
        
    logger.info(f"Batch Complete: {zarr_path}")

if __name__ == "__main__":
    main()