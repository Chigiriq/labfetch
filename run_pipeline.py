import sys
import os
# --- SILENCE HDF5 C-LIBRARY WARNINGS ---
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
import zarr

from pathlib import Path
from shapely.geometry import box 

# --- PROJECT SETUP ---
sys.dont_write_bytecode = True
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fetchers.hrrr_fetcher import HRRRFetcher
from fetchers.rave_fetcher import RAVEFetcher
from fetchers.wfigs_fetcher import WFIGSFetcher
from fetchers.dw_fetcher import DWFetcher
from processors.grid import regrid_rave_to_hrrr, regrid_categorical_to_hrrr

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
    parser.add_argument("--zarr_store", type=str, default=None, help="Specific Zarr store to append to. If not provided, creates a new one.")
    parser.add_argument("--ongoing_days", type=int, default=14, help="Default duration in days to assign to ongoing fires with no end date.")
    
    args = parser.parse_args()

    # Data Output naming
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    
    # Default to a master database name instead of time-bound names
    zarr_name = args.zarr_store if args.zarr_store else "master_wildfire_db.zarr"

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
    
    # Check existing zarr groups for appending and their state
    existing_groups = []
    fire_state = {}

    if zarr_path.exists():
        try:
            zstore = zarr.open(zarr_path, mode='r')
            existing_groups = list(zstore.group_keys())
            logger.info(f"Found existing Zarr store with {len(existing_groups)} fires. Reading temporal states to prevent duplicates...")
            
            for fid in existing_groups:
                try:
                    ds_existing = xr.open_zarr(zarr_path, group=fid, consolidated=False)
                    fire_state[fid] = {
                        "times": pd.DatetimeIndex(ds_existing.time.values),
                        "vars": set(ds_existing.data_vars.keys())
                    }
                    ds_existing.close()
                except Exception as e:
                    logger.warning(f"Could not read state for {fid}: {e}")
                    fire_state[fid] = {"times": pd.DatetimeIndex([]), "vars": set()}
                    
        except Exception as e:
            logger.warning(f"Could not read existing Zarr store: {e}")

    wfigs_fetcher = WFIGSFetcher()
    hrrr_fetcher = HRRRFetcher(save_dir=hrrr_dir)
    rave_fetcher = RAVEFetcher(save_dir=rave_dir)
    dw_fetcher = DWFetcher()

    # --- STEP 1: Discovery & Validation ---
    valid_tasks = []
    incomplete_summary = []

    logger.info("Querying WFIGS for fire incidents...")
    wfigs_gdf = wfigs_fetcher.process(args.start, args.end)
    raw_tasks = []

    if args.bbox:
        logger.info("Manual BBox Mode: Intersecting with WFIGS...")
        coords = [float(x.strip()) for x in args.bbox.split(",")]
        bbox_tuple = (
            coords[2] - args.spatial_pad,  # lon_min
            coords[3] + args.spatial_pad,  # lon_max
            coords[0] - args.spatial_pad,  # lat_min
            coords[1] + args.spatial_pad   # lat_max
        )

        user_poly = box(bbox_tuple[0], bbox_tuple[2], bbox_tuple[1], bbox_tuple[3])
        
        if wfigs_gdf is not None and not wfigs_gdf.empty:
            intersecting_fires = wfigs_gdf[wfigs_gdf.intersects(user_poly)]
            if not intersecting_fires.empty:
                base_tasks = wfigs_fetcher.generate_fire_tasks(intersecting_fires, base_pad=0)
                for t in base_tasks:
                    t["bbox"] = bbox_tuple 
                    t["fire_id"] = f"{t['fire_id']}_custom_cut"
                    t["name"] = f"{t['name']} (Custom BBox)"
                    raw_tasks.append(t)

        if not raw_tasks:
            logger.warning("No fires found in provided bounding box. Creating fallback task to fetch HRRR.")
            raw_tasks = [{
                "fire_id": f"{args.fire_id}_no_wfigs",
                "name": "Manual Override (No WFIGS Fire)",
                "start": args.start,
                "end": args.end,
                "acres": 0,
                "bbox": bbox_tuple,
                "missing_wfigs": True 
            }]
    else:
        if wfigs_gdf is None or wfigs_gdf.empty: 
            logger.warning("No fires found in WFIGS for this range. Exiting.")
            return
        raw_tasks = wfigs_fetcher.generate_fire_tasks(wfigs_gdf, base_pad=args.spatial_pad)
        

    for task in raw_tasks:
        # Handle 'ongoing' fires with no specified end date
        if task.get("end") is None or pd.isna(task.get("end")):
            task["end"] = pd.to_datetime(task["start"]) + pd.Timedelta(days=args.ongoing_days)
            task["ongoing_capped"] = True

        f_start = pd.to_datetime(task["start"]) - pd.Timedelta(hours=args.time_pad)
        f_end = pd.to_datetime(task["end"]) + pd.Timedelta(hours=args.time_pad)

        # Check if requested pipeline range fully encapsulates the fire
        is_fully_contained = (start_dt <= f_start) and (end_dt >= f_end)
        
        if is_fully_contained:
            task["temporal_clip_status"] = "FULLY_CONTAINED"
        else:
            task["temporal_clip_status"] = "CLIPPED"
            start_str = pd.to_datetime(task["start"]).strftime('%Y-%m-%d %H:%M') if pd.notnull(task.get("start")) else "Unknown"

            # Modify existing loggin to show if artificial cap exits
            end_val = pd.to_datetime(task["end"]).strftime('%Y-%m-%d %H:%M')
            end_str = f"{end_val} (Ongoing Capped)" if task.get("ongoing_capped") else end_val

            incomplete_summary.append(f"{task['fire_id']} (Clipped: Fire active {start_str} to {end_str})")

        # ALWAYS append to valid_tasks, no skipping
        valid_tasks.append(task)

    logger.info(f"{len(valid_tasks)} total tasks queued for processing.")

    if valid_tasks:
        logger.info("--- FIRE IDs QUEUED ---")
        for vt in valid_tasks:
            status = "" if vt["temporal_clip_status"] == "FULLY_CONTAINED" else "[CLIPPED]"
            logger.info(f"   + {vt['fire_id']} ({vt['name']}) {status}")
    
    if incomplete_summary:
        logger.info("--- INCOMPLETE TIMEFRAME SUMMARY ---")
        for inc in incomplete_summary:
            logger.info(f"   ~ {inc}")

    if not valid_tasks: 
        logger.warning("No valid tasks remain. Exiting.")
        return
        
    fire_tasks = valid_tasks
    
    # Dynamic Wolrd
    logger.info("Initializing static pre-fire features (Dynamic World)...")
    template_hrrr = hrrr_fetcher.process(args.start, args.start, bbox=None)

    for task in fire_tasks:
        fid = task["fire_id"]
        state = fire_state.get(fid, {"times": pd.DatetimeIndex([]), "vars": set()})
        
        if "dw_landcover" not in state["vars"] and not task.get("missing_wfigs"):
            logger.info(f"[{fid}] Fetching pre-fire Dynamic World landcover...")
            try:
                dw_ds = dw_fetcher.process(task["start"], task["start"], bbox=task["bbox"])
                
                if dw_ds is not None and template_hrrr is not None:
                    hrrr_target = hrrr_fetcher._spatial_subset(template_hrrr, *task["bbox"])
                    hrrr_target = hrrr_target.rename({"latitude": "lat", "longitude": "lon"})
                    
                    weights_file = weights_dir / f"weights_dw_{fid}.nc"
                    dw_rg = regrid_categorical_to_hrrr(dw_ds, hrrr_target, weights_path=weights_file)
                    
                    if "time" in dw_rg.dims:
                        dw_rg = dw_rg.isel(time=0).drop_vars("time")
                        
                    mode = "a" if fid in existing_groups else "w"
                    dw_rg.to_zarr(
                        zarr_path, 
                        group=fid, 
                        mode=mode, 
                        append_dim=None, 
                        consolidated=False
                    )
                    
                    if fid not in existing_groups:
                        existing_groups.append(fid)
                    state["vars"].add("dw_landcover")
                    fire_state[fid] = state
                    logger.info(f"[{fid}] Dynamic World written to Zarr.")
                    
            except Exception as e:
                logger.error(f"[{fid}] Failed to process Dynamic World: {e}")

    if template_hrrr is not None:
        template_hrrr.close()
        hrrr_fetcher.cleanup_timestamp(args.start)

    # Fire is already in the Zarr store: append, Else: write a new group
    fire_initialized = {task["fire_id"]: (task["fire_id"] in existing_groups) for task in fire_tasks}
    times = pd.date_range(args.start, args.end, freq="1h")

    # --- Only download new data ---
    times_to_process = []
    for t in times:
        needs_processing = False
        for task in fire_tasks:
            fid = task["fire_id"]
            state = fire_state.get(fid, {"times": pd.DatetimeIndex([]), "vars": set()})
            if t not in state["times"]:
                needs_processing = True
                break
        
        if needs_processing:
            times_to_process.append(t)
            
    times = pd.DatetimeIndex(times_to_process)
    
    if len(times) == 0:
        logger.info("All requested hours already exist for all fires. Exiting.")
        return

    # --- STEP 2: Rave Prefetch ---
    # rave_fetcher.prefetch(args.start, args.end)
    logger.info(f"Prefetching RAVE data for missing range: {times[0]} to {times[-1]}")
    rave_fetcher.prefetch(times[0], times[-1])

    # --- STEP 3: Multithreaded Proccess Loop ---
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future_fetch = executor.submit(fetch_data_task, hrrr_fetcher, rave_fetcher, times[0])

    for i, t in enumerate(times):
        logger.info(f"Processing hour {i+1}/{len(times)}: {t}")
        
        hrrr_conus, rave_conus = future_fetch.result()
        
        if i + 1 < len(times):
            future_fetch = executor.submit(fetch_data_task, hrrr_fetcher, rave_fetcher, times[i+1])

        if hrrr_conus is None: 
            continue

        for task in fire_tasks:           
            try:
                fid = task["fire_id"]
                state = fire_state.get(fid, {"times": pd.DatetimeIndex([]), "vars": set()})
                hour_exists = t in state["times"]

                hrrr_clip = hrrr_fetcher._spatial_subset(hrrr_conus, *task["bbox"])
                if hrrr_clip is None: 
                    continue

                rave_clip = None
                if rave_conus is not None:
                    try:
                        rave_clip = rave_fetcher._spatial_subset(rave_conus, *task["bbox"])
                    except Exception:
                        pass
                
                hrrr_clip = hrrr_clip.rename({"latitude": "lat", "longitude": "lon"})

                if rave_clip is not None:
                    weights_file = weights_dir / f"weights_{fid}.nc"
                    rave_clip = rave_clip.rename({"grid_latt": "lat", "grid_lont": "lon"})
                    
                    rave_subset = xr.Dataset({"rave_frp": rave_clip["FRP_MEAN"].fillna(0.0)})
                    rave_subset = rave_subset.assign_coords({"lat": rave_clip.lat, "lon": rave_clip.lon})

                    rave_rg = regrid_rave_to_hrrr(rave_subset, hrrr_clip, weights_path=weights_file)
                    merged = xr.merge([hrrr_clip, rave_rg], compat="override")
                else:
                    empty_frp = xr.DataArray(np.nan, coords={"y": hrrr_clip.y, "x": hrrr_clip.x}, dims=["y", "x"])
                    merged = hrrr_clip.assign(rave_frp=empty_frp)
                    merged.attrs["RAVE_STATUS"] = "ERROR_OR_MISSING_DATA"

                # Tag Temporal Status and WFIGS status directly into the Zarr metadata
                merged.attrs["TEMPORAL_CLIP_STATUS"] = task["temporal_clip_status"]
                
                if task.get("ongoing_capped"):
                    merged.attrs["ONGOING_STATUS"] = "ARTIFICIALLY_CAPPED"

                if task.get("missing_wfigs"):
                    merged.attrs["WFIGS_STATUS"] = "NO_FIRE_IN_BBOX"

                if "time" not in merged.dims: merged = merged.expand_dims("time")
                
                current_vars = set(merged.data_vars.keys())
                new_vars = current_vars - state["vars"]

                # --- Append Decision Logic ---
                if hour_exists and not new_vars:
                    # Hour exists and no new columns are being added. Safely skip to avoid duplicates.
                    merged.close()
                    continue
                    
                if hour_exists and new_vars:
                    # Hour exists, but a new column was detected. Drop old columns and append only the new variable.
                    logger.info(f"[{fid}] Appending new variable(s) {new_vars} to existing timestamp {t}")
                    vars_to_drop = current_vars.intersection(state["vars"])
                    merged = merged.drop_vars(vars_to_drop)
                    
                    mode = "a"
                    append_dim = None # Required for variable appending
                else:
                    # Standard Write or Temporal Append
                    mode = "a" if fire_initialized[fid] else "w"
                    append_dim = "time" if mode == "a" else None

                merged.to_zarr(
                    zarr_path, 
                    group=fid, 
                    mode=mode if mode == "w" else "a", 
                    append_dim=append_dim, 
                    consolidated=False
                )
                
                fire_initialized[fid] = True
                
                # Update Tracker
                state["times"] = state["times"].union([t])
                state["vars"].update(new_vars)
                fire_state[fid] = state
                
                merged.close()
                
            except Exception as e:
                logger.error(f"Error on {task['name']} at {t}: {e}")

        hrrr_conus.close()
        if rave_conus is not None: rave_conus.close()
        hrrr_fetcher.cleanup_timestamp(t)
        rave_fetcher.cleanup_timestamp(t)

    executor.shutdown()
    
    for d in [hrrr_dir, rave_dir, weights_dir]:
        if d.exists(): shutil.rmtree(d)
        
    logger.info(f"Batch Complete: {zarr_path}")

if __name__ == "__main__":
    main()