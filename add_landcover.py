import os
import zarr
import xarray as xr
import rioxarray
import xesmf as xe
import pandas as pd
from pathlib import Path

from processors.grid import silence_c_errors

def get_tif_for_fire(start_time: pd.Timestamp) -> str:
    """Returns the correct TIFF file name based on the fire's timeframe."""
    if start_time.year == 2024:
        return "Dynamic_World_US_3km_Late_2024.tif"
    elif start_time.year == 2025:
        return "Dynamic_World_US_3km_Year_2025.tif"
    else:
        return "Dynamic_World_US_3km_Year_2026.tif"

def main():
    zarr_path = "data/master_wildfire_db.zarr"
    tif_directory = "data/landcover"
    
    if not os.path.exists(zarr_path):
        print("Zarr database not found.")
        return

    zstore = zarr.open(zarr_path, mode='a')
    groups = list(zstore.group_keys())
    
    print(f"Found {len(groups)} fires in the database. Appending Landcover data...")

    for fid in groups:
        try:
            # 1. Open the existing fire dataset
            ds = xr.open_zarr(zarr_path, group=fid, consolidated=False)
            
            if "landcover" in ds.data_vars:
                print(f"[{fid}] Landcover already exists, skipping.")
                continue

            # 2. Figure out which Dynamic World file to use based on the fire's start date
            fire_start = pd.to_datetime(ds.time.min().values)
            tif_name = get_tif_for_fire(fire_start)
            tif_path = Path(tif_directory) / tif_name
            
            if not tif_path.exists():
                print(f"[{fid}] Missing {tif_name}. Skipping.")
                continue

            # 3. Load the Dynamic World GeoTIFF using rioxarray
            dw_da = rioxarray.open_rasterio(tif_path)
            dw_ds = (
                dw_da.squeeze("band", drop=True)
                .rename({"x": "lon", "y": "lat"})
                .to_dataset(name="landcover")
            )

            # 4. Create Regridder
            weights_path = f"data/temp_weights/dw_weights_{fid}.nc"
            os.makedirs("data/temp_weights", exist_ok=True)
            
            with silence_c_errors():
                regridder = xe.Regridder(
                    dw_ds, 
                    ds, 
                    "nearest_s2d", #switching to nearest_s2d to preserve classifications
                    filename=weights_path,
                    reuse_weights=False
                )
                
            # 5. Execute Regridding
            dw_regridded = regridder(dw_ds)

            dw_regridded = dw_regridded.assign_coords({"lat": ds.lat, "lon": ds.lon})

            # 6. Append dynamically to the Zarr file
            dw_regridded.to_zarr(zarr_path, group=fid, mode="a", consolidated=False)
            print(f"[{fid}] Successfully appended Dynamic World Landcover ({tif_name}).")
            
            # Cleanup Regridder memory
            del regridder
            if os.path.exists(weights_path):
                os.remove(weights_path)

        except Exception as e:
            print(f"[{fid}] Error processing landcover: {e}")

if __name__ == "__main__":
    main()