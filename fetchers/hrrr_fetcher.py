import pandas as pd
import xarray as xr
import numpy as np
import warnings
import os

from pathlib import Path
from herbie import Herbie

from .base_fetcher import BaseFetcher

# ---- xarray config ----
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

class HRRRFetcher(BaseFetcher):
    DEFAULT_VARS = [
        ":TMP:2 m", 
        ":DPT:2 m", 
        ":UGRD:10 m", 
        ":VGRD:10 m",
        ":PBLH:", 
        ":PRES:surface:", 
        ":HGT:surface:",
    ]

    def __init__(self, model="hrrr", product="sfc", save_dir=None):
        super().__init__(source_name="HRRR")
        self.model = model
        self.product = product
        self.save_dir = Path(save_dir) if save_dir else DATA_ROOT / "hrrr"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _spatial_subset(ds, lon_min, lon_max, lat_min, lat_max):
        if ds.longitude.max() > 180:
            lon_min %= 360
            lon_max %= 360

        mask = (
            (ds.longitude >= lon_min) & (ds.longitude <= lon_max) & 
            (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
        ).compute()

        y, x = np.where(mask.values)
        if len(y) == 0:
            raise ValueError("BBox does not intersect HRRR grid.")

        return ds.isel(y=slice(y.min(), y.max() + 1), x=slice(x.min(), x.max() + 1))

    def fetch_data(self, start_time, end_time, bbox=None, variable=None):
        times = pd.date_range(start_time, end_time, freq="1h")
        combined = None

        for t in times:
            H = Herbie(t, model=self.model, product=self.product, save_dir=str(self.save_dir))
            search = "|".join(self.DEFAULT_VARS) if variable is None else variable
            
            try:
                ds_list = H.xarray(search=search)
            except Exception as e:
                print(f"  -> Fetch failed: {e}")
                continue

            if not isinstance(ds_list, list): ds_list = [ds_list]

            try:
                ds = xr.merge(ds_list, compat="override")
            except Exception as e:
                print(f"  -> Merge failed: {e}")
                continue

            if "step" in ds.dims: ds = ds.isel(step=0)

            if bbox:
                try:
                    ds = self._spatial_subset(ds, *bbox)
                except ValueError:
                    print("  -> BBox empty, skipping.")
                    continue

            if "orog" in ds: ds = ds.rename({"orog": "elevation"})
            elif "hgt" in ds: ds = ds.rename({"hgt": "elevation"})

            if "time" in ds.coords: ds = ds.expand_dims("time")

            if combined is None: combined = ds
            else: combined = xr.concat([combined, ds], dim="time")

        return combined

    def validate_data(self, data: xr.Dataset) -> bool:
        if len(data.data_vars) == 0:
            return False
        return True
    
    def cleanup_timestamp(self, timestamp):
        """Deletes raw HRRR files for a specific timestamp WITHOUT network calls."""
        try:
            # Extract the date and hour signature Herbie uses (e.g., '20250107' and 't13z')
            ts = pd.to_datetime(timestamp)
            date_str = ts.strftime('%Y%m%d')
            hour_str = f"t{ts.strftime('%H')}z"
            
            # Recursively find and delete any matching files in the save directory
            for p in self.save_dir.rglob(f"*{date_str}*{hour_str}*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass