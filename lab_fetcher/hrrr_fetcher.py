from herbie import Herbie
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
import warnings

# ---- xarray config ----
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


class HRRRFetcher:
    DEFAULT_VARS = [
        "TMP:2 m",
        "DPT:2 m",
        "UGRD:10 m",
        "VGRD:10 m",
        "PBLH",
        "PRES:sfc",
    ]

    def __init__(self, model="hrrr", product="sfc", save_dir=None):
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
            (ds.longitude >= lon_min)
            & (ds.longitude <= lon_max)
            & (ds.latitude >= lat_min)
            & (ds.latitude <= lat_max)
        )

        y, x = np.where(mask.values)
        if len(y) == 0:
            raise ValueError("BBox does not intersect HRRR grid.")

        return ds.isel(
            y=slice(y.min(), y.max() + 1),
            x=slice(x.min(), x.max() + 1),
        )

    def fetch_range(self, start_time, end_time, variable=None, bbox=None):

        times = pd.date_range(start_time, end_time, freq="1h")

        combined = None

        for t in times:
            print("Fetching HRRR:", t)

            H = Herbie(
                t,
                model=self.model,
                product=self.product,
                save_dir=str(self.save_dir),
            )

            search = "|".join(self.DEFAULT_VARS) if variable is None else variable
            ds = H.xarray(search=search)

            if isinstance(ds, list):
                ds = xr.merge(ds, compat="override")

            if "step" in ds.dims:
                ds = ds.isel(step=0)

            if bbox:
                ds = self._spatial_subset(ds, *bbox)

            # ---- STREAM CONCAT ----
            if combined is None:
                combined = ds
            else:
                combined = xr.concat([combined, ds], dim="time")

        return combined