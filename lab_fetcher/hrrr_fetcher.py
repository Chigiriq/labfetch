from herbie import Herbie
import pandas as pd
import xarray as xr
from pathlib import Path


class HRRRFetcher:
    DEFAULT_VARS = [
        "TMP:2 m",
        "DPT:2 m",
        "UGRD:10 m",
        "VGRD:10 m",
        "PBLH",
        "PRES:sfc"
    ]

    def __init__(self, model="hrrr", product="sfc", save_dir=None):
        self.model = model
        self.product = product

        # Default save_dir inside project folder
        if save_dir is None:
            # Use current working directory / labfetch/hrrr
            self.save_dir = Path.cwd() / "lab_fetcher" / "hrrr"
        else:
            self.save_dir = Path(save_dir)

        # Make sure the directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _spatial_subset(ds, lon_min, lon_max, lat_min, lat_max):
        return ds.where(
            (ds.longitude >= lon_min) &
            (ds.longitude <= lon_max) &
            (ds.latitude >= lat_min) &
            (ds.latitude <= lat_max),
            drop=True
        )

    def fetch_range(self, start_time, end_time, variable=None, bbox=None):

        times = pd.date_range(start_time, end_time, freq="1h")
        datasets = []

        for t in times:
            print(f"Fetching HRRR: {t}")

            H = Herbie(
                t,
                model=self.model,
                product=self.product,
                save_dir=str(self.save_dir),  # <-- force project save_dir
            )

            if variable is None:
                search_string = "|".join(self.DEFAULT_VARS)
            else:
                search_string = variable

            ds = H.xarray(search=search_string)

            # Handle multi-hypercube return
            if isinstance(ds, list):
                ds = xr.merge(ds, compat="override")

            if "step" in ds.dims:
                ds = ds.isel(step=0)

            if bbox:
                lon_min, lon_max, lat_min, lat_max = bbox
                ds = self._spatial_subset(ds, lon_min, lon_max, lat_min, lat_max)

            datasets.append(ds)

        combined = xr.concat(datasets, dim="time")

        return combined
