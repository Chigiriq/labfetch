import requests
from bs4 import BeautifulSoup
from pathlib import Path
import xarray as xr
import pandas as pd
import warnings
from urllib.parse import urljoin

# ---- xarray + warnings config ----
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


class RAVEFetcher:
    BASE_URL = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km/"

    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else DATA_ROOT / "rave"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _list_directory(self, url):
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        return [
            urljoin(url, a.get("href"))
            for a in soup.find_all("a")
            if a.get("href", "").endswith(".nc")
        ]

    def _collect_nc_files(self, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        files = []
        months = pd.period_range(start_time, end_time, freq="M")

        for p in months:
            url = f"{self.BASE_URL}{p.strftime('%Y')}/{p.strftime('%m')}/"
            print("Checking:", url)

            try:
                for link in self._list_directory(url):
                    name = link.split("/")[-1]
                    try:
                        ts = pd.to_datetime(
                            name.split("_s")[1][:14],
                            format="%Y%m%d%H%M%S",
                        )
                        if start_time <= ts <= end_time:
                            files.append(link)
                    except Exception:
                        pass
            except Exception:
                pass

        return sorted(files)

    @staticmethod
    def _to_0360(lon):
        return lon % 360


    @classmethod
    def _spatial_subset(cls, ds, lon_min, lon_max, lat_min, lat_max):

        # ---- convert bbox to 0-360 ----
        lon_min = cls._to_0360(lon_min)
        lon_max = cls._to_0360(lon_max)

        lon = ds["grid_lont"]
        lat = ds["grid_latt"]

        mask = (
            (lon >= lon_min)
            & (lon <= lon_max)
            & (lat >= lat_min)
            & (lat <= lat_max)
        ).compute()

        # Find exact indices like HRRR does to preserve the curved corners!
        import numpy as np
        y_idx, x_idx = np.where(mask.values)
        if len(y_idx) == 0:
            print("Empty spatial subset — skipping file")
            return None

        # ---- keep coords + center vars ----
        keep_vars = [
            v for v in ds.data_vars
            if {"grid_yt", "grid_xt"}.issubset(ds[v].dims)
        ]

        # Slice the bounding box indices directly instead of dropping
        subset = ds[keep_vars + ["grid_latt", "grid_lont"]].isel(
            grid_yt=slice(y_idx.min(), y_idx.max() + 1),
            grid_xt=slice(x_idx.min(), x_idx.max() + 1)
        )

        return subset

    def fetch_range(self, start_time, end_time, bbox=None):

        files = self._collect_nc_files(start_time, end_time)
        print(f"Found {len(files)} remote files matching date range.")

        combined = None

        for url in files:
            name = url.split("/")[-1]
            local = self.save_dir / name

            if not local.exists():
                print("Downloading", name)
                r = requests.get(url)
                r.raise_for_status()
                local.write_bytes(r.content)
            else:
                print("Using cached:", name)

            ds = xr.open_dataset(local, chunks={})

            if bbox:
                ds = self._spatial_subset(ds, *bbox)
                if ds is None:
                    continue

            # ---- STREAM CONCAT ----
            if combined is None:
                combined = ds
            else:
                combined = xr.concat([combined, ds], dim="time")

        if combined is None:
            raise RuntimeError("No RAVE files found.")

        return combined