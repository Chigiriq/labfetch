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
    def _spatial_subset(ds, lon_min, lon_max, lat_min, lat_max):
        mask = (
            (ds["grid_lont"] >= lon_min)
            & (ds["grid_lont"] <= lon_max)
            & (ds["grid_latt"] >= lat_min)
            & (ds["grid_latt"] <= lat_max)
        )

        center_vars = [
            v for v in ds.data_vars
            if {"grid_yt", "grid_xt"}.issubset(ds[v].dims)
        ]

        return ds[center_vars].where(mask, drop=True)

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

            # ---- STREAM CONCAT ----
            if combined is None:
                combined = ds
            else:
                combined = xr.concat([combined, ds], dim="time")

        if combined is None:
            raise RuntimeError("No RAVE files found.")

        return combined