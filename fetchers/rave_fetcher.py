import requests
from bs4 import BeautifulSoup
from pathlib import Path
import xarray as xr
import pandas as pd
import warnings
from urllib.parse import urljoin
import concurrent.futures
from .base_fetcher import BaseFetcher 

# ---- xarray + warnings config ----
xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

class RAVEFetcher(BaseFetcher): 
    BASE_URL = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km/"

    def __init__(self, save_dir=None):
        super().__init__(source_name="RAVE") 
        self.save_dir = Path(save_dir) if save_dir else DATA_ROOT / "rave"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_map = {} 

    def _list_directory(self, url):
        try:
            r = requests.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            return [
                urljoin(url, a.get("href"))
                for a in soup.find_all("a")
                if a.get("href", "").endswith(".nc")
            ]
        except Exception as e:
            print(f"Warning: Could not list directory {url}: {e}")
            return []

    def _collect_nc_files(self, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        files = []
        # Note: 'ME' is the modern pandas frequency for Month End
        months = pd.period_range(start_time, end_time, freq="M")

        for p in months:
            url = f"{self.BASE_URL}{p.strftime('%Y')}/{p.strftime('%m')}/"
            print("Checking NOAA RAVE Index:", url)

            links = self._list_directory(url)
            
            for link in links:
                name = link.split("/")[-1]
                try:
                    ts_str = name.split("_s")[1][:14]
                    ts = pd.to_datetime(ts_str, format="%Y%m%d%H%M%S")
                    
                    if start_time <= ts <= end_time:
                        files.append(link)
                except Exception:
                    continue

        return sorted(files)

    @staticmethod
    def _to_0360(lon):
        return lon % 360

    @classmethod
    def _spatial_subset(cls, ds, lon_min, lon_max, lat_min, lat_max):
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

        import numpy as np
        y_idx, x_idx = np.where(mask.values)
        if len(y_idx) == 0:
            print("  -> Empty spatial subset — skipping file")
            return None

        keep_vars = [
            v for v in ds.data_vars
            if {"grid_yt", "grid_xt"}.issubset(ds[v].dims)
        ]

        subset = ds[keep_vars + ["grid_latt", "grid_lont"]].isel(
            grid_yt=slice(y_idx.min(), y_idx.max() + 1),
            grid_xt=slice(x_idx.min(), x_idx.max() + 1)
        )
        return subset

    def _download_worker(self, url):
        name = url.split("/")[-1]
        local = self.save_dir / name
        
        if not local.exists():
            print(f"Downloading {name}...")
            try:
                r = requests.get(url)
                r.raise_for_status()
                local.write_bytes(r.content)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                return None
        return local

    def prefetch(self, start_time, end_time):
        files = self._collect_nc_files(start_time, end_time)
        if not files:
            print("No RAVE files found for this range.")
            return {}

        print(f"Found {len(files)} RAVE files. Starting parallel download...")

        local_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self._download_worker, files)
            for res in results:
                if res: local_paths.append(res)

        self.file_map = {}
        for p in local_paths:
            try:
                name = p.name
                ts_str = name.split("_s")[1][:14]
                ts = pd.to_datetime(ts_str, format="%Y%m%d%H%M%S").round("h")
                self.file_map[ts] = p 
            except Exception:
                pass
        
        print(f"Successfully cached {len(self.file_map)} RAVE files.")
        return self.file_map

    def fetch_data(self, start_time, end_time, bbox=None):
        t = pd.to_datetime(start_time)
        
        if t not in self.file_map:
            print(f"  -> File for {t} not found in prefetch map.")
            return None

        path = self.file_map[t]
        try:
            ds = xr.open_dataset(path, chunks={})
            if bbox:
                return self._spatial_subset(ds, *bbox)
            return ds
        except Exception as e:
            print(f"Error opening RAVE file {path}: {e}")
            return None

    def validate_data(self, data: xr.Dataset) -> bool:
        return "FRP_MEAN" in data.data_vars or "rave_frp" in data.data_vars