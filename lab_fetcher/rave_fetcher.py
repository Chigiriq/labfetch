import requests
from bs4 import BeautifulSoup
from pathlib import Path
import xarray as xr
import pandas as pd
from urllib.parse import urljoin


class RAVEFetcher:
    BASE_URL = "https://www.ospo.noaa.gov/pub/Blended/RAVE/RAVE-HrlyEmiss-3km/"

    def __init__(self, save_dir="data/rave"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _list_directory(self, url):
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith(".nc"):
                links.append(urljoin(url, href))

        return links

    def _collect_nc_files(self, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        all_files = []

        months = pd.period_range(start_time, end_time, freq="M")

        for period in months:
            year = period.strftime("%Y")
            month = period.strftime("%m")

            month_url = f"{self.BASE_URL}{year}/{month}/"
            print("Checking:", month_url)

            try:
                links = self._list_directory(month_url)

                for link in links:
                    filename = link.split("/")[-1]

                    try:
                        start_str = filename.split("_s")[1][:14]
                        file_time = pd.to_datetime(
                            start_str,
                            format="%Y%m%d%H%M%S"
                        )

                        if start_time <= file_time <= end_time:
                            all_files.append(link)

                    except Exception:
                        continue

            except Exception:
                continue

        return sorted(all_files)

    @staticmethod
    def _spatial_subset(ds, lon_min, lon_max, lat_min, lat_max):

        # Build 2-D mask only on cell-center grid
        mask = (
            (ds["grid_lont"] >= lon_min) &
            (ds["grid_lont"] <= lon_max) &
            (ds["grid_latt"] >= lat_min) &
            (ds["grid_latt"] <= lat_max)
        )

        # Drop variables not on (grid_yt, grid_xt)
        center_vars = [
            v for v in ds.data_vars
            if {"grid_yt", "grid_xt"}.issubset(ds[v].dims)
        ]

        ds_center = ds[center_vars]

        # Apply mask only to center variables
        ds_subset = ds_center.where(mask, drop=True)

        return ds_subset


    def fetch_range(self, start_time, end_time, bbox=None):

        files = self._collect_nc_files(start_time, end_time)

        print(f"Found {len(files)} remote files matching date range.")

        datasets = []

        for file_url in files:
            filename = file_url.split("/")[-1]
            local_file = self.save_dir / filename

            if not local_file.exists():
                print(f"Downloading {filename}")
                r = requests.get(file_url)
                r.raise_for_status()
                local_file.write_bytes(r.content)
            else:
                print(f"Using cached file: {filename}")

            ds = xr.open_dataset(local_file)

            if bbox is not None:
                lon_min, lon_max, lat_min, lat_max = bbox
                ds = self._spatial_subset(ds, lon_min, lon_max, lat_min, lat_max)

            datasets.append(ds)

        if not datasets:
            raise RuntimeError("No RAVE files found.")

        combined = xr.concat(datasets, dim="time")

        return combined
