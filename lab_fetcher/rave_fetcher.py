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

        # Only scan unique year/month combinations
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
                        # Parse start timestamp from filename
                        # format: _sYYYYMMDDHHMMSSS
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

    def fetch_range(self, start_time, end_time):
        files = self._collect_nc_files(start_time, end_time)

        print(f"Found {len(files)} remote files matching date range.")

        datasets = []
        downloaded = 0
        loaded = 0

        for file_url in files:
            filename = file_url.split("/")[-1]
            local_file = self.save_dir / filename

            if not local_file.exists():
                print(f"Downloading {filename}")
                r = requests.get(file_url)
                r.raise_for_status()
                local_file.write_bytes(r.content)
                downloaded += 1
            else:
                print(f"Using cached file: {filename}")

            try:
                ds = xr.open_dataset(local_file)
                datasets.append(ds)
                loaded += 1
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        print(f"Downloaded: {downloaded}")
        print(f"Loaded into xarray: {loaded}")

        if not datasets:
            raise RuntimeError("No RAVE files found.")

        combined = xr.concat(datasets, dim="time")

        return combined

