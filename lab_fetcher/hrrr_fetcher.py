from herbie import Herbie
import xarray as xr
import pandas as pd
from pathlib import Path


class HRRRFetcher:
    def __init__(self, save_dir="data"): #have to manually move folder up a dir so no double hrrr folder
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, start_time, end_time, product="sfc", variable=":TMP:3 m"):
        """
        Fetch HRRR data between start_time and end_time.

        Returns:
            xarray.Dataset
        """
        times = pd.date_range(start=start_time, end=end_time, freq="1h")


        datasets = []

        for t in times:
            print(f"Fetching HRRR for {t}")

            H = Herbie(
                t,
                model="hrrr",
                product=product,
                save_dir=self.save_dir,
                fxx=0  # analysis hour
            )

            try:
                ds = H.xarray(variable)
                ds = ds.expand_dims(time=[pd.to_datetime(t)])
                datasets.append(ds)
            except Exception as e:
                print(f"Skipping {t}: {e}")

        if not datasets:
            raise RuntimeError("No HRRR data downloaded.")

        combined = xr.concat(datasets, dim="time")

        return combined
