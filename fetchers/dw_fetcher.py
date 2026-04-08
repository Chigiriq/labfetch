import ee
import xarray as xr
import pandas as pd
from .base_fetcher import BaseFetcher

class DWFetcher(BaseFetcher):
    def __init__(self):
        super().__init__(source_name="DynamicWorld")
        try:
            # Xee requires Earth Engine to be initialized. 
            # We use the high-volume endpoint optimized for Xarray data pulling.
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception:
            print(f"[{self.source_name}] Earth Engine not initialized. Attempting to authenticate...")
            try:
                ee.Authenticate()
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            except Exception as e:
                print(f"[{self.source_name}] FATAL: Earth Engine authentication failed.")
                print("Please run `earthengine authenticate` in your terminal to log in.")
                raise e

    def fetch_data(self, start_time: str, end_time: str, bbox: tuple = None) -> xr.Dataset:
        if not bbox:
            print(f"[{self.source_name}] Bounding box is required.")
            return None

        # Calculate a 30-day window ending on the fire's start date
        fire_start = pd.to_datetime(start_time)
        search_start = fire_start - pd.Timedelta(days=30)
        
        lon_min, lon_max, lat_min, lat_max = bbox
        
        # Create an Earth Engine bounding box
        geom = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        # Filter the Dynamic World collection
        ic = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterBounds(geom)
              .filterDate(search_start.strftime('%Y-%m-%d'), fire_start.strftime('%Y-%m-%d')))
              
        try:
            # Check if any images exist in the 30 day window
            if ic.size().getInfo() == 0:
                print(f"[{self.source_name}] No Dynamic World images found for this window.")
                return None
                
            # --- SERVER-SIDE REDUCTION ---
            # We calculate the 'mode' (most frequent landcover) on Google's servers.
            # This completely bypasses clouds and flattens 30 days of data into 1 image.
            mode_image = ic.select('label').mode().set('system:time_start', ee.Date(fire_start.strftime('%Y-%m-%d')).millis())
            
            # Wrap the computed image back into a collection so Xee can read it
            computed_ic = ee.ImageCollection([mode_image])
            
            # Use the Xee engine to pull the EE image directly into an Xarray Dataset
            ds = xr.open_dataset(
                computed_ic, 
                engine='ee', 
                geometry=geom,
                crs='EPSG:4326', 
                scale=0.0001 # ~10m resolution in degrees
            )

            # Pull the data from Google's servers into local RAM
            ds_static = ds.compute()
            
            # Clean up the dataset to ensure it is a flat 2D spatial map
            if "time" in ds_static.dims:
                ds_static = ds_static.isel(time=0).drop_vars("time")
                
            # Xee returns coordinates as 'Y' and 'X'. Rename to match the pipeline
            ds_static = ds_static.rename({"Y": "lat", "X": "lon", "label": "dw_landcover"})
            
            return ds_static

        except Exception as e:
            print(f"[{self.source_name}] Earth Engine Fetch Error: {e}")
            return None

    def validate_data(self, data: xr.Dataset) -> bool:
        return data is not None and "dw_landcover" in data.data_vars