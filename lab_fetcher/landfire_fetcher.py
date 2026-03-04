import requests
import shutil
import math
from pathlib import Path
import xarray as xr
import rioxarray # Required: pip install rioxarray

class LandfireFetcher:
    # URL for LANDFIRE 2016 Remap - Fuel Model 40 (Scott/Burgan)
    # Service: EDW_LF2016_FBFM40_01
    BASE_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_LF2016_FBFM40_01/ImageServer/exportImage"

    def __init__(self, save_dir="data/landfire"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fetch_bbox(self, bbox_tuple):
        """
        Fetches LANDFIRE Fuel Model 40 data for a bbox.
        
        Args:
            bbox_tuple: (min_lon, max_lon, min_lat, max_lat)
        """
        min_lon, max_lon, min_lat, max_lat = bbox_tuple
        
        # 1. Calculate dimensions to maintain approx 30m resolution
        # 1 degree lat approx 111km. 
        deg_width = abs(max_lon - min_lon)
        deg_height = abs(max_lat - min_lat)
        
        # 30 meters in degrees (approx)
        pixel_size_deg = 30 / 111000 
        
        width_px = int(deg_width / pixel_size_deg)
        height_px = int(deg_height / pixel_size_deg)

        # Cap max size to prevent massive downloads (optional safety)
        if width_px * height_px > 25000000: # 25 megapixels
            print("Warning: BBox is very large. Capping Landfire resolution to avoid timeouts.")
            scale = 25000000 / (width_px * height_px)
            width_px = int(width_px * math.sqrt(scale))
            height_px = int(height_px * math.sqrt(scale))

        # 2. Check Cache
        filename = f"lf_fbfm40_{min_lon:.2f}_{min_lat:.2f}_{max_lon:.2f}_{max_lat:.2f}.tif"
        local_path = self.save_dir / filename

        if local_path.exists():
            print(f"Using cached LANDFIRE: {local_path}")
            return self._load_and_clean(local_path)

        # 3. Download if not cached
        print(f"Fetching LANDFIRE (Fuel Model 40) from REST API...")
        print(f"  -> Requesting {width_px}x{height_px} pixels for 30m resolution.")

        params = {
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "bboxSR": "4326",      # Input bbox is Lat/Lon WGS84
            "imageSR": "4326",     # Output should be Lat/Lon
            "size": f"{width_px},{height_px}", 
            "format": "tiff",
            "pixelType": "U8",     # Fuel models are 0-255 integers
            "f": "image"           # Return binary
        }

        try:
            r = requests.get(self.BASE_URL, params=params, stream=True, timeout=30)
            r.raise_for_status()
            
            with open(local_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            
            print("  -> Download complete.")
            return self._load_and_clean(local_path)
            
        except Exception as e:
            print(f"Failed to fetch LANDFIRE data: {e}")
            return None

    def _load_and_clean(self, path):
        """Loads GeoTIFF via rioxarray and renames dims for xESMF"""
        try:
            # Load raster
            da = rioxarray.open_rasterio(path)
            
            # Select first band (it's usually 1 band for fuel model)
            if 'band' in da.dims:
                da = da.isel(band=0)

            # Rename x/y to lon/lat for xESMF compatibility
            da = da.rename({'x': 'lon', 'y': 'lat'})
            
            # Name the variable
            da.name = "fuel_model"
            
            return da.to_dataset()
        except Exception as e:
            print(f"Error loading Landfire TIFF: {e}")
            return None