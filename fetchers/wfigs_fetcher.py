import geopandas as gpd
import pandas as pd
import requests
from urllib.parse import urlencode
from .base_fetcher import BaseFetcher

class WFIGSFetcher(BaseFetcher):
    URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0/query"

    def __init__(self):
        super().__init__(source_name="WFIGS")

    def _parse_date(self, val):
        if not val: return None
        return pd.to_datetime(val, unit='ms') if isinstance(val, (int, float)) else pd.to_datetime(val)

    def fetch_data(self, start_time: str, end_time: str, bbox: tuple = None, min_acres: int = 100):
        start_str = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M:%S')
        end_str = pd.to_datetime(end_time).strftime('%Y-%m-%d %H:%M:%S')

        where_clause = (
            f"FireDiscoveryDateTime >= '{start_str}' "
            f"AND FireDiscoveryDateTime <= '{end_str}' "
            f"AND IncidentTypeCategory = 'WF'"
        )
        
        if min_acres: 
            where_clause += f" AND IncidentSize >= {min_acres}"

        params = {
            "where": where_clause,
            "outFields": "*", 
            "f": "geojson",
            "outSR": "4326" 
        }

        query_url = f"{self.URL}?{urlencode(params)}"
        print(f"  -> Querying WFIGS API...")

        try:
            response = requests.get(query_url)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                print(f"  -> ArcGIS API Error: {data['error']}")
                return None
            if not data.get("features"):
                return None

            gdf = gpd.GeoDataFrame.from_features(data["features"])
            gdf.set_crs(epsg=4326, inplace=True) 
            return gdf
            
        except Exception as e:
            print(f"  -> WFIGS Fetch Error: {e}")
            return None

    def validate_data(self, data: gpd.GeoDataFrame) -> bool:
        return data is not None and not data.empty and "geometry" in data.columns

    def generate_fire_tasks(self, gdf: gpd.GeoDataFrame, pad: float = 0.5):
        tasks = []
        for idx, row in gdf.iterrows():
            if row.geometry is None or row.geometry.is_empty: continue
                
            # Geometry bounds of a point is just (lon, lat, lon, lat)
            minx, miny, maxx, maxy = row.geometry.bounds
            
            # The pad expands it to a 110km x 110km box
            bbox_tuple = (minx - pad, maxx + pad, miny - pad, maxy + pad)
            
            tasks.append({
                "fire_id": row.get("UniqueFireIdentifier", f"fire_{idx}"),
                "name": row.get("IncidentName", "Unknown"),
                "start": self._parse_date(row.get("FireDiscoveryDateTime")),
                "end": self._parse_date(row.get("ContainmentDateTime")), 
                "acres": row.get("IncidentSize", 0),
                "bbox": bbox_tuple
            })
        return tasks