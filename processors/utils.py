import xarray as xr

def normalize_lon(ds, lon="lon"):
    """Convert 0–360 → -180–180 and sort."""
    if ds[lon].max() > 180:
        ds = ds.assign_coords({lon: ((ds[lon] + 180) % 360) - 180})
        ds = ds.sortby(lon)
    return ds