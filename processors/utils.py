import xarray as xr

def normalize_lon(ds, lon="lon"):
    """
    Convert 0-360 → -180-180.
    Supports 2D curvilinear grids by modifying values directly.
    """
    ds = ds.copy()

    if lon in ds.coords or lon in ds.data_vars:
        if ds[lon].max() > 180:
            ds[lon] = ((ds[lon] + 180) % 360) - 180
            
    return ds