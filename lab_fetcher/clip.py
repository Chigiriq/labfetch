from .utils import normalize_lon

def clip_latlon(ds, bbox, lat="lat", lon="lon", pad=0.5):
    """
    Clip dataset to bbox with padding.
    Works for HRRR + RAVE.
    """

    ds = normalize_lon(ds, lon)

    return ds.sel(
        {
            lat: slice(bbox["lat_min"] - pad, bbox["lat_max"] + pad),
            lon: slice(bbox["lon_min"] - pad, bbox["lon_max"] + pad),
        }
    )