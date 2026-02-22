def convert_bbox_to_360(bbox):
    """
    Convert (-180 to 180) bbox to (0 to 360) longitude.
    bbox format:
    (lon_min, lon_max, lat_min, lat_max)
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    return (lon_min_360, lon_max_360, lat_min, lat_max)
