import xarray as xr
import xesmf as xe



def build_hrrr_grid(hrrr_ds):

    return xr.Dataset(
        {
            "lat": hrrr_ds["latitude"],
            "lon": hrrr_ds["longitude"],
        }
    )




def build_rave_grid(rave_ds):
    """
    Build xESMF-compatible grid from RAVE cell centers.
    """

    return xr.Dataset(
        {
            "lat": rave_ds["grid_latt"],
            "lon": rave_ds["grid_lont"],
        }
    )




def regrid_rave_to_hrrr(rave_ds, hrrr_ds, method="bilinear"):

    source_grid = build_rave_grid(rave_ds)
    target_grid = build_hrrr_grid(hrrr_ds)

    regridder = xe.Regridder(
        source_grid,
        target_grid,
        method=method,
        reuse_weights=False,
    )

    regridded = regridder(rave_ds)

    return regridded
