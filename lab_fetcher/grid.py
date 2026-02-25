import xarray as xr
# import xesmf as xe
import warnings
import os
import warnings

xr.set_options(use_new_combine_kwarg_defaults=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# xESMF detection (works on conda Linux -- WSL dependent)
try:
    import xesmf as xe
    _XESMF_AVAILABLE = True
    print("xESMF detected — using high-quality regridding")
except Exception as e:
    _XESMF_AVAILABLE = False
    warnings.warn(f"xESMF unavailable, falling back to interp(): {e}")


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
    """
    Regrid RAVE dataset onto HRRR grid.
    
    If ESMF/xESMF is available, uses it for proper conservative/curved regridding.
    Otherwise falls back to xarray.interp().
    """

    # Only interpolate variables that live on grid
    grid_vars = [
        v for v in rave_ds.data_vars
        if {"grid_yt", "grid_xt"}.issubset(rave_ds[v].dims)
    ]
    rave_center = rave_ds[grid_vars]

    # Normalize RAVE longitudes 0-360 to match HRRR
    # if rave_center.grid_lont.min() < 0:
    #     rave_center = rave_center.assign_coords(
    #         grid_lont=(rave_center.grid_lont % 360)
    #     )
    # Convert RAVE 0–360 → -180–180 to match HRRR
    if rave_center.grid_lont.min() < 0:
        rave_center = rave_center.assign_coords(
            grid_lont=(rave_center.grid_lont % 360)
        )

    if _XESMF_AVAILABLE:
        # xESMF regridding
        source_grid = build_rave_grid(rave_center)
        target_grid = build_hrrr_grid(hrrr_ds)

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method=method,
            reuse_weights=False
        )
        regridded = regridder(rave_center)
        return regridded

    else:
        # fallback: xarray interpolation
        rave_center = rave_center.assign_coords(
            latitude=rave_ds["grid_latt"],
            longitude=rave_ds["grid_lont"]
        )
        regridded = rave_center.interp(
            latitude=hrrr_ds.latitude,
            longitude=hrrr_ds.longitude,
            method="linear"
        )
        warnings.warn(
            "Using xarray.interp() fallback. Curvilinear/conservative regridding may not be exact."
        )
        return regridded





#keeping this here in case we happen to use atmos chem modeling.
# --> main issue is xesmf doesn't work great on windows allegedly? I'll test more later
# def regrid_rave_to_hrrr(rave_ds, hrrr_ds, method="bilinear"):

#     source_grid = build_rave_grid(rave_ds)
#     target_grid = build_hrrr_grid(hrrr_ds)

#     regridder = xe.Regridder(
#         source_grid,
#         target_grid,
#         method=method,
#         reuse_weights=False,
#     )

#     regridded = regridder(rave_ds)

#     return regridded


# def regrid_rave_to_hrrr(rave_ds, hrrr_ds, method="bilinear"):
#     if method == "bilinear":
#         return rave_ds.interp(
#             grid_lont=hrrr_ds.longitude,
#             grid_latt=hrrr_ds.latitude,
#             method="linear"
#         )
#     else:
#         raise ValueError("Conservative regridding requires xESMF.")