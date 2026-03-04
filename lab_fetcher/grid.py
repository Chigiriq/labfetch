import os
import xesmf as xe
from dask.diagnostics import ProgressBar
import warnings
import xarray as xr

warnings.filterwarnings("ignore", message="Input array is not F_CONTIGUOUS")

def regrid_rave_to_hrrr(rave_clip, hrrr_clip, weights_path=None, method="bilinear"):
    """
    Regrids ALL variables in rave_clip to the hrrr_clip grid.
    
    Args:
        rave_clip (xr.Dataset): RAVE dataset (source).
        hrrr_clip (xr.Dataset): HRRR dataset (destination grid).
        weights_path (str or Path): Path to save/load weights file. 
                                    If None, weights are calculated in memory (slower).
        method (str): Regridding method (bilinear, conservative, etc).
    """
    if weights_path:
        reuse = os.path.exists(weights_path)
        filename = str(weights_path)
    else:
        reuse = False
        filename = None

    regridder = xe.Regridder(
        rave_clip,
        hrrr_clip,
        method,
        reuse_weights=reuse,
        filename=filename,
    )

    print(f"  (Regridding RAVE: {list(rave_clip.data_vars)}...)")
    with ProgressBar():
        rave_rg = regridder(rave_clip)

    return rave_rg

def regrid_static_to_hrrr(static_ds, hrrr_clip, weights_path=None):
    """
    Regrids Static Data (Landfire) → HRRR grid.
    
    USES NEAREST_S2D (Nearest Source to Destination) method.
    This is required for categorical data like Fuel Models (you cannot average them).
    """
    
    if weights_path:
        # Append '_static' to distinguish from RAVE weights
        filename = str(weights_path).replace(".nc", "_static.nc")
        reuse = os.path.exists(filename)
    else:
        reuse = False
        filename = None

    # 'nearest_s2d' ensures we pick the dominant fuel model, not an average
    regridder = xe.Regridder(
        static_ds,
        hrrr_clip,
        method="nearest_s2d", 
        reuse_weights=reuse,
        filename=filename,
    )

    print(f"  (Regridding Static Data: {list(static_ds.data_vars)}...)")
    # No progress bar needed for single frame usually
    static_rg = regridder(static_ds)

    return static_rg