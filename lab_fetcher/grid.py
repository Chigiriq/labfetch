import os
import xesmf as xe
from dask.diagnostics import ProgressBar
import warnings

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
    
    # Logic to handle weights file reuse
    if weights_path:
        reuse = os.path.exists(weights_path)
        filename = str(weights_path) # xESMF expects a string
    else:
        # If no path provided, do not save a file, do not reuse
        reuse = False
        filename = None

    # Create the regridder
    regridder = xe.Regridder(
        rave_clip,
        hrrr_clip,
        method,
        reuse_weights=reuse,
        filename=filename,
    )

    print(f"  (Regridding {list(rave_clip.data_vars)}...)")
    with ProgressBar():
        rave_rg = regridder(rave_clip)

    return rave_rg