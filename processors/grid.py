import os
import xesmf as xe
import warnings
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore", message="Input array is not F_CONTIGUOUS") # esmf regrid warning

def regrid_rave_to_hrrr(rave_clip, hrrr_clip, weights_path, method="bilinear"):
    """
    Regrids RAVE to HRRR using persistent weights to save CPU time.
    """
    weights_file = str(weights_path)
    reuse = os.path.exists(weights_file)
    
    if reuse:
        print(f"  -> Reusing existing weights: {weights_path.name}")
    else:
        print(f"  -> Calculating new weights for this fire's BBox...")

    regridder = xe.Regridder(
        rave_clip,
        hrrr_clip,
        method=method,
        reuse_weights=reuse,
        filename=weights_file,
    )

    rave_rg = regridder(rave_clip)
    
    return rave_rg