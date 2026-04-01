import xesmf as xe
import os
import warnings
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def silence_c_errors():
    """Redirects OS-level C stderr to /dev/null to completely silence HDF5 panics."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        # Route all C-level errors into the black hole
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore the normal error channel so Python can still log real errors
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

def regrid_rave_to_hrrr(rave_ds, hrrr_ds, weights_path):
    """
    Regrids RAVE satellite data to the HRRR curvilinear grid.
    Dynamically generates or reuses weights.
    """
    weights_exist = Path(weights_path).exists()
    
    # Mute Python warnings (F_CONTIGUOUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # Mute OS-Level C-Library errors (HDF5-DIAG)
        with silence_c_errors():
            regridder = xe.Regridder(
                rave_ds, 
                hrrr_ds, 
                "bilinear",
                filename=str(weights_path),
                reuse_weights=weights_exist
            )
            
        # Perform the regridding (safely outside the mute block just in case)
        rave_rg = regridder(rave_ds)
    
    # Memory Cleanup
    del regridder
    
    return rave_rg