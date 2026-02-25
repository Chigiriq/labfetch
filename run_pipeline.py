import argparse
import xarray as xr
import sys
# import dask

from pathlib import Path

from dask.diagnostics import ProgressBar

from lab_fetcher.hrrr_fetcher import HRRRFetcher
from lab_fetcher.grid import regrid_rave_to_hrrr
from lab_fetcher.rave_fetcher import RAVEFetcher

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def compute_scalar(da: xr.DataArray):
    """Safely get scalar from dask-backed array with progress."""
    with ProgressBar():
        return da.sum().compute()


def validate_regridding(original, regridded):
    print("\nValidating mass conservation…")

    orig_total = compute_scalar(original)
    new_total = compute_scalar(regridded)

    rel_error = abs(new_total - orig_total) / orig_total

    print(f"Original total: {orig_total}")
    print(f"Regridded total: {new_total}")
    print(f"Relative error: {rel_error:.4%}")

    return rel_error


def main():
    parser = argparse.ArgumentParser(description="Fetch HRRR + RAVE and regrid")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--method", default="bilinear")
    # parser.add_argument("--method", default="conservative")

    args = parser.parse_args()

    print("\nStarting pipeline")
    print(f"Range: {args.start} → {args.end}")

    # ---------------- HRRR ----------------
    print("\nFetching HRRR…")
    hrrr = HRRRFetcher()
    hrrr_ds = hrrr.fetch_range(args.start, args.end)

    # chunk so progress bars actually show work
    hrrr_ds = hrrr_ds.chunk()

    print("HRRR ready")

    # ---------------- RAVE ----------------
    print("\nFetching RAVE…")
    rave = RAVEFetcher()
    rave_ds = rave.fetch_range(args.start, args.end)
    rave_ds = rave_ds.chunk()

    print("RAVE ready")

    # ---------------- Data Check ----------------
    print("\nRAVE lat range:",
      float(rave_ds.grid_latt.min().compute()),
      float(rave_ds.grid_latt.max().compute()))

    print("RAVE lon range:",
        float(rave_ds.grid_lont.min().compute()),
        float(rave_ds.grid_lont.max().compute()))
    
    print("HRRR lat range:",
      float(hrrr_ds.latitude.min().compute()),
      float(hrrr_ds.latitude.max().compute()))

    print("HRRR lon range:",
        float(hrrr_ds.longitude.min().compute()),
        float(hrrr_ds.longitude.max().compute()))

    # ---------------- Regrid ----------------
    print("\nRegridding RAVE → HRRR grid…")
    rave_on_hrrr = regrid_rave_to_hrrr(
        rave_ds,
        hrrr_ds,
        method=args.method,
    )

    # force compute with progress
    with ProgressBar():
        rave_on_hrrr = rave_on_hrrr.compute()

    print("Regridding complete")

    # ---------------- Validation ----------------
    error = validate_regridding(
        rave_ds.PM25,
        rave_on_hrrr.PM25
    )

    if error > 0.05:
        print("WARNING: High mass error")

    # ---------------- Merge ----------------
    print("\nMerging datasets…")
    combined = xr.merge([hrrr_ds, rave_on_hrrr])

    # ---------------- Save ----------------
    print("\nWriting NetCDF…")
    output_dir = DATA_ROOT / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / f"combined_{args.start}_{args.end}.nc"

    with ProgressBar():
        combined.to_netcdf(outfile)

    print(f"\nSaved {outfile}")
    print("Pipeline complete")


if __name__ == "__main__":
    main()