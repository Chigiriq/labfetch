import argparse
import xarray as xr
from  pathlib import Path

from lab_fetcher.hrrr_fetcher import HRRRFetcher
from lab_fetcher.grid import regrid_rave_to_hrrr
from lab_fetcher.rave_fetcher import RAVEFetcher  # assuming you have one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

def validate_regridding(original, regridded):
    orig_total = original.sum().item()
    new_total = regridded.sum().item()

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

    args = parser.parse_args()

    # Fetch HRRR
    hrrr = HRRRFetcher()
    hrrr_ds = hrrr.fetch_range(args.start, args.end)

    # Fetch RAVE
    rave = RAVEFetcher()
    rave_ds = rave.fetch_range(args.start, args.end)

    # Regrid
    rave_on_hrrr = regrid_rave_to_hrrr(
        rave_ds,
        hrrr_ds,
        method=args.method,
    )

    error = validate_regridding(
        rave_ds.PM25,
        rave_on_hrrr.PM25
    )

    if error > 0.05:
        print("WARNING: High mass error")

    # Merge
    combined = xr.merge([hrrr_ds, rave_on_hrrr])

    # Save output
    output_dir = DATA_ROOT / "combined"
    output_dir.mkdir(exist_ok=True)

    outfile = output_dir / f"combined_{args.start}_{args.end}.nc"
    combined.to_netcdf(outfile)

    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()