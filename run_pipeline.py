import argparse
import xarray as xr

from lab_fetcher.hrrr_fetcher import HRRRFetcher
from lab_fetcher.grid import regrid_rave_to_hrrr
from lab_fetcher.rave_fetcher import RAVEFetcher  # assuming you have one


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

    # Merge
    combined = xr.merge([hrrr_ds, rave_on_hrrr])

    # Save output
    outfile = f"combined_{args.start}_{args.end}.nc"
    combined.to_netcdf(outfile)

    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()