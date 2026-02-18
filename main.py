from lab_fetcher.hrrr_fetcher import HRRRFetcher
from lab_fetcher.rave_fetcher import RAVEFetcher


def main():
    start = "2025-01-01 00:00"
    end = "2025-01-01 05:00"

    hrrr = HRRRFetcher()
    hrrr_ds = hrrr.fetch_range(start, end)

    print(hrrr_ds)

    rave = RAVEFetcher()
    rave_ds = rave.fetch_range(start, end)

    print(rave_ds)


if __name__ == "__main__":
    main()
