from labfetch import fetch

ds = fetch(
    source="hrrr",
    start="2025-01-01T00:00",
    end="2025-01-01T06:00"
)
