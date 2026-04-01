# Labfetch

**Labfetch** is an automated, multithreaded meteorological data pipeline designed to ingest High-Resolution Rapid Refresh (HRRR) weather data and RAVE fire emissions data, spatially subset them to active wildfire perimeters, and merge them onto a unified analysis-ready grid.

## Key Features

* **Automated Wildfire Discovery:** Integrates with the WFIGS API to automatically discover active fires (>100 acres) within your specified time window.
* **Producer-Consumer Multithreading:** Network-bound downloads (pre-fetching the next hour) run in parallel with CPU-bound processing (regridding the current hour), cutting execution time in half.
* **Dynamic Padding:** Automatically scales the bounding box based on the total acres burned to capture the surrounding environmental context.
* **Constant Disk Footprint:** Raw GRIB and NetCDF files are instantly deleted after processing. The raw data footprint never exceeds ~2GB, regardless of whether you process one day or one year.
* **Zarr Output:** Aggregates processed variables directly into a high-performance `.zarr` store, grouped by individual Fire IDs for seamless machine learning ingestion.

---

## Pipeline Overview

The program executes the following workflow automatically:

1.  **Discovery & Validation:** Queries WFIGS to identify valid fires overlapping the requested time window (or accepts a manual bounding box).
2.  **RAVE Prefetching:** Caches all required RAVE emissions files in parallel.
3.  **Multithreaded Processing Loop:** * *Background Thread:* Downloads HRRR/RAVE data for hour $t+1$.
    * *Main Thread:* Clips, reprojects (via `xesmf`), and merges data for hour $t$.
4.  **Zarr Aggregation:** Appends the hourly snapshot to the final `.zarr` store under the specific `fire_id` group.
5.  **Disk Cleanup:** Sweeps raw files from the disk before moving to the next iteration.

---

## Important: System Requirements

**This tool relies heavily on the `esmf` and `xesmf` package families.** These geospatial packages **do not** natively support Windows.

* **Linux Users:** You can run this natively.
* **Windows Users:** You **must** run the CLI pipeline through **WSL (Windows Subsystem for Linux)**.
* **Mac Users**: Less functional due to package dependency errors, as a workaround, most x86 based vm's work 

### Environment Files
The specific package list required to build the environment is located in the `envs` folder. **Note:** This environment relies on **conda-forge**.

* **WSL / Linux:** `Reqs.txt`

---

## Installation & Setup

### 1. Setting up Miniconda on WSL
If you are setting up a fresh WSL instance, follow these steps to install Miniconda:

```bash
# 1. Update package list and install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install wget bzip2 -y

# 2. Download the Miniconda installer
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

# 3. Run the installer
bash Miniconda3-latest-Linux-x86_64.sh
```

**During installation:**
* Press **Enter** to review the license.
* Type **yes** to accept the license terms.
* Type **yes** when asked to run `conda init`.

**Finally, restart your shell to apply changes:**
```bash
source ~/.bashrc
```

### 2. Creating the Environment
To ensure all dependencies (especially the complex geospatial libraries like `cartopy` and `xesmf`) are installed correctly, you **must** use the `conda-forge` channel.

```bash
# Navigate to the project root
cd path/to/labfetch

# Create the environment using the requirements file
# IMPORTANT: The '-c conda-forge' flag is required.
conda create -n labforge -c conda-forge --file envs/Reqs.txt

# Activate the environment
conda activate labforge
```

__Any additional package installs should only use conda-forge as a source to avoid environment conflicts.__

---

## Usage

### Command Line Interface (CLI)
The main entry point is `run_pipeline.py`.

#### Command Structure
```bash
python run_pipeline.py --start <TIMESTAMP> --end <TIMESTAMP> [OPTIONS]
```

#### Arguments Description
| Argument | Format | Description |
| :--- | :--- | :--- |
| `--start` | `YYYY-MM-DD HH:MM` | **Required.** The starting date and time for data collection (UTC). |
| `--end` | `YYYY-MM-DD HH:MM` | **Required.** The ending date and time (UTC). |
| `--bbox` | `"lat_min,lat_max,lon_min,lon_max"` | *Optional.* Bypasses WFIGS discovery to process a specific custom bounding box. |
| `--fire_id` | `String` | *Optional.* Used with `--bbox` to name the Zarr group (defaults to "manual_fetch"). |
| `--spatial_pad` | `Float` | *Optional.* Degrees to pad the spatial bounding box (defaults via `config.yaml`). |
| `--time_pad` | `Integer` | *Optional.* Hours to pad before discovery and after containment (defaults via `config.yaml`). |

#### Example 1: WFIGS Auto-Discovery (Recommended)
This command automatically finds all fires >100 acres active within this timeframe and processes them sequentially:

```bash
python run_pipeline.py --start "2025-01-07 10:00" --end "2025-01-08 10:00"
```

#### Example 2: Manual Override (Specific Event)
This bypasses the WFIGS API to track a specific set of coordinates:

```bash
python run_pipeline.py \
  --start "2025-01-07 20:00" \
  --end "2025-01-07 23:00" \
  --bbox "33.9,34.3,-118.7,-118.3" \
  --fire_id "2025-LA-FIRE-CUSTOM"
```

---

## Output & Data Structure

### File Storage
The pipeline strictly manages storage to prevent bloat:
* **Final Output:** `data/{START}-{END}_WF.zarr` (A consolidated Zarr store).
* **Volatile Directories:** `raw_hrrr/`, `raw_rave/`, and `temp_weights/` are created dynamically and wiped clean instantly after processing.

### Dataset Format (Zarr)
Because Zarr is hierarchical, the output file contains separate **Groups** for every fire processed in the batch (e.g., `2025-CALFD-000738`). Each group contains an **xarray Dataset** aligned to the HRRR model's curvilinear grid.

**Dimensions & Coordinates:**
* `time`: Hourly timestamps (UTC).
* `y`, `x`: Projection grid indices (Lambert Conformal Conic).
* `lat`, `lon`: 2D arrays providing decimal latitude/longitude for every grid cell.

### Variable List
| Variable | Source | Description | Units |
| :--- | :--- | :--- | :--- |
| **rave_frp** | RAVE | Fire Radiative Power (Mean) | MW |
| **t2m** | HRRR | Temperature at 2 meters | K |
| **u10** | HRRR | U-Component of Wind at 10m | m/s |
| **v10** | HRRR | V-Component of Wind at 10m | m/s |
| **d2m** | HRRR | Dew Point Temperature | K |
| **sp** | HRRR | Surface Pressure | Pa |
| **elevation**| HRRR | Terrain Height / Orography | m |

---

## Visualization Example

Zarr stores require you to specify the `group` (the specific fire) when loading the data into xarray.

```python
import xarray as xr
import matplotlib.pyplot as plt

# 1. Load Data for a specific Fire ID
# You can use xr.open_zarr("...", group=None) to see available groups
zarr_path = "data/20250107_1000-20250108_1000_WF.zarr"
fire_id = "2025-CALFD-000738" 

ds = xr.open_zarr(zarr_path, group=fire_id)

# 2. Pick specific time indices (e.g., First, Middle, Last)
indices = [0, len(ds.time)//2, len(ds.time)-1]

# 3. Select variables
target_vars = ["rave_frp", "t2m", "elevation", "u10", "v10"]

for var in target_vars:
    if var not in ds: continue
        
    print(f"--- Plotting Evolution of {var} ---")
    
    # Select specific times and plot in a row
    subset = ds[var].isel(time=indices)
    
    subset.plot(
        col="time",
        col_wrap=3,
        x="lon", 
        y="lat",
        cmap="turbo",
        robust=False,
        figsize=(15, 4)
    )
    plt.show()
```