# Labfetch

**Labfetch** is a meteorological data pipeline designed to ingest weather data from **HRRR** (via Herbie) and fire emissions data from **RAVE**, spatially subset them to a specific region of interest, and merge them onto a unified grid for analysis.

## 🔄 Pipeline Overview

The program executes the following workflow automatically:

1.  **Fetch & Clip (HRRR):** Downloads High-Resolution Rapid Refresh (HRRR) weather data (Surface variables + Smoke) and clips it to the user-defined bounding box.
2.  **Fetch & Clip (RAVE):** Downloads RAVE fire emissions data (`FRP`, `PM2.5`) and clips it to the same bounding box.
3.  **Regrid:** Uses `xesmf` to reproject the RAVE data onto the HRRR geospatial grid.
4.  **Merge:** Combines the weather and emissions data into a single NetCDF file (`combined_final.nc`).

---

## ⚠️ Important: System Requirements

**This tool relies heavily on the `esmf` and `xesmf` package families.** These geospatial packages **do not** natively support Windows.

*   **Linux Users:** You can run this natively.
*   **Windows Users:** You **must** run the CLI pipeline through **WSL (Windows Subsystem for Linux)**.

### Environment Files
The specific package list required to build the environment are located in the `envs` folder. **Note:** This environment relies on **conda-forge**.

*   **WSL / Linux:** `Reqs.txt`

---

## 🛠 Installation & Setup

### 1. Setting up Miniconda on WSL
If you are setting up a fresh WSL instance, follow these steps to install Miniconda:

```bash
# 1. Update package list and install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install wget bzip2 -y

# 2. Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 3. Run the installer
bash Miniconda3-latest-Linux-x86_64.sh
```

**During installation:**
*   Press **Enter** to review the license.
*   Type **yes** to accept the license terms.
*   Type **yes** when asked to run `conda init`.

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
conda activate labfetch
```

__Any additional package installs should only use conda-forge as a source to avoid environment issues__

---

## 🚀 Usage

### Command Line Interface (CLI)
The main entry point is `run_pipeline.py`.

#### Command Structure
```bash
python run_pipeline.py --start <TIMESTAMP> --end <TIMESTAMP> --bbox <COORDINATES>
```

#### Arguments Description
| Argument | Format | Description |
| :--- | :--- | :--- |
| `--start` | `YYYY-MM-DD HH:MM` | The starting date and time for data collection (UTC). |
| `--end` | `YYYY-MM-DD HH:MM` | The ending date and time. |
| `--bbox` | `"lat_min,lat_max,lon_min,lon_max"` | The bounding box coordinates. **Note:** Ensure no spaces between commas. |

#### Example Run (Los Angeles Fire Event)
This command grabs data for the LA area during a specific fire event window:

```bash
python run_pipeline.py \
  --start "2025-01-07 20:00" \
  --end "2025-01-07 23:00" \
  --bbox "33.9,34.3,-118.7,-118.3"
```

### Jupyter Notebooks
If you wish to use the notebooks (`validationTest.ipynb` or `test_fetch.ipynb`) for data exploration on Windows:

1.  Create and activate the environment using the Windows requirements file and **conda-forge**:
    ```bash
    conda create -n labforge -c conda-forge --file envs/Reqs.txt
    conda activate labfetch
    ```
2.  Launch Jupyter:
    ```bash
    jupyter lab
    ```

---

## 📂 Output & Data Structure

### 💾 File Storage
The pipeline manages data storage automatically:
*   **Final Output:** `data/combined_final.nc` (Unified NetCDF dataset).
*   **Temporary Workspace:** `temp_processing/` (Stores hourly chunks during execution; automatically cleaned up upon completion).

### 📐 Dataset Format
The output is an **xarray Dataset** aligned to the HRRR model's geospatial grid.

**Dimensions & Coordinates:**
*   `time`: Hourly timestamps (UTC).
*   `y`, `x`: Projection grid indices (Lambert Conformal Conic).
*   `lat`, `lon`: 2D arrays providing decimal latitude/longitude for every grid cell.

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

### 🐍 Visualization Example

You can quickly visualize the evolution of a fire event using `xarray` and `matplotlib`.

### 🐍 Visualization Example

To inspect specific snapshots (Start, Middle, and End) of key variables:

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load Data
ds = xr.open_dataset("data/combined_final.nc")

# 1. Pick specific time indices (e.g., First, Middle, Last)
# This ensures you see the beginning, peak, and end of your range
indices = [0, len(ds.time)//2, len(ds.time)-1]

# 2. Select specific variables to check (don't plot static grids like lat/lon)
target_vars = ["rave_frp", "t2m", "elevation", "u10", "v10"]

for var in target_vars:
    if var not in ds: continue
        
    print(f"--- Plotting Evolution of {var} ---")
    
    # Select the specific times and plot them in a row (col="time")
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