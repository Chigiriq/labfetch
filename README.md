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
The specific package lists required to build the environment are located in the `envs` folder. **Note:** These environments rely on **conda-forge**.

*   **WSL / Linux:** `envs/wslReqs.txt` (Required for the main pipeline)
*   **Windows:** `envs/winReqs.txt` (For local notebook analysis only; *cannot run regridding through CLI*)

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

# Create the environment using the WSL requirements file
# IMPORTANT: The '-c conda-forge' flag is required.
conda create --name labfetch --file envs/wslReqs.txt -c conda-forge

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

1.  Create the environment using the Windows requirements file and **conda-forge**:
    ```bash
    conda create --name labfetch --file envs/winReqs.txt -c conda-forge
    conda activate labfetch
    ```
2.  Install the Jupyter kernel:
    ```bash
    conda install -c conda-forge ipykernel
    python -m ipykernel install --user --name=labfetch
    ```
3.  Launch Jupyter:
    ```bash
    jupyter lab
    ```

---

## 📂 Output
*   **Intermediate Files:** Stored in `temp_processing/` during the run (automatically deleted upon success).
*   **Final Output:** The merged NetCDF file is saved to `data/combined_final.nc`.