#!/usr/bin/env python
# coding: utf-8

# # Create LoadData CSVs with the paths to IC functions for analysis
# 
# In this notebook, we create LoadData CSVs that contains paths to each channel per image set and associated illumination correction `npy` files per channel for CellProfiler to process. 

# ## Import libraries

# In[5]:


import argparse
import pathlib
import pandas as pd
import os
import shutil

import sys

sys.path.append("../utils")
import loaddata_utils as ld_utils
from bandicoot_utils import bandicoot_check


# ## Set paths

# In[6]:


argparse = argparse.ArgumentParser(
    description="Create LoadData CSV files to run CellProfiler on the cluster"
)
argparse.add_argument("--HPC", action="store_true", help="Type of compute to run on")

# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC

print(f"HPC: {HPC}")


# In[7]:


# Set the index directory based on whether HPC is used or not
if HPC:
    # Path for index directory to make loaddata csvs though compute cluster (HPC)
    index_directory = pathlib.Path(
        "/scratch/alpine/jtomkinson@xsede.org/PCCMA_atlas_data/"
    ).resolve(strict=True)
else:
    # Find root directory of the project
    root_dir = pathlib.Path().cwd()

    image_base_dir = bandicoot_check(
        pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
    )
    # Path for index directory  to make loaddata csv locally
    index_directory = pathlib.Path(
        f"{image_base_dir}/PCCMA_data/Atlas_v2_screen/"
    ).resolve(strict=True)

# Set all paths that are common to both HPC and local
config_dir_path = pathlib.Path("../1.whole_image_qc/load_data_config/").resolve(
    strict=True
)
output_csv_dir = pathlib.Path("./loaddata_csvs/").absolute()
output_csv_dir.mkdir(parents=True, exist_ok=True)
illum_directory = pathlib.Path("../2.illumination_correction/illum_directory/").resolve(
    strict=True
)

# Find all 'Images' folders within the directory (one per plate)
images_folders = sorted(index_directory.rglob("Images"))
print(f"Found {len(images_folders)} 'Images' folders in the directory.")


# ## Create LoadData CSVs with illumination functions for all data

# In[8]:


# Define the config path
config_path = config_dir_path / "config.yml"

# Iterate over each plate folder and create a LoadData CSV with illumination correction
for images_folder in images_folders:
    # plate folder is the folder that holds the 'Images' folder
    folder = images_folder.parent

    br00_id = folder.name.split("__")[0]
    plate_name = br00_id

    path_to_output_csv = (
        output_csv_dir / f"{plate_name}_loaddata_original.csv"
    ).absolute()

    path_to_output_with_illum_csv = (
        output_csv_dir / f"{plate_name}_loaddata_with_illum.csv"
    ).absolute()

    illum_output_path = (
        (illum_directory / br00_id).absolute().resolve(strict=True)
    )

    ld_utils.create_loaddata_illum_csv(
        index_directory=images_folder,
        config_path=config_path,
        path_to_output=path_to_output_csv,
        illum_directory=illum_output_path,
        plate_id=br00_id,
        illum_output_path=path_to_output_with_illum_csv,
    )

    print(f"Created LoadData CSV with illumination for {plate_name}")


# In[ ]:


# Folder containing the loaddata CSVs
loaddata_dir = output_csv_dir

# Folder containing the platemaps
platemap_dir = pathlib.Path("../0.download_data/metadata/platemaps/").resolve(
    strict=True
)

# Cell lines and time points (hours) that can not be processed, so no LoadData CSV is created
# and no CellProfiler job is run for them
excluded_time_points = {
    "CF1500": [24, 48, 72],
    "CHLA262": [72],
    "Saos-2": [24],
    "X0092": [24, 48, 72],
}

# Make sure every excluded cell line matches a name in the platemaps (avoids a typo silently excluding nothing)
platemap_cell_lines = set(
    pd.concat(
        pd.read_csv(platemap) for platemap in platemap_dir.glob("*_platemap.csv")
    )["cell_line"]
)
missing_cell_lines = set(excluded_time_points) - platemap_cell_lines
if missing_cell_lines:
    raise ValueError(
        f"Excluded cell lines not found in the platemaps: {sorted(missing_cell_lines)}"
    )

# Output folder (cleared first so CSVs from a previous run/machine or renamed cell lines are not left behind)
cell_line_output_dir = output_csv_dir / "cell_line_loaddata_csvs"
if cell_line_output_dir.exists():
    shutil.rmtree(cell_line_output_dir)
cell_line_output_dir.mkdir()

# Iterate through loaddata csvs
for loaddata_csv in loaddata_dir.glob("*_loaddata_with_illum.csv"):

    print(f"\nProcessing loaddata: {loaddata_csv.name}")

    plate_id = loaddata_csv.name.split("_")[0]

    # Find matching platemap
    platemap_path = next(platemap_dir.glob(f"*{plate_id}*.csv"))
    print(f"Using platemap: {platemap_path.name}")

    loaddata_df = pd.read_csv(loaddata_csv)
    platemap_df = pd.read_csv(platemap_path)

    # Iterate through each cell line
    for cell_line in platemap_df["cell_line"].unique():

        cell_line_clean = cell_line.replace(" ", "_")

        cell_line_rows = platemap_df[platemap_df["cell_line"] == cell_line]

        # Skip the cell line if the time point of this plate is excluded
        time_points = [int(time_point) for time_point in cell_line_rows["time_point"].unique()]
        if any(
            time_point in excluded_time_points.get(cell_line, [])
            for time_point in time_points
        ):
            print(f"Excluded: {cell_line} at time point {time_points} ({plate_id})")
            continue

        # Create folder for this cell line
        cell_line_folder = cell_line_output_dir / cell_line_clean
        cell_line_folder.mkdir(exist_ok=True)

        cell_line_platemap = cell_line_rows[["well_position"]]

        filtered_df = loaddata_df.merge(
            cell_line_platemap,
            left_on="Metadata_Well",
            right_on="well_position",
            how="inner",
        )

        output_file = (
            cell_line_folder / f"{plate_id}_{cell_line_clean}_loaddata_with_illum.csv"
        )

        filtered_df.to_csv(output_file, index=False)

        print(f"Saved: {output_file}")

