#!/usr/bin/env python
# coding: utf-8

# # Create LoadData CSVs to use for image qc metric extraction
# 
# In this notebook, we create a LoadData CSV that contains paths to each channel per image set for CellProfiler to process. 
# We can use this LoadData CSV to run whole image QC to evaluate thresholds.

# ## Import libraries

# In[1]:


import os
import pathlib
import sys

import yaml

sys.path.append("../utils")
import loaddata_utils as ld_utils
from bandicoot_utils import bandicoot_check


# ## Set paths

# In[2]:


# Find root directory of the project
root_dir = pathlib.Path().cwd()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# Paths for parameters to make loaddata csv
index_directory = pathlib.Path(f"{image_base_dir}/PCCMA_data/Atlas_v2_screen/")
config_dir_path = pathlib.Path("./load_data_config").absolute()
output_csv_dir = pathlib.Path("./loaddata_csvs/")
output_csv_dir.mkdir(parents=True, exist_ok=True)

# Find all 'Images' folders within the directory
images_folders = list(index_directory.rglob("Images"))
print(f"Found {len(images_folders)} 'Images' folders in {index_directory}")


# ## Create LoadData CSVs for all data

# In[3]:


# Define the config path
config_path = config_dir_path / "config.yml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Config loaded successfully:")
print(config)


# In[4]:


# Remove any existing LoadData CSVs to avoid stale data from previous runs
for old_csv in output_csv_dir.glob("*.csv"):
    old_csv.unlink()

# Create a LoadData CSV for each of the 9 plates
for images_folder in sorted(images_folders):
    plate_folder = images_folder.parent
    br00_id = plate_folder.name.split("__")[0]

    path_to_output_csv = (output_csv_dir / f"{br00_id}_loaddata.csv").absolute()
    print(f"Creating LoadData CSV for plate {br00_id}")

    ld_utils.create_loaddata_csv(
        index_directory=images_folder,
        config_path=config_path,
        path_to_output=path_to_output_csv,
    )
    print(f"Created LoadData CSV for {br00_id} at {path_to_output_csv}")

