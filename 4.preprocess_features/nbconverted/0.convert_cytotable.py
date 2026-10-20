#!/usr/bin/env python
# coding: utf-8

# # Convert SQLite outputs to parquet files with CytoTable
# 
# In this notebook, we convert the SQLite file from each CellProfiler job (one cell line from one plate/time point) to a single-cell parquet file.
# We save the parquet files in one folder per cell line:
# 
# ```
# data/converted_profiles/<cell_line>/<plate>_<cell_line>_converted.parquet
# ```
# 
# Each cell line folder holds one parquet file per plate (time point), so we can load all plates of a cell line (or all cell lines) with a glob in single-cell QC instead of concatenating the files here.

# ## Import libraries

# In[1]:


import logging
import os
import pathlib
import sys

import pandas as pd

# cytotable will merge objects from SQLite file into single cells and save as parquet file
from cytotable import convert, presets

sys.path.append("../utils")
from bandicoot_utils import bandicoot_check

# Set the logging level to a higher level to avoid outputting unnecessary errors from config file in convert function
logging.getLogger().setLevel(logging.ERROR)


# ## Set paths and variables

# In[2]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"

# update preset to include both the site metadata, cell counts, and PathName columns
joins = presets.config[preset]["CONFIG_JOINS"].replace(
    "Image_Metadata_Well,",
    "Image_Metadata_Well, Image_Metadata_Site, Image_Count_Cells, Image_Metadata_Row, Image_Metadata_Col, ",
)

# Add the PathName columns separately
joins = joins.replace(
    "COLUMNS('Image_FileName_.*'),",
    "COLUMNS('Image_FileName_.*'),\n COLUMNS('Image_PathName_.*'),",
)

# type of file output from cytotable (currently only parquet)
dest_datatype = "parquet"

# name of the SQLite file that each job creates (set in the pipelines)
sqlite_name = "alsf_morphology_features.sqlite"

# Find root directory of the project
root_dir = pathlib.Path().cwd()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# directory with one folder of CellProfiler outputs per job (<plate>_<cell_line>)
sqlite_dir = pathlib.Path(f"{image_base_dir}/PCCMA_data/Atlas_v2_profiles/sqlite_outputs/").resolve(
    strict=True
)

# directory for converted profiles (one folder per cell line)
output_dir = pathlib.Path("data/converted_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# columns to move to the front of the data frame and add the "Metadata" prefix to
prioritized_columns = [
    "Nuclei_Location_Center_X",
    "Nuclei_Location_Center_Y",
    "Cells_Location_Center_X",
    "Cells_Location_Center_Y",
    "Image_Count_Cells",
]

# Each SQLite file is one job (one cell line from one plate/time point)
sqlite_paths = sorted(sqlite_dir.glob(f"*/{sqlite_name}"))
print(f"There are {len(sqlite_paths)} SQLite files to convert.")


# ## Convert SQLite to parquet files
# 
# For each job, we:
# 
# 1. Merge the single cells with CytoTable and save as a parquet file.
# 2. Move the location and cell count columns to the front and add the `Metadata` prefix.
# 3. Save the parquet file in the folder of the cell line.
# 
# We skip any job that already has a parquet file, so we can run the notebook again without converting everything again.

# In[3]:


for sqlite_path in sqlite_paths:
    # name of the job folder is <plate>_<cell_line> (the cell line can include underscores)
    plate, cell_line = sqlite_path.parent.name.split("_", 1)

    # one folder per cell line with one parquet file per plate (time point)
    cell_line_dir = output_dir / cell_line
    cell_line_dir.mkdir(exist_ok=True)
    output_path = cell_line_dir / f"{plate}_{cell_line}_converted.parquet"

    if output_path.exists():
        print(f"Skipping {plate} {cell_line} (already converted)")
        continue

    # convert to a temporary file first so a stopped run does not leave an unfinished parquet file
    temp_path = cell_line_dir / f"{plate}_{cell_line}_temp.parquet"
    temp_path.unlink(missing_ok=True)

    # Merge single cells and output as parquet file
    convert(
        source_path=str(sqlite_path),
        dest_path=str(temp_path),
        dest_datatype=dest_datatype,
        preset=preset,
        joins=joins,
        chunk_size=15000,
    )

    converted_df = pd.read_parquet(temp_path)

    # If any, drop rows where "Metadata_ImageNumber" is NaN (artifact of cytotable)
    converted_df = converted_df.dropna(subset=["Metadata_ImageNumber"])

    # Rearrange columns and add "Metadata" prefix
    converted_df = converted_df[
        prioritized_columns
        + [col for col in converted_df.columns if col not in prioritized_columns]
    ].rename(
        columns=lambda col: "Metadata_" + col if col in prioritized_columns else col
    )

    # assert that there are column names with PathName in the dataset
    assert any("PathName" in col for col in converted_df.columns)

    # Assert that Metadata_Row and Metadata_Col are present for downstream QC
    assert {"Image_Metadata_Row", "Image_Metadata_Col"}.issubset(
        converted_df.columns
    ), "Missing required Metadata columns: Row and/or Col"

    converted_df.to_parquet(output_path, index=False)
    temp_path.unlink()

    print(f"Converted {plate} {cell_line} with the shape: {converted_df.shape}")


# ## Confirm that every job has a parquet file

# In[4]:


parquet_paths = sorted(output_dir.glob("*/*_converted.parquet"))
cell_line_dirs = [folder for folder in output_dir.iterdir() if folder.is_dir()]

print(
    f"There are {len(parquet_paths)} parquet files across {len(cell_line_dirs)} cell lines."
)
assert len(parquet_paths) == len(sqlite_paths), "Not every SQLite file was converted"

# print shape and head of one converted data frame
converted_df = pd.read_parquet(parquet_paths[0])
print(converted_df.shape)
converted_df.head()


# **To confirm the number of single cells is correct above, please use any database browser software to see if the number of rows in the "Per_Cells" compartment matches the number of rows in the data frame.**
