#!/usr/bin/env python
# coding: utf-8

# # Validate the CellProfiler outputs
# 
# In this notebook, we confirm that every CellProfiler job (one cell line from one plate/time point) completed properly.
# For each job, we compare the outputs to the LoadData CSV that was used to run the job and check that:
# 
# 1. The output folder and SQLite file exist and the SQLite file can be read.
# 2. Every image set (well and site) in the LoadData CSV is in the `Per_Image` table (a job that failed or was stopped early will be missing image sets).
# 3. The number of rows in each object table matches the object counts in the `Per_Image` table.
# 4. There is a nuclei and cells outline image for every image set.
# 
# Image sets where no nuclei were segmented are reported but are not treated as a failure.

# ## Import libraries

# In[1]:


import os
import pathlib
import sqlite3

import pandas as pd

import sys

sys.path.append("../utils")
from bandicoot_utils import bandicoot_check


# ## Helper functions

# In[2]:


def validate_job(name: str, loaddata_csv: pathlib.Path) -> dict:
    """Compare the outputs of one CellProfiler job to the LoadData CSV used to run it.

    Parameters
    ----------
    name : str
        name of the job (<plate>_<cell_line>), which is the name of the output folder
    loaddata_csv : pathlib.Path
        path to the LoadData CSV used to run the job

    Returns
    -------
    dict
        summary of the job, where `issues` lists every problem found (empty if the job completed properly)
    """
    job_dir = output_dir / name
    sqlite_path = job_dir / sqlite_name
    issues = []

    # image sets (well and site) that the job should have processed
    loaddata_df = pd.read_csv(loaddata_csv, usecols=["Metadata_Well", "Metadata_Site"])
    expected_image_sets = set(
        zip(loaddata_df["Metadata_Well"], loaddata_df["Metadata_Site"].astype(int))
    )

    result = {
        "job": name,
        "expected_image_sets": len(expected_image_sets),
        "processed_image_sets": 0,
        "nuclei": 0,
        "cells": 0,
        "image_sets_without_nuclei": 0,
        "sqlite_size_mb": 0.0,
    }

    if not sqlite_path.is_file():
        result["issues"] = ["SQLite file not found"]
        return result
    result["sqlite_size_mb"] = round(sqlite_path.stat().st_size / 1e6, 1)

    try:
        # open as read only so nothing is written to the output folder
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro&immutable=1", uri=True)
        try:
            image_df = pd.read_sql(
                "SELECT Image_Metadata_Well, Image_Metadata_Site, "
                + ", ".join(object_tables.values())
                + " FROM Per_Image",
                conn,
            )
            object_rows = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in object_tables
            }
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        result["issues"] = [f"SQLite file could not be read ({error})"]
        return result

    # every image set in the LoadData CSV should be in the Per_Image table exactly once
    processed_image_sets = list(
        zip(image_df["Image_Metadata_Well"], image_df["Image_Metadata_Site"].astype(int))
    )
    result["processed_image_sets"] = len(processed_image_sets)
    missing_image_sets = expected_image_sets - set(processed_image_sets)
    if missing_image_sets:
        issues.append(f"{len(missing_image_sets)} image sets missing from Per_Image")
    if set(processed_image_sets) - expected_image_sets:
        issues.append("image sets in Per_Image that are not in the LoadData CSV")
    if len(processed_image_sets) != len(set(processed_image_sets)):
        issues.append("duplicate image sets in Per_Image")

    # the object counts are empty if an image set was started but the measurements were not saved
    if image_df[list(object_tables.values())].isna().any().any():
        issues.append("image sets with empty object counts in Per_Image")

    # the number of rows in each object table should match the object counts in the Per_Image table
    for table, count_column in object_tables.items():
        expected_rows = int(image_df[count_column].fillna(0).sum())
        if object_rows[table] != expected_rows:
            issues.append(
                f"{table} has {object_rows[table]} rows but Per_Image counts {expected_rows}"
            )

    result["nuclei"] = object_rows["Per_Nuclei"]
    result["cells"] = object_rows["Per_Cells"]
    result["image_sets_without_nuclei"] = int((image_df["Image_Count_Nuclei"] == 0).sum())

    # there should be one outline image per image set for both nuclei and cells
    output_files = {file.name for file in job_dir.iterdir()}
    plate = name.split("_")[0]
    for prefix in outline_prefixes:
        missing_outlines = [
            (well, site)
            for well, site in expected_image_sets
            if f"{prefix}_{plate}_{well}_{site}.tiff" not in output_files
        ]
        if missing_outlines:
            issues.append(f"{len(missing_outlines)} {prefix} images missing")

    result["issues"] = issues
    return result


# ## Set paths and variables

# In[3]:


# Find root directory of the project
root_dir = pathlib.Path().cwd()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# directory with one folder of CellProfiler outputs per job (<plate>_<cell_line>)
output_dir = pathlib.Path(f"{image_base_dir}/PCCMA_data/Atlas_v2_profiles/sqlite_outputs/").resolve(
    strict=True
)

# directory where the LoadData CSVs are located (one folder per cell line, one CSV per plate/time point)
loaddata_dir = pathlib.Path("./loaddata_csvs/cell_line_loaddata_csvs").resolve(
    strict=True
)

# name of the SQLite file and the outline images that each job creates (set in the pipelines)
sqlite_name = "alsf_morphology_features.sqlite"
outline_prefixes = ["NucleiOutlines", "CellsOutlines"]

# object tables in the SQLite file and the matching object count column in the Per_Image table
object_tables = {
    "Per_Nuclei": "Image_Count_Nuclei",
    "Per_Cells": "Image_Count_Cells",
    "Per_Cytoplasm": "Image_Count_Cytoplasm",
}

# Each LoadData CSV is one job (one cell line from one plate/time point)
loaddata_csvs = sorted(loaddata_dir.glob("*/*_loaddata_with_illum.csv"))
print(f"Total number of expected jobs: {len(loaddata_csvs)}")


# ## Check that the output folders match the expected jobs

# In[4]:


# name for the job is the plate (time point) and cell line, which is the name of the output folder
expected_jobs = {
    f"{loaddata_csv.stem.split('_')[0]}_{loaddata_csv.parent.name}": loaddata_csv
    for loaddata_csv in loaddata_csvs
}
output_folders = {folder.name for folder in output_dir.iterdir() if folder.is_dir()}

missing_folders = sorted(set(expected_jobs) - output_folders)
unexpected_folders = sorted(output_folders - set(expected_jobs))

print(f"Number of output folders: {len(output_folders)}")
print(f"Jobs without an output folder: {missing_folders}")
print(f"Output folders without a LoadData CSV: {unexpected_folders}")


# ## Validate the outputs of each job

# In[5]:


results = []
for name, loaddata_csv in expected_jobs.items():
    if name in missing_folders:
        results.append({"job": name, "issues": ["output folder not found"]})
        continue
    results.append(validate_job(name=name, loaddata_csv=loaddata_csv))

results_df = pd.DataFrame(results)
results_df["passed"] = results_df["issues"].apply(lambda issues: len(issues) == 0)

print(f"Jobs that completed properly: {results_df['passed'].sum()} of {len(results_df)}")
print(f"Total image sets processed: {int(results_df['processed_image_sets'].sum())}")
print(f"Total nuclei: {int(results_df['nuclei'].sum())}")
print(f"Total cells: {int(results_df['cells'].sum())}")


# ## Jobs with issues
# 
# Any job listed here did not complete properly and will need to be run again.

# In[6]:


failed_df = results_df[~results_df["passed"]]

if failed_df.empty:
    print("All jobs completed properly!")
else:
    for _, row in failed_df.iterrows():
        print(f"{row['job']}: {'; '.join(row['issues'])}")


# ## Jobs with image sets where no nuclei were segmented
# 
# These jobs completed properly, but have at least one image set (FOV) without any nuclei.
# This is expected for sparse or poor quality FOVs and is reported here to be aware of.

# In[7]:


no_nuclei_df = results_df[results_df["image_sets_without_nuclei"] > 0]

print(f"Jobs with image sets without nuclei: {len(no_nuclei_df)}")
no_nuclei_df[
    ["job", "expected_image_sets", "image_sets_without_nuclei", "nuclei", "cells"]
].sort_values("image_sets_without_nuclei", ascending=False)


# ## Jobs with the fewest nuclei
# 
# A very low number of nuclei for a job can point to a segmentation problem for that cell line or time point.

# In[8]:


results_df[
    ["job", "expected_image_sets", "processed_image_sets", "nuclei", "cells", "sqlite_size_mb"]
].sort_values("nuclei").head(15)

