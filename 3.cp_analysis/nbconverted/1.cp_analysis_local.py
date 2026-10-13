#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction using CellProfiler

# ## Import libraries
# 

# In[1]:


import argparse
import pathlib
import pprint

import sys

sys.path.append("../utils")
import cp_parallel

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[ ]:


# set the run type for the parallelization
run_name = "analysis"

# directory where the CellProfiler pipelines are located (one pipeline per cell line)
pipeline_dir = pathlib.Path("./pipeline").resolve(strict=True)

# set main output dir for all jobs if it doesn't exist
output_dir = pathlib.Path("./sqlite_outputs")
output_dir.mkdir(exist_ok=True)

# directory where the LoadData CSVs are located (one folder per cell line, one CSV per plate/time point)
loaddata_dir = pathlib.Path("./loaddata_csvs/cell_line_loaddata_csvs").resolve(
    strict=True
)

# maximum number of CellProfiler jobs to run at the same time (adjust based on CPUs/RAM)
max_workers = 8

# Each LoadData CSV is one job (one cell line from one plate/time point)
loaddata_csvs = sorted(loaddata_dir.glob("*/*_loaddata_with_illum.csv"))

# Print the number of jobs
print(f"Total number of LoadData CSVs: {len(loaddata_csvs)}")


# ## Create dictionary to process data

# In[ ]:


# create plate info dictionary with all parts of the CellProfiler CLI command to run in parallel
plate_info_dictionary = {}
for loaddata_csv in loaddata_csvs:
    # cell line is the name of the folder holding the LoadData CSV and plate (time point) is the prefix of the file name
    cell_line = loaddata_csv.parent.name
    plate = loaddata_csv.stem.split("_")[0]

    # skip any cell line that does not have a matching pipeline
    path_to_pipeline = pipeline_dir / f"analysis_{cell_line}.cppipe"
    if not path_to_pipeline.exists():
        print(f"Skipping {loaddata_csv.name}: no pipeline found for {cell_line}")
        continue

    # name for the job is the plate and cell line so each job gets its own output folder and log
    name = f"{plate}_{cell_line}"
    plate_info_dictionary[name] = {
        "path_to_loaddata": loaddata_csv,
        "path_to_output": output_dir / name,
        "path_to_pipeline": path_to_pipeline,
    }

# view the dictionary to assess that all info is added correctly
print(f"Total number of jobs to run: {len(plate_info_dictionary)}")
pprint.pprint(plate_info_dictionary, indent=4)


# ## Perform segmentation and feature extraction (analysis)
# 
# Note: This code cell was not ran as we prefer to perform CellProfiler processing tasks via `sh` file (bash script) which is more stable.

# In[ ]:


# run the jobs in batches so the number of CellProfiler processes never exceeds max_workers
job_names = list(plate_info_dictionary)
for start in range(0, len(job_names), max_workers):
    batch = {name: plate_info_dictionary[name] for name in job_names[start : start + max_workers]}
    print(f"Running jobs {start + 1}-{start + len(batch)} of {len(job_names)}")
    cp_parallel.run_cellprofiler_parallel(
        plate_info_dictionary=batch, run_name=run_name
    )

