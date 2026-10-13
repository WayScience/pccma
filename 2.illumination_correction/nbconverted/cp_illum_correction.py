#!/usr/bin/env python
# coding: utf-8

# # Calculate illumination correction functions per channel (all across images)

# ## Import libraries

# In[1]:


import math
import pathlib
import pprint
import re

import sys

sys.path.append("../utils")
import cp_parallel


# ## Helper functions

# In[2]:


def update_blur_threshold(match: re.Match) -> str:
    """Update the blur threshold if it doesn't match the QC value.
    
    Args:
        match (re.Match): A regex match object containing the matched text.
    
    Returns:
        str: The updated text with the new blur threshold if it was changed, otherwise the original
             matched text.
    """
    prefix, channel, current_value = match.group(1), match.group(2), match.group(3)
    qc_value = qc_blur_thresholds.get(channel)
    if qc_value is not None and not math.isclose(float(current_value), qc_value):
        blur_updates[channel] = {"old": float(current_value), "new": qc_value}
        return f"{prefix}{qc_value}"
    return match.group(0)

def update_saturation_threshold(match: re.Match) -> str:
    """Update the saturation threshold if it doesn't match 1.0.

    Args:
        match (re.Match): A regex match object containing the matched text.
    
    Returns:
        str: The updated text with the new saturation threshold if it was changed, otherwise the original
             matched text.
    """
    prefix, channel, current_value = match.group(1), match.group(2), match.group(3)
    if not math.isclose(float(current_value), 1.0):
        saturation_updates[channel] = {"old": float(current_value), "new": 1.0}
        return f"{prefix}1.0"
    return match.group(0)


# ## Set paths and variables

# In[3]:


# set the run type for the parallelization
run_name = "illum_correction"

# set path for CellProfiler pipeline
path_to_pipeline = pathlib.Path("./pipeline/illum.cppipe").resolve(strict=True)

# set main output dir for all plates if it doesn't exist
output_dir = pathlib.Path("./illum_directory")
output_dir.mkdir(exist_ok=True)

# directory where loaddata CSVs are located within the folder
loaddata_dir = pathlib.Path("../1.whole_image_qc/loaddata_csvs/").resolve(strict=True)

# Extract plate names and include as list
plate_names = [file.stem.split("_")[0] for file in loaddata_dir.glob("*.csv")]

# Print the number of plates and their names
print(f"Total number of plates: {len(plate_names)}")
print("Plate Names:")
for name in plate_names:
    print(name)


# ## Sync blur thresholds in the pipeline with whole image QC

# In[4]:


# path to the blur thresholds calculated during whole image QC
path_to_blur_thresholds = pathlib.Path(
    "../1.whole_image_qc/blur_thresholds.txt"
).resolve(strict=True)

# load the blur thresholds calculated during whole image QC (format: "OrigDNA: -2.36")
with open(path_to_blur_thresholds, "r") as f:
    qc_blur_thresholds = {
        channel.strip(): float(value)
        for channel, value in (line.split(":", 1) for line in f if line.strip())
    }

# find each blur (PowerLogLogSlope) line
blur_pattern = re.compile(
    r"(Which measurement\?:ImageQuality_PowerLogLogSlope_(\w+)\n"
    r"\s*Flag images based on low values\?:Yes\n"
    r"\s*Minimum value:)(-?\d+\.\d+)"
)

# Initialize a dictionary to store blur updates
blur_updates = {}

# Set updated text for the pipeline with updated blur thresholds
updated_pipeline_text = blur_pattern.sub(update_blur_threshold, path_to_pipeline.read_text())

# Check if updates were made and write to the pipeline file if necessary
if blur_updates:
    path_to_pipeline.write_text(updated_pipeline_text)
    print("Updated blur thresholds in the pipeline to match whole image QC:")
    pprint.pprint(blur_updates, indent=4)
else:
    print("Blur thresholds in the pipeline already match whole image QC.")


# ## Ensure saturation threshold is set to 1 for all channels

# In[5]:


# find each saturation (PercentMaximal) line
saturation_pattern = re.compile(
    r"(Which measurement\?:ImageQuality_PercentMaximal_(\w+)\n"
    r"\s*Flag images based on low values\?:No\n"
    r"\s*Minimum value:[\d.]+\n"
    r"\s*Flag images based on high values\?:Yes\n"
    r"\s*Maximum value:)(\d+\.?\d*)"
)

# Initialize a dictionary to store saturation updates
saturation_updates = {}

# Create text to update saturation thresholds to 1.0
updated_pipeline_text = saturation_pattern.sub(
    update_saturation_threshold, path_to_pipeline.read_text()
)

# Check if updates were made and write to the pipeline file if necessary
if saturation_updates:
    path_to_pipeline.write_text(updated_pipeline_text)
    print("Updated saturation thresholds in the pipeline to 1:")
    pprint.pprint(saturation_updates, indent=4)
else:
    print("Saturation thresholds in the pipeline are already set to 1 for every channel.")


# ## Create dictionary to process data

# In[6]:


# create plate info dictionary with all parts of the CellProfiler CLI command to run in parallel
plate_info_dictionary = {
    name: {
        "path_to_loaddata": next(loaddata_dir.glob(f"{name}*.csv"), None),
        "path_to_output": output_dir / name,
        "path_to_pipeline": path_to_pipeline,
    }
    for name in plate_names
    if next(loaddata_dir.glob(f"{name}*.csv"), None)
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Calculate IC functions on data
# 
# Note: This code cell was not ran as we prefer to perform CellProfiler processing tasks via `sh` file (bash script) which is more stable.

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=plate_info_dictionary, run_name=run_name
)

