#!/usr/bin/env python
# coding: utf-8

# # Annotate single-cell profiles
# 
# > Shared step used by both the bulk (`3.bulk_processing.ipynb`) and single-cell (`4.single_cell_normalize.ipynb`) branches downstream, so `annotate()` only runs once per plate.
# 
# For each plate, we load the converted profiles of every cell line on the plate, merge the QC annotations of the cell line (from `1.sc_quality_control.ipynb`), and remove the single cells that failed QC.
# We then concatenate the cell lines into one plate data frame and annotate it with the platemap (cell line, time point, and seeding density of each well).

# ## Import libraries

# In[1]:


import csv
import gc
import os
import pathlib
import pprint
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from pycytominer import annotate
from pycytominer.cyto_utils import output

sys.path.append("../utils")
from preprocess_utils import load_plate_passing_qc


# ## Set paths and variables

# In[2]:


# Directory with the converted profiles (one folder per cell line, one parquet file per plate/time point)
converted_dir = pathlib.Path("./data/converted_profiles").resolve(strict=True)

# extract the plate names from the converted profiles (<plate>_<cell_line>_converted.parquet)
plate_names = sorted(
    {file.name.split("_")[0] for file in converted_dir.glob("*/*_converted.parquet")}
)

# Directory with the QC annotations from single-cell QC (one file per cell line)
qc_results_dir = pathlib.Path("./qc_results").resolve(strict=True)

# Output path for annotated profiles. Shared by both the bulk and single-cell
# downstream steps, so annotate() only needs to run once per plate.
output_dir = pathlib.Path("./data/annotated_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# path for platemap directory (one platemap per plate)
platemap_dir = pathlib.Path("../0.download_data/metadata/platemaps").resolve(strict=True)

# Rename map applied after annotate() so both downstream branches see a
# consistent site column name (CytoTable's raw output calls it
# Image_Metadata_Site; pycytominer's own convention is Metadata_Site).
column_name_mapping = {
    "Image_Metadata_Site": "Metadata_Site",
}

plate_names


# ## Set dictionary with plates to process

# In[3]:


# Create plate info dictionary
plate_info_dictionary = {
    plate_id: {
        "platemap_path": str(platemap_dir / f"{plate_id}_platemap.csv"),
    }
    for plate_id in plate_names
}

# Display the dictionary to verify the entries
pprint.pprint(plate_info_dictionary, indent=4)


# In[4]:


# Run configuration, both driven by environment variables so run_pipeline.sh
# can control them without papermill:
#   PLATE_ID  - restrict to a single plate (used to run each plate as its own
#               process so memory doesn't accumulate across plates). Leave
#               unset to process all plates, e.g. when running interactively.
#   OVERWRITE - if set (1/true/yes), reprocess and overwrite a plate's output
#               even if it already exists. Leave unset (the default) to skip
#               plates that already have output.
plate_id_filter = os.environ.get("PLATE_ID")
if plate_id_filter:
    if plate_id_filter not in plate_info_dictionary:
        raise ValueError(f"Unknown plate_id in PLATE_ID env var: {plate_id_filter}")
    plate_info_dictionary = {plate_id_filter: plate_info_dictionary[plate_id_filter]}

overwrite = os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes")

plate_info_dictionary


# ## Annotate profiles

# In[5]:


timing_log_path = output_dir / "timing_log.csv"

for plate_id, info in plate_info_dictionary.items():
    output_annotated_file = str(output_dir / f"{plate_id}_annotated.parquet")

    # Already annotated, so this plate can safely be skipped on a rerun
    # (unless OVERWRITE is set, in which case it's reprocessed regardless)
    if pathlib.Path(output_annotated_file).exists() and not overwrite:
        print(f"Skipping {plate_id}: already annotated (found {output_annotated_file})")
        continue

    print(f"Annotating {plate_id}")
    plate_start_time = time.time()

    # Drop poor-quality segmentations per cell line BEFORE concatenating the cell
    # lines into the plate and annotating (QC annotations are one file per cell line)
    single_cell_df = load_plate_passing_qc(
        plate=plate_id, converted_dir=converted_dir, qc_results_dir=qc_results_dir
    )
    platemap_df = pd.read_csv(info["platemap_path"])

    # Annotation (platemap includes the cell line and time point of each well)
    annotated_df = annotate(
        profiles=single_cell_df,
        platemap=platemap_df,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
    )
    annotated_df.rename(columns=column_name_mapping, inplace=True)

    # Assert "Metadata_Site" is now present after the rename() call above
    assert "Metadata_Site" in annotated_df.columns, f"{plate_id}: Metadata_Site column missing after rename()"

    # Every single cell should match a well in the platemap
    assert len(annotated_df) == len(single_cell_df), f"{plate_id}: single cells were lost when annotating"
    assert annotated_df["Metadata_cell_line"].notna().all(), f"{plate_id}: single cells without a cell line from the platemap"

    output(
        df=annotated_df,
        output_filename=output_annotated_file,
        output_type="parquet",
    )
    annotate_seconds = time.time() - plate_start_time

    # Clear memory
    del single_cell_df, platemap_df, annotated_df
    gc.collect()

    print(f"Annotation completed for {plate_id}! (annotate={annotate_seconds:.1f}s)")

    # Record timing so per-plate performance can be compared across runs
    write_header = not timing_log_path.exists()
    with open(timing_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["plate_id", "annotate_seconds", "timestamp"])
        writer.writerow(
            [
                plate_id,
                f"{annotate_seconds:.1f}",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ]
        )

