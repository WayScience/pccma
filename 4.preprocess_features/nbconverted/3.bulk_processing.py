#!/usr/bin/env python
# coding: utf-8

# # Process bulk profiles
# 
# > NOTE: For bulk profiles, we use MAD robustize as the "standard" normalization.
# > There are no negative control wells in this screen (no treatments), so we normalize to the whole plate.

# ## Import libraries

# In[1]:


import csv
import gc
import os
import pathlib
import pprint
import time
from datetime import datetime, timezone

import pandas as pd
from pycytominer import aggregate, feature_select, normalize
from pycytominer.cyto_utils import output


# ## Set paths and variables

# In[2]:


# Directory with the converted profiles (one folder per cell line, one parquet file per plate/time point)
converted_dir = pathlib.Path("./data/converted_profiles").resolve(strict=True)

# extract the plate names from the converted profiles (<plate>_<cell_line>_converted.parquet)
plate_names = sorted(
    {file.name.split("_")[0] for file in converted_dir.glob("*/*_converted.parquet")}
)

# Directory containing per-plate annotated profiles from 2.annotate.ipynb
annotated_dir = pathlib.Path("./data/annotated_profiles")

# output path for bulk profiles
output_dir = pathlib.Path("./data/bulk_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# path for platemap directory (one platemap per plate)
platemap_dir = pathlib.Path("../0.download_data/metadata/platemaps").resolve(strict=True)

# operations to perform for feature selection
# NOTE: drop_na_columns runs before correlation_threshold on purpose. Several
# Costes correlation features are NaN for some cells, and pycytominer's
# correlation_threshold falls back to a much slower NaN-aware pandas .corr()
# (instead of a fast BLAS np.corrcoef) if any NaNs are still present in the
# feature columns it's given. Dropping NaN columns first keeps it on the fast path.
feature_select_ops = [
    "drop_na_columns",
    "blocklist",  # default block list uses same standard CP naming convention
    "frequency_threshold",
    "variance_threshold",
    "correlation_threshold",
]

plate_names


# ## Set dictionary with plates to process

# In[3]:


# Create plate info dictionary
plate_info_dictionary = {
    plate_id: {
        # Platemap, only used to know which metadata columns annotate() added
        # (see strata_cols below)
        "platemap_path": str(platemap_dir / f"{plate_id}_platemap.csv"),
        # Annotated profiles are produced per-plate by 2.annotate.ipynb
        "annotated_path": (
            str((annotated_dir / f"{plate_id}_annotated.parquet").resolve())
            if (annotated_dir / f"{plate_id}_annotated.parquet").exists()
            else None
        ),
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


# ## Process data with pycytominer

# In[5]:


timing_log_path = output_dir / "timing_log.csv"

for plate_id, info in plate_info_dictionary.items():
    if info["annotated_path"] is None:
        print(
            f"Skipping {plate_id}: not yet annotated "
            "(run 2.annotate.ipynb for this plate first)"
        )
        continue

    # Output file paths for each file
    output_aggregated_file = str(output_dir / f"{plate_id}_bulk_aggregated.parquet")
    output_normalized_file = str(output_dir / f"{plate_id}_bulk_normalized.parquet")
    output_feature_select_file = str(
        output_dir / f"{plate_id}_bulk_feature_selected.parquet"
    )

    # Already fully processed, so this plate can safely be skipped on a rerun
    # (unless OVERWRITE is set, in which case it's reprocessed regardless)
    if pathlib.Path(output_feature_select_file).exists() and not overwrite:
        print(f"Skipping {plate_id}: already processed (found {output_feature_select_file})")
        continue

    print(f"Now performing pycytominer pipeline for {plate_id}")
    plate_start_time = time.time()

    # Load the already-annotated single-cell profile (shared with
    # 4.single_cell_normalize.ipynb, so annotate() only ran once for this plate)
    annotated_df = pd.read_parquet(info["annotated_path"])

    # Step 1: Aggregation
    # aggregate() only keeps `strata` columns plus the aggregated numeric
    # features, dropping everything else. Rather than re-annotating the
    # aggregated (well-level) data afterward to reattach platemap metadata,
    # include every platemap-derived metadata column that's constant within a
    # well directly in `strata`, so it survives aggregation.
    aggregate_start_time = time.time()
    platemap_cols = [
        f"Metadata_{col}" for col in pd.read_csv(info["platemap_path"], nrows=0).columns
    ]
    candidate_metadata_cols = [
        col for col in platemap_cols
        if col in annotated_df.columns
        and col not in ("Metadata_Plate", "Metadata_Well")
    ]
    strata_cols = ["Metadata_Plate", "Metadata_Well"] + [
        col for col in candidate_metadata_cols
        if annotated_df.groupby("Metadata_Well")[col].nunique(dropna=False).le(1).all()
    ]
    aggregated_df = aggregate(
        population_df=annotated_df,
        operation="median",
        strata=strata_cols,
    )
    output(
        df=aggregated_df,
        output_filename=output_aggregated_file,
        output_type="parquet",
    )
    aggregate_seconds = time.time() - aggregate_start_time

    # Clear memory -- annotated_df is full single-cell scale; everything
    # from here on operates on the much smaller well-level aggregated_df
    del annotated_df
    gc.collect()

    # Step 2: Normalization (whole-plate) -- feed aggregate()'s own in-memory
    # result directly instead of re-reading output_aggregated_file back off disk.
    normalize_start_time = time.time()
    normalize(
        profiles=aggregated_df,
        method="mad_robustize",  # use robustize to avoid influence of outliers in the normalization
        samples="all",  # no negative controls in this screen, so normalize to the whole plate
        output_file=output_normalized_file,
        output_type="parquet",
    )
    normalize_seconds = time.time() - normalize_start_time

    # Clear memory
    del aggregated_df
    gc.collect()

    # Step 3: Feature selection
    feature_select_start_time = time.time()
    feature_select(
        output_normalized_file,
        operation=feature_select_ops,
        corr_threshold=0.90,  # keep the same default value for correlation_threshold to be more strict as to identify the most informative features
        freq_cut=0.05,  # keep the same default value for freq_cut
        unique_cut=0.01,  # keep the same default value for unique_cut
        na_cutoff=0,  # update na_cutoff from default to 0 to remove any columns with any NaN values (best for downstream modeling)
        output_file=output_feature_select_file,
        output_type="parquet",
    )
    feature_select_seconds = time.time() - feature_select_start_time

    total_seconds = time.time() - plate_start_time
    print(
        f"Bulk processing completed for {plate_id}! "
        f"(aggregate={aggregate_seconds:.1f}s, "
        f"normalize={normalize_seconds:.1f}s, feature_select={feature_select_seconds:.1f}s, "
        f"total={total_seconds:.1f}s)"
    )

    # Record timing so per-plate performance can be compared across runs
    write_header = not timing_log_path.exists()
    with open(timing_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["plate_id", "aggregate_seconds", "normalize_seconds", "feature_select_seconds", "total_seconds", "timestamp"])
        writer.writerow(
            [
                plate_id,
                f"{aggregate_seconds:.1f}",
                f"{normalize_seconds:.1f}",
                f"{feature_select_seconds:.1f}",
                f"{total_seconds:.1f}",
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ]
        )

