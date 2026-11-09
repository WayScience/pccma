#!/usr/bin/env python
# coding: utf-8

# # Process MorphEm profiles
# 
# In this notebook, we process the nucleocentric MorphEm profiles of each plate (time point) with pycytominer, the same as the CellProfiler profiles in `4.preprocess_features`.
# 
# For each plate, we:
# 
# 1. Load the MorphEm profile of every cell line on the plate and remove the single cells that failed QC, using the QC annotations of the CellProfiler profiles (`4.preprocess_features/qc_results/`).
# 2. Annotate the single cells with the platemap (cell line, time point, and seeding density of each well).
# 3. **Single-cell profiles:** normalize (standardize to the whole plate) and feature select.
# 4. **Bulk profiles:** aggregate to the well (median), normalize (MAD robustize to the whole plate), and feature select.
# 
# There are no negative control wells in this screen (no treatments), so we normalize to the whole plate.
# 
# We set the MorphEm features in every pycytominer function, since pycytominer only finds the CellProfiler compartments (Cells, Nuclei, and Cytoplasm) by default.
# We do not use the blocklist, since it only includes CellProfiler features.
# 
# > **Note:** We processed every plate by running this notebook as a script (`nbconverted/3.process_morphem_profiles.py`) with `run_morphem_pipeline.sh`.
# > The outputs below are from running the notebook afterwards without overwriting, so it skips each plate because the processed profiles already exist.

# ## Import libraries

# In[1]:


import gc
import os
import pathlib
import pprint
import time

import pandas as pd
from path_utils import get_root_dir
from preprocess_utils import load_plate_passing_qc
from pycytominer import aggregate, annotate, feature_select, normalize
from pycytominer.cyto_utils import output

# root of the repository (found from the .git folder), so every path is built from the root
# and the notebook gives the same paths whether we run it from this folder or from the root
root_dir = get_root_dir()
module_dir = root_dir / "5.morphem_featurization"


# ## Set paths and variables

# In[2]:


# Directory with the MorphEm profiles (one folder per cell line, one parquet file per plate/time point)
morphem_dir = (module_dir / "data/morphem_extracted_profiles").resolve(strict=True)

# extract the plate names from the MorphEm profiles (<plate>_<cell_line>_morphem.parquet)
plate_names = sorted(
    {file.name.split("_")[0] for file in morphem_dir.glob("*/*_morphem.parquet")}
)

# Directory with the converted CellProfiler profiles, to confirm that every job on a plate has a MorphEm profile
converted_dir = (root_dir / "4.preprocess_features/data/converted_profiles").resolve(strict=True)

# Directory with the QC annotations of the CellProfiler profiles (one file per cell line)
qc_results_dir = (root_dir / "4.preprocess_features/qc_results").resolve(strict=True)

# path for platemap directory (one platemap per plate)
platemap_dir = (root_dir / "0.download_data/metadata/platemaps").resolve(strict=True)

# output directories
annotated_dir = module_dir / "data/morphem_annotated_profiles"
single_cell_dir = module_dir / "data/morphem_single_cell_profiles"
bulk_dir = module_dir / "data/morphem_bulk_profiles"
for directory in (annotated_dir, single_cell_dir, bulk_dir):
    directory.mkdir(parents=True, exist_ok=True)

# operations to perform for feature selection (no blocklist, as it only includes CellProfiler features)
feature_select_ops = [
    "drop_na_columns",
    "variance_threshold",
    "correlation_threshold",
]

plate_names


# ## Set plates to process

# In[3]:


# Run configuration, both driven by environment variables so run_morphem_pipeline.sh
# can control them without papermill:
#   PLATE_ID  - restrict to a single plate (used to run each plate as its own
#               process so memory doesn't accumulate across plates). Leave
#               unset to process all plates, e.g. when running interactively.
#   OVERWRITE - if set (1/true/yes), reprocess and overwrite a plate's output
#               even if it already exists. Leave unset (the default) to skip
#               plates that already have output.
plate_id_filter = os.environ.get("PLATE_ID")
if plate_id_filter:
    if plate_id_filter not in plate_names:
        raise ValueError(f"Unknown plate_id in PLATE_ID env var: {plate_id_filter}")
    plate_names = [plate_id_filter]

overwrite = os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes")

plate_info_dictionary = {
    plate_id: {
        "platemap_path": str(platemap_dir / f"{plate_id}_platemap.csv"),
        "annotated_path": str(annotated_dir / f"{plate_id}_morphem_annotated.parquet"),
        "sc_normalized_path": str(single_cell_dir / f"{plate_id}_morphem_sc_normalized.parquet"),
        "sc_feature_selected_path": str(single_cell_dir / f"{plate_id}_morphem_sc_feature_selected.parquet"),
        "bulk_aggregated_path": str(bulk_dir / f"{plate_id}_morphem_bulk_aggregated.parquet"),
        "bulk_normalized_path": str(bulk_dir / f"{plate_id}_morphem_bulk_normalized.parquet"),
        "bulk_feature_selected_path": str(bulk_dir / f"{plate_id}_morphem_bulk_feature_selected.parquet"),
    }
    for plate_id in plate_names
}

pprint.pprint(plate_info_dictionary, indent=4)


# ## Process each plate

# In[4]:


for plate_id, info in plate_info_dictionary.items():
    if pathlib.Path(info["bulk_feature_selected_path"]).exists() and pathlib.Path(
        info["sc_feature_selected_path"]
    ).exists() and not overwrite:
        print(f"Skipping {plate_id}: already processed")
        continue

    # every cell line on the plate needs a MorphEm profile, since we normalize to the whole plate
    expected_jobs = {path.name.replace("_converted.parquet", "") for path in converted_dir.glob(f"*/{plate_id}_*_converted.parquet")}
    extracted_jobs = {path.name.replace("_morphem.parquet", "") for path in morphem_dir.glob(f"*/{plate_id}_*_morphem.parquet")}
    missing_jobs = sorted(expected_jobs - extracted_jobs)
    if missing_jobs:
        raise ValueError(f"{plate_id}: missing MorphEm profiles for {missing_jobs} (run 2.extract_morphem_features.ipynb first)")

    print(f"Processing {plate_id}")
    plate_start_time = time.time()

    # every intermediate data frame of the plate is kept in this one dictionary, which we clear once at the
    # end of the plate (each plate also runs as its own process in run_morphem_pipeline.sh)
    plate_dfs = {}

    # Step 1: remove the single cells that failed QC (per cell line) and concatenate the cell lines
    plate_dfs["single_cell"] = load_plate_passing_qc(
        plate=plate_id,
        converted_dir=morphem_dir,
        qc_results_dir=qc_results_dir,
        profile_suffix="morphem",
    )

    # every MorphEm profile should be from the 224 x 224 crop centered on the nucleus (MORPHEM_INPUT_SIZE in
    # utils/morphem_utils.py), so we do not mix profiles from the previous crop, which do not have this column
    if (
        "Metadata_Nucleocentric_Crop_Size" not in plate_dfs["single_cell"].columns
        or plate_dfs["single_cell"]["Metadata_Nucleocentric_Crop_Size"].ne(224).any()
    ):
        raise ValueError(
            f"{plate_id}: some MorphEm profiles are not from the 224 x 224 crop "
            "(run 2.extract_morphem_features.ipynb again with --overwrite)"
        )

    # Step 2: annotate with the platemap (cell line and time point of each well)
    platemap_df = pd.read_csv(info["platemap_path"])
    plate_dfs["annotated"] = annotate(
        profiles=plate_dfs["single_cell"],
        platemap=platemap_df,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
    )
    # use the same site column name as the CellProfiler profiles
    plate_dfs["annotated"] = plate_dfs["annotated"].rename(columns={"Image_Metadata_Site": "Metadata_Site"})

    assert len(plate_dfs["annotated"]) == len(plate_dfs["single_cell"]), f"{plate_id}: single cells were lost when annotating"
    assert plate_dfs["annotated"]["Metadata_cell_line"].notna().all(), f"{plate_id}: single cells without a cell line from the platemap"

    output(df=plate_dfs["annotated"], output_filename=info["annotated_path"], output_type="parquet")

    morphem_features = [col for col in plate_dfs["annotated"].columns if col.startswith("Nucleocentric_")]
    print(f"{plate_id}: {plate_dfs['annotated'].shape[0]} single cells and {len(morphem_features)} MorphEm features")

    # Step 3: single-cell profiles (standardize to the whole plate, then feature select)
    plate_dfs["sc_normalized"] = normalize(
        profiles=plate_dfs["annotated"],
        features=morphem_features,
        method="standardize",
        samples="all",
    )
    output(df=plate_dfs["sc_normalized"], output_filename=info["sc_normalized_path"], output_type="parquet")
    plate_dfs["sc_feature_selected"] = feature_select(
        profiles=plate_dfs["sc_normalized"],
        features=morphem_features,
        operation=feature_select_ops,
        corr_threshold=0.90,
        na_cutoff=0,
    )
    output(df=plate_dfs["sc_feature_selected"], output_filename=info["sc_feature_selected_path"], output_type="parquet")

    # Step 4: bulk profiles
    # include every platemap column that is constant within a well in the strata, so it is kept after aggregation
    platemap_cols = [f"Metadata_{col}" for col in platemap_df.columns]
    strata_cols = ["Metadata_Plate", "Metadata_Well"] + [
        col
        for col in platemap_cols
        if col in plate_dfs["annotated"].columns
        and col not in ("Metadata_Plate", "Metadata_Well")
        and plate_dfs["annotated"].groupby("Metadata_Well")[col].nunique(dropna=False).le(1).all()
    ]
    plate_dfs["bulk_aggregated"] = aggregate(
        population_df=plate_dfs["annotated"],
        strata=strata_cols,
        features=morphem_features,
        operation="median",
    )
    output(df=plate_dfs["bulk_aggregated"], output_filename=info["bulk_aggregated_path"], output_type="parquet")

    plate_dfs["bulk_normalized"] = normalize(
        profiles=plate_dfs["bulk_aggregated"],
        features=morphem_features,
        method="mad_robustize",
        samples="all",
    )
    output(df=plate_dfs["bulk_normalized"], output_filename=info["bulk_normalized_path"], output_type="parquet")
    plate_dfs["bulk_feature_selected"] = feature_select(
        profiles=plate_dfs["bulk_normalized"],
        features=morphem_features,
        operation=feature_select_ops,
        corr_threshold=0.90,
        na_cutoff=0,
    )
    output(df=plate_dfs["bulk_feature_selected"], output_filename=info["bulk_feature_selected_path"], output_type="parquet")

    # free the memory of every data frame of the plate at once
    plate_dfs.clear()
    gc.collect()

    print(f"Processing completed for {plate_id} in {time.time() - plate_start_time:.1f} seconds")
