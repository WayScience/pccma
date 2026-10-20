#!/usr/bin/env python
# coding: utf-8

# # Sphere whole-screen bulk profiles
# 
# > Run once every plate has been bulk-processed (`3.bulk_processing.ipynb`).
# 
# Adapted from the [SK-N-AS repo1 screen](https://github.com/WayScience/pccma_repo1_screen_image_profiling/blob/main/SK-N-AS_repo1_screen/3.process_features/3b.sphering.ipynb), which was adapted from Erik Serrano's work in the [fibrosis drug screen repository](https://github.com/WayScience/targeted_fibrosis_drug_screen/).
# 
# The atlas has no batches / replicate plate groups to pool, so feature selection and the sphering fit/transform are done once across every plate's pooled normalized profile, producing a single spherized profile for the whole screen (not one per plate).
# 
# Unlike the SK-N-AS screen, the atlas has no negative control wells (no treatments), so we fit the sphering transform on all wells across the screen (`reference_query`).
# We use all wells instead of only the U2-OS wells (on every plate) because there are only 84 U2-OS wells, which is fewer than the number of features, making the sphering transform unstable.

# ## Import libraries

# In[1]:


import os
import pathlib

import pandas as pd
from pycytominer import feature_select, normalize


# ## Set paths and variables

# In[2]:


# Directory containing per-plate bulk profiles from 3.bulk_processing.ipynb
output_dir = pathlib.Path("./data/bulk_profiles").resolve(strict=True)

# Population used to fit the sphering transform ("all" for every well across the screen).
# A pandas query can be used instead to fit on a subset of wells (e.g., 'Metadata_cell_line == "U2-OS"').
reference_query = "all"

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

spherized_output_dir = pathlib.Path("./data/spherized_profiles")
spherized_output_dir.mkdir(parents=True, exist_ok=True)

pooled_feature_select_file = output_dir / "pccma_atlas_pooled_bulk_feature_selected.parquet"
output_spherized_file = spherized_output_dir / "pccma_atlas_pooled_bulk_spherized.parquet"


# ## Run configuration

# In[3]:


# Run configuration, driven by an environment variable so run_pipeline.sh can
# control it without papermill:
#   OVERWRITE - if set (1/true/yes), reprocess and overwrite the spherized
#               profile even if it already exists. Leave unset (the default)
#               to skip if it already exists.
overwrite = os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes")


# ## Sphering step

# In[4]:


if output_spherized_file.exists() and not overwrite:
    print(f"Skipping sphering: already spherized (found {output_spherized_file})")
else:
    normalized_paths = sorted(output_dir.glob("*_bulk_normalized.parquet"))

    print(f"Pooling {len(normalized_paths)} plates for sphering...")

    # step 1: concat every plate's normalized profile before feature selection
    concat_df = pd.concat(
        [pd.read_parquet(path) for path in normalized_paths],
        ignore_index=True,
    ).reset_index(drop=True)

    # step 2a: Apply feature selection across the pooled screen to get a
    # common set of features for sphering.
    print("Feature selecting pooled screen...")
    feature_select_df = feature_select(
        profiles=concat_df,
        operation=feature_select_ops,
        na_cutoff=0,  # updated from default to 0 to remove any columns with any NaN values (best for downstream modeling)
        corr_threshold=0.95,  # increased from default to 0.95 to be more strict as to identify the most informative features
        freq_cut=0.05,  # same as default
        output_file=pooled_feature_select_file,
        output_type="parquet",
    )

    # step 2b: Remove features with too little variation inside the exact
    # reference population used to fit spherization (no change when the
    # reference population is all wells, as step 2a already applied it).
    print(
        "Feature selecting pooled screen with frequency threshold within "
        "the reference population only..."
    )
    zero_reference_var_fs_df = feature_select(
        profiles=feature_select_df,
        operation="frequency_threshold",
        freq_cut=0.05,  # same as default
        unique_cut=0.01,  # same as default
        samples=reference_query,
    )

    # step 3: Spherize/whiten the whole pooled screen using the reference population
    print("Sphering pooled screen using the reference population...")
    normalize(
        profiles=zero_reference_var_fs_df,
        method="spherize",
        samples=reference_query,
        spherize_center=True,
        spherize_method="ZCA-cor",  # same as default
        spherize_epsilon=1e-6,  # same as default
        output_file=output_spherized_file,
        output_type="parquet",
    )

    print(f"Saved feature-selected profiles to {pooled_feature_select_file}")
    print(f"Saved whole-screen spherized profile to {output_spherized_file}")

    # Confirm every well across the screen is in the spherized profile
    spherized_df = pd.read_parquet(output_spherized_file)
    assert len(spherized_df) == len(concat_df), "Wells were lost when sphering"
    print(spherized_df.shape)

