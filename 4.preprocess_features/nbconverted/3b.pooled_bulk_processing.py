#!/usr/bin/env python
# coding: utf-8

# # Pool bulk profiles across plates
# 
# This notebook pools the per-plate normalized bulk profiles from `3.bulk_processing.ipynb` into **one file** and runs feature selection once on all wells together, so every plate and time point has the same features.
# 
# **Why:** `3.bulk_processing.ipynb` selects features for each plate on its own, so each plate keeps a different set of features (only 57 are shared by all nine plates). Plots and analyses across plates, such as a UMAP or heat map of all wells, need one common set.
# 
# **Steps**
# 1. Read `<plate>_bulk_normalized.parquet` for every plate. Each plate was already normalized on its own (MAD robustize to all wells of the plate), and this notebook does not change those values.
# 2. Stack the wells of all plates into one table.
# 3. Run pycytominer feature selection on the stacked table (see the settings below).
# 4. Save `pccma_atlas_pooled_bulk_feature_selected.parquet` to `data/bulk_profiles/` and copy it to the bandicoot mount when it is available.
# 
# **Metadata columns.** The output has these columns first, then the features:
# 
# | Column | Description |
# | --- | --- |
# | `Metadata_Plate` | Plate barcode (e.g., `BR00150698`). One plate is one time point. |
# | `Metadata_Well` | Well name (e.g., `E05`). |
# | `Metadata_cell_line` | Cell line name as in the platemap (matches the `cell_line` column of `cell_lines/pccma_cell_line_metadata.csv`, not `cell_line_key`). |
# | `Metadata_row`, `Metadata_column` | Row letter and column number of the well. |
# | `Metadata_seeding_density` | Cells seeded per well. |
# | `Metadata_time_point` | Hours after seeding (24, 48, or 72). |
# | `Metadata_plate_barcode` | Same as `Metadata_Plate`, from the platemap. |
# | `Metadata_plate_coating` | Standard, Synthemax, or Laminin. |
# | `Metadata_pfa_fixation` | PFA fixation of the plate: `single` (plates `BR00150695` to `BR00150700`) or `double` (`BR00150701` to `BR00150703`). |

# ## Import libraries

# In[1]:


import hashlib
import pathlib
import shutil

import pandas as pd
from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features


# ## Set paths and feature selection settings

# In[2]:


# Per-plate normalized bulk profiles from 3.bulk_processing.ipynb: use the local data folder
# and fall back to the bandicoot mount, where the per-plate profiles are stored
local_dir = pathlib.Path("./data/bulk_profiles")
bandicoot_dir = pathlib.Path("~/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/bulk_profiles").expanduser()
input_dir = local_dir if any(local_dir.glob("*_bulk_normalized.parquet")) else bandicoot_dir

output_file = local_dir / "pccma_atlas_pooled_bulk_feature_selected.parquet"

# Same operations and settings as the per-plate feature selection in 3.bulk_processing.ipynb, except for the
# two noted below (drop_outliers and corr_threshold)
feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "frequency_threshold",
    "variance_threshold",
    "drop_outliers",  # added: removes the extreme-value features described below
    "correlation_threshold",
]
freq_cut = 0.05
unique_cut = 0.01
na_cutoff = 0  # remove any column with a missing value
corr_threshold = 0.95  # 0.90 in the per-plate selection

# drop_outliers removes a feature when any well has an absolute value above this cutoff.
# A high cutoff is deliberate. Some features (mostly high-order granularity) have a (near) zero MAD on a plate, so
# MAD robustize divides by almost zero and gives values up to about 1e19. Those are artifacts. The default cutoff (15)
# would also remove real differences between cell lines (587 of the 2,453 features exceed 15). Real values reach about
# 500, while 124 features exceed 1,000, so 1,000 removes the artifacts and keeps the real tail.
outlier_cutoff = 1000

print(f"Reading per-plate profiles from {input_dir}")


# ## Pool the plates

# In[3]:


plate_files = sorted(input_dir.glob("BR*_bulk_normalized.parquet"))
pooled_df = pd.concat([pd.read_parquet(file) for file in plate_files], ignore_index=True)
assert not pooled_df.duplicated(["Metadata_Plate", "Metadata_Well"]).any(), "a plate and well appears more than once"

# metadata columns first, then the features (the plates share the same feature columns)
metadata_cols = [col for col in pooled_df.columns if col.startswith("Metadata_")]
features = infer_cp_features(pooled_df)
pooled_df = pooled_df[metadata_cols + features]

print(f"{len(plate_files)} plates, {len(pooled_df)} wells, {len(features)} features")
print(pooled_df.groupby(["Metadata_Plate", "Metadata_time_point"]).size().rename("wells").to_string())


# ## Feature selection on the pooled wells

# In[4]:


local_dir.mkdir(parents=True, exist_ok=True)

feature_select(
    profiles=pooled_df,
    features=features,
    operation=feature_select_ops,
    freq_cut=freq_cut,
    unique_cut=unique_cut,
    na_cutoff=na_cutoff,
    corr_threshold=corr_threshold,
    outlier_cutoff=outlier_cutoff,
    output_file=str(output_file),
    output_type="parquet",
)

selected_df = pd.read_parquet(output_file)
selected_features = infer_cp_features(selected_df)
n_artifacts = int((pooled_df[features].abs().max() > outlier_cutoff).sum())

print(f"Kept {len(selected_features)} of {len(features)} features ({n_artifacts} had values above {outlier_cutoff:g})")
print(f"Saved {len(selected_df)} wells to {output_file}")


# ## Check the output

# In[5]:


assert len(selected_df) == len(pooled_df), "wells were lost"
assert list(selected_df.columns[: len(metadata_cols)]) == metadata_cols, "metadata columns are not first"
assert selected_df[selected_features].notna().all().all(), "missing values in the features"
assert selected_df[selected_features].abs().max().max() < outlier_cutoff, "an extreme-value feature was kept"

print(f"Largest absolute value of any kept feature: {selected_df[selected_features].abs().max().max():.1f}")


# ## Copy to bandicoot
# 
# Copy the file next to the per-plate profiles on the mount (when it is available) and check the copy is identical,
# since a large write to the mount was once seen to silently lose data.

# In[6]:


if bandicoot_dir.exists():
    destination = bandicoot_dir / output_file.name
    shutil.copy2(output_file, destination)
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256(output_file) == sha256(destination), "the copy on bandicoot differs from the local file"
    print(f"Copied to {destination}")
else:
    print("bandicoot mount not found, only the local file was saved")

