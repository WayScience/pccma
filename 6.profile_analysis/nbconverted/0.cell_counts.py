#!/usr/bin/env python
# coding: utf-8

# # Cell counts per cell line
# 
# Counts the single cells that each cell line contributes to the atlas after single-cell quality control (QC), for each plate (time point).
# 
# **Why:** the number of cells per cell line and time point tells how much data each line has, and the cell lines with few cells are the ones to treat with care in later analyses.
# 
# **Input:** `4.preprocess_features/qc_report/qc_report_per_plate.csv`, which has the number of single cells of every cell line and plate before QC and the number that failed QC.
# The cells that pass QC are the cells in the `*_sc_feature_selected.parquet` profiles, and the notebook checks that the totals match.
# 
# **Output:** `results/cell_counts.csv` with one row per cell line and plate.
# 
# | Column | Description |
# | --- | --- |
# | `cell_line`, `plate`, `time_point` | Cell line name (as in the platemaps), plate barcode, and hours after seeding. |
# | `total_single_cells` | Single cells segmented before QC. |
# | `failed_qc` | Cells removed by QC (clustered nuclei, low solidity nuclei, or cells with multiple nuclei). |
# | `cells_after_qc` | Cells kept in the profiles (`total_single_cells - failed_qc`). |
# | `percentage_failing_qc` | `failed_qc` as a percentage of `total_single_cells`. |
# | `cancer_type`, `cancer_group`, `tissue_site`, `origin` | Cell line annotations from `cell_lines/`. |
# 
# A cell line has no row for a plate that was not processed (CHLA262 at 72 h, Saos-2 at 24 h, and CF1500 and X0092 on all plates; see `3.cp_analysis`).

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
import pyarrow.parquet as pq
import yaml


# ## Set paths

# In[2]:


qc_report_path = pathlib.Path("../4.preprocess_features/qc_report/qc_report_per_plate.csv")
metadata_path = pathlib.Path("../cell_lines/pccma_cell_line_metadata.csv")
groups_path = pathlib.Path("../cell_lines/plotting_helpers/cancer_type_categories.yaml")

# single-cell feature-selected profiles, used only to check the counts
local_dir = pathlib.Path("../4.preprocess_features/data/single_cell_profiles")
bandicoot_dir = pathlib.Path("~/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/single_cell_profiles").expanduser()
profile_dir = local_dir if any(local_dir.glob("*_sc_feature_selected.parquet")) else bandicoot_dir

results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)


# ## Helper function
# 
# Adds the cancer group and tissue site of each cell line, from the groups that `cell_lines/plotting_helpers/cancer_type_categories.yaml` defines.

# In[3]:


def load_cell_line_groups(metadata_path, groups_path):
    """One row per cell line with its cancer type, cancer group and tissue site.

    The cancer groups and tissue sites come from cancer_type_categories.yaml, which lists
    which `cancer_type` values of the metadata belong to each group.
    """
    metadata = pd.read_csv(metadata_path).drop_duplicates("cell_line")  # U2-OS has one row per plate condition
    categories = yaml.safe_load(open(groups_path))
    groups = pd.DataFrame(
        [
            (cancer_type, group["name"], site["name"])
            for site in categories["sites"]
            for group in site["groups"]
            for cancer_type in group["cancer_types"]
        ],
        columns=["cancer_type", "cancer_group", "tissue_site"],
    )
    return metadata[["cell_line", "cancer_type", "origin"]].merge(groups, on="cancer_type", how="left")


# ## Cell counts per cell line and plate

# In[4]:


qc_report = pd.read_csv(qc_report_path).rename(columns={
    "Image_Metadata_Plate": "plate",
    "total_failed_qc": "failed_qc",
    "percentage_failing_cells": "percentage_failing_qc",
})
qc_report["cells_after_qc"] = qc_report["total_single_cells"] - qc_report["failed_qc"]

counts = qc_report[["cell_line", "plate", "time_point", "total_single_cells", "failed_qc", "cells_after_qc", "percentage_failing_qc"]].merge(
    load_cell_line_groups(metadata_path, groups_path), on="cell_line", how="left"
)

assert not counts.duplicated(["cell_line", "plate"]).any(), "a cell line and plate appears more than once"
assert counts["cancer_group"].notna().all(), "a cell line has no cancer group"

counts = counts.sort_values(["cancer_group", "cell_line", "time_point"]).reset_index(drop=True)
counts.to_csv(results_dir / "cell_counts.csv", index=False)
print(f"{counts['cell_line'].nunique()} cell lines, {len(counts)} cell line and plate combinations")


# ## Check against the profiles
# 
# The cells after QC of each plate should equal the number of rows in that plate's single-cell profiles.

# In[5]:


after_qc = counts.groupby("plate")["cells_after_qc"].sum()
for plate, n_cells in after_qc.items():
    n_rows = pq.ParquetFile(profile_dir / f"{plate}_sc_feature_selected.parquet").metadata.num_rows
    assert n_cells == n_rows, f"{plate}: {n_cells} cells after QC in the report but {n_rows} rows in the profiles"
print("The cell counts match the single-cell profiles of all", len(after_qc), "plates")


# ## Summary

# In[6]:


print("Cells after QC per time point:")
print(counts.groupby("time_point")["cells_after_qc"].agg(["sum", "median", "min", "max"]).rename(columns={"sum": "total"}).to_string())

per_line = counts.groupby("cell_line")["cells_after_qc"].sum().sort_values()
print(f"\nCells after QC per cell line (all time points): median {per_line.median():,.0f}, from {per_line.min():,} ({per_line.index[0]}) to {per_line.max():,} ({per_line.index[-1]})")
print(f"Cell line and plate combinations with fewer than 1,000 cells: {(counts['cells_after_qc'] < 1000).sum()}")

