#!/usr/bin/env python
# coding: utf-8

# # Generate a single-cell QC report
# 
# The QC report summarizes how many single cells failed each coSMicQC condition (and any condition) per cell line and plate (time point), using the QC annotations from `1.sc_quality_control.ipynb`.
# We also save a summary per cell line across all time points, including whether the cell line uses its own thresholds in `qc_thresholds.yml`.

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
import yaml


# ## Set paths and variables

# In[2]:


# path for platemap directory (one platemap per plate)
platemap_dir = pathlib.Path("../0.download_data/metadata/platemaps").resolve(strict=True)

# Directory with the QC annotations from single-cell QC (one file per cell line)
qc_results_dir = pathlib.Path("./qc_results").resolve(strict=True)

# Directory with the converted profiles (one folder per cell line), used to make sure every cell line has QC annotations
converted_dir = pathlib.Path("./data/converted_profiles").resolve(strict=True)

# output path for reports
output_dir = pathlib.Path("./qc_report")
output_dir.mkdir(parents=True, exist_ok=True)

# Thresholds used for QC (to report which cell lines use their own thresholds)
with open("qc_thresholds.yml") as file:
    thresholds_config = yaml.safe_load(file)
cell_lines_with_overrides = set(thresholds_config.get("cell_lines") or {})

# Every cell line must have QC annotations
cell_lines = sorted(folder.name for folder in converted_dir.iterdir() if folder.is_dir())
missing_cell_lines = [
    cell_line
    for cell_line in cell_lines
    if not (qc_results_dir / f"{cell_line}_qc_annotations.parquet").exists()
]
assert not missing_cell_lines, f"Cell lines without QC annotations: {missing_cell_lines}"

print(f"There are {len(cell_lines)} cell lines with QC annotations.")


# ## Load the QC annotations and add the platemap metadata

# In[3]:


# Load the QC annotations of every cell line (one row per single cell)
qc_df = pd.concat(
    [
        pd.read_parquet(qc_results_dir / f"{cell_line}_qc_annotations.parquet").assign(
            cell_line_folder=cell_line
        )
        for cell_line in cell_lines
    ],
    ignore_index=True,
)
outlier_columns = [col for col in qc_df.columns if col.endswith("_is_outlier")]
qc_df["failed_qc"] = qc_df[outlier_columns].any(axis=1)

# Load the platemaps of every plate to add the cell line, seeding density, and time point of each well
platemap_df = pd.concat(
    [pd.read_csv(path) for path in sorted(platemap_dir.glob("*_platemap.csv"))],
    ignore_index=True,
)
qc_df = qc_df.merge(
    platemap_df[["plate_barcode", "well_position", "cell_line", "seeding_density", "time_point"]],
    left_on=["Image_Metadata_Plate", "Image_Metadata_Well"],
    right_on=["plate_barcode", "well_position"],
    how="left",
    validate="many_to_one",
)

# Every single cell should match a well in the platemaps, and the cell line in the platemap
# should match the cell line folder (cell line names use underscores instead of spaces in the folders)
assert qc_df["cell_line"].notna().all(), "Single cells without a well in the platemaps"
assert (
    qc_df["cell_line"].str.replace(" ", "_") == qc_df["cell_line_folder"]
).all(), "Cell line in the platemap does not match the cell line folder"

print(qc_df.shape)


# ## QC report per cell line and plate (time point)

# In[4]:


# Number of single cells, number that failed each condition and any condition, and percentage failed
failure_stats = (
    qc_df.groupby(
        ["cell_line", "time_point", "seeding_density", "Image_Metadata_Plate"]
    )
    .agg(
        total_single_cells=("failed_qc", "size"),
        **{
            col.replace("Metadata_cqc_", "failed_").replace("_is_outlier", ""): (col, "sum")
            for col in outlier_columns
        },
        total_failed_qc=("failed_qc", "sum"),
        percentage_failing_cells=("failed_qc", "mean"),
    )
    .reset_index()
)
failure_stats["percentage_failing_cells"] = (
    failure_stats["percentage_failing_cells"] * 100
).round(2)
failure_stats["own_thresholds"] = failure_stats["cell_line"].str.replace(" ", "_").isin(
    cell_lines_with_overrides
)

failure_stats.to_csv(output_dir / "qc_report_per_plate.csv", index=False)

print(failure_stats.shape)
failure_stats.head()


# ## QC report per cell line (all time points)

# In[5]:


cell_line_stats = (
    qc_df.groupby("cell_line")
    .agg(
        total_single_cells=("failed_qc", "size"),
        **{
            col.replace("Metadata_cqc_", "failed_").replace("_is_outlier", ""): (col, "sum")
            for col in outlier_columns
        },
        total_failed_qc=("failed_qc", "sum"),
        percentage_failing_cells=("failed_qc", "mean"),
    )
    .reset_index()
)
cell_line_stats["percentage_failing_cells"] = (
    cell_line_stats["percentage_failing_cells"] * 100
).round(2)
cell_line_stats["own_thresholds"] = cell_line_stats["cell_line"].str.replace(" ", "_").isin(
    cell_lines_with_overrides
)
cell_line_stats = cell_line_stats.sort_values("percentage_failing_cells", ascending=False)

cell_line_stats.to_csv(output_dir / "qc_report_per_cell_line.csv", index=False)

print(
    f"Total single cells: {cell_line_stats['total_single_cells'].sum()}, "
    f"failed QC: {cell_line_stats['total_failed_qc'].sum()} "
    f"({cell_line_stats['total_failed_qc'].sum() / cell_line_stats['total_single_cells'].sum() * 100:.2f}%)"
)
cell_line_stats.head(10)

