# Preprocess CellProfiler features

In this module, we convert the CellProfiler outputs from `3.cp_analysis` to single-cell parquet files, remove poor quality single cells with [coSMicQC](https://github.com/cytomining/coSMicQC), and process the profiles with [pycytominer](https://github.com/cytomining/pycytominer) into single-cell and bulk profiles.

We process each plate (time point) on its own, since there are no negative control wells in this screen (no treatments), so we normalize each plate to all of its single cells or wells.

## Notebooks

We run the notebooks in order with `run_preprocessing_pipeline.sh`:

| Notebook | Description |
| --- | --- |
| `0.convert_cytotable.ipynb` | Converts the SQLite file of each CellProfiler job (one cell line from one plate) to a single-cell parquet file with CytoTable, and adds the `Metadata` prefix to the location and cell count columns. Skips any job that is already converted. |
| `1.sc_quality_control.ipynb` | Finds poor quality single cells with coSMicQC for one cell line at a time (all plates of the cell line together), using the thresholds in `qc_thresholds.yml`. Saves the QC annotations of each single cell and figures of the outliers. |
| `2.annotate.ipynb` | For each plate, removes the single cells that failed QC, concatenates the cell lines, and annotates the single cells with the platemap (cell line, time point, and seeding density of each well). |
| `3.bulk_processing.ipynb` | Creates the bulk profiles of each plate: aggregate to the well (median), normalize (MAD robustize to the whole plate), and feature select. |
| `4.single_cell_normalize.ipynb` | Normalizes the single-cell profiles of each plate (standardize to the whole plate). |
| `5.single_cell_feature_select.ipynb` | Feature selects the normalized single-cell profiles of each plate. |
| `6.sc_qc_report.ipynb` | Summarizes how many single cells failed each QC condition per cell line and plate, and per cell line across all plates. |

## Single-cell quality control

We use coSMicQC to find single cells with poor segmentations, which are often caused by high confluence (overlapping nuclei).
We calculate z-scores per cell line across all of its plates (time points), and a single cell fails QC when it is an outlier for any of the three conditions:

| Condition | Features | What it finds |
| --- | --- | --- |
| Clustered nuclei | `Nuclei_Intensity_MassDisplacement_DNA` and `Nuclei_Intensity_IntegratedIntensity_DNA` (both high) | Overlapping nuclei segmented as one nucleus. |
| Low solidity nuclei | `Nuclei_AreaShape_Solidity` (low) | Very irregular nuclei, likely mis-segmented. |
| Cells with multiple nuclei | `Cells_Intensity_IntegratedIntensity_DNA` (high) | Cells segmented around more than one nucleus. |

`qc_thresholds.yml` holds the default thresholds and the thresholds of each cell line that differs from the default.
We set the thresholds of a cell line after reviewing its example outliers, and the YAML file includes the reason for each change.
47 of the 54 cell lines use their own thresholds for at least one condition.

Across the 1,420,883 single cells, 216,171 (15.2%) fail QC (2.4% to 49.2% per cell line).
The QC report (`qc_report/qc_report_per_plate.csv` and `qc_report/qc_report_per_cell_line.csv`) has the number of single cells that failed each condition.

> **Note:** We do not remove poor quality FOVs (blurry or over-saturated) before this module.
> Single-cell QC removes the poor segmentations from these FOVs, so we keep the good single cells in them.

## Profile processing

We process the profiles with pycytominer, normalizing to the whole plate since there are no negative controls:

| Profile | Steps |
| --- | --- |
| Single-cell | Standardize (to all single cells on the plate), then feature select. |
| Bulk | Aggregate to the well (median), MAD robustize (to all wells on the plate), then feature select. |

For feature selection, we drop features with any missing values, features on the default blocklist, and features with low frequency, low variance, or high correlation (Pearson correlation above 0.90).

## Inputs and outputs

The module depends on:

- SQLite files from `3.cp_analysis` (one per cell line and plate) on the Bandicoot mount (`PCCMA_data/Atlas_v2_profiles/sqlite_outputs/`)
- Platemaps from `0.download_data/metadata/platemaps/`

| Folder | Description | Tracked by git |
| --- | --- | --- |
| `data/converted_profiles/<cell_line>/` | One converted single-cell profile per plate (`<plate>_<cell_line>_converted.parquet`, all single cells before QC). | No |
| `data/annotated_profiles/` | One annotated single-cell profile per plate (single cells that passed QC). | No |
| `data/single_cell_profiles/` | Normalized and feature-selected single-cell profiles per plate. | No |
| `data/bulk_profiles/` | Aggregated, normalized, and feature-selected bulk profiles per plate. | No |
| `qc_results/` | QC annotations of each single cell (`<cell_line>_qc_annotations.parquet`) and the thresholds used (`<cell_line>_qc_thresholds.json`). | No |
| `qc_figures/` | Figures of the outliers of each cell line. | No |
| `qc_report/` | QC report per cell line and plate, and per cell line. | Yes |
| `executed_notebooks/1.sc_quality_control/` | The executed QC notebook of each cell line, to review the outliers. | No |

`5.morphem_featurization` also uses the converted profiles (center and bounding box of each nucleus) and the QC annotations, so it removes the same single cells as this module.

## Run the module

```bash
cd 4.preprocess_features
./run_preprocessing_pipeline.sh
```

The script creates or updates the `environments/preprocessing` uv environment from its lock file, converts the notebooks to scripts, and runs each step.

- **Conversion** only runs when the Bandicoot mount with the SQLite files is available. Otherwise, the script uses the existing converted profiles.
- **QC** runs one cell line at a time with papermill, so it keeps the executed notebook of each cell line for review.
- **Annotation and processing** run one plate per `python` process, so the memory is released after each plate.

Every step skips a cell line or plate that already has output, unless its inputs are newer (e.g., a plate was converted again or QC was run again).
Set `OVERWRITE=1` (`OVERWRITE=1 ./run_preprocessing_pipeline.sh`) to process everything again.
