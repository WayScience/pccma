# Illumination correction (IC)

In this module, we calculate an illumination correction function per channel for each plate with CellProfiler.

The notebook first updates the pipeline (`pipeline/illum.cppipe`) to match the whole image QC results before running CellProfiler:

- It sets the blur thresholds to the values in `1.whole_image_qc/blur_thresholds.txt`.
- It sets the saturation threshold to 1 for all channels.

The pipeline skips FOVs flagged for blur or saturation, so that poor quality FOVs do not contribute to the functions.

## Contents

| File or folder | Description |
| --- | --- |
| `cp_illum_correction.ipynb` | Updates the QC thresholds in the pipeline and runs CellProfiler for all plates in parallel. |
| `pipeline/illum.cppipe` | CellProfiler pipeline to calculate the illumination correction functions. |
| `illum_directory/` | One folder per plate with an illumination correction function (`npy` file) per channel (e.g., `BR00150695/BR00150695_IllumDNA.npy`). |
| `logs/` | CellProfiler log per plate (not tracked in the repository). |

We use the LoadData CSVs from `1.whole_image_qc/loaddata_csvs/` as the input, so you must run the whole image QC module first.
The `3.cp_analysis` module applies the illumination correction functions to the images.

## Run the module

The module uses the `pccma_atlas_cp_env` environment (`environments/cellprofiler_env.yml`).

```bash
cd 2.illumination_correction
source run_ic.sh
```
