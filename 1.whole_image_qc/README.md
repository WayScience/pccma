# Whole image quality control (QC)

In this module, we extract whole image quality metrics with CellProfiler and determine thresholds to identify poor quality fields of view (FOVs).

We do not evaluate each cell line individually as we expect that image quality metrics are universal and not impacted by the cell lines like morphology features are.

We evaluate two metrics per channel:

- **Blur** (`PowerLogLogSlope`): Detects out-of-focus FOVs.
Thresholds are determined per channel across all plates using [coSMicQC](https://github.com/cytomining/coSMicQC).
- **Saturation** (`PercentMaximal`): Detects over-saturated FOVs or FOVs with large artifacts.
A FOV fails if 1% or more of the pixels are at the maximum intensity.

## Notebooks

| Notebook | Description | Environment |
| --- | --- | --- |
| `0.create_loaddata_csvs.ipynb` | Creates a LoadData CSV per plate with the paths to each channel per image set. | `pccma_atlas_cp_env` |
| `1.extract_image_quality.ipynb` | Runs the CellProfiler pipeline (`pipeline/whole_img_qc.cppipe`) to extract the image quality metrics for each plate. | `pccma_atlas_cp_env` |
| `2.evaluate_qc.ipynb` | Determines the blur threshold per channel, evaluates saturation, and visualizes the FOVs that fail QC. | `pccma_atlas_whole_img_qc_env` |

## Outputs

| File or folder | Description |
| --- | --- |
| `loaddata_csvs/` | One LoadData CSV per plate. |
| `whole_img_qc_output/` | Image quality metrics from CellProfiler per plate (not tracked in the repository). |
| `blur_thresholds.txt` | Blur threshold per channel. |
| `qc_figures/` | Figures of the blur metric distributions. |
| `logs/` | CellProfiler log per plate (not tracked in the repository). |

The `load_data_config/config.yml` file sets the channels and metadata included in the LoadData CSVs and is also used in `3.cp_analysis`.
The thresholds in `blur_thresholds.txt` are used in `2.illumination_correction` to update the pipeline so that FOVs that fail QC are not used to calculate the illumination correction functions.

## Run the module

```bash
cd 1.whole_image_qc
source run_qc.sh
```

> **Note:** The path to the images is set in `0.create_loaddata_csvs.ipynb` and will need to be updated to where the images are stored.
