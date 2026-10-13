# Platemap metadata

In this module, we generate the platemap for each plate in the Atlas dataset and create figures of the plate layouts.

> **Note:** This module does not download the images.
> The images are stored outside of this repository and the paths to the images are set in the LoadData CSV notebooks in the later modules.

## Contents

The platemaps come from the experimental layout file, `Rambutan_Atlasv2_April2026.xlsx`, which includes a metadata sheet per plate.
The layout file lives in [`cell_lines/raw_data/`](../cell_lines/raw_data/) with the other raw cell line data, and the notebook reads it from there.

The other files are in the `metadata` folder.

| File or folder | Description |
| --- | --- |
| `generate_platemaps_from_xlsx.ipynb` | Converts the metadata sheet for each plate in the layout file into a platemap CSV. |
| `generate_platemap_figures.ipynb` | Creates a figure of the 384-well layout for each plate. |
| `platemaps/` | One platemap CSV per plate (e.g., `BR00150695_platemap.csv`). |
| `platemap_figures/` | One PNG per plate. |

## Platemap columns

| Column | Description |
| --- | --- |
| `cell_line` | Name of the cell line in the well. |
| `row` and `column` | Row letter and column number of the well. |
| `well_position` | Well name (e.g., `C03`). |
| `seeding_density` | Number of cells seeded per well. |
| `time_point` | Time point of the plate in hours (24, 48, or 72). |
| `plate_barcode` | Barcode of the plate (e.g., `BR00150695`). |
| `plate_coating` | Coating of the well (Standard, Synthemax, or Laminin). |
| `pfa_fixation` | PFA fixation of the plate (single or double). |

Only wells with cells are in the platemaps.
The platemaps are used in `3.cp_analysis` to split the images from each plate by cell line.

## Run the notebooks

The notebooks use the `pccma_atlas_platemaps_env` environment (`environments/platemaps_env.yml`).

1. Run `generate_platemaps_from_xlsx.ipynb` to create the platemaps.
2. Run `generate_platemap_figures.ipynb` to create the figures.

> **Note:** The figures notebook uses the LoadData CSVs from `1.whole_image_qc/loaddata_csvs/` to identify which wells were imaged, so the LoadData CSVs must be created first.
> Wells that were imaged but are not in the platemap are shown as media only wells.
