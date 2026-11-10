# Profile analysis

In this module, we collect the data that the figures of the PCCMA paper will plot.
Each notebook computes one set of tables from the profiles and saves them in its own folder, `results/<notebook name>/`, with a file name that describes the table (for example, `results/0.cell_counts/cell_counts_per_cell_line_and_plate.csv`).
The header of each notebook describes every table and its columns.
Charts in the notebooks are for demonstration and are displayed in the notebook, not saved to disk.

## Inputs

We use the profiles from [`4.preprocess_features`](../4.preprocess_features/), which are on the bandicoot mount in `PCCMA_data/Atlas_v2_profiles/`.

| Analysis type | Profiles |
| --- | --- |
| Univariate | Normalized profiles without feature selection: `bulk_profiles/BR*_bulk_normalized.parquet` (bulk) and `single_cell_profiles/BR*_sc_normalized.parquet` (single-cell). |
| Multivariate | Feature-selected profiles: `BR*_bulk_feature_selected.parquet` and `BR*_sc_feature_selected.parquet` for each plate, and the pooled bulk profiles (`pccma_atlas_pooled_bulk_feature_selected.parquet`) for analyses across plates. |

The cell line metadata is in [`cell_lines/pccma_cell_line_metadata.csv`](../cell_lines/pccma_cell_line_metadata.csv).
We join the profiles to it on the `cell_line` column.

## Notebooks

| Notebook | What it computes | Why we generate it | Profiles |
| --- | --- | --- | --- |
| `0.cell_counts` | Number of cells per cell line and time point after QC. | Describes how many cells each cell line contributes. | Single-cell |
| `1.replicate_correlation` | Pairwise Pearson correlation between replicate wells of each cell line. | Shows that replicate wells agree, with a shuffled-feature baseline. | Bulk |
| `2.bulk_embedding` | Similarity matrix between cell lines (heat map) and a UMAP of the bulk profiles. | Shows how the cell lines relate to each other. | Pooled bulk |
| `3.single_cell_embedding` | A UMAP of subsampled single cells for each cancer type, with example cell images for chosen points. | Shows the diversity of single cells within and across cancer types. | Single-cell, multivariate |
| `4.feature_comparison` | The distribution of nucleus size for each cell line, and the variance of each feature across cell lines with a statistical test. | Compares one size feature across all cell lines and identifies a second feature that differs between cell lines. | Normalized, univariate |
| `5.neuroblastoma_single_cell` | A UMAP of the neuroblastoma single cells with representative images for the clusters. | Focuses on one cancer type to show what the clusters look like. | Single-cell, multivariate |
| `6.metastatic_vs_primary` | A linear model per cell line comparing metastatic with primary disease, adjusting for other variables. | Tests the comparison without picking up confounding variables. | Normalized, univariate |

## Shared code

A function that more than one notebook needs goes in the `pccma_utils` package ([`utils/pccma_utils`](../utils/pccma_utils/)), which has tests in [`utils/tests`](../utils/tests/).
A function used by a single notebook stays in that notebook.
