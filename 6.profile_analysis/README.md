# Profile analysis

In this module, we collect the data that the figures of the PCCMA paper will plot.
Each notebook computes one set of tables from the profiles and saves them in a `results/` folder.
We make the figures later from these tables, so every panel traces back to one notebook.

We add each notebook in its own pull request, so the notebooks below are planned and are not in this module yet.

## Inputs

We use the profiles from [`4.preprocess_features`](../4.preprocess_features/), which are on the bandicoot mount in `PCCMA_data/Atlas_v2_profiles/`.

| Analysis type | Profiles |
| --- | --- |
| Univariate | Normalized profiles without feature selection: `bulk_profiles/BR*_bulk_normalized.parquet` (bulk) and `single_cell_profiles/BR*_sc_normalized.parquet` (single-cell). |
| Multivariate | Feature-selected profiles: `BR*_bulk_feature_selected.parquet` and `BR*_sc_feature_selected.parquet` for each plate, and the pooled bulk profiles (`pccma_atlas_pooled_bulk_feature_selected.parquet`) for analyses across plates. |

The cell line metadata is in [`cell_lines/pccma_cell_line_metadata.csv`](../cell_lines/pccma_cell_line_metadata.csv).
We join the profiles to it on the `cell_line` column.

## Planned notebooks

| Notebook | What it computes | Why we generate it | Profiles |
| --- | --- | --- | --- |
| `0.cell_counts` | Number of cells per cell line and time point after QC. | Describes how many cells each cell line contributes. | Single-cell |
| `1.replicate_correlation` | Pairwise Pearson correlation between replicate wells of each cell line. | Shows that replicate wells agree, with a shuffled-feature baseline. | Bulk, multivariate |
| `2.bulk_embedding` | Similarity matrix between cell lines (heat map) and a UMAP of the bulk profiles. | Shows how the cell lines relate to each other. | Pooled bulk, multivariate |
| `3.single_cell_embedding` | A UMAP of subsampled single cells for each cancer type, with example cell images for chosen points. | Shows the diversity of single cells within and across cancer types. | Single-cell, multivariate |
| `4.feature_comparison` | The distribution of nucleus size for each cell line, and the variance of each feature across cell lines with a statistical test. | Compares one size feature across all cell lines and identifies a second feature that differs between cell lines. | Normalized, univariate |
| `5.neuroblastoma_single_cell` | A UMAP of the neuroblastoma single cells with representative images for the clusters. | Focuses on one cancer type to show what the clusters look like. | Single-cell, multivariate |
| `6.metastatic_vs_primary` | A linear model per cell line comparing metastatic with primary disease, adjusting for other variables. | Tests the comparison without picking up confounding variables. | Normalized, univariate |

We still need to decide:

- Which groups the KS tests cover.
- Which variables the linear model in `6.metastatic_vs_primary` adjusts for, and which cancer type it uses.
- Whether the second feature in `4.feature_comparison` comes from the variance test or is a mitotic feature such as nucleus texture.

## Notebook conventions

- Each notebook explains its steps in markdown and sets its paths and parameters at the top.
- Helper functions sit in their own cells near the top of the notebook.
  If a second notebook needs the same function, it moves to [`utils`](../utils/).
- Notebooks are converted to scripts in `nbconverted/`, like the other modules.
