"""Annotate cell lines with their cancer group and tissue site."""

import pandas as pd
import yaml


def load_cell_line_groups(metadata_path, groups_path):
    """One row per cell line with its cancer type, origin, cancer group and tissue site.

    `metadata_path` is the cell line metadata (`cell_lines/pccma_cell_line_metadata.csv`) and `groups_path`
    is `cell_lines/plotting_helpers/cancer_type_categories.yaml`, which lists which `cancer_type` values of the
    metadata belong to each cancer group, and which tissue site each group belongs to.

    The metadata has one row per plate condition (U2-OS is on several), so the first row of each cell line is
    kept. A cell line whose cancer type is not in the yaml file gets missing (NaN) group and site.
    """
    metadata = pd.read_csv(metadata_path).drop_duplicates("cell_line")
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
