"""Tests of pccma_utils.cell_lines.load_cell_line_groups, with a small metadata table and categories file."""

import pandas as pd
import pytest
from pccma_utils.cell_lines import load_cell_line_groups

METADATA = """cell_line,cancer_type,origin
A673,Ewing's Sarcoma,Primary
U2-OS,Osteosarcoma,Primary
U2-OS,Osteosarcoma,Primary
CHLA10,Primitive Neuroectodermal Tumor (PNET),Metastasis
MYSTERY,Unlisted Tumor,Primary
"""

CATEGORIES = """sites:
  - name: Bone
    groups:
      - name: Ewing sarcoma family
        cancer_types:
          - "Ewing's Sarcoma"
          - Primitive Neuroectodermal Tumor (PNET)
      - name: Osteosarcoma
        cancer_types:
          - Osteosarcoma
"""


@pytest.fixture
def groups(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    groups_path = tmp_path / "categories.yaml"
    metadata_path.write_text(METADATA)
    groups_path.write_text(CATEGORIES)
    return load_cell_line_groups(metadata_path, groups_path).set_index("cell_line")


def test_keeps_one_row_per_cell_line(groups):
    # U2-OS has two rows in the metadata (one per plate condition)
    assert groups.index.is_unique
    assert set(groups.index) == {"A673", "U2-OS", "CHLA10", "MYSTERY"}


def test_adds_the_cancer_group_and_tissue_site(groups):
    assert groups.loc["A673", "cancer_group"] == "Ewing sarcoma family"
    assert groups.loc["A673", "tissue_site"] == "Bone"
    assert groups.loc["U2-OS", "cancer_group"] == "Osteosarcoma"


def test_several_cancer_types_can_share_one_cancer_group(groups):
    # Ewing's Sarcoma and the PNET cancer type are both in the Ewing sarcoma family
    assert groups.loc["CHLA10", "cancer_group"] == groups.loc["A673", "cancer_group"]


def test_keeps_the_cancer_type_and_origin_of_the_metadata(groups):
    assert groups.loc["CHLA10", "cancer_type"] == "Primitive Neuroectodermal Tumor (PNET)"
    assert groups.loc["CHLA10", "origin"] == "Metastasis"


def test_a_cancer_type_missing_from_the_categories_gets_no_group(groups):
    assert pd.isna(groups.loc["MYSTERY", "cancer_group"])
    assert pd.isna(groups.loc["MYSTERY", "tissue_site"])
