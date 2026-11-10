"""Tests of pccma_utils.palettes.site_colors against the tissue sites of the cell line categories file."""

import pathlib
import re

import yaml
from pccma_utils.palettes import site_colors

CATEGORIES_PATH = pathlib.Path(__file__).parents[2] / "cell_lines" / "plotting_helpers" / "cancer_type_categories.yaml"


def test_every_tissue_site_has_a_color_in_the_same_order():
    sites = [site["name"] for site in yaml.safe_load(open(CATEGORIES_PATH))["sites"]]
    assert list(site_colors) == sites


def test_colors_are_distinct_hex_codes():
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", color) for color in site_colors.values())
    assert len(set(site_colors.values())) == len(site_colors)
