"""Tests of pccma_utils.profiles.numeric_features, with small pyarrow schemas."""

import pyarrow as pa
from pccma_utils.profiles import numeric_features


def test_returns_the_numeric_columns_that_are_not_metadata_or_image_columns():
    schema = pa.schema([
        ("Metadata_Plate", pa.string()),
        ("Metadata_Nuclei_Location_Center_X", pa.float64()),  # numeric, but metadata
        ("Image_FileName_OrigDNA", pa.string()),
        ("Image_Count_Cells", pa.int64()),  # numeric, but an image-level column
        ("Nuclei_AreaShape_Area", pa.float64()),
        ("Cells_Neighbors_NumberOfNeighbors_Adjacent", pa.int32()),
        ("Cells_Label", pa.string()),  # not numeric
    ])
    assert numeric_features(schema) == {"Nuclei_AreaShape_Area", "Cells_Neighbors_NumberOfNeighbors_Adjacent"}


def test_the_result_can_be_intersected_across_plates():
    plate_a = pa.schema([("Nuclei_Area", pa.float64()), ("Cells_Area", pa.float64())])
    plate_b = pa.schema([("Nuclei_Area", pa.float64()), ("Cytoplasm_Area", pa.float64())])
    assert set.intersection(numeric_features(plate_a), numeric_features(plate_b)) == {"Nuclei_Area"}


def test_a_schema_without_features_gives_an_empty_set():
    assert numeric_features(pa.schema([("Metadata_Well", pa.string())])) == set()
