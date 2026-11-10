"""Tests of pccma_utils.single_cells, with small synthetic profiles and images."""

import numpy as np
import pandas as pd
import pytest
from pccma_utils import single_cells
from pccma_utils.single_cells import CELL_COLUMNS, closest_to_centers, crop_cells_as_png, png_to_array, sample_cells

FEATURES = ["Cells_AreaShape_Area", "Nuclei_AreaShape_Area"]
CELLS_PER_LINE = {"A": 7, "B": 3, "C": 12}  # cells of each cell line on each plate


def write_plate(folder, plate):
    """Write the single-cell profile of one plate (parquet) with three cell lines of different sizes."""
    rng = np.random.default_rng(len(plate))
    lines = [line for line, n in CELLS_PER_LINE.items() for _ in range(n)]
    df = pd.DataFrame({column: np.arange(len(lines), dtype=float) for column in CELL_COLUMNS})
    df["Metadata_Plate"], df["Metadata_cell_line"] = plate, lines
    for feature in FEATURES:
        df[feature] = rng.normal(size=len(lines))
    path = folder / f"{plate}_sc_feature_selected.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def plate_files(tmp_path):
    return [write_plate(tmp_path, plate) for plate in ("BR001", "BR002")]


def reference_sample(plate_files, cap, seed, cell_lines=None):
    """The sampling as it was written in the notebooks before it moved to the package."""
    rng = np.random.default_rng(seed)
    sampled = []
    for plate_file in plate_files:
        plate_df = pd.read_parquet(plate_file, columns=CELL_COLUMNS + FEATURES)
        if cell_lines is not None:
            plate_df = plate_df[plate_df["Metadata_cell_line"].isin(cell_lines)]
        chosen = [
            rng.choice(np.asarray(rows), size=min(len(rows), cap), replace=False)
            for rows in plate_df.groupby("Metadata_cell_line").indices.values()
        ]
        sampled.append(plate_df.iloc[np.concatenate(chosen)])
    return pd.concat(sampled, ignore_index=True)


@pytest.mark.parametrize("cell_lines", [None, ["A", "C"]])
def test_sample_cells_takes_up_to_the_cap_of_each_line_and_plate(plate_files, cell_lines):
    cells, counts = sample_cells(plate_files, FEATURES, 5, seed=0, cell_lines=cell_lines)

    lines = ["A", "B", "C"] if cell_lines is None else cell_lines
    expected = {"A": 5, "B": 3, "C": 5}  # the cap, or all the cells of a line that has fewer
    assert cells.groupby(["Metadata_Plate", "Metadata_cell_line"]).size().to_dict() == {
        (plate, line): expected[line] for plate in ("BR001", "BR002") for line in lines
    }
    assert list(cells.columns) == CELL_COLUMNS + FEATURES
    assert counts["plate_file"].tolist() == [file.name for file in plate_files]
    assert counts["sampled"].tolist() == [sum(expected[line] for line in lines)] * 2
    assert counts["available"].tolist() == [sum(CELLS_PER_LINE[line] for line in lines)] * 2


def test_sample_cells_draws_each_cell_once_and_the_same_cells_for_the_same_seed(plate_files):
    cells, _ = sample_cells(plate_files, FEATURES, 5, seed=1)
    again, _ = sample_cells(plate_files, FEATURES, 5, seed=1)
    other, _ = sample_cells(plate_files, FEATURES, 5, seed=2)

    assert not cells.duplicated(subset=["Metadata_Plate"] + CELL_COLUMNS[1:]).any()
    pd.testing.assert_frame_equal(cells, again)
    assert not cells["Nuclei_AreaShape_Area"].equals(other["Nuclei_AreaShape_Area"])


@pytest.mark.parametrize("cell_lines", [None, ["B", "C"]])
def test_sample_cells_gives_the_cells_that_the_notebooks_drew_before(plate_files, cell_lines):
    cells, _ = sample_cells(plate_files, FEATURES, 5, seed=0, cell_lines=cell_lines)

    pd.testing.assert_frame_equal(cells, reference_sample(plate_files, 5, 0, cell_lines))


def test_closest_to_centers_finds_the_cells_nearest_to_each_center():
    embedding = np.array([[0.0, 0.0], [1.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.0, 3.0], [10.0, 1.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])

    one = closest_to_centers(embedding, labels, centers)
    assert one.to_dict("list") == {"position": [0, 3], "cluster": [1, 2], "rank": [1, 1]}

    two = closest_to_centers(embedding, labels, centers, n=2)
    assert two.to_dict("list") == {"position": [0, 2, 3, 5], "cluster": [1, 1, 2, 2], "rank": [1, 2, 1, 2]}

    # a cluster with fewer cells than requested gives all of them
    assert closest_to_centers(embedding, labels, centers, n=10)["position"].tolist() == [0, 2, 1, 3, 5, 4]


def test_crop_cells_as_png_gives_the_clipped_image_of_each_cell(monkeypatch):
    calls = []

    def fake_crop(image_dir, well, site, x, y, size):
        calls.append((image_dir, well, site, x, y, size))
        return np.full((2, 2, 3), 2.0) if well == "B02" else np.full((2, 2, 3), 0.5)  # 2.0 is brighter than white

    monkeypatch.setattr(single_cells, "find_image_dir", lambda screen_dir, plate: f"{screen_dir}/{plate}/Images")
    monkeypatch.setattr(single_cells, "crop_cell", fake_crop)
    cells = pd.DataFrame(
        {
            "Metadata_Plate": ["BR001", "BR002"], "Metadata_Well": ["A01", "B02"], "Metadata_Site": [1, 2],
            "Metadata_Nuclei_Location_Center_X": [10.0, 20.0], "Metadata_Nuclei_Location_Center_Y": [30.0, 40.0],
        }
    )

    images = crop_cells_as_png(cells, "screen", crop_size=160)

    assert calls == [("screen/BR001/Images", "A01", 1, 10.0, 30.0, 160), ("screen/BR002/Images", "B02", 2, 20.0, 40.0, 160)]
    assert all(png.startswith(b"\x89PNG") for png in images)
    assert png_to_array(images[0]).shape == (2, 2, 4)
    assert png_to_array(images[0])[0, 0, 0] == pytest.approx(0.5, abs=0.01)
    assert png_to_array(images[1])[0, 0, 0] == 1.0  # clipped to white


def test_the_png_bytes_fit_in_a_parquet_column(tmp_path):
    png = crop_cells_as_png.__globals__["io"].BytesIO()
    single_cells.imsave(png, np.random.default_rng(0).random((4, 4, 3)), format="png")
    table = pd.DataFrame({"cluster": [1, 2], "image_png": [png.getvalue(), png.getvalue()]})

    table.to_parquet(tmp_path / "cells.parquet", compression="zstd")

    back = pd.read_parquet(tmp_path / "cells.parquet")
    assert back["image_png"].tolist() == table["image_png"].tolist()
    assert png_to_array(back["image_png"][0]).shape == (4, 4, 4)
