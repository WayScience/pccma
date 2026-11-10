"""Tests of pccma_utils.cell_images.

The tests write small synthetic Phenix images to a temporary folder (a dark 200 x 200 field with one bright
square in the DNA channel), so they run without the raw images of the atlas.
"""

import numpy as np
import pytest
import tifffile
from pccma_utils.cell_images import CHANNELS, crop_cell, find_image_dir, read_field

FIELD_SIZE = 200
DNA_CHANNEL = CHANNELS["DNA"][0]


def write_field(image_dir, well_row, well_column, site, channel, image):
    """Write one channel of one field with the file name of the Phenix images."""
    name = f"r{well_row:02d}c{well_column:02d}f{site:02d}p01-ch{channel}sk1fk1fl1.tiff"
    tifffile.imwrite(image_dir / name, image.astype(np.uint16))


@pytest.fixture
def image_dir(tmp_path):
    """Images folder of plate BR001 with well I05 (row 9, column 5), field 2.

    Every channel is dark noise-free background (100) except the DNA channel, which has a bright 20 x 20 square
    whose center is at pixel (x=120, y=50).
    """
    folder = tmp_path / "BR001__2026-07-02T18_25_42-Measurement 1" / "Images"
    folder.mkdir(parents=True)
    for _, (channel, _) in CHANNELS.items():
        image = np.full((FIELD_SIZE, FIELD_SIZE), 100)
        if channel == DNA_CHANNEL:
            image[40:60, 110:130] = 4000  # rows 40-59 (y), columns 110-129 (x)
        write_field(folder, 9, 5, 2, channel, image)
    return folder


def test_find_image_dir_finds_the_images_folder_of_the_plate(tmp_path, image_dir):
    assert find_image_dir(tmp_path, "BR001") == image_dir


def test_find_image_dir_uses_the_last_measurement(tmp_path):
    for measurement in ("Measurement 1", "Measurement 2"):
        (tmp_path / f"BR002__2026-07-02T18_25_42-{measurement}" / "Images").mkdir(parents=True)
    assert find_image_dir(tmp_path, "BR002").parent.name.endswith("Measurement 2")


def test_find_image_dir_raises_for_an_unknown_plate(tmp_path):
    with pytest.raises(FileNotFoundError, match="BR999"):
        find_image_dir(tmp_path, "BR999")


def test_read_field_builds_the_phenix_file_name_from_the_well(image_dir):
    # well I05 is row 9, column 5, so the file is r09c05f02p01-ch5sk1fk1fl1.tiff
    field = read_field(image_dir, "I05", 2, DNA_CHANNEL)
    assert field.shape == (FIELD_SIZE, FIELD_SIZE)
    assert field.dtype == np.float32
    assert field[50, 120] == 4000  # the bright square, indexed as [y, x]
    assert field[0, 0] == 100


def test_crop_cell_returns_a_color_image_between_0_and_1(image_dir):
    crop = crop_cell(image_dir, "I05", 2, x=120, y=50, size=64)
    assert crop.shape == (64, 64, 3)
    assert crop.min() >= 0 and crop.max() <= 1


def test_crop_cell_is_centered_on_the_requested_pixel(image_dir):
    # The bright DNA square is at (x=120, y=50), and DNA is shown in blue (channel 2 of the RGB image),
    # so the blue channel of the crop is brightest around the center of the crop.
    crop = crop_cell(image_dir, "I05", 2, x=120, y=50, size=64)
    blue = crop[..., 2]
    brightest_row, brightest_column = np.unravel_index(blue.argmax(), blue.shape)
    assert abs(brightest_row - 32) <= 10 and abs(brightest_column - 32) <= 10
    assert blue[32, 32] > blue[0, 0]


def test_crop_cell_near_the_edge_is_padded_and_keeps_its_size(image_dir):
    # a cell 5 pixels from the top left corner: the crop reaches outside the image and is padded
    crop = crop_cell(image_dir, "I05", 2, x=5, y=5, size=64)
    assert crop.shape == (64, 64, 3)
    # the padded top left part of the crop is outside the image, so it is as dark as the background
    assert crop[:10, :10].max() < 0.1
