"""Tests of crop_warehouse (the nucleocentric JPG crops of the single cells).

The tests write small synthetic Phenix-style images to a temporary folder, like the tests of
pccma_utils.cell_images, so they run without the raw images of the atlas.
"""

import io

import numpy as np
import pytest
from PIL import Image

from crop_warehouse import (
    CHANNELS,
    CROP_SIZE,
    JPEG_QUALITY,
    composite_crops,
    correct_illumination,
)
from nucleocentric_crops import (
    crop_with_padding,
    get_nucleocentric_crop_box,
    rescale_crop_to_8bit,
)


def test_crop_size_matches_morphem():
    """The warehouse crops are the same size as the MorphEm input."""
    assert CROP_SIZE == 224


def test_correct_illumination_divides_and_clips():
    """Illumination correction divides by the function and clips to the 16-bit range."""
    image = np.array([[1000, 2000], [32768, 65535]], dtype=np.uint16)
    illum = np.full((2, 2), 2.0, dtype=np.float32)
    corrected = correct_illumination(image, illum)
    assert corrected.dtype == np.float32
    assert corrected[0, 0] == 500
    assert corrected[0, 1] == 1000
    assert corrected[1, 0] == 16384
    assert corrected[1, 1] == 32767.5  # division only lowers values; the clip is a safety net


def test_rescale_only_saturated_at_255():
    """Only the camera-saturated pixels become 255; the others are within 0 to 254."""
    crop = np.array([[0.0, 100.0], [32767.0, 65535.0]], dtype=np.float32)
    saturated = crop == 65535
    rescaled = rescale_crop_to_8bit(crop, saturated)
    assert rescaled[1, 1] == 255
    assert rescaled[1, 0] == 254  # the highest unsaturated value
    assert rescaled[0, 0] == 0
    assert rescaled.dtype == np.uint8


def test_crop_box_centers_on_nucleus():
    """The crop is centered on the nucleus, even when it goes past the edge of the image."""
    y_min, y_max, x_min, x_max = get_nucleocentric_crop_box(center_x=10.0, center_y=10.0)
    assert (x_max - x_min) == CROP_SIZE
    assert (y_max - y_min) == CROP_SIZE
    assert x_min < 0  # near the edge: not moved back inside
    image = np.zeros((200, 200), dtype=np.uint16)
    image[10 - 5 : 10 + 5, 10 - 5 : 10 + 5] = 1000
    crop, pad_fraction = crop_with_padding(image, (y_min, y_max, x_min, x_max))
    assert crop.shape == (CROP_SIZE, CROP_SIZE)
    assert pad_fraction > 0


def test_composite_crops_shape_and_colors():
    """The composite is an RGB image that is dark for an empty crop and bright for a full one."""
    dark = {channel: np.zeros((8, 8), dtype=np.uint8) for channel in CHANNELS}
    bright = {channel: np.full((8, 8), 200, dtype=np.uint8) for channel in CHANNELS}
    dark_composite = composite_crops(dark)
    bright_composite = composite_crops(bright)
    assert dark_composite.shape == (8, 8, 3)
    assert dark_composite.max() < 10
    assert bright_composite.min() > 100


def test_composite_crops_channel_color():
    """Only the DNA channel bright: the composite is blue like the DNA color of the figures."""
    crops = {channel: np.zeros((8, 8), dtype=np.uint8) for channel in CHANNELS}
    crops["DNA"] = np.full((8, 8), 200, dtype=np.uint8)
    composite = composite_crops(crops)
    middle = composite[4, 4]
    assert middle[2] > middle[0]  # blue above red
    assert middle[2] > middle[1]  # blue above green


@pytest.fixture
def jpg_roundtrip():
    """Encode and decode one grayscale crop and one composite through JPEG, like the warehouse."""

    def roundtrip(array):
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="jpeg", quality=JPEG_QUALITY)
        buffer.seek(0)
        return np.asarray(Image.open(buffer))

    return roundtrip


def test_jpeg_roundtrip_keeps_pixel_values(jpg_roundtrip):
    """JPEG q95 keeps the 8-bit pixel values of a microscopy crop within a few gray levels."""
    # a smooth blob in a dark background, like a fluorescence crop
    y, x = np.mgrid[0:CROP_SIZE, 0:CROP_SIZE]
    crop = 40 + 180 * np.exp(-(((x - 112) ** 2) + (y - 112) ** 2) / (2 * 40**2))
    crop = np.round(crop).astype(np.uint8)
    decoded = jpg_roundtrip(crop)
    assert decoded.shape == crop.shape
    assert np.abs(decoded.astype(int) - crop.astype(int)).max() <= 3
