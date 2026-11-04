"""Functions to cut the image of a single cell out of the raw images of a plate."""

import pathlib

import numpy as np
import tifffile

# Channel number of each stain in the raw image file names, and the display color of each (RGB, 0-1).
# The mapping comes from 1.whole_image_qc/load_data_config/config.yml (Alexa 488 = ER, Alexa 568 = AGP,
# Alexa 647 = Mito, Hoechst = DNA, Alexa 488 Long = RNA).
CHANNELS = {
    "DNA": (5, (0.0, 0.35, 1.0)),
    "ER": (2, (0.0, 1.0, 0.0)),
    "RNA": (6, (1.0, 1.0, 0.0)),
    "AGP": (3, (1.0, 0.15, 0.0)),
    "Mito": (4, (1.0, 0.0, 1.0)),
}


def find_image_dir(screen_dir, plate):
    """Folder with the raw images of a plate (the folder name is the barcode followed by the measurement)."""
    matches = sorted(pathlib.Path(screen_dir).glob(f"{plate}__*/Images"))
    if not matches:
        raise FileNotFoundError(f"no image folder for plate {plate} in {screen_dir}")
    return matches[-1]


def read_field(image_dir, well, site, channel):
    """Raw image of one channel of one field of view (the site is the field number)."""
    row, column = ord(well[0]) - ord("A") + 1, int(well[1:])
    path = image_dir / f"r{row:02d}c{column:02d}f{int(site):02d}p01-ch{channel}sk1fk1fl1.tiff"
    return tifffile.imread(path).astype(np.float32)


def crop_cell(image_dir, well, site, x, y, size=160, low=1, high=99.8):
    """Color image (height x width x 3, 0-1) of the cell with its nucleus center at pixel (x, y).

    The five channels are cropped around the center, each is stretched between the `low` and `high`
    percentiles of its own crop, and they are combined in color (screen blend). The raw images are used,
    without illumination correction. The part of the crop outside the image is black.
    """
    half = size // 2
    x, y = int(round(x)), int(round(y))
    composite = np.ones((size, size, 3), dtype=np.float32)
    for channel, color in CHANNELS.values():
        image = read_field(image_dir, well, site, channel)
        padded = np.pad(image, half, mode="constant", constant_values=image.min())
        crop = padded[y : y + size, x : x + size]  # the padding of `half` pixels shifts the center to (x, y)
        lower, upper = np.percentile(crop, [low, high])
        stretched = np.clip((crop - lower) / max(upper - lower, 1e-6), 0, 1)
        composite *= 1 - 0.85 * stretched[..., None] * np.array(color, dtype=np.float32)
    return 1 - composite
