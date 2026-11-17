"""Nucleocentric crop geometry and 8-bit rescaling, numpy only.

These functions were part of `morphem_utils`; they moved here so that modules which only crop
images (the image-crop warehouse in `5a.object_image_crops`) can use the exact same crop
geometry, padding, and 8-bit rescaling as the MorphEm feature extraction
(`5.morphem_featurization`) without installing torch. `morphem_utils` re-exports them, so the
MorphEm notebooks keep importing them from one place.

See `morphem_utils` for the design notes of the nucleocentric crop (224 x 224 centered on the
nucleus, padding at the edge of the image, and only the camera-saturated pixels at 255).
"""

import numpy as np

# size of the input images that MorphEm expects (the side of every nucleocentric crop)
MORPHEM_INPUT_SIZE = 224


def rescale_crop_to_8bit(crop: np.ndarray, saturated: np.ndarray) -> np.ndarray:
    """
    Rescale one channel of a crop to 8-bit, where only the saturated pixels become 255.

    MorphEm was trained on 8-bit images and SaturationNoiseInjector replaces the pixels with the
    value 255 with noise, so we keep 255 for the pixels that were saturated in the camera and
    linearly rescale the other pixels of the crop from their minimum and maximum to 0 to 254.
    PerImageNormalize (instance normalization) removes any linear rescaling, so rescaling each crop
    on its own only sets the 8-bit precision and does not change the scale of the features.
    We do not rescale the whole FOV with percentile clipping, since it sets many pixels of the
    cells to 255 in FOVs with few cells, which MorphEm then replaces with noise.

    Parameters
    ----------
    crop : np.ndarray
        2D crop of one channel (e.g., 16-bit after illumination correction).
    saturated : np.ndarray
        2D boolean mask of the crop with the pixels that were saturated in the raw image.

    Returns
    -------
    np.ndarray
        2D uint8 crop.
    """
    # Load in crop as float32 for rescaling
    crop = crop.astype(np.float32)
    # Rescale to 8-bit to match MorphEm's expectations on what it was trained on
    rescaled = np.zeros(crop.shape, dtype=np.uint8)
    # Clip the unsaturated pixels to the range [0, 254] after rescaling
    unsaturated = crop[~saturated]
    if unsaturated.size > 0 and unsaturated.max() > unsaturated.min():
        lower, upper = unsaturated.min(), unsaturated.max()
        rescaled = np.round(np.clip((crop - lower) / (upper - lower), 0, 1) * 254).astype(np.uint8)
    rescaled[saturated] = 255
    return rescaled


def get_nucleocentric_crop_box(
    center_x: float, center_y: float, crop_size: int = MORPHEM_INPUT_SIZE
) -> tuple[int, int, int, int]:
    """
    Find the square crop of `crop_size` pixels centered on the nucleus.

    Every crop has the same size as the MorphEm input (224 x 224 pixels), so MorphEm does not resize
    the crops and every single cell has the same scale. We do not move crops at the edge of the image
    back inside the image, so the nucleus is always in the center of the crop. The part of the crop
    outside the image is padded by crop_with_padding.

    Parameters
    ----------
    center_x, center_y : float
        Center of the nucleus from CellProfiler (Nuclei_Location_Center_X/Y).
    crop_size : int
        Side of the square crop in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        (y_min, y_max, x_min, x_max) of the crop, where the max values are exclusive. The values can
        be outside the image when the nucleus is near the edge.
    """
    x_min = round(center_x - crop_size / 2)
    y_min = round(center_y - crop_size / 2)
    return y_min, y_min + crop_size, x_min, x_min + crop_size


def crop_with_padding(
    image: np.ndarray,
    crop_box: tuple[int, int, int, int],
    pad_value: float | bool | None = None,
) -> tuple[np.ndarray, float]:
    """
    Crop an image and pad the part of the crop that is outside the image.

    Parameters
    ----------
    image : np.ndarray
        2D image (one channel of a FOV, or a mask of the FOV).
    crop_box : tuple[int, int, int, int]
        (y_min, y_max, x_min, x_max) of the crop from get_nucleocentric_crop_box.
    pad_value : float | bool | None
        Value of the padded pixels. If None, use the minimum of the pixels inside the image, so the
        padded pixels become 0 when we rescale the crop to 8-bit (the same as a dark background).

    Returns
    -------
    tuple[np.ndarray, float]
        The crop and the fraction of the crop that is padding (0 when the crop is inside the image).
    """
    y_min, y_max, x_min, x_max = crop_box
    height, width = image.shape
    inside = image[max(y_min, 0) : min(y_max, height), max(x_min, 0) : min(x_max, width)]
    # in theory this should never happen: the crop is centered on a nucleus inside the image, so at least
    # about half of the crop is always inside the image (we check in case of a bad nucleus center)
    if inside.size == 0:
        raise ValueError(f"The crop {crop_box} is completely outside the image of shape {image.shape}")
    if pad_value is None:
        pad_value = inside.min()
    pad_widths = (
        (max(-y_min, 0), max(y_max - height, 0)),
        (max(-x_min, 0), max(x_max - width, 0)),
    )
    crop = np.pad(inside, pad_widths, mode="constant", constant_values=pad_value)
    pad_fraction = 1 - inside.size / crop.size
    return crop, pad_fraction


def count_other_nuclei(
    crop_boxes: np.ndarray, center_x: np.ndarray, center_y: np.ndarray
) -> np.ndarray:
    """
    Count the other nuclei in the crop of each single cell of one FOV.

    A nucleus counts when its center is inside the crop, so a neighbor only counts when a large part
    of it is in the crop (not just the tip of a protrusion).

    Parameters
    ----------
    crop_boxes : np.ndarray
        (number of single cells, 4) array of (y_min, y_max, x_min, x_max) of each crop.
    center_x, center_y : np.ndarray
        Center of the nucleus of each single cell, in the same order as crop_boxes.

    Returns
    -------
    np.ndarray
        Number of other nuclei in each crop (not counting the nucleus of the single cell).
    """
    y_min, y_max, x_min, x_max = np.asarray(crop_boxes).T
    inside = (
        (center_x[None, :] >= x_min[:, None])
        & (center_x[None, :] < x_max[:, None])
        & (center_y[None, :] >= y_min[:, None])
        & (center_y[None, :] < y_max[:, None])
    )
    np.fill_diagonal(inside, False)
    return inside.sum(axis=1)


def _sum_in_boxes(
    summed: np.ndarray, y_min: np.ndarray, y_max: np.ndarray, x_min: np.ndarray, x_max: np.ndarray
) -> np.ndarray:
    # sum of a mask inside each box from its summed-area table (boxes are clipped to the image)
    height, width = summed.shape[0] - 1, summed.shape[1] - 1
    y_min, y_max = np.clip(y_min, 0, height), np.clip(y_max, 0, height)
    x_min, x_max = np.clip(x_min, 0, width), np.clip(x_max, 0, width)
    y_max, x_max = np.maximum(y_max, y_min), np.maximum(x_max, x_min)
    return summed[y_max, x_max] - summed[y_min, x_max] - summed[y_max, x_min] + summed[y_min, x_min]


def get_other_cell_overlap(
    crop_boxes: np.ndarray, cell_boxes: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure how much of each crop and each single cell overlaps the other single cells of one FOV.

    We use the bounding boxes of the cells from CellProfiler, so the overlap is an upper bound of the
    overlap of the cells themselves (a bounding box also covers the background around a cell).

    Parameters
    ----------
    crop_boxes : np.ndarray
        (number of single cells, 4) array of (y_min, y_max, x_min, x_max) of each crop.
    cell_boxes : np.ndarray
        (number of single cells, 4) array of (y_min, y_max, x_min, x_max) of the bounding box of each
        cell (Cells_AreaShape_BoundingBox*, max values exclusive), in the same order as crop_boxes.
    image_shape : tuple[int, int]
        Shape of the FOV (height, width).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The fraction of each crop covered by the bounding boxes of the other cells, and the fraction
        of the bounding box of each cell that overlaps the bounding boxes of the other cells.
    """
    height, width = image_shape
    crop_boxes = np.asarray(crop_boxes, dtype=int)
    cell_boxes = np.clip(np.asarray(cell_boxes, dtype=int), 0, [height, height, width, width])
    cell_y_min, cell_y_max, cell_x_min, cell_x_max = cell_boxes.T
    crop_y_min, crop_y_max, crop_x_min, crop_x_max = crop_boxes.T

    # number of cell bounding boxes that cover each pixel of the FOV
    corners = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(corners, (cell_y_min, cell_x_min), 1)
    np.add.at(corners, (cell_y_min, cell_x_max), -1)
    np.add.at(corners, (cell_y_max, cell_x_min), -1)
    np.add.at(corners, (cell_y_max, cell_x_max), 1)
    coverage = corners.cumsum(axis=0).cumsum(axis=1)[:height, :width]

    # summed-area tables of the pixels covered by at least one and at least two bounding boxes
    def summed_area(mask: np.ndarray) -> np.ndarray:
        summed = np.zeros((height + 1, width + 1), dtype=np.int64)
        summed[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
        return summed

    at_least_one, at_least_two = summed_area(coverage >= 1), summed_area(coverage >= 2)

    # pixels of the crop covered by another cell: outside the cell's own box, any box counts;
    # inside its own box, only pixels covered by at least two boxes (its own and another one)
    own_y_min, own_y_max = np.maximum(cell_y_min, crop_y_min), np.minimum(cell_y_max, crop_y_max)
    own_x_min, own_x_max = np.maximum(cell_x_min, crop_x_min), np.minimum(cell_x_max, crop_x_max)
    covered_in_crop = (
        _sum_in_boxes(at_least_one, crop_y_min, crop_y_max, crop_x_min, crop_x_max)
        - _sum_in_boxes(at_least_one, own_y_min, own_y_max, own_x_min, own_x_max)
        + _sum_in_boxes(at_least_two, own_y_min, own_y_max, own_x_min, own_x_max)
    )
    crop_area = (crop_y_max - crop_y_min) * (crop_x_max - crop_x_min)

    cell_area = np.maximum((cell_y_max - cell_y_min) * (cell_x_max - cell_x_min), 1)
    overlap_in_cell = _sum_in_boxes(at_least_two, cell_y_min, cell_y_max, cell_x_min, cell_x_max)

    return covered_in_crop / crop_area, overlap_in_cell / cell_area
