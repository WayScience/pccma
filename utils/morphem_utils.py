"""Utility functions to extract nucleocentric MorphEm features from 2D Cell Painting images.

MorphEm is a self-supervised Vision Transformer (ViT-Small, DINO Bag of Channels) that
the Caicedo Lab trained on the CHAMMI-75 dataset (Agrawal et al., ICLR 2026).
It takes one channel at a time and outputs a 384-dimensional embedding (CLS token) per channel.

The preprocessing transforms (SaturationNoiseInjector and PerImageNormalize) are copied from the
MorphEm model card: https://huggingface.co/CaicedoLab/MorphEm
The nucleocentric approach (a crop centered on each nucleus) is adapted from Mike Lippincott's code
in the NF1 3D organoid profiling pipeline:
https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/utils/src/image_analysis_3D/featurization_utils/morphem_featurization.py
https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/3.cellprofiling/scripts/nucleo_centric_featurization.py

We make the following changes for 2D images and to process many nuclei at once:

- MorphEm has one input channel (`in_chans: 1` in the model config), so we pass each channel
  once instead of copying it to three channels.
- We apply the noise and normalization to each crop on its own, so we can run a batch of crops
  through the model (SaturationNoiseInjector only changes the first image of a batch).
- We convert each crop from 16-bit to 8-bit, since MorphEm was trained on 8-bit images and
  SaturationNoiseInjector finds saturated pixels by the value 255 (only the pixels that were
  saturated in the camera become 255).
- We crop 224 x 224 pixels centered on each nucleus (the MorphEm input size) instead of the nucleus
  bounding box plus 25 pixels, so MorphEm does not resize the crops and every single cell has the
  same scale. We pad crops at the edge of the image instead of moving them, so the nucleus is always
  in the center.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# Hugging Face model ID and revision (commit) of MorphEm, set so the features are reproducible
MORPHEM_MODEL_ID = "CaicedoLab/MorphEm"
MORPHEM_REVISION = "0e8d58787421f83f975634d72420d85c5dfc9c2c"

# size of the input images that MorphEm expects
MORPHEM_INPUT_SIZE = 224


def get_morphem_model(device: str | None = None) -> torch.nn.Module:
    """
    Load the MorphEm model from Hugging Face in evaluation mode.

    Parameters
    ----------
    device : str | None
        Device to load the model on ("cuda" or "cpu"). If None, use CUDA when available.

    Returns
    -------
    torch.nn.Module
        The MorphEm model in evaluation mode.
    """
    # import here so the rest of the utilities do not need transformers
    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        MORPHEM_MODEL_ID, revision=MORPHEM_REVISION, trust_remote_code=True
    )
    return model.to(device).eval()


# Noise Injector transformation (from the MorphEm model card)
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation (from the MorphEm model card)
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


# transforms in the same order as the MorphEm model card, applied to one crop of shape (1, H, W)
# (our crops are already 224 x 224, so Resize does not change them)
MORPHEM_TRANSFORM = v2.Compose(
    [
        SaturationNoiseInjector(),
        PerImageNormalize(),
        v2.Resize(size=(MORPHEM_INPUT_SIZE, MORPHEM_INPUT_SIZE), antialias=True),
    ]
)


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


def preprocess_crops(crops: list[np.ndarray]) -> torch.Tensor:
    """
    Apply the MorphEm transforms to each crop on its own and stack them into one batch.

    Parameters
    ----------
    crops : list[np.ndarray]
        8-bit 2D crops of one channel.

    Returns
    -------
    torch.Tensor
        Batch of crops with shape (N, 1, 224, 224).
    """
    return torch.stack(
        [
            MORPHEM_TRANSFORM(torch.from_numpy(crop).float().unsqueeze(0))
            for crop in crops
        ]
    )


def featurize_crops(
    crops: list[np.ndarray],
    model: torch.nn.Module,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract the MorphEm CLS-token features from crops of one channel.

    Parameters
    ----------
    crops : list[np.ndarray]
        8-bit 2D crops of one channel.
    model : torch.nn.Module
        The MorphEm model (see get_morphem_model).
    device : str
        Device that the model is on ("cuda" or "cpu").
    batch_size : int
        Number of crops to run through the model at once.

    Returns
    -------
    np.ndarray
        Features with shape (number of crops, 384).
    """
    features = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = preprocess_crops(crops[start : start + batch_size]).to(device)
            output = model.forward_features(batch)
            features.append(output["x_norm_clstoken"].cpu().numpy())
    return np.concatenate(features, axis=0)
