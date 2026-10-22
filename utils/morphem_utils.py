"""Utility functions to extract nucleocentric MorphEm features from 2D Cell Painting images.

MorphEm is a self-supervised Vision Transformer (ViT-Small, DINO Bag of Channels) that
the Caicedo Lab trained on the CHAMMI-75 dataset (Agrawal et al., ICLR 2026).
It takes one channel at a time and outputs a 384-dimensional embedding (CLS token) per channel.

The preprocessing transforms (SaturationNoiseInjector and PerImageNormalize) are copied from the
MorphEm model card: https://huggingface.co/CaicedoLab/MorphEm
The nucleocentric approach (expand the nucleus bounding box by 25 pixels in each direction and
square the crop) is adapted from Mike Lippincott's code in the NF1 3D organoid profiling pipeline:
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
    center_x: float,
    center_y: float,
    bbox_min_x: float,
    bbox_max_x: float,
    bbox_min_y: float,
    bbox_max_y: float,
    expand_pixels: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int, bool]:
    """
    Find a square crop centered on the nucleus that includes the nucleus bounding box plus
    `expand_pixels` in each direction.

    The side of the square is the longest side of the nucleus bounding box plus two times
    `expand_pixels`. When the square goes past the edge of the image, we move it back inside the
    image instead of making it smaller or padding it, so every crop is square and only holds
    real pixels (the nucleus is then off center).

    Parameters
    ----------
    center_x, center_y : float
        Center of the nucleus from CellProfiler (Nuclei_Location_Center_X/Y).
    bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y : float
        Bounding box of the nucleus from CellProfiler (Nuclei_AreaShape_BoundingBox*).
    expand_pixels : int
        Number of pixels to add on each side of the bounding box.
    image_shape : tuple[int, int]
        Shape of the image (height, width).

    Returns
    -------
    tuple[int, int, int, int, bool]
        (y_min, y_max, x_min, x_max, shifted) of the crop, where the max values are exclusive and
        shifted is True when the crop was moved to stay inside the image.
    """
    height, width = image_shape
    side = int(max(bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y)) + 2 * expand_pixels
    side = min(side, height, width)

    # top left corner of the square centered on the nucleus
    x_min = round(center_x - side / 2)
    y_min = round(center_y - side / 2)

    # move the square back inside the image if needed
    x_min_inside = min(max(x_min, 0), width - side)
    y_min_inside = min(max(y_min, 0), height - side)
    shifted = (x_min_inside != x_min) or (y_min_inside != y_min)

    return y_min_inside, y_min_inside + side, x_min_inside, x_min_inside + side, shifted


def preprocess_crops(crops: list[np.ndarray]) -> torch.Tensor:
    """
    Apply the MorphEm transforms to each crop on its own and stack them into one batch.

    Parameters
    ----------
    crops : list[np.ndarray]
        8-bit 2D crops of one channel (crops can have different sizes).

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
