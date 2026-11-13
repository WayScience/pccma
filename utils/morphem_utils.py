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

# the numpy-only crop geometry functions live in nucleocentric_crops, which does not need torch,
# so modules that only crop images (5a.object_image_crops) can import them without the MorphEm
# environment; they are re-exported here so the MorphEm notebooks import them from one place
from nucleocentric_crops import (  # noqa: F401
    MORPHEM_INPUT_SIZE,
    count_other_nuclei,
    crop_with_padding,
    get_nucleocentric_crop_box,
    get_other_cell_overlap,
    rescale_crop_to_8bit,
)

# Hugging Face model ID and revision (commit) of MorphEm, set so the features are reproducible
MORPHEM_MODEL_ID = "CaicedoLab/MorphEm"
MORPHEM_REVISION = "0e8d58787421f83f975634d72420d85c5dfc9c2c"


def get_morphem_model(device: str) -> torch.nn.Module:
    """
    Load the MorphEm model from Hugging Face in evaluation mode.

    Parameters
    ----------
    device : str
        Device to load the model on ("cuda" or "cpu"). We do not fall back to the CPU when CUDA is not
        available, so a job without a GPU fails instead of running for days.

    Returns
    -------
    torch.nn.Module
        The MorphEm model in evaluation mode.
    """
    # import here so the rest of the utilities do not need transformers
    from transformers import AutoModel

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
