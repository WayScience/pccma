#!/usr/bin/env python
# coding: utf-8

# # Extract nucleocentric MorphEm features
# 
# In this notebook, we extract deep learning features with [MorphEm](https://huggingface.co/CaicedoLab/MorphEm) (Agrawal et al., 2026) from a crop centered on each nucleus (nucleocentric).
# We process one cell line from one plate (time point) per job, the same as the CellProfiler jobs in `3.cp_analysis`.
# 
# We do not segment the images again.
# We use the center and bounding box of each nucleus from CellProfiler (in the converted profiles from `4.preprocess_features/0.convert_cytotable.ipynb`), so every row in the MorphEm profiles matches a single cell in the CellProfiler profiles.
# 
# For each field of view (FOV), we:
# 
# 1. Load the five channels, find the saturated pixels, and apply the illumination correction functions (divide, the same as in the CellProfiler pipelines).
# 2. Crop a square around the center of each nucleus that is the size of the nucleus bounding box plus 25 pixels in each direction (adapted from the NF1 3D organoid profiling pipeline).
# 3. Rescale each crop to 8-bit, since MorphEm was trained on 8-bit images, where only the pixels that were saturated in the camera become 255 (MorphEm replaces them with noise).
# 4. Extract the 384 MorphEm features (CLS token) from each channel of each crop, which gives 1,920 features per single cell.
# 
# We name the features like the CellProfiler features (compartment, feature group, measurement, then channel), so tools that expect the channel at the end of the name also work: `Nucleocentric_MorphEm_Feature<n>_<channel>` (e.g., `Nucleocentric_MorphEm_Feature0_DNA`, with `<n>` from 0 to 383).
# 
# We save the features with the same metadata columns as the converted profiles (plate, well, site, image number, and object numbers), so we can remove the single cells that failed QC with the QC annotations from `4.preprocess_features/1.sc_quality_control.ipynb`.

# ## Import libraries

# In[ ]:


import argparse
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch

sys.path.append("../utils")
from bandicoot_utils import bandicoot_check
from morphem_utils import (
    MORPHEM_REVISION,
    featurize_crops,
    get_morphem_model,
    get_nucleocentric_crop_box,
    rescale_crop_to_8bit,
)

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## Set paths and variables

# In[ ]:


if not in_notebook:
    print("Running as script")
    parser = argparse.ArgumentParser(
        description="Extract nucleocentric MorphEm features for one cell line from one plate"
    )
    parser.add_argument("--plate", type=str, required=True, help="Plate name (e.g., BR00150698)")
    parser.add_argument("--cell_line", type=str, required=True, help="Name of the cell line folder (e.g., A673)")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of crops to run through MorphEm at once")
    parser.add_argument("--overwrite", action="store_true", help="Extract the features again if the output already exists")
    args = parser.parse_args()
    plate, cell_line, batch_size, overwrite = args.plate, args.cell_line, args.batch_size, args.overwrite
else:
    print("Running in a notebook")
    plate = "BR00150698"
    cell_line = "A673"
    batch_size = 256
    overwrite = False

# number of pixels to add on each side of the nucleus bounding box (same as the NF1 pipeline as a start)
expand_pixels = 25

# channel names in the converted profiles (Image_FileName_Orig<channel>) and illumination functions (<plate>_Illum<channel>.npy)
channels = ["DNA", "ER", "RNA", "AGP", "Mito"]

# set the seed, since MorphEm replaces the saturated pixels with random noise
seed = 0
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Find root directory of the project
root_dir = pathlib.Path().cwd()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# Directory with the images (one folder per plate), used when the paths in the profiles are from the HPC cluster
images_dir = pathlib.Path(f"{image_base_dir}/PCCMA_data/Atlas_v2_screen")

# converted profile of the job with the center and bounding box of each nucleus from CellProfiler
converted_path = pathlib.Path(
    f"../4.preprocess_features/data/converted_profiles/{cell_line}/{plate}_{cell_line}_converted.parquet"
).resolve(strict=True)

# illumination correction functions of the plate
illum_dir = pathlib.Path(f"../2.illumination_correction/illum_directory/{plate}").resolve(strict=True)

# one folder per cell line with one parquet file per plate (time point), the same as the converted profiles
output_dir = pathlib.Path(f"./data/morphem_profiles/{cell_line}")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{plate}_{cell_line}_morphem.parquet"

print(f"Processing {cell_line} from {plate}")


# ## Load the single cells from the converted profile
# 
# We only load the metadata columns, the center and bounding box of each nucleus, and the image paths.

# In[ ]:


# columns that identify each single cell (the QC annotations are merged on these columns)
id_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Image_Metadata_Row",
    "Image_Metadata_Col",
    "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number",
    "Metadata_Cells_Number_Object_Number",
    "Metadata_Cytoplasm_Parent_Cells",
    "Metadata_Cytoplasm_Parent_Nuclei",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
    "Metadata_Cells_Location_Center_X",
    "Metadata_Cells_Location_Center_Y",
]

# nucleus bounding box from CellProfiler (we add the Metadata prefix so pycytominer keeps them as metadata)
bbox_columns = [
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Nuclei_AreaShape_BoundingBoxMaximum_X",
    "Nuclei_AreaShape_BoundingBoxMinimum_Y",
    "Nuclei_AreaShape_BoundingBoxMaximum_Y",
]

image_columns = [f"Image_FileName_Orig{channel}" for channel in channels] + [
    f"Image_PathName_Orig{channel}" for channel in channels
]

if output_path.exists() and not overwrite:
    print(f"Skipping {plate} {cell_line} (found {output_path})")
    single_cell_df = None
else:
    single_cell_df = pd.read_parquet(
        converted_path, columns=id_columns + bbox_columns + image_columns
    )

    # use the local path to the images when the profiles have the paths from the HPC cluster
    for channel in channels:
        path_column = f"Image_PathName_Orig{channel}"
        single_cell_df[path_column] = single_cell_df[path_column].apply(
            lambda path: path
            if pathlib.Path(path).exists()
            else re.sub(r"^.*PCCMA_atlas_data/", f"{images_dir}/", path)
        )

    # every single cell should have a unique ID in the job
    assert not single_cell_df.duplicated(
        ["Metadata_ImageNumber", "Metadata_Cells_Number_Object_Number"]
    ).any(), "Duplicate single cells in the converted profile"

    print(
        f"{len(single_cell_df)} single cells from {single_cell_df['Metadata_ImageNumber'].nunique()} FOVs"
    )


# ## Load the illumination correction functions and MorphEm

# In[ ]:


if single_cell_df is not None:
    illum_functions = {
        channel: np.load(illum_dir / f"{plate}_Illum{channel}.npy") for channel in channels
    }

    model = get_morphem_model(device)
    print(f"Loaded MorphEm (revision {MORPHEM_REVISION})")


# ## Extract the nucleocentric features from each FOV

# In[ ]:


feature_names = None

if single_cell_df is not None:
    start_time = time.time()
    fov_dfs = []

    for fov_number, (image_number, fov_df) in enumerate(
        single_cell_df.groupby("Metadata_ImageNumber", sort=True), start=1
    ):
        fov_record = fov_df.iloc[0]

        # load each channel, find the saturated pixels, and apply the illumination correction function
        fov_images = {}
        fov_saturated = {}
        for channel in channels:
            image_path = pathlib.Path(
                fov_record[f"Image_PathName_Orig{channel}"]
            ) / fov_record[f"Image_FileName_Orig{channel}"]
            image = tifffile.imread(image_path)
            assert image.shape == illum_functions[channel].shape, f"{image_path} does not match the shape of the illumination function"
            fov_saturated[channel] = image == np.iinfo(image.dtype).max
            fov_images[channel] = image.astype(np.float32) / illum_functions[channel]
        image_shape = fov_images[channels[0]].shape

        # find the nucleocentric crop of each nucleus (the same crop for every channel)
        crop_boxes = [
            get_nucleocentric_crop_box(
                center_x=record["Metadata_Nuclei_Location_Center_X"],
                center_y=record["Metadata_Nuclei_Location_Center_Y"],
                bbox_min_x=record["Nuclei_AreaShape_BoundingBoxMinimum_X"],
                bbox_max_x=record["Nuclei_AreaShape_BoundingBoxMaximum_X"],
                bbox_min_y=record["Nuclei_AreaShape_BoundingBoxMinimum_Y"],
                bbox_max_y=record["Nuclei_AreaShape_BoundingBoxMaximum_Y"],
                expand_pixels=expand_pixels,
                image_shape=image_shape,
            )
            for record in fov_df.to_dict(orient="records")
        ]

        # extract the features of each channel (one block of 384 features per channel)
        channel_feature_dfs = []
        saturated_fractions = {}
        for channel in channels:
            # fraction of the crop that was saturated in the camera (MorphEm replaces these pixels with noise)
            saturated_fractions[
                f"Metadata_Nucleocentric_SaturatedFraction_{channel}"
            ] = [
                fov_saturated[channel][y_min:y_max, x_min:x_max].mean()
                for y_min, y_max, x_min, x_max, _ in crop_boxes
            ]
            crops = [
                rescale_crop_to_8bit(
                    fov_images[channel][y_min:y_max, x_min:x_max],
                    saturated=fov_saturated[channel][y_min:y_max, x_min:x_max],
                )
                for y_min, y_max, x_min, x_max, _ in crop_boxes
            ]
            features = featurize_crops(crops, model=model, device=device, batch_size=batch_size)
            channel_feature_dfs.append(
                pd.DataFrame(
                    features,
                    columns=[
                        f"Nucleocentric_MorphEm_Feature{i}_{channel}"
                        for i in range(features.shape[1])
                    ],
                    index=fov_df.index,
                )
            )

        crop_df = pd.DataFrame(
            crop_boxes,
            columns=[
                "Metadata_Nucleocentric_Crop_Minimum_Y",
                "Metadata_Nucleocentric_Crop_Maximum_Y",
                "Metadata_Nucleocentric_Crop_Minimum_X",
                "Metadata_Nucleocentric_Crop_Maximum_X",
                "Metadata_Nucleocentric_Crop_Shifted",
            ],
            index=fov_df.index,
        )
        saturated_df = pd.DataFrame(saturated_fractions, index=fov_df.index)
        fov_dfs.append(pd.concat([crop_df, saturated_df] + channel_feature_dfs, axis=1))

        if fov_number % 10 == 0:
            print(
                f"Processed {fov_number} FOVs ({time.time() - start_time:.1f} seconds)"
            )

    features_df = pd.concat(fov_dfs).loc[single_cell_df.index]
    feature_names = [col for col in features_df.columns if col.startswith("Nucleocentric_")]
    print(
        f"Extracted {len(feature_names)} features from {len(features_df)} single cells in {time.time() - start_time:.1f} seconds"
    )


# ## Save the MorphEm profile
# 
# We keep the image file names and paths so we can display the crops later (e.g., with CytoDataFrame).
# We also keep the fraction of each crop that was saturated in the camera per channel (`Metadata_Nucleocentric_SaturatedFraction_<channel>`), since the saturated pixels are where MorphEm adds noise, so we can find the single cells that might be too saturated.

# In[ ]:


if feature_names is not None:
    morphem_df = pd.concat(
        [
            single_cell_df[id_columns + image_columns],
            single_cell_df[bbox_columns].rename(columns=lambda col: f"Metadata_{col}"),
            features_df,
        ],
        axis=1,
    )
    morphem_df["Metadata_Nucleocentric_Expand_Pixels"] = expand_pixels

    # every single cell from the converted profile should have features
    assert len(morphem_df) == len(single_cell_df), "Single cells were lost during featurization"
    assert not morphem_df[feature_names].isna().any().any(), "Some features are NaN"

    # save to a temporary file first so a stopped run does not leave an unfinished parquet file
    temp_path = output_dir / f"{plate}_{cell_line}_temp.parquet"
    morphem_df.to_parquet(temp_path, index=False)
    temp_path.replace(output_path)

    print(
        f"Saved {output_path} with the shape {morphem_df.shape} "
        f"({morphem_df['Metadata_Nucleocentric_Crop_Shifted'].sum()} crops moved inside the image at the edges)"
    )


# ## Show example crops (notebook only)
# 
# We show the crops of a few single cells with the nucleus bounding box in red, to confirm that the crops are centered on the nuclei.

# In[ ]:


if in_notebook and feature_names is not None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    examples_df = morphem_df[
        morphem_df["Metadata_ImageNumber"] == morphem_df["Metadata_ImageNumber"].iloc[0]
    ].sample(n=min(4, len(morphem_df)), random_state=seed)

    image_cache = {}
    fig, axes = plt.subplots(len(examples_df), len(channels), figsize=(2.2 * len(channels), 2.2 * len(examples_df)), squeeze=False)
    for row, record in enumerate(examples_df.to_dict(orient="records")):
        for col, channel in enumerate(channels):
            if channel not in image_cache:
                image_path = pathlib.Path(record[f"Image_PathName_Orig{channel}"]) / record[f"Image_FileName_Orig{channel}"]
                image = tifffile.imread(image_path)
                image_cache[channel] = (
                    image.astype(np.float32) / illum_functions[channel],
                    image == np.iinfo(image.dtype).max,
                )
            y_min, y_max = record["Metadata_Nucleocentric_Crop_Minimum_Y"], record["Metadata_Nucleocentric_Crop_Maximum_Y"]
            x_min, x_max = record["Metadata_Nucleocentric_Crop_Minimum_X"], record["Metadata_Nucleocentric_Crop_Maximum_X"]
            corrected, saturated = image_cache[channel]
            ax = axes[row, col]
            # show the same 8-bit crop that MorphEm receives (saturated pixels are 255)
            ax.imshow(
                rescale_crop_to_8bit(corrected[y_min:y_max, x_min:x_max], saturated[y_min:y_max, x_min:x_max]),
                cmap="gray", vmin=0, vmax=255,
            )
            ax.add_patch(
                patches.Rectangle(
                    (record["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"] - x_min, record["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"] - y_min),
                    record["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X"] - record["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"],
                    record["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y"] - record["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"],
                    fill=False, edgecolor="red", linewidth=1,
                )
            )
            ax.set_axis_off()
            if row == 0:
                ax.set_title(channel)
    plt.suptitle(f"Nucleocentric crops (+{expand_pixels} pixels) for {cell_line} from {plate}")
    plt.tight_layout()
    plt.show()

