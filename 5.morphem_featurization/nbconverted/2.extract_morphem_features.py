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
# 2. Crop 224 x 224 pixels centered on each nucleus (the MorphEm input size, so MorphEm does not resize the crops and every single cell has the same scale).
# We chose this crop with `0.evaluate_crop_sizes.ipynb` and `1.view_224_crops.ipynb`: it holds the whole cell for 99.2% of single cells, and most crops also include neighboring cells.
# When the nucleus is near the edge of the image, we do not move the crop back inside the image, so the nucleus is always in the center; we pad the part of the crop outside the image with the minimum of the crop (0 after rescaling, like the dark background).
# 3. Rescale each crop to 8-bit, since MorphEm was trained on 8-bit images, where only the pixels that were saturated in the camera become 255 (MorphEm replaces them with noise).
# 4. Extract the 384 MorphEm features (CLS token) from each channel of each crop, which gives 1,920 features per single cell.
# 
# We name the features like the CellProfiler features (compartment, feature group, measurement, then channel), so tools that expect the channel at the end of the name also work: `Nucleocentric_MorphEm_Feature<n>_<channel>` (e.g., `Nucleocentric_MorphEm_Feature0_DNA`, with `<n>` from 0 to 383).
# 
# We save the features with the same metadata columns as the converted profiles (plate, well, site, image number, and object numbers), so we can remove the single cells that failed QC with the QC annotations from `4.preprocess_features/1.sc_quality_control.ipynb`.
# We also save whether each crop is at the edge of the image and the fraction of the crop that is padding (`Metadata_Nucleocentric_Crop_At_Edge` and `Metadata_Nucleocentric_Pad_Fraction`), and how crowded each crop is: the number of other nuclei in the crop, the fraction of the crop covered by the bounding boxes of other cells, and the fraction of the cell's bounding box that overlaps the bounding boxes of other cells (`Metadata_Nucleocentric_Other_Nuclei_Count`, `Metadata_Nucleocentric_Other_Cells_Fraction`, and `Metadata_Nucleocentric_Cell_Overlap_Fraction`).
# We do not remove any single cells with these columns, but we can use them in later analyses (e.g., to remove crowded crops).
# 
# > **Note:** We extracted the MorphEm features for every job by running this notebook as a script (`nbconverted/2.extract_morphem_features.py`) with `run_morphem_pipeline.sh`.
# > The outputs below are from running the notebook afterwards without overwriting, so it skips the extraction because the MorphEm profile already exists.

# ## Import libraries

# In[1]:


import argparse
import os
import pathlib
import re
import time

import numpy as np
import pandas as pd
import tifffile
import torch
from bandicoot_utils import bandicoot_check
from morphem_utils import (
    MORPHEM_INPUT_SIZE,
    MORPHEM_REVISION,
    count_other_nuclei,
    crop_with_padding,
    featurize_crops,
    get_morphem_model,
    get_nucleocentric_crop_box,
    get_other_cell_overlap,
    rescale_crop_to_8bit,
)
from path_utils import get_root_dir

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

# root of the repository (found from the .git folder), so every path is built from the root
# and the notebook gives the same paths whether we run it from this folder or from the root
root_dir = get_root_dir()
module_dir = root_dir / "5.morphem_featurization"


# ## Define functions

# In[2]:


def correct_illumination(image: np.ndarray, illum_function: np.ndarray) -> np.ndarray:
    # the same as CorrectIlluminationApply in CellProfiler (divide, then set values below 0 to 0 and above
    # the maximum to the maximum), but in the 16-bit range instead of 0 to 1 (CellProfiler divides the
    # images by 65535 when loading them), which is a constant factor that the 8-bit rescaling of each crop removes
    return np.clip(image.astype(np.float32) / illum_function, 0, np.iinfo(image.dtype).max)


# ## Set paths and variables

# In[3]:


if not in_notebook:
    print("Running as script")
    parser = argparse.ArgumentParser(
        description="Extract nucleocentric MorphEm features for one cell line from one plate"
    )
    parser.add_argument("--plate", type=str, required=True, help="Plate name (e.g., BR00150698)")
    parser.add_argument("--cell_line", type=str, required=True, help="Name of the cell line folder (e.g., A673)")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of crops to run through MorphEm at once")
    parser.add_argument("--overwrite", action="store_true", help="Extract the features again if the output already exists")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to run MorphEm on (cpu is only for testing small jobs, the whole screen would take days)",
    )
    args = parser.parse_args()
    plate, cell_line, batch_size, overwrite, device = args.plate, args.cell_line, args.batch_size, args.overwrite, args.device
else:
    print("Running in a notebook")
    plate = "BR00150698"
    cell_line = "A673"
    batch_size = 256
    overwrite = False
    device = "cuda"

# side of the square crop centered on each nucleus (the MorphEm input size, chosen in 0.evaluate_crop_sizes.ipynb)
crop_size = MORPHEM_INPUT_SIZE

# channel names in the converted profiles (Image_FileName_Orig<channel>) and illumination functions (<plate>_Illum<channel>.npy)
channels = ["DNA", "ER", "RNA", "AGP", "Mito"] # constant list

# set the seed, since MorphEm replaces the saturated pixels with random noise
seed = 0
torch.manual_seed(seed)

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# Directory with the images (one folder per plate), used when the paths in the profiles are from the HPC cluster
images_dir = pathlib.Path(f"{image_base_dir}/PCCMA_data/Atlas_v2_screen")

# converted profile of the job with the center and bounding box of each nucleus from CellProfiler
converted_path = (
    root_dir / "4.preprocess_features/data/converted_profiles" / cell_line / f"{plate}_{cell_line}_converted.parquet"
).resolve(strict=True)

# illumination correction functions of the plate
illum_dir = (root_dir / "2.illumination_correction/illum_directory" / plate).resolve(strict=True)

# one folder per cell line with one parquet file per plate (time point), the same as the converted profiles
output_dir = module_dir / "data/morphem_extracted_profiles" / cell_line
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{plate}_{cell_line}_morphem.parquet"

# illumination correction function of each channel (<plate>_Illum<channel>.npy)
illum_paths = {
    channel: illum_dir / f"{plate}_Illum{channel}.npy"
    for channel in channels
}

# extract the features unless the output already exists (every step below only runs when this is True)
extract_features = overwrite or not output_path.exists()
if not extract_features:
    print(f"Skipping {plate} {cell_line} (found {output_path})")
else:
    # stop before loading the profile or the model if any illumination correction function is missing
    missing_illum = [
        str(path) for path in illum_paths.values() if not path.is_file()
    ]
    if missing_illum:
        raise FileNotFoundError(
            "Cannot extract MorphEm features due to missing illumination functions; "
            f"missing illumination function(s): {missing_illum}"
        )
    # stop instead of falling back to the CPU, which would take days for the whole screen
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available (no GPU or the NVIDIA driver is not working). "
            "Run on a machine with a GPU, or use --device cpu to test a small job on the CPU."
        )
    print(f"Using device: {device}")
    print(f"Processing {cell_line} from {plate}")


# ## Load the single cells from the converted profile
# 
# We only load the metadata columns, the center and bounding box of each nucleus, and the image paths.

# In[4]:


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

# nucleus and cell bounding boxes from CellProfiler (we add the Metadata prefix so pycytominer keeps them as metadata)
# the cell bounding boxes are used to measure how much each crop and cell overlaps the other cells
bbox_columns = [
    f"{compartment}_AreaShape_BoundingBox{bound}_{axis}"
    for compartment in ("Nuclei", "Cells")
    for bound in ("Minimum", "Maximum")
    for axis in ("X", "Y")
]

image_columns = [f"Image_FileName_Orig{channel}" for channel in channels] + [
    f"Image_PathName_Orig{channel}" for channel in channels
]

if extract_features:
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

# In[5]:


if extract_features:
    illum_functions = {
        channel: np.load(path) for channel, path in illum_paths.items()
    }

    model = get_morphem_model(device)
    print(f"Loaded MorphEm (revision {MORPHEM_REVISION})")


# ## Extract the nucleocentric features from each FOV

# In[6]:


if extract_features:
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
            fov_images[channel] = correct_illumination(image, illum_functions[channel])

        # find the crop centered on each nucleus (the same crop for every channel, can go past the edge of the image)
        crop_boxes = [
            get_nucleocentric_crop_box(
                center_x=record["Metadata_Nuclei_Location_Center_X"],
                center_y=record["Metadata_Nuclei_Location_Center_Y"],
                crop_size=crop_size,
            )
            for record in fov_df.to_dict(orient="records")
        ]
        # fraction of each crop that is outside the image (padding), which is the same for every channel
        pad_fractions = [
            crop_with_padding(fov_saturated[channels[0]], crop_box, pad_value=False)[1]
            for crop_box in crop_boxes
        ]

        # how crowded each crop is (every single cell segmented in the FOV, before QC):
        # the number of other nuclei in the crop, the fraction of the crop covered by the bounding boxes of
        # other cells, and the fraction of the cell's bounding box that overlaps the bounding boxes of other cells
        other_nuclei_counts = count_other_nuclei(
            crop_boxes,
            fov_df["Metadata_Nuclei_Location_Center_X"].to_numpy(),
            fov_df["Metadata_Nuclei_Location_Center_Y"].to_numpy(),
        )
        cell_boxes = fov_df[
            [
                "Cells_AreaShape_BoundingBoxMinimum_Y",
                "Cells_AreaShape_BoundingBoxMaximum_Y",
                "Cells_AreaShape_BoundingBoxMinimum_X",
                "Cells_AreaShape_BoundingBoxMaximum_X",
            ]
        ].to_numpy()
        other_cells_fractions, cell_overlap_fractions = get_other_cell_overlap(
            crop_boxes, cell_boxes, fov_saturated[channels[0]].shape
        )

        # extract the features of each channel (one block of 384 features per channel)
        channel_feature_dfs = []
        saturated_fractions = {}
        zero_fractions = {}
        for channel in channels:
            # the padding is never saturated, and the padded pixels become 0 after rescaling to 8-bit
            saturated_crops = [
                crop_with_padding(fov_saturated[channel], crop_box, pad_value=False)[0]
                for crop_box in crop_boxes
            ]
            # fraction of the crop that was saturated in the camera (MorphEm replaces these pixels with noise)
            saturated_fractions[
                f"Metadata_Nucleocentric_SaturatedFraction_{channel}"
            ] = [saturated_crop.mean() for saturated_crop in saturated_crops]
            crops = [
                rescale_crop_to_8bit(
                    crop_with_padding(fov_images[channel], crop_box)[0],
                    saturated=saturated_crop,
                )
                for crop_box, saturated_crop in zip(crop_boxes, saturated_crops)
            ]
            # fraction of the 8-bit crop that is 0 (the padding and the darkest pixels of the crop), which MorphEm sees as empty
            zero_fractions[f"Metadata_Nucleocentric_ZeroFraction_{channel}"] = [
                (crop == 0).mean() for crop in crops
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
            ],
            index=fov_df.index,
        )
        crop_df["Metadata_Nucleocentric_Pad_Fraction"] = pad_fractions
        crop_df["Metadata_Nucleocentric_Crop_At_Edge"] = crop_df["Metadata_Nucleocentric_Pad_Fraction"] > 0
        crop_df["Metadata_Nucleocentric_Other_Nuclei_Count"] = other_nuclei_counts
        crop_df["Metadata_Nucleocentric_Other_Cells_Fraction"] = other_cells_fractions
        crop_df["Metadata_Nucleocentric_Cell_Overlap_Fraction"] = cell_overlap_fractions
        saturated_df = pd.DataFrame(saturated_fractions, index=fov_df.index)
        zero_df = pd.DataFrame(zero_fractions, index=fov_df.index)
        fov_dfs.append(pd.concat([crop_df, saturated_df, zero_df] + channel_feature_dfs, axis=1))

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

# In[7]:


if extract_features:
    morphem_df = pd.concat(
        [
            single_cell_df[id_columns + image_columns],
            single_cell_df[bbox_columns].rename(columns=lambda col: f"Metadata_{col}"),
            features_df,
        ],
        axis=1,
    )
    morphem_df["Metadata_Nucleocentric_Crop_Size"] = crop_size

    # every single cell from the converted profile should have features
    assert len(morphem_df) == len(single_cell_df), "Single cells were lost during featurization"
    assert not morphem_df[feature_names].isna().any().any(), "Some features are NaN"

    # save to a temporary file first so a stopped run does not leave an unfinished parquet file
    temp_path = output_dir / f"{plate}_{cell_line}_temp.parquet"
    morphem_df.to_parquet(temp_path, index=False)
    temp_path.replace(output_path)

    at_edge = morphem_df["Metadata_Nucleocentric_Crop_At_Edge"]
    print(f"Saved {output_path} with the shape {morphem_df.shape}")
    print(
        f"{at_edge.sum()} of {len(morphem_df)} crops ({at_edge.mean():.1%}) are at the edge of the image and padded "
        f"(median padding of those crops: {morphem_df.loc[at_edge, 'Metadata_Nucleocentric_Pad_Fraction'].median():.1%})"
    )
    print(
        f"Median per crop: {morphem_df['Metadata_Nucleocentric_Other_Nuclei_Count'].median():.0f} other nuclei, "
        f"{morphem_df['Metadata_Nucleocentric_Other_Cells_Fraction'].median():.1%} of the crop covered by other cells, "
        f"{morphem_df['Metadata_Nucleocentric_Cell_Overlap_Fraction'].median():.1%} of the cell overlapping other cells (bounding boxes)"
    )


# ## Show example crops (notebook only)
# 
# We show the crops of a few single cells with the nucleus bounding box in red, to confirm that the crops are centered on the nuclei (as a sanity check).

# In[8]:


if in_notebook and extract_features:
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
                    correct_illumination(image, illum_functions[channel]),
                    image == np.iinfo(image.dtype).max,
                )
            crop_box = (
                record["Metadata_Nucleocentric_Crop_Minimum_Y"], record["Metadata_Nucleocentric_Crop_Maximum_Y"],
                record["Metadata_Nucleocentric_Crop_Minimum_X"], record["Metadata_Nucleocentric_Crop_Maximum_X"],
            )
            y_min, x_min = crop_box[0], crop_box[2]
            corrected, saturated = image_cache[channel]
            ax = axes[row, col]
            # show the same 8-bit crop that MorphEm receives (saturated pixels are 255, padding is 0)
            ax.imshow(
                rescale_crop_to_8bit(
                    crop_with_padding(corrected, crop_box)[0],
                    crop_with_padding(saturated, crop_box, pad_value=False)[0],
                ),
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
    plt.suptitle(f"Nucleocentric crops ({crop_size} x {crop_size} pixels) for {cell_line} from {plate}")
    plt.tight_layout()
    plt.show()
