# Nucleocentric MorphEm features

In this module, we extract deep learning features with [MorphEm](https://huggingface.co/CaicedoLab/MorphEm) from a crop centered on each nucleus (nucleocentric) and process them into single-cell and bulk profiles with pycytominer.

MorphEm is a self-supervised Vision Transformer (ViT-Small) that the Caicedo Lab trained with the DINO Bag of Channels recipe on the CHAMMI-75 dataset of microscopy images.
It takes one channel at a time and outputs a 384-dimensional embedding (CLS token) per channel, so each "single cell" has 1,920 features (five channels).

We do not segment the images again.
We use the center and bounding box of each nucleus from CellProfiler (`3.cp_analysis`), so every single cell in the MorphEm profiles matches a single cell in the CellProfiler profiles.
This lets us remove the same single cells that failed QC in `4.preprocess_features` and compare the two types of profiles.

## Nucleocentric crops

For each field of view (FOV), we:

1. Load the five channels, find the pixels that are saturated in the camera (65,535), and apply the illumination correction functions from `2.illumination_correction` (divide, the same as in the CellProfiler pipelines).
CellProfiler divides each image by 65,535 when loading it (`Image_Scaling_Orig<channel>`), divides by the illumination function, and clips the values to 0 to 1.
We do the same in the 16-bit range (divide, then clip to 0 to 65,535), which only differs by the constant 65,535 that the 8-bit rescaling of each crop removes.
The job stops before loading anything if any illumination correction function of the plate is missing.
2. Crop 224 x 224 pixels centered on the nucleus (`Nuclei_Location_Center_X/Y`), the size of the MorphEm input.
When the crop goes past the edge of the image, we do not move it back inside the image, so the nucleus is always in the center of the crop.
We pad the part of the crop outside the image with the minimum of the crop, which becomes 0 after rescaling to 8-bit (like the dark background).
3. Rescale each crop to 8-bit, where the pixels that were saturated in the camera become 255 and we linearly rescale the other pixels to 0 to 254.
4. Apply the MorphEm transforms to each crop (replace the saturated pixels with noise and normalize the crop; the crops are already 224 x 224, so MorphEm does not resize them) and extract the CLS token from each channel.

### Crop size

We crop 224 x 224 pixels centered on each nucleus.
We chose this crop with `0.evaluate_crop_sizes.ipynb` and `1.view_224_crops.ipynb` (before the extraction), using all 1,420,883 single cells from CellProfiler before QC:

- **The whole cell fits:** the crop holds the whole cell (nucleus and cell bounding boxes) for 99.2% of single cells (91.0% to 100% per cell line).
- **Every cell has the same scale:** the crop is the size of the MorphEm input, so MorphEm does not resize it and the size of a cell in the crop is its real size.
- **Most crops include neighboring cells:** 99.3% of crops include at least one other nucleus (median of 21 other nuclei, from 2 to 53 per cell line).
We expect this, and the nucleus of the single cell is always in the center of the crop.

### Crops at the edge of the image

**424,549 of the 1,420,883 crops (29.9%) are at the edge of the image** (21.2% to 37.1% per cell line, in `crop_evaluation/crop_size_summary.csv`).
We do not move these crops back inside the image, since that would put the nucleus off center (by up to 106 pixels).
Instead, we keep every crop centered on its nucleus and pad the part outside the image.
The padding is a median of 20.1% of these crops (95th percentile 41.5%, maximum 70.3% for a nucleus in the corner of the image).

### Crowding and empty pixels of each crop

Since most crops include neighboring cells or padding, we save how crowded and how empty each crop is, so we can use it in later analyses (e.g., to remove or compare crowded crops).
We measure these from every single cell crop that CellProfiler segmented in the FOV (before QC):

| Column | Type | Description |
| --- | --- | --- |
| `Metadata_Nucleocentric_Other_Nuclei_Count` | `int64` | Number of other nuclei whose center is inside the crop. |
| `Metadata_Nucleocentric_Other_Cells_Fraction` | `float64` | Fraction of the crop covered by the bounding boxes of other cells. |
| `Metadata_Nucleocentric_Cell_Overlap_Fraction` | `float64` | Fraction of the bounding box of the centered single cell that overlaps the bounding boxes of other cells. |
| `Metadata_Nucleocentric_Crop_At_Edge` | `bool` | Whether the crop goes past the edge of the image. |
| `Metadata_Nucleocentric_Pad_Fraction` | `float64` | Fraction of the crop that is padding (outside the image). |
| `Metadata_Nucleocentric_ZeroFraction_<channel>` | `float64` | Fraction of the 8-bit crop of each channel that is 0 (the padding and the darkest pixels of the crop), which MorphEm sees as empty. It is at least the padding fraction, and can be much higher in sparse FOVs, where most of the crop is background. |

The two bounding box fractions are an upper bound of the overlap of the cells themselves, since a bounding box also covers the background around a cell.

### How we pass the crops to MorphEm

- We pass each channel once as a single channel, since MorphEm has one input channel (`in_chans: 1`).
- We apply the MorphEm transforms to each crop on its own, then run a batch of crops through the model, since `SaturationNoiseInjector` from the model card only changes the first image in a batch (the batched features are the same as running one crop at a time).
- We rescale each crop to 8-bit on its own, where only the pixels that were saturated in the camera become 255.
MorphEm was trained on 8-bit images and replaces the pixels with the value 255 with noise, and its normalization removes any linear rescaling, so rescaling each crop on its own only sets the 8-bit precision.

We load MorphEm from a fixed revision of the Hugging Face repository (`MORPHEM_REVISION` in `utils/morphem_utils.py`) and set the random seed, since MorphEm replaces the saturated pixels with random noise.

## Notebooks

We run the notebooks in order: first we check which crop holds the whole cell and how many neighboring cells it includes (`0` and `1`), then we extract the features with the chosen crop (`2`) and process them (`3`).
`run_morphem_pipeline.sh` only runs the extraction and processing, since the crop evaluation is a one-time check of the crop size.

| Notebook | Environment | Description |
| --- | --- | --- |
| `0.evaluate_crop_sizes.ipynb` | `environments/preprocessing` (uv) | Run before the extraction to choose the crop (sanity check, not run by `run_morphem_pipeline.sh`). Checks whether a 224 x 224 crop centered on the nucleus holds the whole cell (99.2% of the 1.4 million single cells before QC fit), counts the crops at the edge of the image (29.9%, padded instead of moved), and counts the other single cells in each crop. Saves `figures/crop_size_224_fit_per_cell_line.png`, a landscape version for slides (`figures/crop_size_224_fit_per_cell_line_slide.png`), `figures/crop_neighbors_per_cell_line_slide.png`, `crop_evaluation/crop_size_summary.csv`, and `crop_evaluation/crop_neighbor_summary.csv`. |
| `1.view_224_crops.ipynb` | `environments/preprocessing` (uv) | Run before the extraction to choose the crop (sanity check, not run by `run_morphem_pipeline.sh`). Shows example 224 x 224 crops centered on the nucleus for every cell line (single cells that passed QC, picked at the 25th, 50th, and 75th percentiles of the number of other nuclei in the crop), as composites of DNA, ER, and AGP. Saves `figures/example_224_crops_typical_per_cell_line.png` (one crop per cell line) and `figures/example_224_crops_per_cell_line.png` (three crops per cell line). Run `0.evaluate_crop_sizes.ipynb` first. |
| `2.extract_morphem_features.ipynb` | `environments/morphem` (uv) | Extracts the MorphEm features for one cell line from one plate (`--plate` and `--cell_line`) and saves `data/morphem_profiles/<cell_line>/<plate>_<cell_line>_morphem.parquet`. When run as a notebook, it also shows example crops. |
| `3.process_morphem_profiles.ipynb` | `environments/preprocessing` (uv) | For each plate: removes the single cells that failed QC from coSMicQC (using `4.preprocess_features/qc_results/`), annotates with the platemap, and creates the single-cell (standardize, feature select) and bulk (median, MAD robustize, feature select) profiles. |

We set the MorphEm features in every pycytominer function, since pycytominer only finds the CellProfiler compartments (Cells, Nuclei, and Cytoplasm) by default.
We do not use the blocklist in feature selection, since it only includes CellProfiler features.

## Feature names

We name the MorphEm features in the same order as the CellProfiler features (compartment, feature group, measurement, then channel), so any downstream code that expects the channel at the end of the name also works:

```
Nucleocentric_MorphEm_Feature<n>_<channel>
```

| Part | CellProfiler equivalent | Values |
| --- | --- | --- |
| `Nucleocentric` | Compartment (e.g., `Nuclei`) | The 224 x 224 crop centered on the nucleus |
| `MorphEm` | Feature group (e.g., `Intensity`) | The model that extracted the features |
| `Feature<n>` | Measurement (e.g., `MeanIntensity`) | Position in the 384-dimensional MorphEm embedding (0 to 383) |
| `<channel>` | Channel (e.g., `DNA`) | `DNA`, `ER`, `RNA`, `AGP`, or `Mito` |

For example, `Nucleocentric_MorphEm_Feature0_DNA` to `Nucleocentric_MorphEm_Feature383_Mito`, which gives 1,920 features per single cell.
Every name has four parts separated by underscores, so the channel is always the last part (`name.split("_")[-1]`).

We extract the features of each channel separately (Bag of Channels), so the same `Feature<n>` in two channels is the same position in the embedding of the same model, but from different images.
They are not paired measurements like the CellProfiler intensity features of two channels, so we treat each column as its own feature.

## Inputs and outputs

The module depends on the outputs of the previous modules:

- Converted profiles from `4.preprocess_features/data/converted_profiles/` (center and bounding box of each nucleus, image paths, and the IDs of each single cell)
- coSMicQC annotations from `4.preprocess_features/qc_results/`
- Illumination correction functions from `2.illumination_correction/illum_directory/`
- Platemaps from `0.download_data/metadata/platemaps/`

| Folder | Description |
| --- | --- |
| `data/morphem_profiles/<cell_line>/` | One MorphEm profile per plate (all single cells, before QC). |
| `data/annotated_profiles/` | One annotated single-cell profile per plate (single cells that passed QC). |
| `data/single_cell_profiles/` | Normalized and feature-selected single-cell profiles per plate. |
| `data/bulk_profiles/` | Aggregated, normalized, and feature-selected bulk profiles per plate. |

Every MorphEm profile has the same metadata as the converted profiles (`Image_Metadata_Plate`, `Image_Metadata_Well`, `Image_Metadata_Site`, `Metadata_ImageNumber`, and the object numbers), along with the nucleus and cell bounding boxes, the crop coordinates (`Metadata_Nucleocentric_Crop_*`, which are outside the image for crops at the edge), the crop size (`Metadata_Nucleocentric_Crop_Size`, 224), the edge and crowding columns of each crop (see [Crowding and empty pixels of each crop](#crowding-and-empty-pixels-of-each-crop)), and the fraction of each crop that was saturated in the camera per channel (`Metadata_Nucleocentric_SaturatedFraction_<channel>`).
`3.process_morphem_profiles.ipynb` stops if any MorphEm profile on a plate is not from the 224 x 224 crop, so we do not mix profiles from different crops.
The saturated pixels are where MorphEm adds noise and where the features depend the most on the preprocessing, so we can use these columns to find single cells that might be too saturated (we do not remove them).

## Run the module

Run `4.preprocess_features/run_preprocessing_pipeline.sh` first, then (from any folder):

```bash
./5.morphem_featurization/run_morphem_pipeline.sh
```

You only need [uv](https://docs.astral.sh/uv/) installed; you do not need to create or activate an environment.
The script runs every step with `uv run --locked --project environments/<environment>`, which creates or updates the environment from its lock file and runs the step in it.
It extracts the features for every job (`2.extract_morphem_features.ipynb`), then processes each plate (`3.process_morphem_profiles.ipynb`).
You do not need to run `0.evaluate_crop_sizes.ipynb` or `1.view_224_crops.ipynb` first; they are a one-time check of the crop size, and the script does not run them.
It skips any job or plate that already has output, unless its inputs are newer (set `OVERWRITE=1` to process everything again).
The script saves its logs in `logs/` (not tracked by git), and the messages from the script start with the time and a level (`INFO` or `ERROR`):

| Log | Contents |
| --- | --- |
| `run_morphem_pipeline_<date>_<time>.log` | One per run: everything the script and the notebooks print (the same as the terminal). |
| `<plate>_<cell_line>_morphem_extraction.log` | One per extraction job (cell line and plate), the same as the CellProfiler runs. |
| `<plate>_morphem_processing.log` | One per processed plate. |

A job log is only written when the job runs, so skipped jobs keep the log from the run that made their output, and running a job again replaces its log.
`3.process_morphem_profiles.ipynb` stops if any cell line on a plate does not have a MorphEm profile, since we normalize to the whole plate.

> **Note:** Run the extraction on a GPU.
> The extraction stops if CUDA is not available instead of falling back to the CPU (use `--device cpu`, or `DEVICE=cpu` for `run_morphem_pipeline.sh`, only to test a small job).
> There are about 1.4 million single cells, but MorphEm only takes one channel at a time.
> Unlike a segmentation mask, where one mask is applied to all channels, each single cell gets a separate crop image for each of the five channels (the same crop box, cut from each channel).
> This means there are about 1.4 million cells × 5 channels ≈ 7.1 million crops that MorphEm processes.
> On a CPU test, MorphEm extracts about 23 crops per second, which would take more than three days.

## Paths and shared utilities

The notebooks do not add `utils` to `sys.path`.
Both uv environments install `utils` as a package in editable mode (`pccma-atlas-utils` in `[tool.uv.sources]`), so the notebooks import the functions directly (e.g., `from morphem_utils import featurize_crops`).

At the top of each notebook, we find the root of the repository with `get_root_dir()` (from `utils/path_utils.py`, which searches up for the `.git` folder) and build every path from it (`root_dir` for other modules and `module_dir` for this module).
The paths do not depend on the folder we run from, so the notebooks give the same paths when run interactively from this folder and when `run_morphem_pipeline.sh` runs the scripts from the root.
