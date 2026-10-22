# Nucleocentric MorphEm features

In this module, we extract deep learning features with [MorphEm](https://huggingface.co/CaicedoLab/MorphEm) from a crop centered on each nucleus (nucleocentric) and process them into single-cell and bulk profiles with pycytominer.

MorphEm is a self-supervised Vision Transformer (ViT-Small) that the Caicedo Lab trained with the DINO Bag of Channels recipe on the CHAMMI-75 dataset of microscopy images.
It takes one channel at a time and outputs a 384-dimensional embedding (CLS token) per channel, so each single cell has 1,920 features (five channels).

We do not segment the images again.
We use the center and bounding box of each nucleus from CellProfiler (`3.cp_analysis`), so every single cell in the MorphEm profiles matches a single cell in the CellProfiler profiles.
This lets us remove the same single cells that failed QC in `4.preprocess_features` and compare the two types of profiles.

## Nucleocentric crops

For each field of view (FOV), we:

1. Load the five channels, find the pixels that are saturated in the camera (65,535), and apply the illumination correction functions from `2.illumination_correction` (divide, the same as in the CellProfiler pipelines).
2. Crop a square centered on the nucleus (`Nuclei_Location_Center_X/Y`) with a side equal to the longest side of the nucleus bounding box (`Nuclei_AreaShape_BoundingBox*`) plus 25 pixels in each direction.
When the square goes past the edge of the image, we move it back inside the image, so every crop is square and only holds real pixels (`Metadata_Nucleocentric_Crop_Shifted` is True for these crops).
3. Rescale each crop to 8-bit, where the pixels that were saturated in the camera become 255 and we linearly rescale the other pixels to 0 to 254.
4. Apply the MorphEm transforms to each crop (replace the saturated pixels with noise, normalize the crop, and resize to 224 x 224) and extract the CLS token from each channel.

We adapted the nucleocentric approach (+25 pixels around the nucleus bounding box) from the NF1 3D organoid profiling pipeline, with the following changes for 2D images and to process many nuclei at once:

| Change | Reason |
| --- | --- |
| We pass each channel once as a single channel. | MorphEm has one input channel (`in_chans: 1`), so we do not need to copy the channel three times. |
| We apply the transforms to each crop on its own, then run a batch of crops through the model. | `SaturationNoiseInjector` from the model card only changes the first image in a batch. We confirmed that the batched features are the same as running one crop at a time. |
| We convert each crop to 8-bit where only the saturated pixels become 255. | MorphEm was trained on 8-bit images and replaces pixels with the value 255 with noise. When we rescaled the whole FOV with percentile clipping, many pixels of the cells became 255 in FOVs with few cells, so the model would replace them with noise. The normalization in MorphEm removes any linear rescaling, so rescaling each crop on its own only sets the 8-bit precision. |
| We move crops at the edge of the image back inside the image instead of cutting them off. | The crops stay square, so resizing to 224 x 224 does not stretch them. |

We load MorphEm from a fixed revision of the Hugging Face repository (`MORPHEM_REVISION` in `utils/morphem_utils.py`) and set the random seed, since MorphEm replaces the saturated pixels with random noise.

## Notebooks

| Notebook | Environment | Description |
| --- | --- | --- |
| `0.extract_morphem_features.ipynb` | `environments/morphem` (uv) | Extracts the MorphEm features for one cell line from one plate (`--plate` and `--cell_line`) and saves `data/morphem_profiles/<cell_line>/<plate>_<cell_line>_morphem.parquet`. When run as a notebook, it also shows example crops. |
| `1.process_morphem_profiles.ipynb` | `environments/preprocessing` (uv) | For each plate: removes the single cells that failed QC (using `4.preprocess_features/qc_results/`), annotates with the platemap, and creates the single-cell (standardize, feature select) and bulk (median, MAD robustize, feature select) profiles. |

We set the MorphEm features (`Nucleocentric_<channel>_MorphEm_Feature<n>`) in every pycytominer function, since pycytominer only finds the CellProfiler compartments (Cells, Nuclei, and Cytoplasm) by default.
We do not use the blocklist in feature selection, since it only includes CellProfiler features.

## Inputs and outputs

The module depends on the outputs of the previous modules:

- Converted profiles from `4.preprocess_features/data/converted_profiles/` (center and bounding box of each nucleus, image paths, and the IDs of each single cell)
- QC annotations from `4.preprocess_features/qc_results/`
- Illumination correction functions from `2.illumination_correction/illum_directory/`
- Platemaps from `0.download_data/metadata/platemaps/`

| Folder | Description |
| --- | --- |
| `data/morphem_profiles/<cell_line>/` | One MorphEm profile per plate (all single cells, before QC). |
| `data/annotated_profiles/` | One annotated single-cell profile per plate (single cells that passed QC). |
| `data/single_cell_profiles/` | Normalized and feature-selected single-cell profiles per plate. |
| `data/bulk_profiles/` | Aggregated, normalized, and feature-selected bulk profiles per plate. |

Every MorphEm profile has the same metadata as the converted profiles (`Image_Metadata_Plate`, `Image_Metadata_Well`, `Image_Metadata_Site`, `Metadata_ImageNumber`, and the object numbers), along with the nucleus bounding box, the crop coordinates, and the fraction of each crop that was saturated in the camera per channel (`Metadata_Nucleocentric_<channel>_Saturated_Fraction`).
The saturated pixels are where MorphEm adds noise and where the features depend the most on the preprocessing, so we can use these columns to find single cells that might be too saturated (we do not remove them).

## Run the module

Run `4.preprocess_features/run_pipeline.sh` first, then:

```bash
cd 5.morphem_featurization
./run_morphem_pipeline.sh
```

The script creates or updates the two uv environments from their lock files (`uv sync --locked`), extracts the features for every job, then processes each plate.
It skips any job or plate that already has output, unless its inputs are newer (set `OVERWRITE=1` to process everything again).
`1.process_morphem_profiles.ipynb` stops if any cell line on a plate does not have a MorphEm profile, since we normalize to the whole plate.

> **Note:** Run the extraction on a GPU.
> There are about 1.4 million single cells (7.1 million crops across five channels).
> On a CPU, MorphEm extracts about 23 crops per second, which would take more than three days.

## Citations

If you use these features, please cite MorphEm and CHAMMI-75:

```bibtex
@inproceedings{agrawal2026chammi,
  title={{CHAMMI}-75: Pre-training multi-channel models with heterogeneous microscopy images},
  author={Vidit Agrawal and John Peters and Tyler N. Thompson and Mohammad Vali Sanian and Chau Pham and Nikita Moshkov and Arshad Kazi and Aditya Pillai and Jack Freeman and Byunguk Kang and Samouil L. Farhi and Ernest Fraenkel and Ron M. Stewart and Lassi Paavolainen and Bryan A. Plummer and Juan C. Caicedo},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=SLjqdj3LPk}
}
```

- MorphEm model card: <https://huggingface.co/CaicedoLab/MorphEm>
- CHAMMI-75 code: <https://github.com/CaicedoLab/CHAMMI-75>
- NF1 3D organoid profiling pipeline (Lippincott, Tomkinson, et al.), where we adapted the MorphEm featurization and nucleocentric crops:
  - <https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/utils/src/image_analysis_3D/featurization_utils/morphem_featurization.py>
  - <https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/3.cellprofiling/scripts/nucleo_centric_featurization.py>
