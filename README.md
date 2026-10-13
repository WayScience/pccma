# Pediatric Cancer Cell Morphology Atlas

Image analysis and image-based profiling workflows to extract morphology profiles from the Pediatric Cancer Cell Morphology Atlas (PCCMA) dataset.

## Atlas data

The Atlas is a Cell Painting dataset of 56 cell lines imaged at three time points (24, 48, and 72 hours).

- **Plates:** Nine 384-well plates, where each plate is one time point.
- **Cell lines:** 56 cell lines are each on one set of three plates (one plate per time point) with four wells per plate.
U2-OS is included on all nine plates as a control cell line.
- **Conditions:** Seeding density (1,000 to 12,000 cells/well) and plate coating (Standard, Synthemax, or Laminin) are set per cell line and are recorded in the platemaps.
- **Imaging:** Nine fields of view (FOVs) per well at 20x magnification.
- **Channels:** Five Cell Painting channels are used for analysis.
Brightfield channels are not included.

| Channel name | Stain target | Name in pipelines |
| --- | --- | --- |
| HOECHST 33342 | Nuclei | DNA |
| Alexa 488 | Endoplasmic reticulum | ER |
| Alexa 488 Long | Nucleoli and cytoplasmic RNA | RNA |
| Alexa 568 | Actin, Golgi, and plasma membrane | AGP |
| Alexa 647 | Mitochondria | Mito |

| Plates | Time points | Number of cell lines | Wells with cells per plate |
| --- | --- | --- | --- |
| BR00150695, BR00150696, BR00150697 | 24, 48, 72 hours | 14 | 56 |
| BR00150698, BR00150699, BR00150700 | 24, 48, 72 hours | 19 | 84 |
| BR00150701, BR00150702, BR00150703 | 24, 48, 72 hours | 25 | 108 |

### Empty wells that were imaged

Some wells were imaged but do not contain cells (media only), so they are not included in the platemaps (`0.download_data/metadata/platemaps/`).
These wells were visually confirmed to be empty.
Because the per cell line LoadData CSVs created in `3.cp_analysis` are built from the platemaps, these wells are not processed during segmentation and feature extraction.

| Plates | Time points | Empty wells imaged | Number of wells |
| --- | --- | --- | --- |
| BR00150695, BR00150696, BR00150697 | 24, 48, 72 hours | Rows C-F in columns 9-10 and rows G-N in columns 7-10 | 40 |
| BR00150698, BR00150699, BR00150700 | 24, 48, 72 hours | Rows I-N in columns 9-10 | 12 |
| BR00150701, BR00150702, BR00150703 | 24, 48, 72 hours | Rows C-H in columns 3-4 | 12 |

## Repository structure

The modules are numbered in the order that they are run.

| Module | Purpose |
| --- | --- |
| [0.download_data](./0.download_data/) | Generate the platemap for each plate from the experimental layout file and create platemap figures. |
| [1.whole_image_qc](./1.whole_image_qc/) | Extract whole image quality metrics with CellProfiler and determine the blur and saturation thresholds used to identify poor quality FOVs. |
| [2.illumination_correction](./2.illumination_correction/) | Calculate an illumination correction function per channel for each plate with CellProfiler. |
| [3.cp_analysis](./3.cp_analysis/) | Perform illumination correction, segmentation, and feature extraction with CellProfiler, where each cell line has its own pipeline. |
| [environments](./environments/) | Conda environment files and a script to create the environments on an HPC cluster. |
| [utils](./utils/) | Shared functions for creating LoadData CSVs and running CellProfiler in parallel. |

## Quality control (QC) of poor quality images

We handle poor quality FOVs (blurry or over-saturated) differently depending on the module:

- **Illumination correction:** We filtered out poor quality FOVs prior to calculation of the illumination correction functions to avoid issues in the output.
- **Segmentation and feature extraction:** We do process poor quality FOVs.
QC is applied later on, which allows us to salvage any good cells from these FOVs.

## Running the analysis

Each module contains notebooks and a bash script that converts the notebooks to Python scripts and runs them.

### Environments

Create the environments from the files in the `environments` folder.
The CellProfiler modules (whole image QC metric extraction, illumination correction, and analysis) use `pccma_atlas_cp_env`.

```bash
# create one environment
conda env create -f environments/cellprofiler_env.yml
```

On an HPC cluster with SLURM, the environments can be created with the provided script from within the `environments` folder:

```bash
cd environments
# create all environments
sbatch hpc_create_envs.sh
# or create only the CellProfiler environment
sbatch hpc_create_envs.sh cellprofiler_env.yml
```

### Segmentation and feature extraction

The `3.cp_analysis` module processes each cell line from each plate (time point) as its own job.
There are 174 cell line and plate combinations (55 cell lines on three plates and U2-OS on nine plates), of which 166 are processed.
The cell lines and time points that are not processed are listed in the [module README](./3.cp_analysis/README.md).
Each job outputs a SQLite file to `3.cp_analysis/sqlite_outputs/<plate>_<cell_line>/`.

To run on an HPC cluster with SLURM, submit the parent script from within the module, which creates the LoadData CSVs and submits one job per cell line per plate:

```bash
cd 3.cp_analysis
sbatch cp_analysis_hpc_parent.sh
```

To run on a local machine:

```bash
cd 3.cp_analysis
source cp_analysis_local.sh
```

> **Note:** The path to the images is set in `3.cp_analysis/0.create_loaddata_csvs.ipynb` for both the HPC cluster and a local machine and will need to be updated to where the images are stored.
