# CellProfiler analysis (segmentation and feature extraction)

In this module, we perform illumination correction, segmentation, and feature extraction with CellProfiler.

Each cell line has its own pipeline (`pipeline/analysis_<cell_line>.cppipe`) because the segmentation parameters are optimized per cell line.
In each pipeline, we:

1. Apply the illumination correction function to each channel.
2. Segment nuclei, cells, and cytoplasm.
3. Extract morphology features and export to a SQLite file.

> **Note:** Poor quality FOVs (blurry or over-saturated) are not filtered out in these pipelines.
> These FOVs are only filtered out when calculating the illumination correction functions.
> We process all FOVs here and apply QC later on to salvage any good cells from the poor quality FOVs.

## One job per cell line per plate

Each cell line from each plate (time point) is processed as its own job.
There are 174 cell line and plate combinations in total:

- 55 cell lines on three plates each (165)
- U2-OS on all nine plates (9)

Of these, 166 are processed as jobs and eight are excluded (see below).

### Excluded cell lines and time points

Some cell lines and time points can not be processed, so no LoadData CSV is created and no job is run for them.

| Cell line | Time points not processed | Plates | Number of jobs excluded |
| --- | --- | --- | --- |
| CF1500 | 24, 48, 72 hours (all) | BR00150698, BR00150699, BR00150700 | 3 |
| CHLA262 | 72 hours | BR00150703 | 1 |
| Saos-2 | 24 hours | BR00150695 | 1 |
| X0092 | 24, 48, 72 hours (all) | BR00150698, BR00150699, BR00150700 | 3 |

The exclusions are set in the `excluded_time_points` dictionary in `0.create_loaddata_csvs.ipynb`, which applies to both the HPC cluster and a local machine.

## Notebooks

| Notebook | Description |
| --- | --- |
| `0.create_loaddata_csvs.ipynb` | Creates a LoadData CSV per plate with the paths to the images and illumination correction functions, then splits each CSV by cell line using the platemaps (excluded cell lines and time points are skipped). |
| `1.cp_analysis_hpc.ipynb` | Runs CellProfiler for one LoadData CSV (one cell line from one plate), which is passed with `--input_csv`. |
| `1.cp_analysis_local.ipynb` | Runs CellProfiler for all LoadData CSVs in batches on a local machine. |
| `2.validate_cp_outputs.ipynb` | Confirms that every job completed properly by comparing the SQLite file and outline images of each job to the LoadData CSV used to run it. |

All notebooks use the `pccma_atlas_cp_env` environment (`environments/cellprofiler_env.yml`).

## Inputs and outputs

| File or folder | Description |
| --- | --- |
| `pipeline/` | One CellProfiler pipeline per cell line. |
| `loaddata_csvs/` | One LoadData CSV per plate. |
| `loaddata_csvs/cell_line_loaddata_csvs/<cell_line>/` | One LoadData CSV per plate for the cell line (e.g., `BR00150698_A673_loaddata_with_illum.csv`). |
| `sqlite_outputs/<plate>_<cell_line>/` | SQLite file and segmentation outline images per job (not tracked in the repository). |
| `logs/` | CellProfiler log per job (not tracked in the repository). |

The name of the cell line folder must match the name of the pipeline.
Any LoadData CSV for a cell line without a matching pipeline is skipped.

The module depends on the outputs of the previous modules:

- Platemaps from `0.download_data/metadata/platemaps/`
- LoadData config from `1.whole_image_qc/load_data_config/config.yml`
- Illumination correction functions from `2.illumination_correction/illum_directory/`

## Run on an HPC cluster (SLURM)

Submit the parent script from within the module:

```bash
cd 3.cp_analysis
sbatch cp_analysis_hpc_parent.sh
```

The parent script (`cp_analysis_hpc_parent.sh`):

1. Converts the notebooks to Python scripts.
2. Creates the LoadData CSVs with the paths to the images on the cluster (`--HPC` flag).
No jobs are submitted if this step fails.
3. Submits one child job (`cp_analysis_hpc_child.sh`) per LoadData CSV, where the job name is `<plate>_<cell_line>`.

> **Note:** A job can finish without an error from SLURM even if CellProfiler failed.
> After all jobs finish, confirm that there are 166 folders in `sqlite_outputs/` that each contain a SQLite file and check the `logs/` folder for errors.

> **Note:** Running the parent script again will submit all jobs again, including the jobs that already finished.

## Run on a local machine

```bash
cd 3.cp_analysis
source cp_analysis_local.sh
```

The number of jobs that run at the same time is set with `max_workers` in `1.cp_analysis_local.ipynb`.

## Image paths

The path to the images is set in `0.create_loaddata_csvs.ipynb` for both the HPC cluster and a local machine and will need to be updated to where the images are stored.
The LoadData CSVs in this repository contain the paths for the local machine and are created again with the cluster paths when running the parent script.
