#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --account=amc-general
#SBATCH --time=12:00:00
#SBATCH --output=run_CP_child-%x-%j.out

# 1 task at 6 GB RAM to process one LoadData CSV (one cell line, one plate/time point) per job.
#
# Each image set needs about 3.5 MB of image data; CellProfiler itself needs ~2 GB baseline.
# Standard jobs process 4 wells x 9 FOVs = 36 image sets (~2.1 GB estimated).
# The 6 U2-OS control-plate jobs process 12 wells x 9 FOVs = 108 image sets (~2.4 GB estimated).
# Requesting 6 GB / 12 hours to be safe, since this pipeline runs substantially more measurement
# modules per image set than the reference row-batch job this estimate was scaled from
# (Colocalization, Granularity, Texture, Intensity Distribution, Neighbors, overlays, DB export).

# activate cellprofiler environment
module load miniforge
conda init bash
conda activate pccma_atlas_cp_env

# input csv (one cell line from one plate/time point) passed as first argument
csv=$1

# run your python analysis script with the input csv
# (notebooks are converted once in the parent script so the jobs do not overwrite each other's scripts)
python nbconverted/1.cp_analysis_hpc.py --input_csv "$csv"

# deactivate conda environment
conda deactivate

echo "CellProfiler analysis done for LoadData CSV: $csv"
