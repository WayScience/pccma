#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --account=amc-general
#SBATCH --time=30:00
#SBATCH --output=cp_parent-%j.out

# activate cellprofiler environment
module load miniforge
conda init bash
conda activate pccma_atlas_cp_env

# convert all notebooks to python scripts (if any exist)
jupyter nbconvert --to=script --FilesWriter.build_directory=nbconverted/ *.ipynb

# run the LoadData CSV creation script once before submitting jobs
# (paths to the images on scratch and the illumination functions in this repo are written into the CSVs)
python nbconverted/0.create_loaddata_csvs.py --HPC || {
    echo "LoadData CSV creation failed, no CellProfiler jobs submitted!"
    exit 1
}

# directory with one folder per cell line, each holding one LoadData CSV per plate (time point)
data_dir="$(pwd)/loaddata_csvs/cell_line_loaddata_csvs"

# directory with one CellProfiler pipeline per cell line
pipeline_dir="$(pwd)/pipeline"

# get a list of all LoadData CSV files (one per cell line per plate/time point)
mapfile -t loaddata_csvs < <(find "$data_dir" -mindepth 2 -maxdepth 2 -type f -name "*_loaddata_with_illum.csv" | sort)

echo "Number of LoadData CSV files: ${#loaddata_csvs[@]}"

submitted=0
skipped=0

# loop over each CSV and submit one child job per cell line per plate/time point
for loaddata_file in "${loaddata_csvs[@]}"; do
    # cell line is the folder name and plate is the prefix of the file name
    cell_line=$(basename "$(dirname "$loaddata_file")")
    plate=$(basename "$loaddata_file" | cut -d "_" -f 1)

    # skip any cell line that does not have a matching pipeline
    if [ ! -f "${pipeline_dir}/analysis_${cell_line}.cppipe" ]; then
        echo "Skipping $(basename "$loaddata_file"): no pipeline found for ${cell_line}"
        skipped=$((skipped + 1))
        continue
    fi

    # check job count for this user
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    sbatch --job-name="${plate}_${cell_line}" cp_analysis_hpc_child.sh "$loaddata_file"
    submitted=$((submitted + 1))
done

conda deactivate

echo "Submitted ${submitted} CellProfiler jobs (${skipped} skipped for missing pipelines)!"
