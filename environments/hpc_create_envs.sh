#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=envs-%j.out

# Adapted from Mike Lippincott's script in NF1_3D_organoid_profiling_pipeline repository on GitHub

# Usage (run from within the environments folder):
#   sbatch hpc_create_envs.sh                       -> create/update all environments
#   sbatch hpc_create_envs.sh cellprofiler_env.yml  -> create/update only the given environment file(s)

module load miniforge

# use the environment file(s) passed as arguments, otherwise use all yml files in this folder
if [ "$#" -gt 0 ]; then
    yaml_files="$@"
else
    yaml_files=$(ls *.yml 2>/dev/null)
fi

# stop if there are no environment files to use
if [ -z "$yaml_files" ]; then
    echo "No environment files found (run this script from within the environments folder)"
    exit 1
fi

# make sure every environment file exists before creating anything
for yaml_file in $yaml_files; do
    if [ ! -f "$yaml_file" ]; then
        echo "Environment file not found: $yaml_file (run this script from within the environments folder)"
        exit 1
    fi
done

# read the first line of the yaml file
for yaml_file in $yaml_files; do
    # read the first line of the yaml file
    first_line=$(head -n 1 $yaml_file)
    # parse the first line to get the environment name
    environment_name=$(echo $first_line | cut -d ' ' -f 2)
    # check if the environment exists
    if conda env list | grep -q $environment_name; then
        mamba env update -f $yaml_file
    else
        mamba env create -f $yaml_file
    fi
done

echo "Environments created"
