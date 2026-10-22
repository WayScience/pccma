#!/bin/bash
#
# Run the nucleocentric MorphEm pipeline, in order:
#   0.extract_morphem_features.ipynb  -> one MorphEm profile per cell line and plate (one job per
#                                        converted profile from 4.preprocess_features, on a GPU if available)
#   1.process_morphem_profiles.ipynb  -> per plate: remove the single cells that failed QC (from
#                                        4.preprocess_features/qc_results), annotate, and create the
#                                        single-cell and bulk profiles
#
# Run 4.preprocess_features/run_pipeline.sh first, since we use the converted profiles (center and
# bounding box of each nucleus) and the QC annotations from that module.
#
# Each job and plate runs as its own `python` process, so the memory is released when it exits.
# By default, every step skips a job or plate that already has output. Set OVERWRITE=1
# (e.g. `OVERWRITE=1 ./run_morphem_pipeline.sh`) to reprocess and overwrite everything instead.

set -uo pipefail

overwrite="${OVERWRITE:-}"

failed=()

# create or update a uv environment from its lock file, then activate it
activate_env() {
    local env_dir="../environments/$1"
    if ! uv sync --locked --project "$env_dir"; then
        echo "uv sync FAILED for ${env_dir}; aborting."
        exit 1
    fi
    source "${env_dir}/.venv/bin/activate"
}

# -----------------------------
# Step 0: extract the MorphEm features (one job per cell line per plate)
# -----------------------------
activate_env morphem

jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb
if [ $? -ne 0 ]; then
    echo "jupyter nbconvert FAILED; aborting pipeline."
    exit 1
fi

mapfile -t converted_files < <(find ../4.preprocess_features/data/converted_profiles -mindepth 2 -maxdepth 2 -name "*_converted.parquet" | sort)
echo "Number of jobs found: ${#converted_files[@]}"

overwrite_flag=""
if [ -n "$overwrite" ]; then
    overwrite_flag="--overwrite"
fi

for converted_file in "${converted_files[@]}"; do
    # cell line is the folder name and plate is the prefix of the file name (<plate>_<cell_line>_converted.parquet)
    cell_line=$(basename "$(dirname "$converted_file")")
    plate=$(basename "$converted_file" | cut -d "_" -f 1)
    morphem_file="./data/morphem_profiles/${cell_line}/${plate}_${cell_line}_morphem.parquet"

    # extract again when the converted profile is newer than the MorphEm profile (e.g., CellProfiler was run again)
    job_flag="$overwrite_flag"
    if [ -f "$morphem_file" ] && [ "$converted_file" -nt "$morphem_file" ]; then
        echo "⚠️ ${converted_file} is newer than ${morphem_file}, extracting again"
        job_flag="--overwrite"
    fi

    python nbconverted/0.extract_morphem_features.py --plate "$plate" --cell_line "$cell_line" $job_flag
    if [ $? -ne 0 ]; then
        echo "MorphEm extraction FAILED for ${plate} ${cell_line}"
        failed+=("extract:${plate}_${cell_line}")
    fi
done

deactivate

# -----------------------------
# Step 1: process the MorphEm profiles (one plate at a time)
# -----------------------------
activate_env preprocessing

mapfile -t plate_ids < <(find data/morphem_profiles -mindepth 2 -maxdepth 2 -name "*_morphem.parquet" -printf "%f\n" | cut -d_ -f1 | sort -u)
echo "Number of plates found: ${#plate_ids[@]}"

for plate_id in "${plate_ids[@]}"; do
    plate_output="./data/bulk_profiles/${plate_id}_morphem_bulk_feature_selected.parquet"
    plate_overwrite="$overwrite"

    # process the plate again when a MorphEm profile or the QC annotations of a cell line on the plate
    # are newer than the output (e.g., new jobs were extracted or QC was run again)
    if [ -f "$plate_output" ] && [ -z "$overwrite" ]; then
        for morphem_file in data/morphem_profiles/*/"${plate_id}"_*_morphem.parquet; do
            plate_cell_line=$(basename "$(dirname "$morphem_file")")
            qc_file="../4.preprocess_features/qc_results/${plate_cell_line}_qc_annotations.parquet"
            if [ "$morphem_file" -nt "$plate_output" ] || [ "$qc_file" -nt "$plate_output" ]; then
                echo "⚠️ inputs for ${plate_id} are newer than ${plate_output}, processing again"
                plate_overwrite=1
                break
            fi
        done
    fi

    PLATE_ID="$plate_id" OVERWRITE="$plate_overwrite" python nbconverted/1.process_morphem_profiles.py
    if [ $? -ne 0 ]; then
        echo "Processing FAILED for plate: ${plate_id}"
        failed+=("process:${plate_id}")
    fi
done

deactivate

echo "======================================"
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failures:"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "✅ MorphEm pipeline completed successfully."
