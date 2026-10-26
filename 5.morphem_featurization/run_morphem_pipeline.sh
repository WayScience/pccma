#!/bin/bash
#
# Extract the MorphEm features (2.extract_morphem_features.ipynb, one job per cell line and plate),
# then process them into single-cell and bulk profiles (3.process_morphem_profiles.ipynb, one job per plate).
#
# Usage, from any folder (set OVERWRITE=1 to run every job again):
#   ./5.morphem_featurization/run_morphem_pipeline.sh
#
# - Run 4.preprocess_features/run_pipeline.sh first.
# - You only need uv installed. `uv run` creates the environments from environments/ (no need to activate them).
# - You do not need to run 0.evaluate_crop_sizes.ipynb or 1.view_224_crops.ipynb (one-time check of the crop size).
# - Jobs that already have output are skipped, unless their inputs are newer.
# - Logs are saved in 5.morphem_featurization/logs/: one for the whole run and one for each job.

set -uo pipefail

# run from the root of the repository
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
module="5.morphem_featurization"
log_dir="${module}/logs"
mkdir -p "$log_dir"

# show everything in the terminal and save it to the log of the run
run_log="${log_dir}/run_morphem_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1

overwrite="${OVERWRITE:-}"
failed=()

# ignore any environment that is already active, since uv run uses the ones in environments/
unset VIRTUAL_ENV

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# true when the output exists and is newer than every input (then the job is skipped)
is_up_to_date() {
    local output="$1"
    shift
    [ -z "$overwrite" ] && [ -f "$output" ] || return 1
    for input in "$@"; do
        [ "$input" -nt "$output" ] && return 1
    done
    return 0
}

# run a command in a uv environment (morphem or preprocessing) and save its output to a job log
run_job() {
    local env_name="$1" job_log="$2"
    shift 2
    uv run --locked --project "environments/${env_name}" "$@" 2>&1 | tee "$job_log"
}

log "INFO: Converting the notebooks to scripts"
if ! uv run --locked --project environments/morphem jupyter nbconvert --to script --output-dir="${module}/nbconverted/" "${module}"/*.ipynb; then
    log "ERROR: jupyter nbconvert failed"
    exit 1
fi

# Step 1: extract the MorphEm features (one job per converted profile from 4.preprocess_features)
for converted_file in $(find 4.preprocess_features/data/converted_profiles -name "*_converted.parquet" | sort); do
    cell_line=$(basename "$(dirname "$converted_file")")
    plate=$(basename "$converted_file" | cut -d "_" -f 1)
    output="${module}/data/morphem_profiles/${cell_line}/${plate}_${cell_line}_morphem.parquet"

    if is_up_to_date "$output" "$converted_file"; then
        log "INFO: Skipping ${plate} ${cell_line} (already extracted)"
        continue
    fi

    log "INFO: Extracting ${plate} ${cell_line}"
    if ! run_job morphem "${log_dir}/${plate}_${cell_line}_morphem_extraction.log" \
        python "${module}/nbconverted/2.extract_morphem_features.py" --plate "$plate" --cell_line "$cell_line" --overwrite; then
        log "ERROR: Extraction failed for ${plate} ${cell_line}"
        failed+=("${plate} ${cell_line} (extraction)")
    fi
done

# Step 2: process the MorphEm profiles (one job per plate)
for plate in $(ls "${module}"/data/morphem_profiles/*/*_morphem.parquet | xargs -n 1 basename | cut -d "_" -f 1 | sort -u); do
    output="${module}/data/bulk_profiles/${plate}_morphem_bulk_feature_selected.parquet"
    sc_output="${module}/data/single_cell_profiles/${plate}_morphem_sc_feature_selected.parquet"

    # inputs: the MorphEm profile and the QC results of each cell line on the plate
    inputs=()
    for morphem_file in "${module}"/data/morphem_profiles/*/"${plate}"_*_morphem.parquet; do
        cell_line=$(basename "$(dirname "$morphem_file")")
        inputs+=("$morphem_file" "4.preprocess_features/qc_results/${cell_line}_qc_annotations.parquet")
    done

    if is_up_to_date "$output" "${inputs[@]}" && [ -f "$sc_output" ]; then
        log "INFO: Skipping ${plate} (already processed)"
        continue
    fi

    log "INFO: Processing ${plate}"
    if ! PLATE_ID="$plate" OVERWRITE=1 run_job preprocessing "${log_dir}/${plate}_morphem_processing.log" \
        python "${module}/nbconverted/3.process_morphem_profiles.py"; then
        log "ERROR: Processing failed for ${plate}"
        failed+=("${plate} (processing)")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    log "ERROR: ${#failed[@]} jobs failed (see their logs in ${log_dir}):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
log "INFO: MorphEm pipeline completed (log: ${run_log})"
