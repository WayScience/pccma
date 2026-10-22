#!/bin/bash
#
# Run the feature-processing pipeline, in order:
#   0.convert_cytotable.ipynb           -> one converted single-cell parquet per cell line and plate
#                                          (only when the bandicoot mount with the SQLite files is available)
#   1.sc_quality_control.ipynb          -> one QC annotations parquet per cell line (via papermill, which
#                                          keeps the executed notebook per cell line for review), using the
#                                          thresholds reviewed in qc_thresholds.yml
#   2.annotate.ipynb                    -> one annotated single-cell profile per plate (cells that
#                                          failed QC removed per cell line), shared input for both
#                                          branches below
#   3.bulk_processing.ipynb             -> aggregated/normalized/feature-selected bulk profiles
#   4.single_cell_normalize.ipynb       -> normalized single-cell profiles
#   5.single_cell_feature_select.ipynb  -> feature-selected single-cell profiles
#   6.sc_qc_report.ipynb                -> QC report per cell line and plate (qc_report/)
#
# Steps 2 to 5 run one plate per `python` invocation (PLATE_ID env var) rather than
# looping over all plates inside a single long-lived process, so each plate's memory
# is fully released back to the OS when its process exits.
#
# By default, every step skips a plate that already has output. Set OVERWRITE=1
# (e.g. `OVERWRITE=1 ./run_pipeline.sh`) to reprocess and overwrite every plate's
# output instead.

set -uo pipefail

overwrite="${OVERWRITE:-}"

# -----------------------------
# Total runtime tracking (printed on exit, success or failure, via trap)
# -----------------------------
pipeline_start_time=$(date +%s)

print_total_time() {
    local elapsed=$(( $(date +%s) - pipeline_start_time ))
    echo "======================================"
    printf 'Total pipeline time: %dh %dm %ds\n' \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
    echo "======================================"
}
trap print_total_time EXIT

# -----------------------------
# Initialize environment
# -----------------------------
# create or update the uv environment from the lock file, then activate it
env_dir="../environments/preprocessing"
if ! uv sync --locked --project "$env_dir"; then
    echo "uv sync FAILED for ${env_dir}; aborting."
    exit 1
fi
source "${env_dir}/.venv/bin/activate"

if ! python -c "import pandas, pyarrow, pycytominer, cytotable, cosmicqc" >/dev/null 2>&1; then
    echo "Active Python ($(command -v python)) is missing required packages (pandas/pyarrow/pycytominer/cytotable/cosmicqc); aborting."
    exit 1
fi

# convert notebooks to scripts
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb
if [ $? -ne 0 ]; then
    echo "jupyter nbconvert FAILED; aborting pipeline."
    exit 1
fi

failed_plates=()

# -----------------------------
# Step 0: convert the SQLite files to parquet with CytoTable (skips jobs that are already converted)
# -----------------------------
echo "======================================"
echo "convert_cytotable (all jobs)"
echo "======================================"

if [ -d "$HOME/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/sqlite_outputs" ]; then
    python nbconverted/0.convert_cytotable.py
    if [ $? -ne 0 ]; then
        # stop here so QC does not cache annotations built from only part of the plates
        echo "convert_cytotable FAILED; aborting pipeline."
        exit 1
    fi
else
    echo "⚠️ bandicoot mount not found, using the existing converted profiles in data/converted_profiles"
fi

# -----------------------------
# Step 1: single-cell QC per cell line via papermill (static image display, as the
# interactive widget can stall papermill)
# -----------------------------
echo "======================================"
echo "sc_quality_control (one cell line at a time)"
echo "======================================"

qc_notebook_dir="./executed_notebooks/1.sc_quality_control"
mkdir -p "$qc_notebook_dir"

mapfile -t cell_lines < <(find data/converted_profiles -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

for cell_line in "${cell_lines[@]}"; do
    qc_output="./qc_results/${cell_line}_qc_annotations.parquet"

    # rerun QC when a converted plate is newer than the QC output (e.g., plates added
    # by a later conversion), as the z-scores need to include every plate
    if [ -f "$qc_output" ] && [ -z "$overwrite" ]; then
        newer_plates=$(find "data/converted_profiles/${cell_line}" -name "*_converted.parquet" -newer "$qc_output")
        if [ -z "$newer_plates" ]; then
            echo "✅ ${cell_line} already QC'd (found ${qc_output})"
            continue
        fi
        echo "⚠️ ${cell_line} has converted plates newer than ${qc_output}, rerunning QC"
    fi

    echo ">>> Running QC for ${cell_line}"
    papermill 1.sc_quality_control.ipynb "${qc_notebook_dir}/${cell_line}.ipynb" \
        -p cell_line "$cell_line" \
        -p interactive_display False
    if [ $? -ne 0 ]; then
        echo "QC FAILED for cell line: ${cell_line}"
        failed_plates+=("sc_quality_control:${cell_line}")
    else
        echo "QC done for cell line: ${cell_line}"
    fi
done

# plate names from the converted profiles (<cell_line>/<plate>_<cell_line>_converted.parquet)
mapfile -t plate_ids < <(find data/converted_profiles -mindepth 2 -maxdepth 2 -name "*_converted.parquet" -printf "%f\n" | cut -d_ -f1 | sort -u)

echo "Number of plates found: ${#plate_ids[@]}"

# Run one step for every plate, one plate per process
#   $1 = step name, $2 = script, $3 = output file pattern (PLATE is replaced with the plate id)
run_step() {
    local step_name="$1" script="$2" output_pattern="$3"

    echo "======================================"
    echo "${step_name} (one plate at a time)"
    echo "======================================"

    for plate_id in "${plate_ids[@]}"; do
        local step_output="${output_pattern//PLATE/$plate_id}"
        local plate_overwrite="$overwrite"

        # redo a plate when the QC annotations of any cell line on it are newer than the
        # step output (e.g., QC was rerun after plates were added), so the plate does not
        # keep outdated QC labels
        if [ -f "$step_output" ] && [ -z "$overwrite" ]; then
            local newer_qc=""
            for converted_file in data/converted_profiles/*/"${plate_id}"_*_converted.parquet; do
                local plate_cell_line
                plate_cell_line=$(basename "$(dirname "$converted_file")")
                local qc_output="./qc_results/${plate_cell_line}_qc_annotations.parquet"
                if [ "$qc_output" -nt "$step_output" ]; then
                    newer_qc="$qc_output"
                    break
                fi
            done

            if [ -z "$newer_qc" ]; then
                echo "✅ ${plate_id} already done (found ${step_output})"
                continue
            fi
            echo "⚠️ ${newer_qc} is newer than ${step_output}, rerunning ${step_name} for ${plate_id}"
            plate_overwrite=1
        fi

        echo ">>> Running ${step_name} for ${plate_id}"
        PLATE_ID="$plate_id" OVERWRITE="$plate_overwrite" python "$script"
        local exit_code=$?

        if [ "$exit_code" -ne 0 ]; then
            echo "${step_name} FAILED for plate: ${plate_id} (exit code $exit_code)"
            failed_plates+=("${step_name}:${plate_id}")
        else
            echo "${step_name} done for plate: ${plate_id}"
        fi
    done
}

run_step "annotate" nbconverted/2.annotate.py "./data/annotated_profiles/PLATE_annotated.parquet"
run_step "bulk_processing" nbconverted/3.bulk_processing.py "./data/bulk_profiles/PLATE_bulk_feature_selected.parquet"

run_step "single_cell_normalize" nbconverted/4.single_cell_normalize.py "./data/single_cell_profiles/PLATE_sc_normalized.parquet"
run_step "single_cell_feature_select" nbconverted/5.single_cell_feature_select.py "./data/single_cell_profiles/PLATE_sc_feature_selected.parquet"

# -----------------------------
# QC report (all cell lines at once, only reads the QC annotations)
# -----------------------------
echo "======================================"
echo "sc_qc_report (all cell lines)"
echo "======================================"

python nbconverted/6.sc_qc_report.py
if [ $? -ne 0 ]; then
    echo "QC report FAILED"
    failed_plates+=("sc_qc_report:all")
else
    echo "QC report done (qc_report/)"
fi

deactivate

echo "======================================"
echo "Pipeline finished."
if [ ${#failed_plates[@]} -gt 0 ]; then
    echo "Failures:"
    printf '  %s\n' "${failed_plates[@]}"
    exit 1
fi

echo "✅ All steps completed successfully."
