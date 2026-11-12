#!/bin/bash
# Generate the local crop warehouse; validation never modifies profiles or UMAP results.
# Optional: PCCMA_PYTHON=/path/to/python to use an existing environment instead of uv.
# OVERWRITE=1 forces regeneration. See README.md for input/output path overrides.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
module="5a.object_image_crops"
mkdir -p "${module}/logs"
run_log="${module}/logs/run_crop_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$run_log") 2>&1
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"

run_python() {
    if [ -n "${PCCMA_PYTHON:-}" ]; then
        "$PCCMA_PYTHON" "$@"
    else
        uv run --locked --project environments/preprocessing python "$@"
    fi
}

shopt -s nullglob
converted_files=(4.preprocess_features/data/converted_profiles/*/*_converted.parquet)
if [ "${#converted_files[@]}" -eq 0 ]; then
    printf 'ERROR: No converted profiles found; run conversion and single-cell QC first.\n' >&2
    exit 1
fi
run_python -m nbconvert --to python --output-dir="${module}/nbconverted" "${module}"/*.ipynb

extra=()
case "${OVERWRITE:-}" in
    1|true|yes) extra+=(--overwrite) ;;
esac
failed=0
for converted in "${converted_files[@]}"; do
    name="${converted##*/}"
    plate="${name%%_*}"
    directory="${converted%/*}"
    cell_line="${directory##*/}"
    printf 'Generating %s %s\n' "$plate" "$cell_line"
    # The generator checks freshness of profiles, QC, TIFFs, illumination and code, and all six paths.
    if ! run_python "${module}/nbconverted/0.generate_object_crops.py" \
        --plate "$plate" --cell_line "$cell_line" "${extra[@]+"${extra[@]}"}"; then
        printf 'ERROR: Crop job failed: %s %s\n' "$plate" "$cell_line" >&2
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    printf 'ERROR: Not publishing an index from failed crop jobs.\n' >&2
    exit 1
fi
run_python "${module}/nbconverted/1.build_crop_index.py"
run_python "${module}/nbconverted/2.validate_joins.py"
printf 'Local crop warehouse validated. No files have been synced to bandicoot.\n'
