# Object image crops

This lettered module produces an independent image folder within the atlas data warehouse.
It uses CellProfiler's object coordinates and the existing per-cell QC decisions; it does not
run segmentation, MorphEm inference, profile processing, or UMAP again. Existing parquet files
are never modified.

Each QC-passing cell has six 224 x 224 JPGs centered on its nucleus: DNA, ER, RNA, AGP and Mito
in grayscale, plus one RGB composite. The crop uses the same illumination correction, edge
padding and pre-JPEG 8-bit rescaling as `5.morphem_featurization`. Shared NumPy-only geometry
lives in `utils/nucleocentric_crops.py`; `morphem_utils` re-exports it for existing callers.

JPEG quality is 95; RGB uses no chroma subsampling. JPEG is lossy. These are display derivatives,
not lossless model inputs: JPEG can change pixel values, including the value 255 used to mark
camera saturation before compression. Do not use the JPGs to reproduce MorphEm embeddings or
quantitative intensity measurements without a separate validation. Composite colors use the
profile-analysis palette and screen blending, with a per-crop 1st–99.8th percentile stretch.

## Local output

Default root: `5a.object_image_crops/data/object_image_crops/` (gitignored).

    crops/<plate>/<cell_line>/<well>/f<site>/
        <plate>_<well>_f<site>_nuc<n>_cell<n>_DNA.jpg
        ..._ER.jpg, ..._RNA.jpg, ..._AGP.jpg, ..._Mito.jpg
        ..._composite.jpg
    manifests/<plate>_<cell_line>_manifest.parquet
    crops_index.parquet
    validation.json

Plate/cell-line/well/site folders avoid putting millions of JPGs into one directory. Paths in
the manifests are relative to this root, so moving the warehouse does not break them. The manifest
carries both the original `Image_Metadata_*` names and processed `Metadata_*` names, object
numbers, nucleus coordinates, cell line, time point, crop bounds/padding, JPEG settings, source
TIFF references and six paths. Viewing the images needs neither profiles nor UMAP data.

The job manifest is published last. Interrupted jobs are regenerated; missing crop files or newer
TIFFs, QC, converted profiles, illumination functions or crop code invalidate a cached job.
An overwritten job replaces its own image folder, including removal of crops for newly QC-failing
cells. Zero-passing-cell jobs still receive an empty manifest with a schema. The combined index
requires a manifest for every converted-profile job.

## Run

Inputs are the converted profiles and QC annotations in `4.preprocess_features`, illumination
functions in `2.illumination_correction`, raw TIFFs referenced by the converted profiles, and
platemaps in `0.download_data/metadata/platemaps`. Processed profiles are used only for validation.

From the repository root:

```bash
./5a.object_image_crops/run_crop_pipeline.sh
```

The script uses the locked `environments/preprocessing` environment through uv. Alternatively,
use an existing Python with NumPy, pandas, PyArrow, Pillow, tifffile, imagecodecs and nbconvert:

```bash
PCCMA_PYTHON=/path/to/python ./5a.object_image_crops/run_crop_pipeline.sh
```

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `OVERWRITE=1` | Regenerate every crop job. |
| `PCCMA_IMAGES_DIR` | Raw-image root for relocating recorded TIFF paths, preserving the exact plate-measurement folder. Defaults to `~/mnt/bandicoot/PCCMA_data/Atlas_v2_screen`. Existing recorded TIFF paths are used first. |
| `PCCMA_CROPS_DIR` | Local output root, e.g. a sufficiently large local work disk. |
| `PCCMA_PROFILES_DIR` | CellProfiler single-cell profiles for validation; defaults to local profiles, then the bandicoot profiles folder. |

Do a one-job pilot before a full run. After converting the notebooks, run:

```bash
PYTHONPATH=utils python 5a.object_image_crops/nbconverted/0.generate_object_crops.py \
    --plate BR00150698 --cell_line A673
```

| Notebook | Purpose |
| --- | --- |
| `0.generate_object_crops.ipynb` | One plate/cell-line job, reading each field once and writing six images per QC-passing cell. Includes a notebook-only preview. |
| `1.build_crop_index.ipynb` | Combine manifests, annotate wells, validate unique keys and all six paths, and publish the index atomically. Requires all converted-profile jobs; not a partial-run index builder. |
| `2.validate_joins.ipynb` | Exact two-way CellProfiler profile coverage; optional MorphEm and UMAP joins; all paths; fresh pixel recomputation for up to three cells per plate. Writes a validation report. |

Absent MorphEm/UMAP inputs are explicitly reported as skipped, not verified. Missing CellProfiler
validation profiles fail rather than claiming a vacuous success. UMAP joins use exact saved
nucleus coordinates, not rounded or nearest-neighbor matches, and ambiguous keys are rejected.
Pixel validation compares decoded images against a fresh JPEG roundtrip in the same environment;
codec-version differences may require validation with the original generation environment.

The tracked cell-count summary lists 1,204,712 post-QC cells. Six images each means 7,228,272 JPGs.
Actual space and runtime must be measured on a real-image pilot; no full-dataset size or runtime
has been established. Check free space and file-count limits before a full run.

## Join without modifying source tables

Both key naming schemes are in the index. Processed single-cell profiles use `Metadata_Plate`,
`Metadata_Well` and `Metadata_Site`, not the converted profiles' `Image_Metadata_*` names.
Select only join keys and paths to avoid suffixing overlapping metadata columns:

```python
import pandas as pd
from crop_warehouse import FINAL_PROFILE_KEY, PATH_COLUMNS, join_crop_paths

index = pd.read_parquet('5a.object_image_crops/data/object_image_crops/crops_index.parquet')
profiles = pd.read_parquet('path/to/BR00150698_sc_feature_selected.parquet')
joined = profiles.merge(index[FINAL_PROFILE_KEY + PATH_COLUMNS],
                        on=FINAL_PROFILE_KEY, how='left', validate='one_to_one')

umap = pd.read_parquet('6.profile_analysis/results/3.single_cell_embedding/'
                       'single_cell_umap_coordinates_by_cancer_group.parquet')
with_images = join_crop_paths(umap, index)  # retains all six image paths
```

Import helpers from the installed utils package or run with `PYTHONPATH=utils` from the repo root.
Bulk profiles and bulk UMAP points represent groups, not individual cells; they do not have a
one-to-one crop join. Filter the crop index by plate/well/cell line to inspect their constituent cells.

## Sync to bandicoot after validation

Proposed destination: `~/mnt/bandicoot/PCCMA_data/Atlas_v2_object_crops/`, beside `Atlas_v2_screen`
and `Atlas_v2_profiles`. Confirm that the real lab mount is available first; an empty local
`~/mnt/bandicoot` placeholder is not a mount. No automatic sync is performed by the pipeline.

After a complete run and review of `validation.json`, preview the copy (set the source to
`PCCMA_CROPS_DIR` if you changed it):

```bash
rsync -an --stats --exclude='.staging/' --exclude='*.tmp.*' \
    5a.object_image_crops/data/object_image_crops/ \
    ~/mnt/bandicoot/PCCMA_data/Atlas_v2_object_crops/
```

Remove `n` from `-an` to copy. Then verify destination contents with a checksum dry run:

```bash
rsync -rnc --itemize-changes --exclude='.staging/' --exclude='*.tmp.*' \
    5a.object_image_crops/data/object_image_crops/ \
    ~/mnt/bandicoot/PCCMA_data/Atlas_v2_object_crops/
```

The verification should report no changed/missing source files. These commands do not delete
extra destination files; the index defines the current published cell set. Do not sync while
crop generation is running. Consumers prepend the destination root to the same relative paths.

## Tests

```bash
python -m pytest utils/tests
```

Pipeline tests create explicitly synthetic TIFFs/parquet inputs under pytest's temporary folder,
then execute the real notebook scripts and shell runner, without a mount, torch or model downloads.
They do not substitute for a real atlas pilot or demonstrate biological embedding equivalence.
