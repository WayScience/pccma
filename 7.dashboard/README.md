# Single-cell crop dashboard

A static, single-page report consuming the existing analysis tables and the
post-QC image-crop warehouse. It does not recompute UMAPs, regenerate crops,
modify profiles, or need a server-side application. The browser uses native
Canvas with no JavaScript packages, CDN downloads, or analytics.

## Data and interpretation

The current plate is `BR00150695` (24 hours), from crop run
`BR00150695-20261113T155150Z-g790ecb4`.

- The crop index contains 80,493 post-QC cells. QC was inherited from the
  published profiles, not recalculated.
- The existing single-cell analysis contributes 6,500 sampled cells from
  13 cell lines in seven cancer groups for this plate. The other 75,252
  analysis rows are outside this report's plate scope, not missing crops.
- Each cancer group has its **own** original UMAP, fitted using that group's
  sampled cells across the original plates and time points. We retain its
  coordinates and show only the selected plate. This is not a new UMAP
  fitted to this plate. Coordinates/distances across different groups
  are not comparable, and UMAP proximity is exploratory, not a statistical test.
- The original sampling draws up to 500 cells per cell line per plate;
  this viewer does not show all 80,493 warehouse cells.
- Crops are the original 224 × 224, nucleus-centered JPEG display assets:
  composite plus DNA, ER, RNA, AGP, and Mito. JPEGs and per-crop display
  scaling are not quantitative intensity measurements or lossless model inputs.

The UMAP table does not retain CellProfiler object numbers. The exporter
matches its exact, unrounded plate/well/site/nucleus-X/nucleus-Y key to the
crop index, requiring unique keys on both sides and complete coverage. It
then retains the cell and nucleus object numbers from that index. It never
uses nearest-neighbor matching, rounded positions, row order, or the notebook's
older embedded PNG examples.

## Use the page

Choose a cancer group, hover a point to inspect its cell, and click/tap to
pin it. The inspector gives the well, field, cell/nucleus identifiers and
channel crops. Change point colors or click a legend entry to show/hide its
points. Keyboard cell navigation provides an alternative to precise hovering.
Pan/zoom and Reset view help inspect crowded regions.

## Build a complete bundle

Run from the repository root with Python 3.11 or 3.12 and `uv`:

```bash
uv sync --project 7.dashboard --locked

RUN=BR00150695-20261113T155150Z-g790ecb4
uv run --project 7.dashboard --locked python 7.dashboard/build_report.py \
  --crop-index "5a.object_image_crops/data/pilot_runs/$RUN/crops_index.parquet" \
  --warehouse "/path/to/extracted/warehouse" \
  --plate BR00150695 \
  --run-id "$RUN" \
  --output "7.dashboard/dist/$RUN"
```

`--warehouse` is the directory **containing `crops/`**, not `crops/` itself.
The existing tree on Willow is:

```text
/home/davebunten/work/pccma-crops-runs/BR00150695-20261113T155150Z-g790ecb4/warehouse
```

The archival copy is under the corresponding run in
`PCCMA_data/Atlas_v2_object_crops/runs/` on bandicoot; extract its `warehouse/`
tree into a new location, not over existing results. See the warehouse module
for its publication and integrity records. Source images are not bundled in
this Git checkout.

For another coordinate table, supply `--umap /path/to/coordinates.parquet`.
Every output directory must be new; the exporter refuses to replace an
existing bundle. It stages and validates the complete output before publishing
that local directory. Only the sampled cells' six existing JPEGs are copied,
byte-for-byte, preserving index-relative paths; they are decoded to check
geometry/mode and hashed after copying. Missing, corrupt, unsafe, ambiguous,
or metadata-mismatched inputs fail the build. A 950 MB total bundle budget
provides headroom for static hosting; an oversized bundle fails instead of
silently dropping cells or channels.

```text
7.dashboard/
  build_report.py       Exact join, asset validation/copy, HTML export
  web/                  Editable page template, CSS and JavaScript
  tests/                Exporter, interaction helper, and browser tests
  pyproject.toml        Small independent build/test environment
  uv.lock               Locked dependencies
  dist/<run>/           Generated bundle (gitignored)
    index.html          One page with embedded data, JS and CSS
    .nojekyll           Serve generated assets unchanged
    manifest.json       Input/code hashes, scope, image hashes and counts
    assets/crops/...    Only the selected cells' original JPEGs
```

The manifest records file basenames and hashes, not private input-directory
paths. The page identifies the plate, run, scope and input hashes. All data
and image paths are relative to the page, so the whole bundle can move intact.

## Coordinate-only diagnostic preview

If the warehouse cannot be reached, an explicit diagnostic mode builds the
real joined coordinates without any image URLs. It displays a visible warning;
it is **not a complete crop dashboard** and does not substitute synthetic cells,
contact-sheet fragments or notebook example images.

```bash
RUN=BR00150695-20261113T155150Z-g790ecb4
uv run --project 7.dashboard --locked python 7.dashboard/build_report.py \
  --crop-index "5a.object_image_crops/data/pilot_runs/$RUN/crops_index.parquet" \
  --run-id "$RUN" \
  --allow-missing-images \
  --output "7.dashboard/previews/$RUN-coordinates"
```

This option does not excuse missing/corrupt files in a supplied warehouse tree.
A complete build must be rerun into a new output directory after image access
is restored.

## GitHub Pages and local viewing

Open the generated `index.html` directly, or serve **only the generated bundle**:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory "7.dashboard/dist/$RUN"
```

For GitHub Pages, use the generated bundle directory as the Pages artifact
root (for example, the `path` passed to `actions/upload-pages-artifact` in
an authorized deployment workflow). Publish **all** its files, including
`assets/`, not only `index.html` or the source `web/template.html`.
Root hosting and project-prefix hosting need no path rewrite. No deployment
workflow or Pages settings are changed automatically by this module.
Do not upload the full warehouse or commit the ignored generated bundle just
to preview the source changes. Publishing the report also publishes its
selected cell images; obtain the normal dataset/publication approval first.

## Tests

```bash
uv run --project 7.dashboard --locked python -m pytest 7.dashboard/tests -q
node --test 7.dashboard/tests/test_web.mjs
```

Browser tests use Playwright; install its Chromium once:

```bash
uv run --project 7.dashboard --locked playwright install chromium
```

Synthetic images in tests are labeled fixtures and verify export and UI
mechanics only. Verification on the actual plate's coordinates is separate
from real warehouse-image decoding and hover display. A coordinate-only
preview passing is not evidence that the real crops have been bundled.

## Sources

- [Single-cell embedding analysis](../6.profile_analysis/3.single_cell_embedding.ipynb)
- [Object crop warehouse and provenance](../5a.object_image_crops/README.md)
- [Cell Painting assay](https://doi.org/10.1038/nprot.2016.105)
- [UMAP method](https://doi.org/10.21105/joss.00861)
- [GitHub Pages publishing limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
