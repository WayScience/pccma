"""Shared functions of the PCCMA notebooks.

The modules are small and each has tests in `utils/tests`:

- `cell_images`: cut the image of a single cell out of the raw images of a plate.
- `cell_lines`: annotate cell lines with their cancer group and tissue site.
- `profiles`: helpers for the columns of the profile files.

Install the folder in editable mode (the uv environments do this) or add it to `PYTHONPATH`, and import,
for example, `from pccma_utils.cell_lines import load_cell_line_groups`.
"""
