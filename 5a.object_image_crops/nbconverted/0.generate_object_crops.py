#!/usr/bin/env python
# coding: utf-8

# # Generate post-QC nucleocentric JPG crops
#
# Write five grayscale channels and one RGB composite for each QC-passing cell. Crop geometry,
# illumination correction and pre-JPEG rescaling match the MorphEm extraction. JPEG is lossy;
# these are display derivatives, not replacements for the original TIFFs or model inputs.
# One job processes a cell line on a plate; source images are read once per field of view.
#

# ## Imports and job settings
#

# In[ ]:


import argparse
import os
import pathlib
import shutil
import time

import numpy as np
import pandas as pd
from crop_warehouse import (
    CHANNELS, PATH_COLUMNS, PROFILE_KEY_COLUMNS, UMAP_KEY_COLUMNS,
    crop_id, make_cell_crops, manifest_row, read_corrected_field,
    require_unique_keys, resolve_image_path, save_cell_crops, select_passing_cells,
    validate_crop_files, requested_plate, guard_plate_scope,
)
from path_utils import get_root_dir

try:
    get_ipython()
    in_notebook = True
except NameError:
    in_notebook = False

root_dir = get_root_dir()
module_dir = root_dir / '5a.object_image_crops'
images_dir = pathlib.Path(os.environ.get('PCCMA_IMAGES_DIR',
    '~/mnt/bandicoot/PCCMA_data/Atlas_v2_screen')).expanduser().resolve()
warehouse_dir = pathlib.Path(os.environ.get('PCCMA_CROPS_DIR',
    str(module_dir / 'data/object_image_crops'))).expanduser().resolve()

if not in_notebook:
    parser = argparse.ArgumentParser(description='Write post-QC nucleocentric JPG crops for one job')
    parser.add_argument('--plate', required=True)
    parser.add_argument('--cell_line', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    plate, cell_line, overwrite = args.plate, args.cell_line, args.overwrite
else:
    plate, cell_line, overwrite = 'BR00150698', 'A673', False

selected_plate = requested_plate()
if selected_plate and plate != selected_plate:
    raise ValueError(f'Job plate {plate} is outside requested scope {selected_plate}')
guard_plate_scope(warehouse_dir, selected_plate)

# Identifiers are path components, never paths supplied by a job.
for value in (plate, cell_line):
    if pathlib.Path(value).name != value or value in ('.', '..'):
        raise ValueError(f'Invalid job identifier: {value}')
converted_path = root_dir / '4.preprocess_features/data/converted_profiles' / cell_line / f'{plate}_{cell_line}_converted.parquet'
qc_path = root_dir / '4.preprocess_features/qc_results' / f'{cell_line}_qc_annotations.parquet'
illum_dir = root_dir / '2.illumination_correction/illum_directory' / plate
illum_paths = {ch: illum_dir / f'{plate}_Illum{ch}.npy' for ch in CHANNELS}
manifest_path = warehouse_dir / 'manifests' / f'{plate}_{cell_line}_manifest.parquet'


# ## Load metadata and apply QC
#

# In[ ]:


image_columns = [f'Image_{kind}_Orig{ch}' for ch in CHANNELS for kind in ('FileName', 'PathName')]
columns = sorted(set(PROFILE_KEY_COLUMNS + UMAP_KEY_COLUMNS + image_columns))
cells = pd.read_parquet(converted_path, columns=columns)
qc = pd.read_parquet(qc_path)
qc = qc.loc[qc['Image_Metadata_Plate'] == plate]
single_cell_df = select_passing_cells(cells, qc)
if not cells['Image_Metadata_Plate'].eq(plate).all():
    raise ValueError('Converted profile contains a different plate')
require_unique_keys(single_cell_df, PROFILE_KEY_COLUMNS, 'Post-QC cells')
require_unique_keys(single_cell_df, UMAP_KEY_COLUMNS, 'Post-QC positions')
if not np.isfinite(single_cell_df[UMAP_KEY_COLUMNS[3:]].to_numpy()).all():
    raise ValueError('Non-finite nucleus centers')
single_cell_df = single_cell_df.assign(Metadata_cell_line=cell_line)
if single_cell_df.apply(crop_id, axis=1).duplicated().any():
    raise ValueError('Duplicate crop filenames in a job')
for well in single_cell_df['Image_Metadata_Well']:
    if pathlib.Path(well).name != well or well in ('.', '..'):
        raise ValueError(f'Invalid well path component: {well}')

# A field must refer to the same five TIFFs for every cell in it.
for _, field in single_cell_df.groupby('Metadata_ImageNumber'):
    if (field[image_columns].nunique(dropna=False) != 1).any():
        raise ValueError('Inconsistent source images within a field')

inputs = [converted_path, qc_path, *illum_paths.values(),
          module_dir / '0.generate_object_crops.ipynb',
          pathlib.Path(__import__('crop_warehouse').__file__),
          pathlib.Path(__import__('nucleocentric_crops').__file__)]
for record in single_cell_df.drop_duplicates('Metadata_ImageNumber').to_dict('records'):
    inputs.extend(resolve_image_path(record, ch, images_dir) for ch in CHANNELS)
write_crops = True
if manifest_path.is_file() and not overwrite:
    fresh = all(path.is_file() and path.stat().st_mtime_ns <= manifest_path.stat().st_mtime_ns for path in inputs)
    if fresh:
        existing = pd.read_parquet(manifest_path)
        try:
            validate_crop_files(existing, warehouse_dir)
            require_unique_keys(existing, PROFILE_KEY_COLUMNS, 'Existing manifest')
            same_keys = existing[PROFILE_KEY_COLUMNS].merge(single_cell_df[PROFILE_KEY_COLUMNS],
                on=PROFILE_KEY_COLUMNS, how='outer', validate='one_to_one', indicator=True)
            write_crops = not same_keys['_merge'].eq('both').all()
        except ValueError:
            write_crops = True
print(f'{plate} {cell_line}: {len(single_cell_df)} of {len(cells)} cells passed QC')
if not write_crops:
    print(f'Skipping {plate} {cell_line}: manifest and all six crop files are up to date')


# ## Generate one field at a time and publish the job manifest last
#

# In[ ]:


if write_crops:
    start = time.monotonic()
    # Invalidate published summaries before changing files. A failed run is not a complete warehouse.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    (warehouse_dir / 'crops_index.parquet').unlink(missing_ok=True)
    (warehouse_dir / 'validation.json').unlink(missing_ok=True)
    functions = {ch: np.load(path) for ch, path in illum_paths.items()}
    job_dir = warehouse_dir / 'crops' / plate / cell_line
    stage = warehouse_dir / '.staging' / f'{plate}_{cell_line}'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    rows = []
    for _, field in single_cell_df.groupby('Metadata_ImageNumber', sort=True):
        corrected, saturated = read_corrected_field(field.iloc[0].to_dict(), functions, images_dir)
        for record in field.to_dict('records'):
            crops, padding = make_cell_crops(record, corrected, saturated)
            subdir = pathlib.Path(record['Image_Metadata_Well']) / f"f{int(record['Image_Metadata_Site']):02d}"
            written = save_cell_crops(crops, stage / subdir, crop_id(record))
            final_paths = {key: str(job_dir / pathlib.Path(path).relative_to(stage)) for key, path in written.items()}
            row = manifest_row(record, final_paths, warehouse_dir)
            row['Metadata_Nucleocentric_Pad_Fraction'] = padding
            rows.append(row)

    # Preserve the schema even for a job where all cells failed QC.
    template = {key: 0 for key in set(PROFILE_KEY_COLUMNS + UMAP_KEY_COLUMNS)}
    template.update(Image_Metadata_Plate=plate, Image_Metadata_Well='', Metadata_cell_line=cell_line)
    template.update({column: '' for column in image_columns})
    empty = manifest_row(template, {key: str(warehouse_dir / key) for key in PATH_COLUMNS}, warehouse_dir)
    empty['Metadata_Nucleocentric_Pad_Fraction'] = 0.0
    manifest_df = pd.DataFrame(rows, columns=list(empty)) if rows else pd.DataFrame(columns=list(empty))
    require_unique_keys(manifest_df, PROFILE_KEY_COLUMNS, 'Job manifest')
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    stage.replace(job_dir)
    validate_crop_files(manifest_df, warehouse_dir)
    temporary = manifest_path.with_suffix('.tmp.parquet')
    manifest_df.to_parquet(temporary, index=False)
    temporary.replace(manifest_path)
    print(f'Saved {len(rows)} cells / {len(rows) * len(PATH_COLUMNS)} JPGs in {time.monotonic() - start:.1f}s')


# ## Preview composites (notebook only)
#

# In[ ]:


if in_notebook:
    import matplotlib.pyplot as plt
    from PIL import Image

    preview = pd.read_parquet(manifest_path).head(4)
    if not preview.empty:
        fig, axes = plt.subplots(1, len(preview), squeeze=False, figsize=(4 * len(preview), 4))
        for ax, row in zip(axes.flat, preview.to_dict('records')):
            with Image.open(warehouse_dir / row['composite']) as image:
                ax.imshow(np.array(image))
            ax.set_title(f"{row['Metadata_Well']} / cell {row['Metadata_Cells_Number_Object_Number']}")
            ax.axis('off')
        plt.show()
