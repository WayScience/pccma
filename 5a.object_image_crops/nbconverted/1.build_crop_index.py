#!/usr/bin/env python
# coding: utf-8

# # Build the independent crop index
#
# Concatenate job manifests, verify that every converted-profile job has a manifest, annotate
# wells from platemaps, and check every path before atomically publishing the index.
#

# ## Paths and inputs
#

# In[ ]:


import os
import pathlib
import pandas as pd
from crop_warehouse import (FINAL_PROFILE_KEY, POSITION_KEY, PATH_COLUMNS, require_unique_keys,
    validate_crop_files, requested_plate, guard_plate_scope)
from path_utils import get_root_dir

root_dir = get_root_dir()
module_dir = root_dir / '5a.object_image_crops'
warehouse_dir = pathlib.Path(os.environ.get('PCCMA_CROPS_DIR',
    str(module_dir / 'data/object_image_crops'))).expanduser().resolve()
index_path = warehouse_dir / 'crops_index.parquet'
manifest_dir = warehouse_dir / 'manifests'
converted_dir = root_dir / '4.preprocess_features/data/converted_profiles'
selected_plate = requested_plate()
guard_plate_scope(warehouse_dir, selected_plate)
pattern = f'*/{selected_plate}_*_converted.parquet' if selected_plate else '*/*_converted.parquet'
converted = sorted(converted_dir.glob(pattern))
if not converted:
    raise FileNotFoundError(f'No converted profiles in {converted_dir}')
expected = {path.name.replace('_converted.parquet', '_manifest.parquet') for path in converted}
pattern = f'{selected_plate}_*_manifest.parquet' if selected_plate else '*_manifest.parquet'
manifest_paths = sorted(manifest_dir.glob(pattern))
if expected != {path.name for path in manifest_paths}:
    raise ValueError('Missing or stale job manifests; generate every converted-profile job first')


# ## Build, annotate and validate
#

# In[ ]:


index_df = pd.concat([pd.read_parquet(path) for path in manifest_paths], ignore_index=True)
# Drop placeholder annotations before adding authoritative platemap values, not afterwards.
index_df = index_df.drop(columns=['Metadata_cell_line', 'Metadata_time_point'])
platemaps = sorted((root_dir / '0.download_data/metadata/platemaps').glob('BR*_platemap.csv'))
if not platemaps:
    raise FileNotFoundError('No platemaps')
wells = pd.concat([pd.read_csv(path)[['plate_barcode', 'well_position', 'cell_line', 'time_point']]
                   for path in platemaps], ignore_index=True).rename(columns={
    'plate_barcode': 'Metadata_Plate', 'well_position': 'Metadata_Well',
    'cell_line': 'Metadata_cell_line', 'time_point': 'Metadata_time_point'})
require_unique_keys(wells, ['Metadata_Plate', 'Metadata_Well'], 'Platemaps')
index_df = index_df.merge(wells, on=['Metadata_Plate', 'Metadata_Well'], how='left', validate='many_to_one')
if index_df[['Metadata_cell_line', 'Metadata_time_point']].isna().any().any():
    raise ValueError('Crop cells missing platemap annotations')
require_unique_keys(index_df, FINAL_PROFILE_KEY, 'Crop index')
require_unique_keys(index_df, POSITION_KEY, 'Crop positions')
validate_crop_files(index_df, warehouse_dir)
temporary = index_path.with_suffix('.tmp.parquet')
index_df.to_parquet(temporary, index=False)
temporary.replace(index_path)
(warehouse_dir / 'validation.json').unlink(missing_ok=True)
print(f'Saved {index_path}: {len(index_df)} cells, {len(index_df) * len(PATH_COLUMNS)} JPGs, {len(manifest_paths)} jobs')
