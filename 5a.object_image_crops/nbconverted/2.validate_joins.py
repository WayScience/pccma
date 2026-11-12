#!/usr/bin/env python
# coding: utf-8

# # Validate crop files, joins and sampled pixels
#
# Require exact post-QC coverage of the available CellProfiler single-cell profiles. Check MorphEm
# profiles and UMAP subsets when present; report absent optional inputs as skipped, never verified.
# Recompute sampled crops from TIFFs and compare against a fresh JPEG encode/decode in this environment.
# No profiles or embeddings are modified, recomputed or required for crop generation itself.
#

# ## Paths and crop integrity
#

# In[ ]:


import json
import os
import pathlib
import pandas as pd
from crop_warehouse import (
    FINAL_PROFILE_KEY, POSITION_KEY, PATH_COLUMNS, join_crop_paths, require_unique_keys,
    validate_crop_files, validate_profile_join, validate_sample_pixels,
)
from path_utils import get_root_dir

root_dir = get_root_dir()
module_dir = root_dir / '5a.object_image_crops'
warehouse_dir = pathlib.Path(os.environ.get('PCCMA_CROPS_DIR',
    str(module_dir / 'data/object_image_crops'))).expanduser().resolve()
report_path = warehouse_dir / 'validation.json'
report_path.unlink(missing_ok=True)
index_df = pd.read_parquet(warehouse_dir / 'crops_index.parquet')
require_unique_keys(index_df, FINAL_PROFILE_KEY, 'Crop index')
require_unique_keys(index_df, POSITION_KEY, 'Crop positions')
validate_crop_files(index_df, warehouse_dir)
report = {'cells': len(index_df), 'jpg_files': len(index_df) * len(PATH_COLUMNS)}

local = root_dir / '4.preprocess_features/data/single_cell_profiles'
mount = pathlib.Path('~/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/single_cell_profiles').expanduser()
profile_dir = pathlib.Path(os.environ.get('PCCMA_PROFILES_DIR',
    str(local if any(local.glob('BR*_sc_feature_selected.parquet')) else mount))).expanduser()


# ## CellProfiler profiles: both directions, one-to-one
#

# In[ ]:


profile_paths = sorted(profile_dir.glob('BR*_sc_feature_selected.parquet'))
if not profile_paths:
    raise FileNotFoundError(f'No CellProfiler single-cell profiles in {profile_dir}')
profile_plates = {path.name.split('_')[0] for path in profile_paths}
if not set(index_df['Metadata_Plate']).issubset(profile_plates):
    raise ValueError('Crop plates without CellProfiler profiles')
for path in profile_paths:
    plate = path.name.split('_')[0]
    profiles = pd.read_parquet(path, columns=FINAL_PROFILE_KEY)
    validate_profile_join(index_df.loc[index_df['Metadata_Plate'] == plate], profiles, f'CellProfiler {plate}')
    print(f'CellProfiler {plate}: {len(profiles)} profile rows matched one-to-one')
report['cellprofiler'] = 'verified'


# ## Optional MorphEm and UMAP joins
#

# In[ ]:


morphem_dir = root_dir / '5.morphem_featurization/data/morphem_single_cell_profiles'
morphem_paths = sorted(morphem_dir.glob('BR*_morphem_sc_feature_selected.parquet'))
if morphem_paths:
    checked = []
    for path in morphem_paths:
        plate = path.name.split('_')[0]
        profiles = pd.read_parquet(path, columns=FINAL_PROFILE_KEY)
        validate_profile_join(index_df.loc[index_df['Metadata_Plate'] == plate], profiles, f'MorphEm {plate}')
        checked.append(plate)
    report['morphem'] = {'verified_plates': checked,
        'missing_plates': sorted(profile_plates - set(checked))}
    print(f'MorphEm: checked {len(checked)} available plates')
else:
    report['morphem'] = 'skipped: profiles unavailable'
    print('MorphEm joins skipped: profiles unavailable')

umap_dir = root_dir / '6.profile_analysis/results/3.single_cell_embedding'
report['umap'] = {}
for filename in ('single_cell_umap_coordinates_by_cancer_group.parquet', 'single_cell_umap_example_cells.parquet'):
    path = umap_dir / filename
    if not path.is_file():
        report['umap'][filename] = 'skipped: table unavailable'
        print(f'UMAP join skipped: {filename} unavailable')
        continue
    table = pd.read_parquet(path)
    joined = join_crop_paths(table, index_df)
    report['umap'][filename] = len(joined)
    print(f'UMAP {filename}: all {len(joined)} cells matched one-to-one')


# ## Recompute sampled pixels and save the validation report
#

# In[ ]:


images_dir = pathlib.Path(os.environ.get('PCCMA_IMAGES_DIR',
    '~/mnt/bandicoot/PCCMA_data/Atlas_v2_screen')).expanduser()
report['pixel_samples'] = validate_sample_pixels(index_df, warehouse_dir,
    root_dir / '2.illumination_correction/illum_directory', images_dir)
temporary = report_path.with_suffix('.tmp.json')
temporary.write_text(json.dumps(report, indent=2) + '\n')
temporary.replace(report_path)
print(f"Checked all six images for {report['pixel_samples']} sampled cells against recomputed JPEG pixels")
print(f'Saved {report_path}')
