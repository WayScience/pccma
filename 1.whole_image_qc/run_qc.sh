#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the CellProfiler environment
conda activate pccma_atlas_cp_env

# convert all notebooks to script files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for creating LoadData CSVs and extracting image quality metrics with CellProfiler
python nbconverted/0.create_loaddata_csvs.py
python nbconverted/1.extract_image_quality.py

# deactivate the CellProfiler environment
conda deactivate

# activate the whole image QC environment (coSMicQC and CytoDataFrame are not in the CellProfiler environment)
conda activate pccma_atlas_whole_img_qc_env

# run Python script for evaluating the QC metrics and determining thresholds
python nbconverted/2.evaluate_qc.py

# deactivate the whole image QC environment
conda deactivate

echo "QC evaluation complete and LoadData CSVs saved to output directory."
