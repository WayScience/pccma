#!/bin/bash

# initialize the correct shell for your machine to allow conda to work (see README for note on shell names)
conda init bash
# activate the CellProfiler environment
conda activate pccma_atlas_cp_env

# convert all notebooks to script files into the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# run Python scripts for generating IC functions
python nbconverted/cp_illum_correction.py

# deactivate the CellProfiler environment
conda deactivate

echo "Illumination correction functions generated and saved to output directory."
