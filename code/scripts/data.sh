#!/bin/bash

# Set your Kaggle API credentials
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."

# Download the dataset
kaggle competitions download -c open-problems-single-cell-perturbations
unzip -q open-problems-single-cell-perturbations.zip -d opxmoe_data

# Remove the zip file
rm open-problems-single-cell-perturbations.zip

echo "Dataset downloaded and unzipped successfully!"