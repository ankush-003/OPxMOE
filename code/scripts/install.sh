#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Python packages
echo "Installing Python packages..."
pip install kaggle kagglehub rdkit captum numpy pandas scikit-learn gensim torch matplotlib seaborn tqdm git+https://github.com/samoturk/mol2vec;

# Install Jupyter if not already installed
echo "Checking for Jupyter..."
if ! command -v jupyter &> /dev/null; then
    echo "Jupyter not found, installing..."
    pip install jupyter
else
    echo "Jupyter is already installed."
fi

echo "All installations are complete!"