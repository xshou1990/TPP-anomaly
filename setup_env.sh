#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

# Clean up any existing environment
echo "Cleaning up any existing environment..."
conda env remove --name tpp_env --yes 2>/dev/null || true
rm -rf /home/triple-jay/anaconda3/envs/tpp_env 2>/dev/null || true

# Create fresh environment using explicit create command
echo "Creating fresh tpp_env environment..."
conda create --name tpp_env --yes python=3.9
conda activate tpp_env

# Install packages from environment.yml if exists
if [ -f "environment.yml" ]; then
    echo "Installing packages from environment.yml..."
    conda env update --name tpp_env --file environment.yml --prune
else
    echo "No environment.yml found, installing base packages..."
    conda install --yes numpy pandas scikit-learn tqdm matplotlib seaborn
fi

# Install additional requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    # First try conda, then pip
    conda install --yes --file requirements.txt 2>/dev/null || \
    pip install -r requirements.txt
fi

# Verify installation
echo "Verifying environment..."
conda list --name tpp_env

echo -e "\nEnvironment setup complete!"
echo "To activate the environment, run: conda activate tpp_env"