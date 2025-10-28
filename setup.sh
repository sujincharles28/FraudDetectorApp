#!/bin/bash
# This is the master setup script for the Fraud Detector App

# Stop immediately if any command fails
set -e

ENV_NAME="fraud-env"

echo "--- Starting Fraud Detector App Setup ---"

# === STEP 1: INSTALL SYSTEM DEPENDENCIES ===
echo "[STEP 1/3] Installing system dependencies (Tesseract OCR)..."
echo "This may ask for your password."
sudo apt-get update
sudo apt-get install -y tesseract-ocr
echo "Tesseract-OCR installed successfully."
echo ""

# === STEP 2: CREATE CONDA ENVIRONMENT ===
# We assume the user has miniconda/anaconda installed

# Check if the environment already exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "[STEP 2/3] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "[STEP 2/3] Creating Conda environment '$ENV_NAME' with Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y
    echo "Conda environment created successfully."
fi
echo ""

# === STEP 3: INSTALL PYTHON PACKAGES ===
echo "[STEP 3/3] Installing Python packages into '$ENV_NAME' from requirements.txt..."

# Use 'conda run' to execute pip install *inside* the target environment
# This is the most robust way to do this from a script.
# We use --no-cache-dir to avoid old, cached versions (like the textual 6.4.0 bug)
conda run -n $ENV_NAME pip install --no-cache-dir -r requirements.txt

echo "All Python packages installed successfully."
echo ""

# === SETUP COMPLETE ===
echo "--- SETUP COMPLETE ---"
echo ""
echo "To run the application, follow these two steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Run the TUI:              python tui2.py"
