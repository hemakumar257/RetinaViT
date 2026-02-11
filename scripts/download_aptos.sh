#!/bin/bash
# scripts/download_aptos.sh

# Target directory
TARGET_DIR="datasets/raw/aptos2019"
mkdir -p "$TARGET_DIR"

echo "Checking for Kaggle CLI..."
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI could not be found. Please install it with 'pip install kaggle'."
    exit 1
fi

echo "Downloading APTOS 2019 Blindness Detection dataset..."
# Note: Requires kaggle.json in ~/.kaggle/
kaggle competitions download -c aptos2019-blindness-detection -p "$TARGET_DIR"

echo "Unzipping dataset..."
unzip -o "$TARGET_DIR/aptos2019-blindness-detection.zip" -d "$TARGET_DIR"
rm "$TARGET_DIR/aptos2019-blindness-detection.zip"

echo "APTOS 2019 download and extraction complete."
