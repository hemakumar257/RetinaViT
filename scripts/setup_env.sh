#!/bin/bash
# Setup environment for RetinaViT

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

echo "Environment setup complete."
