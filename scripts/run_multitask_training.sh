#!/bin/bash

# Multi-task training command (example)
# Jointly train classification (APTOS) and segmentation (IDRiD)
python -m retinavit.training.train_multitask --config configs/training/multitask.yaml --model_config configs/model/multitask_vit.yaml
