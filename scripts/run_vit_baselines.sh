#!/bin/bash

# Example commands for Phase 8 ViT Baselines

# 1. Train Standard ViT Base on APTOS
python -m retinavit.training.train_vit --dataset aptos --model vit_base --training standard

# 2. Train Quality-Aware ViT Base on APTOS
python -m retinavit.training.train_vit --dataset aptos --model vit_quality_base --training standard

# 3. Run Cross-Dataset Evaluation (on Messidor-2 and IDRiD)
python -m retinavit.training.train_vit --dataset aptos --model vit_quality_base --eval_only
