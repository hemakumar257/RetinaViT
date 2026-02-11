#!/bin/bash

# Example commands for Phase 9 ProtoViT Few-shot Learning

# 1. Train ProtoViT in 5-way 5-shot mode on APTOS
python -m retinavit.training.train_protovit --dataset aptos --model protovit --epochs 50

# 2. Evaluate 1-shot and 2-shot performance
# Replace PATH_TO_CHECKPOINT with your best model
python -m retinavit.training.eval_protovit --dataset aptos --model protovit --checkpoint experiments/models/protovit_aptos_best.pth

# 3. Cross-dataset mixing (config based)
# In configs/model/protovit.yaml, set fewshot.datasets: ["aptos", "messidor2"]
python -m retinavit.training.train_protovit --dataset aptos --model protovit --epochs 20
