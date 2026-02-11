#!/bin/bash

# Unified Evaluation for all model types
python -m retinavit.training.evaluation --task classification --dataset aptos
python -m retinavit.training.evaluation --task segmentation --dataset idrid
python -m retinavit.training.evaluation --task fewshot --dataset aptos
