#!/bin/bash

# Robustness Testing Suite
python -m retinavit.training.robustness --mode quality_degradation
python -m retinavit.training.robustness --mode cross_dataset --eval_datasets "messidor2,idrid"
python -m retinavit.training.robustness --mode adversarial
