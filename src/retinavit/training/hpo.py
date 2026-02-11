import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
import os
import pandas as pd

# from .train_vit import train_one_epoch 
from .train_multitask import MultitaskTrainer
from ..models.multitask import MultitaskRetinaViT
from ..models.vit_utils import get_model_complexity

def objective_multitask(trial, cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Sample Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    seg_weight = trial.suggest_float("seg_weight", 0.1, 5.0)
    
    # Update config
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["weight_decay"] = weight_decay
    cfg["training"]["losses"]["classification"]["gamma"] = focal_gamma
    cfg["training"]["losses"]["segmentation"]["weight"] = seg_weight
    
    # 2. Build Model & Trainer
    model = MultitaskRetinaViT(cfg).to(device)
    trainer = MultitaskTrainer(model, cfg, device)
    
    # 3. Short Training Run (e.g., 2 epochs for HPO)
    best_val_acc = 0
    for epoch in range(2):
        # trainer.train_epoch(...) 
        # For demo, we simulate a result based on params
        val_acc = 0.5 + 0.3 * np.random.random() 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    # 4. Multi-objective: Accuracy and Complexity
    complexity = get_model_complexity(model)
    
    return best_val_acc, complexity

def run_hpo(cfg: Dict[str, Any]):
    study_name = cfg.get("hpo", {}).get("study_name", "retinavit_hpo")
    n_trials = cfg.get("hpo", {}).get("n_trials", 10)
    
    # Multi-objective study
    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler()
    )
    
    study.optimize(lambda trial: objective_multitask(trial, cfg), n_trials=n_trials)
    
    print("Number of finished trials: ", len(study.trials))
    
    # Save results
    os.makedirs("experiments/hpo", exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(f"experiments/hpo/{study_name}.csv", index=False)
    print(f"HPO results saved to experiments/hpo/{study_name}.csv")
    
    return study

if __name__ == "__main__":
    # Example usage
    config = {
        "training": {"learning_rate": 1e-4, "losses": {"classification": {}, "segmentation": {}}},
        "hpo": {"n_trials": 3, "study_name": "test_study"},
        "model": {"name": "vit_tiny"}
    }
    # run_hpo(config)
