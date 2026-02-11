import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
from typing import Dict, Any

from retinavit.utils.config import load_config
from retinavit.data.dataloaders import build_retina_dataloader
from retinavit.models.vit import build_vit_base, build_vit_small, build_vit_large
from retinavit.models.vit_utils import load_pretrained_weights
from retinavit.training.metrics import compute_metrics, get_stratified_metrics
from retinavit.training.losses import HybridLoss
from retinavit.data.curriculum import QualityCurriculumSampler

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Check if quality aware
        if hasattr(model, "retina_vit") or "quality_label" in batch:
            outputs = model(images, batch)
        else:
            outputs = model(images)
            
        if isinstance(criterion, HybridLoss):
            # HybridLoss expects dicts
            loss_dict = criterion({"logits": outputs}, {"labels": labels})
            loss = loss_dict["total_loss"]
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_quality = []
    total_loss = 0
    
    for batch in tqdm(loader, desc="Validating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        if hasattr(model, "retina_vit") or "quality_label" in batch:
            outputs = model(images, batch)
        else:
            outputs = model(images)
            
        if isinstance(criterion, HybridLoss):
            loss_dict = criterion({"logits": outputs}, {"labels": labels})
            loss = loss_dict["total_loss"]
        else:
            loss = criterion(outputs, labels)
            
        total_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_quality.extend(batch["quality_label"])
        
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    stratified = get_stratified_metrics(y_true, y_pred, all_quality, y_prob)
    
    metrics.update({f"{k}/{kv}": vv for k, v in stratified.items() for kv, vv in v.items()})
    metrics["loss"] = total_loss / len(loader)
    
    return metrics

@torch.no_grad()
def evaluate_cross_dataset(model, eval_datasets, cfg, device):
    """
    Evaluates a trained model on multiple external datasets.
    """
    results = {}
    for ds_name in eval_datasets:
        print(f"\nMoving to Cross-Dataset Evaluation: {ds_name}")
        # Build a temporary config for the target dataset
        eval_cfg = cfg.copy()
        eval_cfg["dataset"]["name"] = ds_name
        
        # Build loader
        eval_loader = build_retina_dataloader(eval_cfg, split="test")
        
        metrics = validate(model, eval_loader, nn.CrossEntropyLoss(), device)
        results[ds_name] = metrics
        
        # Log to wandb
        wandb.log({f"ext_{ds_name}/{k}": v for k, v in metrics.items()})
        print(f"Results for {ds_name}: Acc {metrics['accuracy']:.4f} | Kappa {metrics['kappa']:.4f}")
        
    return results

def main(dataset_name="aptos", model_name="vit_base", epochs=10, eval_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    cfg = load_config(dataset=dataset_name, model=model_name, training="standard")
    
    # Initialize W&B
    wandb.init(project="RetinaViT", config=cfg, name=f"{model_name}_{dataset_name}")
    
    # Build Model
    if "small" in model_name:
        model = build_vit_small(cfg)
    elif "large" in model_name:
        model = build_vit_large(cfg)
    else:
        model = build_vit_base(cfg)
        
    model = model.to(device)
    
    # Init Weights
    load_pretrained_weights(model, cfg)
    
    if eval_only:
        ckpt_path = f"experiments/models/{model_name}_{dataset_name}_best.pth"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint from {ckpt_path}")
        
        eval_datasets = cfg.get("evaluation", {}).get("cross_dataset", {}).get("eval_datasets", ["messidor2", "idrid"])
        evaluate_cross_dataset(model, eval_datasets, cfg, device)
        return

    # Build Dataloaders
    train_loader = build_retina_dataloader(cfg, split="train")
    val_loader = build_retina_dataloader(cfg, split="val")
    
    # Training Setup
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-4)))
    
    if cfg.get("training", {}).get("use_hybrid_loss", False):
        criterion = HybridLoss(cfg).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Curriculum Sampler
    if cfg.get("curriculum", {}).get("enable", False):
        sampler = QualityCurriculumSampler(train_loader.dataset, cfg)
        train_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=sampler)
    else:
        sampler = None
    
    # Training Loop
    best_kappa = -1
    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)
            
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Logging
        log_data = {"epoch": epoch, "train/loss": train_loss}
        log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
        wandb.log(log_data)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Acc {val_metrics['accuracy']:.4f} | Val Kappa {val_metrics['kappa']:.4f}")
        
        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            os.makedirs("experiments/models", exist_ok=True)
            torch.save(model.state_dict(), f"experiments/models/{model_name}_{dataset_name}_best.pth")
            
    # Final Cross-Dataset Eval
    if cfg.get("evaluation", {}).get("cross_dataset", {}).get("enable", False):
        eval_datasets = cfg["evaluation"]["cross_dataset"].get("eval_datasets", ["messidor2", "idrid"])
        evaluate_cross_dataset(model, eval_datasets, cfg, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aptos")
    parser.add_argument("--model", type=str, default="vit_base")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    
    # Use config builder with training argument
    main(dataset_name=args.dataset, model_name=args.model, epochs=args.epochs, eval_only=args.eval_only)
