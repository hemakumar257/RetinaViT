import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import wandb
from typing import Dict, Any

from retinavit.utils.config import load_config
from retinavit.data.dataloaders import build_fewshot_dataloader
from retinavit.models.protovit import ProtoViT
from retinavit.models.vit_utils import load_pretrained_weights
from retinavit.training.losses import HybridLoss

def train_one_episode(model, episode, optimizer, criterion, device):
    model.train()
    
    # Episode structure from episodic_collate_fn:
    # images: [Ns + Nq, C, H, W]
    # labels: [Ns + Nq]
    # support_len: int
    
    images = episode["images"].to(device)
    labels = episode["labels"].to(device)
    n_support = episode["support_len"]
    
    # Split support and query
    support_images = images[:n_support]
    support_labels = labels[:n_support]
    query_images = images[n_support:]
    query_labels = labels[n_support:]
    
    # Quality info (lists of strings/floats in the batch dict)
    support_quality = {
        "quality_label": episode["quality_labels"][:n_support],
        "quality_score": episode.get("quality_scores", [0.0]*len(episode["images"]))[:n_support]
    }
    query_quality = {
        "quality_label": episode["quality_labels"][n_support:],
        "quality_score": episode.get("quality_scores", [0.0]*len(episode["images"]))[n_support:]
    }
    
    optimizer.zero_grad()
    
    outputs = model(
        support_images, support_labels,
        query_images, query_labels,
        support_quality=support_quality,
        query_quality=query_quality
    )
    
    logits = outputs["logits"] # [Nq, Nw]
    
    # Map absolute labels to relative range [0, Nw-1]
    if isinstance(criterion, HybridLoss):
        loss_dict = criterion(
            {"query_features": outputs["query_features"], "prototypes": outputs["prototypes"]},
            {"fs_labels": query_labels}
        )
        loss = loss_dict["total_loss"]
    else:
        loss = criterion(logits, query_labels)
        
    loss.backward()
    optimizer.step()
    
    # Accuracy
    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels).float().mean()
    
    return loss.item(), acc.item()

def main(dataset_name="aptos", model_name="protovit", epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    cfg = load_config(dataset=dataset_name, model=model_name, training="standard")
    fs_cfg = cfg.get("fewshot", {})
    
    # Initialize W&B
    wandb.init(project="RetinaViT-ProtoViT", config=cfg, name=f"ProtoViT_{dataset_name}")
    
    # Build Episodic DataLoader
    train_loader = build_fewshot_dataloader(cfg, split="train")
    val_loader = build_fewshot_dataloader(cfg, split="val")
    
    # Build ProtoViT Model
    model = ProtoViT(cfg).to(device)
    
    # Init Weights
    load_pretrained_weights(model, cfg)
    
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-4)))
    
    if cfg.get("training", {}).get("use_hybrid_loss", False):
        criterion = HybridLoss(cfg).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Meta-Train]")
        for episode in pbar:
            loss, acc = train_one_episode(model, episode, optimizer, criterion, device)
            train_losses.append(loss)
            train_accs.append(acc)
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})
            
        epoch_loss = np.mean(train_losses)
        epoch_acc = np.mean(train_accs)
        
        # Meta-Validation
        model.eval()
        val_accs = []
        with torch.no_grad():
            for episode in tqdm(val_loader, desc="Meta-Val", leave=False):
                images = episode["images"].to(device)
                labels = episode["labels"].to(device)
                n_support = episode["support_len"]
                
                outputs = model(
                    images[:n_support], labels[:n_support],
                    images[n_support:], labels[n_support:],
                    support_quality={"quality_label": episode["quality_labels"][:n_support]},
                    query_quality={"quality_label": episode["quality_labels"][n_support:]}
                )
                preds = torch.argmax(outputs["logits"], dim=1)
                acc = (preds == labels[n_support:]).float().mean()
                val_accs.append(acc.item())
        
        val_acc = np.mean(val_accs)
        
        # Logging
        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "train/acc": epoch_acc,
            "val/acc": val_acc
        })
        
        print(f"Epoch {epoch}: Train Loss {epoch_loss:.4f} | Train Acc {epoch_acc:.4f} | Val Acc {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("experiments/models", exist_ok=True)
            torch.save(model.state_dict(), f"experiments/models/protovit_{dataset_name}_best.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aptos")
    parser.add_argument("--model", type=str, default="protovit")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    main(dataset_name=args.dataset, model_name=args.model, epochs=args.epochs)
