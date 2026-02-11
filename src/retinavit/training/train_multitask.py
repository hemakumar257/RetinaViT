import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader

from retinavit.models.vit_utils import load_pretrained_weights
from retinavit.training.losses import HybridLoss

class MultitaskTrainer:
    """
    Trainer for multitask learning with classification and segmentation.
    Supports gradient balancing via uncertainty weighting.
    """
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        
        self.multitask_cfg = cfg.get("training", {}).get("multitask", {})
        self.strategy = self.multitask_cfg.get("weighting", {}).get("strategy", "uncertainty")
        
        # Initialize uncertainty parameters if needed
        if self.strategy == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2, device=device))
            self.optimizer = optim.AdamW(
                list(model.parameters()) + [self.log_vars],
                lr=float(cfg["training"].get("learning_rate", 1e-4))
            )
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-4)))
            
        if cfg.get("training", {}).get("use_hybrid_loss", False):
            self.hybrid_criterion = HybridLoss(cfg).to(device)
            self.criterion_clf = None
            self.criterion_seg = None
        else:
            self.hybrid_criterion = None
            self.criterion_clf = nn.CrossEntropyLoss()
            self.criterion_seg = nn.BCEWithLogitsLoss() 
            
        load_pretrained_weights(model, cfg)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)
        masks = batch.get("masks") # May be None for clf-only datasets
        
        outputs = self.model(images)
        
        if self.hybrid_criterion:
            loss_dict = self.hybrid_criterion(outputs, {"labels": labels, "masks": masks, "images": images})
            loss_clf = loss_dict.get("loss_clf", torch.tensor(0.0, device=self.device))
            loss_seg = loss_dict.get("loss_seg", torch.tensor(0.0, device=self.device))
            total_loss = loss_dict["total_loss"]
        else:
            logits = outputs["logits"]
            seg_mask = outputs["segmentation"]
            
            # 1. Classification Loss
            loss_clf = self.criterion_clf(logits, labels)
            
            # 2. Segmentation Loss (only if mask exists)
            loss_seg = torch.tensor(0.0, device=self.device)
            if masks is not None:
                masks = masks.to(self.device)
                loss_seg = self.criterion_seg(seg_mask, masks)
                
            # 3. Balanced Total Loss
            if self.strategy == "uncertainty":
                precision1 = torch.exp(-self.log_vars[0])
                precision2 = torch.exp(-self.log_vars[1])
                total_loss = precision1 * loss_clf + self.log_vars[0] + \
                             precision2 * loss_seg + self.log_vars[1]
            else:
                w_clf = self.multitask_cfg.get("initial_weights", {}).get("classification", 1.0)
                w_seg = self.multitask_cfg.get("initial_weights", {}).get("segmentation", 1.0)
                total_loss = w_clf * loss_clf + w_seg * loss_seg
            
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "loss_clf": loss_clf.item(),
            "loss_seg": loss_seg.item(),
            "w_clf": torch.exp(-self.log_vars[0]).item() if self.strategy == "uncertainty" else 1.0,
            "w_seg": torch.exp(-self.log_vars[1]).item() if self.strategy == "uncertainty" else 1.0
        }

def main():
    # Placeholder for actual training script instantiation
    print("Multitask Trainer initialized.")

if __name__ == "__main__":
    main()
