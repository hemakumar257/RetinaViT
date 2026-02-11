import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to handle class imbalance.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceBoundaryLoss(nn.Module):
    """
    Combined Dice and Boundary loss for lesion segmentation.
    Boundary loss is approximated here using a simplified gradient-based term.
    """
    def __init__(self, dice_weight: float = 0.5, boundary_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def boundary_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Simple boundary approximation using Laplacian-style edge detection difference
        inputs_sigmoid = torch.sigmoid(inputs)
        
        def get_edges(x):
            laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(x.device)
            return F.conv2d(x, laplacian, padding=1).abs()
            
        input_edges = get_edges(inputs_sigmoid)
        target_edges = get_edges(targets)
        return F.mse_loss(input_edges, target_edges)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(inputs, targets) + \
               self.boundary_weight * self.boundary_loss(inputs, targets)

class ProtoContrastiveLoss(nn.Module):
    """
    Prototypical loss with InfoNCE contrastive regularization.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_feats: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. Prototypical Loss (Standard CE over distances)
        # query_feats: [Nq, D], prototypes: [Nw, D]
        dists = torch.cdist(query_feats.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        proto_logits = -dists / self.temperature
        loss_proto = F.cross_entropy(proto_logits, labels)
        
        # 2. Contrastive Regularization (InfoNCE)
        # Each query should be close to its prototype and far from others
        query_norm = F.normalize(query_feats, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        
        sim_matrix = torch.mm(query_norm, proto_norm.t()) / self.temperature
        loss_contrastive = F.cross_entropy(sim_matrix, labels)
        
        return loss_proto + 0.1 * loss_contrastive

class HybridLoss(nn.Module):
    """
    Unified hybrid loss wrapper for multitask learning.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        loss_cfg = cfg.get("training", {}).get("losses", {})
        
        self.weights = {
            "classification": loss_cfg.get("classification", {}).get("weight", 1.0),
            "segmentation": loss_cfg.get("segmentation", {}).get("weight", 1.0),
            "fewshot": loss_cfg.get("fewshot", {}).get("weight", 1.0),
            "quality": loss_cfg.get("quality", {}).get("weight", 0.5)
        }
        
        # Classification
        clf_type = loss_cfg.get("classification", {}).get("type", "ce")
        if clf_type == "focal":
            f_cfg = loss_cfg.get("classification", {})
            self.clf_loss = FocalLoss(alpha=f_cfg.get("alpha", 1.0), gamma=f_cfg.get("gamma", 2.0))
        else:
            self.clf_loss = nn.CrossEntropyLoss()
            
        # Segmentation
        self.seg_loss = DiceBoundaryLoss()
        
        # Few-shot
        self.fs_loss = ProtoContrastiveLoss()
        
        # Quality (MSE on features/reconstruction)
        self.quality_loss = nn.MSELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        losses_dict = {}
        
        # 1. Classification
        if "logits" in outputs and "labels" in targets:
            l_clf = self.clf_loss(outputs["logits"], targets["labels"])
            total_loss += self.weights["classification"] * l_clf
            losses_dict["loss_clf"] = l_clf
            
        # 2. Segmentation
        if "segmentation" in outputs and "masks" in targets:
            l_seg = self.seg_loss(outputs["segmentation"], targets["masks"])
            total_loss += self.weights["segmentation"] * l_seg
            losses_dict["loss_seg"] = l_seg
            
        # 3. Few-shot
        if "query_features" in outputs and "prototypes" in outputs and "fs_labels" in targets:
            l_fs = self.fs_loss(outputs["query_features"], outputs["prototypes"], targets["fs_labels"])
            total_loss += self.weights["fewshot"] * l_fs
            losses_dict["loss_fs"] = l_fs
            
        # 4. Quality Reconstruction (Self-supervised/Auxiliary)
        if "reconstruction" in outputs and "images" in targets:
            l_qual = self.quality_loss(outputs["reconstruction"], targets["images"])
            # Apply more strongly on poor images if quality_labels provided
            if "quality_labels" in targets:
                # This logic would need a weighted mean based on labels
                pass
            total_loss += self.weights["quality"] * l_qual
            losses_dict["loss_quality"] = l_qual
            
        losses_dict["total_loss"] = total_loss
        return losses_dict
