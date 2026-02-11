import torch
import pytest
from retinavit.training.losses import FocalLoss, DiceBoundaryLoss, ProtoContrastiveLoss, HybridLoss

def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0)
    inputs = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    loss = loss_fn(inputs, targets)
    assert loss >= 0
    assert not torch.isnan(loss)

def test_dice_boundary_loss():
    loss_fn = DiceBoundaryLoss()
    inputs = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss = loss_fn(inputs, targets)
    assert loss >= 0

def test_proto_contrastive_loss():
    loss_fn = ProtoContrastiveLoss()
    q_feats = torch.randn(10, 64)
    prototypes = torch.randn(5, 64)
    labels = torch.randint(0, 5, (10,))
    loss = loss_fn(q_feats, prototypes, labels)
    assert loss >= 0

def test_hybrid_loss():
    cfg = {
        "training": {
            "losses": {
                "classification": {"type": "focal", "weight": 1.0},
                "segmentation": {"weight": 0.5}
            }
        }
    }
    loss_fn = HybridLoss(cfg)
    outputs = {
        "logits": torch.randn(4, 5),
        "segmentation": torch.randn(4, 1, 64, 64)
    }
    targets = {
        "labels": torch.randint(0, 5, (4,)),
        "masks": torch.zeros(4, 1, 64, 64)
    }
    losses = loss_fn(outputs, targets)
    assert "total_loss" in losses
    assert losses["total_loss"] > 0

if __name__ == "__main__":
    test_focal_loss()
    test_dice_boundary_loss()
    test_proto_contrastive_loss()
    test_hybrid_loss()
    print("Loss tests passed!")
