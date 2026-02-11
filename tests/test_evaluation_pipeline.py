import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset
from retinavit.training.evaluation import evaluate_classification, evaluate_segmentation

class MockModel(nn.Module):
    def __init__(self, task="clf"):
        super().__init__()
        self.task = task
        if task == "clf":
            self.head = nn.Linear(3, 5) # 3 channels after mean(dim=[-1, -2])
        else:
            self.head = nn.Conv2d(1, 1, 3, padding=1)
            
    def forward(self, x):
        if self.task == "clf":
            return self.head(x.mean(dim=[-1, -2]))
        else:
            return {"segmentation": self.head(x)}

def test_evaluate_classification_smoke():
    model = MockModel(task="clf")
    # Mock data [B, C, H, W]
    x = torch.randn(10, 3, 224, 224)
    y = torch.randint(0, 5, (10,))
    ds = TensorDataset(x, y)
    
    # Custom loader to match expected batch dict
    def collate(batch):
        xs, ys = zip(*batch)
        return {"image": torch.stack(xs), "label": torch.stack(ys)}
        
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    
    metrics = evaluate_classification(model, loader, device=torch.device("cpu"), n_bootstrap=10)
    assert "accuracy" in metrics
    assert "kappa_ci" in metrics

def test_evaluate_segmentation_smoke():
    model = MockModel(task="seg")
    x = torch.randn(4, 1, 64, 64)
    y = torch.zeros(4, 1, 64, 64)
    y[:, :, 10:20, 10:20] = 1.0 # synthetic mask
    
    ds = TensorDataset(x, y)
    def collate(batch):
        xs, ys = zip(*batch)
        return {"image": torch.stack(xs), "masks": torch.stack(ys)}
    
    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    metrics = evaluate_segmentation(model, loader, device=torch.device("cpu"))
    assert "dice_mean" in metrics
    assert metrics["dice_mean"] >= 0

if __name__ == "__main__":
    test_evaluate_classification_smoke()
    test_evaluate_segmentation_smoke()
    print("Evaluation tests passed!")
