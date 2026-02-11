import torch
import torch.nn as nn
from retinavit.training.train_multitask import MultitaskTrainer

def test_gradient_balancing():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 10)
            self.backbone.patch_embed = nn.Identity()
            self.backbone.patch_embed.patch_size = [16, 16]
            self.classifier = nn.Linear(10, 5)
            self.seg_head = nn.Identity()
            
        def forward(self, x):
            feat = self.backbone(x)
            return {
                "logits": self.classifier(feat),
                "segmentation": torch.randn(x.size(0), 1, 224, 224)
            }

    cfg = {
        "training": {
            "learning_rate": 1e-3,
            "multitask": {
                "weighting": {"strategy": "uncertainty"}
            }
        }
    }
    
    device = torch.device("cpu")
    model = DummyModel()
    trainer = MultitaskTrainer(model, cfg, device)
    
    # Check weight updates
    initial_w_clf = torch.exp(-trainer.log_vars[0]).item()
    
    batch = {
        "images": torch.randn(4, 10),
        "labels": torch.zeros(4, dtype=torch.long),
        "masks": torch.randn(4, 1, 224, 224)
    }
    
    # Simulate a few steps
    for _ in range(5):
        stats = trainer.train_step(batch)
        
    final_w_clf = torch.exp(-trainer.log_vars[0]).item()
    
    # Weights should change
    assert initial_w_clf != final_w_clf
    assert stats["total_loss"] > 0

if __name__ == "__main__":
    test_gradient_balancing()
    print("Gradient balancing test passed!")
