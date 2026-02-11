import torch
import torch.nn as nn
import numpy as np
import pytest
from retinavit.explainability.vit_explain import ViTExplainer

class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock timm structure
        self.patch_embed = nn.Identity()
        self.patch_embed.num_patches = 16
        self.patch_embed.patch_size = (16, 16)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn_drop': nn.Identity()
            }) for _ in range(2)
        ])
        self.head = nn.Linear(10, 5)
        self.backbone = self # simpler for mock
        
    def forward(self, x):
        # Dummy softmax for hook to capture something
        # In real timm, the hook captures the input to attn_drop
        B = x.shape[0]
        # Simulate attn matrix [B, H, N, N]
        attn = torch.rand(B, 1, 17, 17) 
        for b in self.blocks:
            b.attn_drop(attn)
        return {"logits": torch.randn(B, 5)}

def test_attention_rollout_shape():
    model = MockViT()
    explainer = ViTExplainer(model, device=torch.device("cpu"))
    images = torch.randn(1, 3, 224, 224)
    
    rollout = explainer.compute_attention_rollout(images)
    assert rollout is not None
    assert rollout.shape == (1, 4, 4) # sqrt(16) patches

def test_grad_cam_shape():
    model = MockViT()
    explainer = ViTExplainer(model, device=torch.device("cpu"))
    images = torch.randn(1, 3, 32, 32)
    
    cam = explainer.compute_grad_cam(images)
    assert cam is not None
    assert cam.shape == (1, 4, 4)

if __name__ == "__main__":
    test_attention_rollout_shape()
    test_grad_cam_shape()
    print("ViT Explainability tests passed!")
