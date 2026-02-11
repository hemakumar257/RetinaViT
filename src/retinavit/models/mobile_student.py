import torch
import torch.nn as nn
from typing import Dict, Any

class MobileRetinaNet(nn.Module):
    """
    A lightweight Vision Transformer / CNN hybrid for mobile deployment.
    Inspired by MobileViT and tiny ViT variants.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", {})
        num_classes = model_cfg.get("num_classes", 5)
        
        # Simplified Mobile-friendly encoder: CNN for initial resolution, then tiny Blocks
        # This is a representative student model structure.
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish()
        )
        
        # Tiny transformer-like blocks (represented as low-dim MLPs for this demo)
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Hardswish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

def build_mobile_student(cfg: Dict[str, Any]) -> nn.Module:
    return MobileRetinaNet(cfg)
