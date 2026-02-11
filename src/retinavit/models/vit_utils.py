import torch
import torch.nn as nn
import os
from typing import Dict, Any, Optional

def load_pretrained_weights(model: nn.Module, cfg: Dict[str, Any]):
    """
    Utility to initialize model from public weights or custom SSL checkpoints.
    """
    init_cfg = cfg.get("init", {})
    init_type = init_cfg.get("type", "scratch")
    
    if init_type == "scratch":
        print("Initializing model from scratch.")
        return
        
    elif init_type == "ssl_fundus":
        path = init_cfg.get("ssl_checkpoint", "checkpoints/ssl_vit_encoder.pt")
        if os.path.exists(path):
            print(f"Loading custom SSL weights from {path}")
            state_dict = torch.load(path, map_location="cpu")
            # If model is RetinaViT, we target model.backbone
            if hasattr(model, "backbone"):
                model.backbone.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
        else:
            print(f"WARNING: SSL checkpoint {path} not found. Initializing from scratch.")
            
    elif init_type == "public_vit":
        # timm already handles this via pretrained=True in constructor
        # but we can add more logic here for specific mappings
        print(f"Using public weights for {cfg['model']['name']}")
        
def get_model_complexity(model: nn.Module):
    """
    Returns proxy for model complexity (parameters).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
