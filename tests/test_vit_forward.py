import pytest
import torch
from retinavit.models.vit import build_vit_base, QualityAwareRetinaViT

def test_vit_base_forward():
    cfg = {
        "model": {
            "name": "vit_base",
            "image_size": [224, 224],
            "num_classes": 5,
            "pretrained": False,
            "quality_aware": {"enable": False}
        }
    }
    model = build_vit_base(cfg)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 5)

def test_quality_aware_vit_forward():
    cfg = {
        "model": {
            "name": "vit_base",
            "image_size": [224, 224],
            "num_classes": 5,
            "pretrained": False,
            "quality_aware": {
                "enable": True,
                "use_quality_token": True,
                "use_attention_modulation": True,
                "use_adaptive_patch_dropout": True
            }
        }
    }
    # QualityAwareRetinaViT is returned if quality_aware enable is true
    model = QualityAwareRetinaViT(cfg)
    dummy_input = torch.randn(2, 3, 224, 224)
    quality_info = {
        "quality_label": ["Good", "Poor"],
        "quality_score": [0.9, 0.2]
    }
    
    output = model(dummy_input, quality_info)
    assert output.shape == (2, 5)
