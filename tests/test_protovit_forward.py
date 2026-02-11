import torch
import pytest
from retinavit.models.protovit import ProtoViT

def test_protovit_forward():
    # Tiny backbone config
    cfg = {
        "model": {
            "backbone": {
                "name": "vit_tiny_patch16_224", # Wrapper will handle variants
                "image_size": [224, 224],
                "patch_size": 16,
                "embed_dim": 192,
                "depth": 2,
                "num_heads": 3,
                "mlp_dim": 768,
                "dropout": 0.1,
                "pretrained": False,
                "num_classes": 5
            },
            "protovit": {
                "feature_type": "class_token",
                "metric_type": "euclidean",
                "cross_attention": {
                    "enable": True,
                    "num_layers": 1,
                    "num_heads": 2
                }
            }
        }
    }
    
    model = ProtoViT(cfg)
    
    # 3-way 2-shot episode
    n_way = 3
    k_shot = 2
    n_query = 5
    
    s_imgs = torch.randn(n_way * k_shot, 3, 224, 224)
    s_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
    
    q_imgs = torch.randn(n_way * n_query, 3, 224, 224)
    q_labels = torch.repeat_interleave(torch.arange(n_way), n_query)
    
    outputs = model(s_imgs, s_labels, q_imgs, q_labels)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (n_way * n_query, n_way)
    assert outputs["prototypes"].shape == (n_way, 192)

def test_protovit_with_quality():
    cfg = {
        "model": {
            "backbone": {
                "name": "vit_tiny_patch16_224",
                "image_size": [224, 224],
                "patch_size": 16,
                "embed_dim": 192,
                "depth": 2,
                "num_heads": 3,
                "mlp_dim": 768,
                "dropout": 0.1,
                "pretrained": False,
                "quality_aware": {
                    "enable": True,
                    "use_quality_token": True,
                    "use_attention_modulation": True
                }
            },
            "protovit": {
                "feature_type": "class_token",
                "metric_type": "euclidean"
            }
        }
    }
    
    model = ProtoViT(cfg)
    
    # 2-way 1-shot
    s_imgs = torch.randn(2, 3, 224, 224)
    s_labels = torch.arange(2)
    s_quality = {"quality_label": ["Good", "Poor"]}
    
    q_imgs = torch.randn(2, 3, 224, 224)
    q_quality = {"quality_label": ["Fair", "Good"]}
    
    outputs = model(s_imgs, s_labels, q_imgs, support_quality=s_quality, query_quality=q_quality)
    assert outputs["logits"].shape == (2, 2)

if __name__ == "__main__":
    print("Running test_protovit_forward...")
    test_protovit_forward()
    print("Success")
    print("Running test_protovit_with_quality...")
    test_protovit_with_quality()
    print("Success")
