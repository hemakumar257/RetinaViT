import torch
import torch.nn as nn
from retinavit.training.hpo import run_hpo

def test_hpo_smoke():
    cfg = {
        "training": {
            "learning_rate": 1e-4, 
            "weight_decay": 1e-5,
            "losses": {
                "classification": {"type": "ce", "weight": 1.0},
                "segmentation": {"weight": 1.0}
            }
        },
        "hpo": {
            "n_trials": 2,
            "study_name": "smoke_test"
        },
        "model": {
            "name": "vit_tiny_patch16_224",
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
                "num_classes": 5
            },
            "multitask": {
                "num_seg_classes": 1,
                "lesion_attention": True,
                "inter_layers": [0, 1],
                "decoder_channels": [32, 16, 8, 4]
            }
        }
    }
    
    study = run_hpo(cfg)
    assert len(study.trials) == 2

if __name__ == "__main__":
    test_hpo_smoke()
    print("HPO smoke test passed!")
