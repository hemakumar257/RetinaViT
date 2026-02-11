import torch
import pytest
from retinavit.models.multitask import MultitaskRetinaViT

def test_multitask_forward():
    cfg = {
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
                "num_seg_classes": 2,
                "lesion_attention": True,
                "inter_layers": [0, 1], # For tiny-2
                "decoder_channels": [64, 32, 16, 8]
            }
        }
    }
    
    model = MultitaskRetinaViT(cfg)
    x = torch.randn(2, 3, 224, 224)
    
    outputs = model(x)
    
    assert "logits" in outputs
    assert "segmentation" in outputs
    
    # Classification logits: [Batch, NumClasses]
    assert outputs["logits"].shape == (2, 5)
    
    # Segmentation mask: [Batch, NumSegClasses, H, W]
    # Note: Head includes a final 2x upsample to reach original res
    assert outputs["segmentation"].shape == (2, 2, 224, 224)

if __name__ == "__main__":
    test_multitask_forward()
    print("Multitask forward test passed!")
