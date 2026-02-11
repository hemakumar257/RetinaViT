import torch
from retinavit.models.temporal import TemporalRetinaModule

def test_temporal_module():
    cfg = {
        "temporal": {
            "input_dim": 128,
            "model": {
                "num_layers": 1,
                "num_heads": 2,
                "causal": True
            }
        }
    }
    
    model = TemporalRetinaModule(cfg)
    
    # Batch size 2, 5 visits, feature dim 128
    x = torch.randn(2, 5, 128)
    
    outputs = model(x)
    
    assert "progression_logits" in outputs
    assert "patient_logits" in outputs
    
    # Progression: [B, T, 3]
    assert outputs["progression_logits"].shape == (2, 5, 3)
    
    # Patient level: [B, 5]
    assert outputs["patient_logits"].shape == (2, 5)

if __name__ == "__main__":
    test_temporal_module()
    print("Temporal module test passed!")
