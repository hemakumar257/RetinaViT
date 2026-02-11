import pytest
import torch
from unittest.mock import MagicMock
from retinavit.data.curriculum import QualityCurriculumSampler

def test_curriculum_weights():
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    # 40 Good, 30 Fair, 30 Poor
    def get_q(i):
        if i < 40: return "Good"
        if i < 70: return "Fair"
        return "Poor"
    dataset.get_quality_label = get_q
    
    cfg = {
        "curriculum": {
            "stages": [
                {"epoch_end": 2, "weights": {"Good": 1.0, "Fair": 0.1, "Poor": 0.0}},
                {"epoch_end": 5, "weights": {"Good": 1.0, "Fair": 1.0, "Poor": 1.0}}
            ]
        }
    }
    
    sampler = QualityCurriculumSampler(dataset, cfg)
    
    # Stage 1: Only Good and a bit of Fair
    sampler.set_epoch(0)
    indices = list(sampler)
    # Check that indices are mostly < 70
    fair_sampled = [idx for idx in indices if 40 <= idx < 70]
    poor_sampled = [idx for idx in indices if idx >= 70]
    
    assert len(poor_sampled) == 0
    assert len(fair_sampled) > 0
    
    # Stage 2: All
    sampler.set_epoch(4)
    indices = list(sampler)
    poor_sampled = [idx for idx in indices if idx >= 70]
    assert len(poor_sampled) > 0

if __name__ == "__main__":
    test_curriculum_weights()
    print("Curriculum tests passed!")
