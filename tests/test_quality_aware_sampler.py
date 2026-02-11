import pytest
import torch
import pandas as pd
from retinavit.data.samplers import QualityAwareSampler
from unittest.mock import MagicMock

def test_quality_aware_weights():
    dataset = MagicMock()
    dataset.df = pd.DataFrame({"quality_label": ["Good", "Poor", "Good", "Poor"]})
    dataset.__len__.return_value = 4
    
    weights_map = {"Good": 1.0, "Poor": 10.0}
    sampler = QualityAwareSampler(dataset, weights_map)
    
    expected_weights = torch.DoubleTensor([1.0, 10.0, 1.0, 10.0])
    assert torch.allclose(sampler.weights, expected_weights)
