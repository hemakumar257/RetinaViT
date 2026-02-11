import pytest
import pandas as pd
from retinavit.data.samplers import EpisodicSampler
from unittest.mock import MagicMock

def test_episodic_sampler_structure():
    dataset = MagicMock()
    # 4 classes, 10 samples each
    df = pd.DataFrame({
        "label": [0]*10 + [1]*10 + [2]*10 + [3]*10
    })
    dataset.df = df
    dataset.__len__.return_value = 40
    
    n_way, k_shot, n_query = 2, 2, 3
    sampler = EpisodicSampler(dataset, n_way=n_way, k_shot=k_shot, n_query=n_query, n_episodes=5)
    
    assert len(sampler) == 5
    episodes = list(iter(sampler))
    assert len(episodes[0]) == n_way * (k_shot + n_query)
