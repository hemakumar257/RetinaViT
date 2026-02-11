import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from retinavit.training.train_protovit import train_one_episode
from retinavit.models.protovit import ProtoViT

def test_episodic_training_step():
    # Mock model
    model = MagicMock(spec=ProtoViT)
    model.train.return_value = None
    model.return_value = {
        "logits": torch.randn(5, 5) # 5 queries, 5 classes
    }
    
    optimizer = MagicMock()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")
    
    # Episode batch
    episode = {
        "images": torch.randn(10, 3, 64, 64), # 5 support + 5 query
        "labels": torch.repeat_interleave(torch.arange(5), 2), # Pairs of (S, Q)
        "support_len": 5,
        "quality_labels": ["Good"] * 10,
        "quality_scores": [0.9] * 10
    }
    # Fix labels to match [S1..Sk, Q1..Qq] reordered expectation in train_protovit
    # items 0-4 are support, items 5-9 are query
    episode["labels"] = torch.cat([torch.arange(5), torch.arange(5)])
    
    loss, acc = train_one_episode(model, episode, optimizer, criterion, device)
    
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    optimizer.zero_grad.assert_called()
    optimizer.step.assert_called()

if __name__ == "__main__":
    print("Running test_episodic_training_step...")
    test_episodic_training_step()
    print("Success")
