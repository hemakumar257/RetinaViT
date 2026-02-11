import pytest
import torch
import pandas as pd
from retinavit.data.temporal import PatientSequenceDataset
from unittest.mock import MagicMock

def test_patient_grouping():
    dataset = MagicMock()
    dataset.df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P3", "P2"],
        "label": [0, 1, 0, 2, 0]
    })
    dataset.dataset_name = "test_ds"
    dataset.__len__.return_value = 5
    # Mock __getitem__ to return dummy img
    dataset.__getitem__.side_effect = lambda i: {"image": torch.zeros((1, 10, 10)), "label": dataset.df.iloc[i]["label"], "quality_label": "Good"}
    
    seq_ds = PatientSequenceDataset(dataset, max_seq_len=2)
    assert len(seq_ds) == 3 # 3 unique patients
    
    sample = seq_ds[0] # P1
    assert sample["patient_id"] == "P1"
    assert sample["images"].shape[0] == 2
    assert sample["labels"].tolist() == [0, 1]
