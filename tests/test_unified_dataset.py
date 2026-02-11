import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from retinavit.data.retina_dataset import RetinaDataset

def test_retina_dataset_dict_output(tmp_path):
    # Mock config
    cfg = {
        "dataset": {"name": "aptos2019", "target_size": [256, 256]},
        "paths": {"data_dir": str(tmp_path)},
        "augmentation": {"enable_medical_aug": False},
        "smartphone_simulation": {"enable": False},
        "quality_assessment": {"quality_label_csv": str(tmp_path / "quality.csv")}
    }
    
    # Create mock data
    raw_dir = tmp_path / "raw" / "aptos2019"
    raw_dir.mkdir(parents=True)
    pd.DataFrame({"id_code": ["test1"], "diagnosis": [0]}).to_csv(raw_dir / "train.csv", index=False)
    pd.DataFrame({"image_path": ["test1.png"], "quality_label": ["Good"]}).to_csv(tmp_path / "quality.csv", index=False)
    
    # Dataset should handle missing images by returning dummy
    dataset = RetinaDataset(cfg)
    assert len(dataset) == 1
    sample = dataset[0]
    
    assert isinstance(sample, dict)
    assert "image" in sample
    assert sample["image"].shape == (3, 256, 256)
    assert sample["label"] == 0
    assert sample["quality_label"] == "Good"
    assert "patient_id" in sample

def test_combined_retina_dataset():
    from retinavit.data.retina_dataset import CombinedRetinaDataset
    from unittest.mock import MagicMock
    
    ds1 = MagicMock(spec=RetinaDataset)
    ds1.dataset_name = "aptos2019"
    ds1.df = pd.DataFrame({"image_path": ["a.png"], "label": [0], "quality_label": ["Good"], "patient_id": ["P1"]})
    ds1.__len__.return_value = 1
    ds1.__getitem__.return_value = {"image": torch.zeros(1), "dataset": "aptos2019"}
    
    ds2 = MagicMock(spec=RetinaDataset)
    ds2.dataset_name = "idrid"
    ds2.df = pd.DataFrame({"image_path": ["i.png"], "label": [1], "quality_label": ["Poor"], "patient_id": ["P2"]})
    ds2.__len__.return_value = 1
    ds2.__getitem__.return_value = {"image": torch.ones(1), "dataset": "idrid"}
    
    combined = CombinedRetinaDataset([ds1, ds2])
    assert len(combined) == 2
    assert combined[0]["dataset"] == "aptos2019"
    assert combined[1]["dataset"] == "idrid"
