import pytest
import os
import shutil
from pathlib import Path
from retinavit.utils.config import load_config, validate_config

def test_load_config():
    # Create dummy data for presence check
    data_dir = Path("datasets/raw/aptos2019")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dummy.txt").write_text("dummy")
    
    # Load standardized combination
    cfg = load_config(dataset="aptos", model="vit_base", training="standard")
    
    # Verify core keys
    assert "seed" in cfg
    assert "device" in cfg
    assert "dataset" in cfg
    assert "model" in cfg
    assert "training" in cfg
    
    # Verify merged values
    assert cfg["dataset"]["name"] == "aptos2019"
    assert cfg["model"]["backbone"] == "vit_base_patch16_224"
    assert cfg["training"]["batch_size"] == 32
    
    # Clean up
    shutil.rmtree("datasets/raw/aptos2019")

def test_config_validation():
    # Valid config
    data_dir = Path("datasets/raw/aptos2019")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dummy.txt").write_text("dummy")
    
    cfg = load_config(dataset="aptos", model="vit_base", training="standard")
    validate_config(cfg) # Should not raise
    
    # Invalid dropout
    cfg["model"]["dropout"] = 1.5
    with pytest.raises(ValueError, match="model.dropout must be between 0 and 1"):
        validate_config(cfg)
        
    # Invalid batch size
    cfg["model"]["dropout"] = 0.1
    cfg["training"]["batch_size"] = -1
    with pytest.raises(ValueError, match="training.batch_size must be greater than 0"):
        validate_config(cfg)
        
    # Clean up
    shutil.rmtree("datasets/raw/aptos2019")

def test_missing_config():
    with pytest.raises(FileNotFoundError):
        load_config(dataset="non_existent")
