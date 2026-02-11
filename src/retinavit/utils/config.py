import yaml
import os
from pathlib import Path
from typing import Any, Dict

def merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges two dictionaries."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def load_config(
    dataset: str = None,
    model: str = None,
    training: str = None,
    config_root: str = "configs"
) -> Dict[str, Any]:
    """Loads and merges hierarchical configurations."""
    root = Path(config_root)
    
    # Start with base config
    with open(root / "base.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Merge dataset
    if dataset:
        ds_path = root / "dataset" / f"{dataset}.yaml"
        if ds_path.exists():
            with open(ds_path, "r") as f:
                merge_dicts(config, yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Dataset config not found: {ds_path}")
            
    # Merge model
    if model:
        model_path = root / "model" / f"{model}.yaml"
        if model_path.exists():
            with open(model_path, "r") as f:
                merge_dicts(config, yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Model config not found: {model_path}")
            
    # Merge training
    if training:
        train_path = root / "training" / f"{training}.yaml"
        if train_path.exists():
            with open(train_path, "r") as f:
                merge_dicts(config, yaml.safe_load(f))
        else:
            raise FileNotFoundError(f"Training config not found: {train_path}")
            
    validate_config(config)
    
    # Dataset presence verification
    verify_dataset_presence(config)
    
    return config

def verify_dataset_presence(cfg: Dict[str, Any]):
    """Verifies that the required dataset directory exists and is not empty."""
    # This now correctly uses the 'datasets' directory as defined in base.yaml
    data_dir = Path(cfg["paths"]["data_dir"]) / "raw" / cfg["dataset"]["name"]
    
    # Map config names to directory names if they differ
    # For now, we assume they align or can be mapped here
    ds_map = {
        "aptos2019": "aptos2019",
        "messidor2": "messidor2",
        "idrid": "idrid"
    }
    
    ds_dir_name = ds_map.get(cfg["dataset"]["name"], cfg["dataset"]["name"])
    data_dir = Path(cfg["paths"]["data_dir"]) / "raw" / ds_dir_name

    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"Dataset '{cfg['dataset']['name']}' not found at {data_dir}. "
            f"Please refer to docs/datasets/{ds_dir_name}.md for download instructions."
        )

def validate_config(cfg: Dict[str, Any]):
    """Validates the final configuration object."""
    # Required keys check
    required_keys = [
        "seed", "device", "paths", "dataset", "model", "training"
    ]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required configuration key: {key}")
            
    # Validate dataset splits
    if "splits" not in cfg["dataset"]:
        raise ValueError("dataset.splits is missing")

    # Validate Augmentation (Optional but checked if present)
    if "augmentation" in cfg:
        aug = cfg["augmentation"]
        for key in ["anatomical_shift_prob", "simulated_exudates_prob", "camera_artifacts_prob"]:
            if key in aug and not (0 <= aug[key] <= 1):
                raise ValueError(f"augmentation.{key} must be between 0 and 1")

    # Validate Smartphone Simulation
    if "smartphone_simulation" in cfg:
        sim = cfg["smartphone_simulation"]
        for key in ["motion_blur_prob", "low_light_prob", "jpeg_artifacts_prob", "offcenter_crop_prob"]:
            if key in sim and not (0 <= sim[key] <= 1):
                raise ValueError(f"smartphone_simulation.{key} must be between 0 and 1")
            
    # Sub-key checks
    if "data_dir" not in cfg["paths"]:
        raise KeyError("Missing paths.data_dir in config")
        
    # Constraint checks
    if cfg["training"].get("batch_size", 0) <= 0:
        raise ValueError("training.batch_size must be greater than 0")
        
    dropout = cfg["model"].get("dropout", 0.0)
    if not (0 <= dropout <= 1):
        raise ValueError("model.dropout must be between 0 and 1")
        
    print("Configuration validated successfully.")
