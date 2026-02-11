import wandb
import mlflow
import os
from typing import Any, Dict

def init_experiment(cfg: Dict[str, Any], project_name: str = "RetinaViT"):
    """Initializes W&B and MLflow for experiment tracking."""
    
    # Derive run name
    ds_name = cfg["dataset"].get("name", "unknown_ds")
    model_name = cfg["model"].get("backbone", "unknown_model")
    train_name = "training"
    run_name = f"{ds_name}_{model_name}_{train_name}"
    
    # Initialize MLflow
    mlruns_dir = cfg["paths"].get("mlruns_dir", "mlruns")
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
    mlflow.set_experiment(project_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(flatten_dict(cfg))
    
    # Initialize W&B
    # Fail gracefully if not logged in or in offline mode
    wandb_mode = os.getenv("WANDB_MODE", "online")
    wandb.init(
        project=project_name,
        name=run_name,
        config=cfg,
        mode=wandb_mode
    )
    
    print(f"Experiment initialized: {run_name}")

def log_metrics(metrics: Dict[str, Any], step: int):
    """Logs metrics to both W&B and MLflow."""
    wandb.log(metrics, step=step)
    mlflow.log_metrics(metrics, step=step)

def log_artifacts(path: str, artifact_name: str):
    """Logs artifacts to both W&B and MLflow."""
    # MLflow
    mlflow.log_artifact(path)
    
    # W&B
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(path)
    wandb.log_artifact(artifact)

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flattens a nested dictionary for easier logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
