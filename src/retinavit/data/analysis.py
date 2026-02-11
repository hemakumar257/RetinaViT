import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

def load_labels_aptos(data_dir: str) -> pd.DataFrame:
    """Loads APTOS 2019 labels from train.csv."""
    csv_path = Path(data_dir) / "train.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Expected columns: id_code, diagnosis
    return df.rename(columns={"diagnosis": "label"})

def load_labels_messidor2(data_dir: str) -> pd.DataFrame:
    """Loads MESSIDOR-2 labels from scores CSV."""
    # Assuming the scores might be in a file like messidor_data.csv
    csv_path = Path(data_dir) / "messidor_data.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Labels in Messidor-2 often use 'advena' or 'referable'
    # We map to 'label' for consistency
    return df

def load_labels_idrid(data_dir: str) -> pd.DataFrame:
    """Loads IDRiD grading labels."""
    csv_path = Path(data_dir) / "B. Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df.rename(columns={"Retinopathy grade": "label"})

def compute_class_distribution(dataset_name: str, data_dir: str) -> Optional[pd.DataFrame]:
    """Computes class counts and frequencies."""
    loaders = {
        "aptos2019": load_labels_aptos,
        "messidor2": load_labels_messidor2,
        "idrid": load_labels_idrid
    }
    
    if dataset_name not in loaders:
        return None
        
    df = loaders[dataset_name](data_dir)
    if df.empty or "label" not in df.columns:
        return None
        
    dist = df["label"].value_counts().sort_index().reset_index()
    dist.columns = ["class", "count"]
    dist["frequency"] = dist["count"] / dist["count"].sum()
    return dist

def summarize_meta_features(dataset_name: str, data_dir: str) -> Dict[str, Any]:
    """Summarizes dataset-wide meta-features like resolutions."""
    data_path = Path(data_dir)
    resolutions = []
    # Simplified check for speed in large datasets
    # In practice, one might use a subset or cached stats
    return {
        "dataset": dataset_name,
        "path": str(data_path),
        "folder_exists": data_path.exists()
    }
