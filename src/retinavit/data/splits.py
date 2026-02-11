import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from typing import Dict, Any, List, Tuple

def get_patient_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates indices for K-fold cross-validation, ensuring all images 
    from a single patient stay within the same fold (no overlap).
    """
    if "patient_id" not in df.columns:
        # Fallback to standard CV if no patient ID
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(kf.split(df))
        
    # StratifiedGroupKFold balances labels while keeping patients grouped
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Stratify by 'label'
    y = df["label"].values
    groups = df["patient_id"].values
    
    folds = []
    for train_idx, val_idx in sgkf.split(df, y, groups):
        folds.append((train_idx, val_idx))
        
    return folds

def get_site_folds(df: pd.DataFrame, n_folds: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits by 'dataset' (site) for external validation stress-testing.
    Each fold uses N-1 datasets for training and 1 for validation.
    """
    if "dataset" not in df.columns:
        return []
        
    datasets = df["dataset"].unique()
    folds = []
    for target_ds in datasets:
        val_idx = df.index[df["dataset"] == target_ds].values
        train_idx = df.index[df["dataset"] != target_ds].values
        folds.append((train_idx, val_idx))
    return folds
