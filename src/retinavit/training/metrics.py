import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
from typing import Dict, List, Any

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Computes standard classification metrics.
    y_true: [N]
    y_pred: [N] (logits or hard labels)
    y_prob: [N, C] (probabilities)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic")
    }
    
    if y_prob is not None:
        try:
            metrics["auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auc_macro"] = 0.0 # Handle cases with only one class in batch
            
    return metrics

def get_stratified_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    quality_labels: List[str],
    y_prob: np.ndarray = None
) -> Dict[str, Dict[str, float]]:
    """
    Computes metrics stratified by quality band.
    """
    stratified = {}
    quality_labels = np.array(quality_labels)
    unique_q = np.unique(quality_labels)
    
    for q in unique_q:
        mask = (quality_labels == q)
        if not np.any(mask):
            continue
            
        q_true = y_true[mask]
        q_pred = y_pred[mask]
        q_prob = y_prob[mask] if y_prob is not None else None
        
        # Only compute if we have enough samples for meaningful metrics
        if len(np.unique(q_true)) > 1:
            stratified[f"quality_{q.lower()}"] = compute_metrics(q_true, q_pred, q_prob)
        else:
            # Fallback for single class buckets
            stratified[f"quality_{q.lower()}"] = {"accuracy": accuracy_score(q_true, q_pred)}
            
    return stratified
