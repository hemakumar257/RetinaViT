import torch
import numpy as np
from typing import Tuple

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error.
    probs: [N, C]
    labels: [N]
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Get samples in this bin
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def get_reliability_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Returns data for reliability diagrams.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        
        if np.any(in_bin):
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper)/2)
            bin_sizes.append(0)
            
    return bin_accs, bin_confs, bin_sizes
