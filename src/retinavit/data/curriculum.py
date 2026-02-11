import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Dict, Any, List

class QualityCurriculumSampler(Sampler):
    """
    Sampler that adjusts quality-based sampling weights over epochs.
    Early epochs focus on Good images, later epochs include Poor images.
    """
    def __init__(self, dataset, cfg: Dict[str, Any]):
        self.dataset = dataset
        self.cfg = cfg.get("curriculum", {})
        self.stages = self.cfg.get("stages", [])
        self.current_epoch = 0
        
        # Group indices by quality
        self.indices_by_quality = {"Good": [], "Fair": [], "Poor": [], "Unknown": []}
        for idx in range(len(dataset)):
            q = dataset.get_quality_label(idx)
            self.indices_by_quality.get(q, self.indices_by_quality["Unknown"]).append(idx)
            
        self.num_samples = len(dataset)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_current_weights(self) -> Dict[str, float]:
        for stage in self.stages:
            if self.current_epoch < stage["epoch_end"]:
                return stage["weights"]
        return {"Good": 1.0, "Fair": 1.0, "Poor": 1.0, "Unknown": 1.0}

    def __iter__(self):
        weights = self._get_current_weights()
        
        # Build probability distribution over all indices
        probs = np.zeros(self.num_samples)
        for q, idxs in self.indices_by_quality.items():
            w = weights.get(q, 1.0)
            if len(idxs) > 0:
                probs[idxs] = w / len(idxs)
                
        # Normalize
        probs /= probs.sum()
        
        # Sample
        # If we have restricted categories (w=0 for some), we might have fewer candidates than num_samples
        num_candidates = (probs > 0).sum()
        if num_candidates == 0:
            return iter(range(self.num_samples))
            
        # Use replacement if we want to maintain epoch size but have restricted candidates
        sampled_indices = np.random.choice(
            self.num_samples, 
            size=self.num_samples, 
            p=probs, 
            replace=(num_candidates < self.num_samples)
        )
        return iter(sampled_indices.tolist())

    def __len__(self):
        return self.num_samples
