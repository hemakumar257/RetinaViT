import torch
import random
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import List, Dict, Any, Iterator

class QualityAwareSampler(Sampler):
    """
    Oversamples images based on their quality labels.
    """
    def __init__(self, dataset, quality_weights: Dict[str, float]):
        self.dataset = dataset
        self.quality_weights = quality_weights
        
        # Compute weights for each sample
        self.weights = self._compute_weights()
        self.sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.dataset),
            replacement=True
        )

    def _compute_weights(self) -> torch.Tensor:
        weights = []
        for i in range(len(self.dataset.df)):
            q_label = self.dataset.df.iloc[i]["quality_label"]
            weight = self.quality_weights.get(q_label, 1.0)
            weights.append(weight)
        return torch.DoubleTensor(weights)

    def __iter__(self) -> Iterator[int]:
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.dataset)

class EpisodicSampler(Sampler):
    """
    Samples episodes for N-way K-shot learning.
    """
    def __init__(
        self,
        dataset,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Group indices by class
        self.m_ind = {}
        for i in range(len(self.dataset.df)):
            label = self.dataset.df.iloc[i]["label"]
            if label not in self.m_ind:
                self.m_ind[label] = []
            self.m_ind[label].append(i)
        
        self.labels = list(self.m_ind.keys())

    def __iter__(self):
        for _ in range(self.n_episodes):
            episode_indices = []
            # Sample N classes
            chosen_classes = random.sample(self.labels, self.n_way)
            
            for c in chosen_classes:
                indices = self.m_ind[c]
                # Sample K support + Q query
                if len(indices) < (self.k_shot + self.n_query):
                    # Oversample if not enough samples
                    sampled = random.choices(indices, k=self.k_shot + self.n_query)
                else:
                    sampled = random.sample(indices, self.k_shot + self.n_query)
                episode_indices.extend(sampled)
            
            yield episode_indices

    def __len__(self):
        return self.n_episodes
