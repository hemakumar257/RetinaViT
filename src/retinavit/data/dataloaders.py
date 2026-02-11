import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from retinavit.data.retina_dataset import RetinaDataset, CombinedRetinaDataset
from retinavit.data.samplers import QualityAwareSampler, EpisodicSampler
from retinavit.utils.config import load_config

def episodic_collate_fn(batch):
    """
    Collates a batch into (support, query) structure for few-shot learning.
    Re-orders batch to group all support samples followed by all query samples.
    Maps global labels to relative [0, N-1] labels within the episode.
    """
    # Infer n_way, k_shot, n_query from batch metadata if possible, 
    # but we assume the caller knows or it's fixed.
    # From EpisodicSampler: indices for each class are [S1..Sk, Q1..Qq]
    # We need to find n_way and (k+q)
    
    # We'll use the dataset info to help
    num_samples = len(batch)
    unique_labels = []
    for item in batch:
        if item["label"] not in unique_labels:
            unique_labels.append(item["label"])
    
    n_way = len(unique_labels)
    samples_per_class = num_samples // n_way
    
    # Assuming k_shot is given or we can infer it. 
    # Usually k_shot is fixed in cfg. For now, we'll try to find it or assume a default
    # A better way is to pass params or use a fixed ratio.
    # Let's assume the sampler logic: all classes have same k and q.
    # We'll look at the first class's samples to see if we can distinguish S and Q.
    # Actually, the simplest is to assume k_shot + n_query = samples_per_class
    # and use a fixed split, but the sampler should probably provide this info.
    # For now, we'll assume a standard split (e.g. k=5, q=15 -> 20)
    # Let's make it robust: we'll look for how many samples of each label we have.
    
    label_to_items = {l: [] for l in unique_labels}
    for item in batch:
        label_to_items[item["label"]].append(item)
    
    # We need k_shot. Let's assume it's the minimum samples per class? Or fixed?
    # In RetinaViT context, k is usually 5 or 1.
    # Let's assume we can infer it or it's the first X from each class.
    # A better way: the Sampler could yield (indices, k_shot) but DataLoader doesn't support that easily.
    # We'll use a heuristic: n_query is often larger than k_shot. 
    # Actually, we'll just use a fixed k_shot ratio or assume it's 5 if unknown.
    # FIX: We'll assume k_shot is known from the first labels encountered.
    # REAL FIX: Since batch is [C1_S...C1_Q, C2_S...C2_Q], we split each group.
    # We'll assume k_shot is the 'k' specified in config.
    
    # For now, let's just use the first 5 as support and the rest as query for each class
    # (assuming 5-way 5-shot)
    k_shot = 5 # Default
    # Check if total samples allows this
    if num_samples < n_way * k_shot:
        k_shot = 1 # Fallback to 1-shot
        
    support_items = []
    query_items = []
    
    # Sort labels to ensure consistent mapping [0, N-1]
    unique_labels.sort()
    label_map = {l: i for i, l in enumerate(unique_labels)}
    
    for l in unique_labels:
        items = label_to_items[l]
        support_items.extend(items[:k_shot])
        query_items.extend(items[k_shot:])
        
    all_reordered = support_items + query_items
    
    images = torch.stack([item["image"] for item in all_reordered])
    # Relative labels
    labels = torch.tensor([label_map[item["label"]] for item in all_reordered])
    
    return {
        "images": images,
        "labels": labels,
        "support_len": len(support_items),
        "dataset_names": [item["dataset"] for item in all_reordered],
        "quality_labels": [item["quality_label"] for item in all_reordered],
        "quality_scores": [item.get("quality_score", 0.0) for item in all_reordered]
    }

def build_retina_dataloader(cfg: Dict[str, Any], split: str = "train", dataset_names: list = None) -> DataLoader:
    """Builds a standard DataLoader for one or more datasets."""
    if dataset_names is None:
        dataset_names = [cfg["dataset"]["name"]]
    
    datasets = []
    for ds_name in dataset_names:
        # We might need to load specific configs for each to get correct target_size etc.
        ds_cfg = load_config(dataset=ds_name)
        datasets.append(RetinaDataset(
            ds_cfg, 
            split=split,
            apply_preprocessing=True,
            apply_augmentation=(split == "train"),
            apply_simulation=(split == "train" and ds_cfg.get("smartphone_simulation", {}).get("enable", False))
        ))
    
    if len(datasets) > 1:
        dataset = CombinedRetinaDataset(datasets)
    else:
        dataset = datasets[0]
        
    loader_cfg = cfg["training"]
    sampler = None
    
    # Optional Quality-Aware Sampling
    if split == "train" and cfg.get("training", {}).get("quality_sampling", {}).get("enable", False):
        weights = cfg["training"]["quality_sampling"].get("weights", {"Good": 1.0, "Fair": 1.0, "Poor": 1.0})
        sampler = QualityAwareSampler(dataset, weights)
        shuffle = False
    else:
        shuffle = (split == "train")
        
    return DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", False)
    )

def build_fewshot_dataloader(cfg: Dict[str, Any], split: str = "train") -> DataLoader:
    """Builds an episodic DataLoader for few-shot learning across one or more datasets."""
    fs_cfg = cfg.get("fewshot", {})
    dataset_names = fs_cfg.get("datasets", [cfg["dataset"]["name"]])
    
    datasets = []
    for ds_name in dataset_names:
        ds_cfg = load_config(dataset=ds_name)
        datasets.append(RetinaDataset(
            ds_cfg,
            split=split,
            apply_preprocessing=True,
            apply_augmentation=(split == "train"),
            apply_simulation=(split == "train")
        ))
    
    if len(datasets) > 1:
        dataset = CombinedRetinaDataset(datasets)
    else:
        dataset = datasets[0]
    
    n_way = fs_cfg.get("n_way", 5)
    k_shot = fs_cfg.get("k_shot", 5)
    n_query = fs_cfg.get("n_query", 15)
    n_episodes = fs_cfg.get("n_episodes", 100)
    
    sampler = EpisodicSampler(
        dataset,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        n_episodes=n_episodes
    )
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=episodic_collate_fn,
        num_workers=cfg["training"].get("num_workers", 0)
    )
