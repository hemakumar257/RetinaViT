import torch
import yaml
from pathlib import Path
from retinavit.data.dataloaders import build_retina_dataloader, build_fewshot_dataloader
from retinavit.utils.config import load_config

def debug_dataset():
    print("=== Debugging Unified Dataset ===")
    
    # Try loading each dataset
    datasets = ["aptos", "messidor2", "idrid"]
    
    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")
        try:
            # Mocking config enough to load
            cfg = load_config(dataset=ds_name, training="standard")
            
            # Standard Dataloader
            loader = build_retina_dataloader(cfg, split="train")
            print(f"Dataset Size: {len(loader.dataset)}")
            
            # Fetch one batch
            batch = next(iter(loader))
            print(f"Batch Image Shape: {batch['image'].shape}")
            print(f"Batch Labels: {batch['label']}")
            print(f"Batch Quality: {batch['quality_label'][:5]}")
            
            # Few-shot Dataloader test
            fs_loader = build_fewshot_dataloader(cfg, split="train")
            fs_batch = next(iter(fs_loader))
            print(f"Few-shot Episode Shape: {fs_batch['images'].shape}")
            print(f"Few-shot Labels count: {len(fs_batch['labels'])}")

        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

    print("\n--- Testing Multi-Dataset Combined Loading ---")
    try:
        cfg = load_config(dataset="aptos", training="standard")
        # Ensure we use names that exist in the system
        cfg["fewshot"]["datasets"] = ["aptos2019", "idrid"] 
        loader = build_fewshot_dataloader(cfg, split="train")
        batch = next(iter(loader))
        print(f"Combined Batch Datasets: {set(batch['dataset_names'])}")
    except Exception as e:
        print(f"Multi-dataset test failed (expected if data missing): {e}")

if __name__ == "__main__":
    debug_dataset()
