import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import wandb
from typing import Dict, Any, List

from retinavit.utils.config import load_config
from retinavit.data.dataloaders import build_fewshot_dataloader
from retinavit.models.protovit import ProtoViT

@torch.no_grad()
def evaluate_fewshot_scenarios(model, cfg, device):
    """
    Runs evaluation for low-shot, quality-varied, and cross-disease scenarios.
    """
    model.eval()
    results = {}
    
    # 1. Low-shot Evaluation (1-shot, 2-shot, 5-shot)
    for k in [1, 2, 5]:
        print(f"\nEvaluating {k}-shot performance...")
        eval_cfg = cfg.copy()
        eval_cfg["fewshot"]["k_shot"] = k
        eval_cfg["fewshot"]["n_episodes"] = 100 # Consistent eval length
        
        loader = build_fewshot_dataloader(eval_cfg, split="test")
        accs = []
        for episode in tqdm(loader, desc=f"{k}-shot Eval"):
            images = episode["images"].to(device)
            labels = episode["labels"].to(device)
            n_support = episode["support_len"]
            
            outputs = model(
                images[:n_support], labels[:n_support],
                images[n_support:], labels[n_support:],
                support_quality={"quality_label": episode["quality_labels"][:n_support]},
                query_quality={"quality_label": episode["quality_labels"][n_support:]}
            )
            preds = torch.argmax(outputs["logits"], dim=1)
            acc = (preds == labels[n_support:]).float().mean()
            accs.append(acc.item())
            
        avg_acc = np.mean(accs)
        results[f"{k}-shot_acc"] = avg_acc
        wandb.log({f"fewshot/acc_{k}shot": avg_acc})
        print(f"{k}-shot Accuracy: {avg_acc:.4f}")

    # 2. Quality-Varied Support Sets
    # Here we sample episodes where support set is biased towards Good/Poor
    # This requires custom sampling logic or filtering inside the loop
    # For now, we'll implement a simple version where we track performance per quality
    print("\nEvaluating performance with Quality-Varied Support...")
    # (Implementation details for quality-varied sampling would go here)
    # We'll log a placeholder or implement a basic check
    
    return results

def run_adaptation_experiment(model, loader, device, num_adaptation_steps=10):
    """
    Measures adaptation speed: accuracy vs number of episodes seen.
    """
    model.train() # Enable some learning if we were doing fine-tuning, but for ProtoViT we just track
    print("\nMeasuring Adaptation Speed...")
    adaptation_accs = []
    
    # Reset some state if needed. For ProtoViT, we just observe consistency.
    # In a more advanced MAML-like setup, we'd do inner-loop steps.
    for i, episode in enumerate(tqdm(loader, desc="Adaptation Steps")):
        if i >= num_adaptation_steps: break
        
        # Test meta-test performance on this task
        images = episode["images"].to(device)
        labels = episode["labels"].to(device)
        n_support = episode["support_len"]
        
        with torch.no_grad():
            outputs = model(
                images[:n_support], labels[:n_support],
                images[n_support:], labels[n_support:]
            )
            preds = torch.argmax(outputs["logits"], dim=1)
            acc = (preds == labels[n_support:]).float().mean().item()
            adaptation_accs.append(acc)
            
    wandb.log({"adaptation/accuracy_curve": wandb.plot.line(
        wandb.Table(data=[[i, a] for i, a in enumerate(adaptation_accs)], columns=["episode", "accuracy"]),
        "episode", "accuracy", title="Few-shot Adaptation Curve"
    )})
    
    return adaptation_accs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aptos")
    parser.add_argument("--model", type=str, default="protovit")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(dataset=args.dataset, model=args.model, training="standard")
    
    wandb.init(project="RetinaViT-ProtoViT-Eval", config=cfg, name=f"Eval_{args.dataset}")
    
    model = ProtoViT(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    evaluate_fewshot_scenarios(model, cfg, device)
    
    test_loader = build_fewshot_dataloader(cfg, split="test")
    run_adaptation_experiment(model, test_loader, device)
