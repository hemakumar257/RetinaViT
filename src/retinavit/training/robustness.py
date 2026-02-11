import torch
import numpy as np
import copy
from tqdm import tqdm
from typing import Dict, Any, List

from .evaluation import evaluate_classification, evaluate_segmentation
from ..data.smartphone_simulation import SmartphoneSimulator

def evaluate_under_quality_degradation(model, dataloader, cfg, device) -> Dict[str, Any]:
    """
    Evaluates model performance across increasing levels of quality degradation.
    """
    levels = [0, 1, 2, 3, 4, 5] # severity levels
    results = {"classification": {}, "segmentation": {}}
    
    # Original results
    base_results = evaluate_classification(model, dataloader, device)
    results["classification"][0] = base_results["accuracy"]
    
    # Create simulator
    simulator = SmartphoneSimulator(cfg)
    
    for level in levels[1:]:
        print(f"Evaluating robustness at degradation level {level}...")
        
        # We need a modified dataloader or apply transform on the fly
        # For simplicity, we'll wrap the dataset or just manually process in a loop
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Severity {level}"):
                images = batch["image"].clone()
                labels = batch["label"]
                
                # Apply degradation (simulator works on numpy)
                imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                degraded_imgs = []
                for i in range(len(imgs_np)):
                    # Control severity via simulator params if possible, 
                    # or just apply multiple times
                    img = imgs_np[i]
                    for _ in range(level):
                        img = simulator.simulate(img)
                    degraded_imgs.append(img)
                
                # Back to tensor
                degraded_imgs = np.stack(degraded_imgs).transpose(0, 3, 1, 2)
                degraded_tensor = torch.from_numpy(degraded_imgs).float().to(device) / 255.0
                
                outputs = model(degraded_tensor)
                preds = torch.argmax(outputs if not isinstance(outputs, dict) else outputs["logits"], dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(labels)
                
        acc = (torch.cat(all_preds) == torch.cat(all_targets)).float().mean().item()
        results["classification"][level] = acc
        
    return results

def evaluate_cross_dataset(model, datasets: List[str], base_cfg, device):
    """
    Evaluates a model across multiple external datasets.
    """
    from ..data.dataloaders import build_retina_dataloader
    results = {}
    
    for ds_name in datasets:
        print(f"Cross-dataset evaluation on {ds_name}...")
        cfg = copy.deepcopy(base_cfg)
        cfg["dataset"]["name"] = ds_name
        
        loader = build_retina_dataloader(cfg, split="test")
        metrics = evaluate_classification(model, loader, device)
        results[ds_name] = metrics
        
    return results

def generate_adversarial_examples(model, images, labels, eps=0.01, steps=10, alpha=0.002):
    """
    Simple PGD attack for medical image robustness evaluation.
    """
    model.eval()
    adv_images = images.clone().detach()
    adv_images.requires_grad = True
    
    # Initial random noise
    adv_images.data = adv_images.data + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images.data = torch.clamp(adv_images.data, 0, 1)
    
    for i in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        logits = outputs if not isinstance(outputs, dict) else outputs["logits"]
        
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
    return adv_images

def evaluate_adversarial_robustness(model, dataloader, device, eps=0.01):
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Adversarial (PGD) Test"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Clean Eval
        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs if not isinstance(outputs, dict) else outputs["logits"], dim=1)
            correct_clean += (preds == labels).sum().item()
            
        # Adv Eval
        adv_images = generate_adversarial_examples(model, images, labels, eps=eps)
        with torch.no_grad():
            outputs_adv = model(adv_images)
            preds_adv = torch.argmax(outputs_adv if not isinstance(outputs_adv, dict) else outputs_adv["logits"], dim=1)
            correct_adv += (preds_adv == labels).sum().item()
            
        total += labels.size(0)
        
    return {
        "clean_acc": correct_clean / total,
        "adv_acc": correct_adv / total,
        "robustness_drop": (correct_clean - correct_adv) / total
    }
