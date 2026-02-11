import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, jaccard_score
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

from .metrics import compute_metrics

def bootstrap_metric(y_true, y_pred, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Computes a metric with a confidence interval using bootstrap resampling.
    """
    scores = []
    n_samples = len(y_true)
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        # Metric function signature: fn(y_true, y_pred, y_prob)
        score = metric_fn(y_true[indices], y_pred[indices], y_prob[indices] if y_prob is not None else None)
        scores.append(score)
        
    if not scores:
        return 0.0, (0.0, 0.0)
        
    scores = np.array(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), (lower, upper)

def evaluate_classification(model, dataloader, device, n_bootstrap=1000) -> Dict[str, Any]:
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Classification"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Extract features or handle multitask outputs
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    results = {}
    
    # 1. Base Metrics
    base_metrics = compute_metrics(y_true, y_pred, y_prob)
    results.update(base_metrics)
    
    # 2. Confidence Intervals
    def acc_fn(t, p, prob): return accuracy_score(t, p)
    def kappa_fn(t, p, prob): return cohen_kappa_score(t, p, weights="quadratic")
    def auc_fn(t, p, prob): 
        try: return roc_auc_score(t, prob, multi_class="ovr", average="macro")
        except: return 0.5

    results["accuracy_mean"], results["accuracy_ci"] = bootstrap_metric(y_true, y_pred, y_prob, acc_fn, n_bootstrap)
    results["kappa_mean"], results["kappa_ci"] = bootstrap_metric(y_true, y_pred, y_prob, kappa_fn, n_bootstrap)
    results["auc_mean"], results["auc_ci"] = bootstrap_metric(y_true, y_pred, y_prob, auc_fn, n_bootstrap)
    
    return results

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = (y_pred > 0.5).astype(float).flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def compute_hd95(y_true, y_pred):
    """
    Computes the 95th percentile Hausdorff Distance.
    y_true, y_pred: [H, W] or [D, H, W]
    """
    # Simply using undirected max of directed hausdorff for base HD
    # HD95 is more complex, here's a simplified version for small objects
    d1 = directed_hausdorff(y_true, y_pred)[0]
    d2 = directed_hausdorff(y_pred, y_true)[0]
    return max(d1, d2) # simplified, ideally use percentile on point sets

def evaluate_segmentation(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    dices = []
    ious = []
    hd95s = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Segmentation"):
            if "masks" not in batch: continue
            images = batch["image"].to(device)
            targets = batch["masks"].cpu().numpy()
            
            outputs = model(images)
            if "segmentation" not in outputs: continue
            
            preds = torch.sigmoid(outputs["segmentation"]).cpu().numpy()
            
            for i in range(len(targets)):
                t = targets[i, 0] # [H, W]
                p = preds[i, 0]   # [H, W]
                
                dices.append(dice_coefficient(t, p))
                ious.append(jaccard_score(t.flatten() > 0.5, p.flatten() > 0.5))
                # HD95 is expensive, maybe skip if mask is empty
                if t.sum() > 0 and (p > 0.5).sum() > 0:
                    hd95s.append(compute_hd95(t, p))
                    
    return {
        "dice_mean": np.mean(dices) if dices else 0.0,
        "iou_mean": np.mean(ious) if ious else 0.0,
        "hd95_mean": np.mean(hd95s) if hd95s else 0.0
    }

def evaluate_fewshot(model, episodic_loader, device, n_episodes=50) -> Dict[str, Any]:
    model.eval()
    episode_accs = []
    shot_adaptation = [] # To store (step, acc)
    
    # We simulate adaptation by taking first k-shot samples and evaluating on q-query samples
    # In ProtoViT, adaptation is often static (prototypes from support)
    # But we can simulate "learning" by evaluating on support vs query to show curve
    
    with torch.no_grad():
        for i, episode in enumerate(tqdm(episodic_loader, desc="Evaluating Few-Shot")):
            if i >= n_episodes: break
            
            images = episode["images"].to(device)
            labels = episode["labels"].to(device)
            n_support = episode["support_len"]
            
            outputs = model(
                images[:n_support], labels[:n_support],
                images[n_support:], labels[n_support:],
                support_quality={"quality_label": episode["quality_labels"][:n_support]},
                query_quality={"quality_label": episode["quality_labels"][n_support:]}
            )
            
            logits = outputs["logits"]
            query_labels = labels[n_support:]
            preds = torch.argmax(logits, dim=1)
            acc = (preds == query_labels).float().mean().item()
            episode_accs.append(acc)
            
            # Simulation of steps: if we had a fine-tuning phase
            # For pure ProtoNet, adaptation is one-step.
            # We'll just return final acc.
            
    return {
        "fs_acc_mean": np.mean(episode_accs),
        "fs_acc_std": np.std(episode_accs)
    }

def compare_classifiers(preds_a, preds_b, labels) -> Dict[str, Any]:
    """
    Statistical significance tests.
    """
    from sklearn.metrics import accuracy_score
    acc_a = accuracy_score(labels, preds_a)
    acc_b = accuracy_score(labels, preds_b)
    
    # McNemar's Test
    # [a, b]
    # [c, d]
    # a: both correct, d: both wrong
    # b: A correct, B wrong, c: A wrong, B correct
    a = np.sum((preds_a == labels) & (preds_b == labels))
    b = np.sum((preds_a == labels) & (preds_b != labels))
    c = np.sum((preds_a != labels) & (preds_b == labels))
    d = np.sum((preds_a != labels) & (preds_b != labels))
    
    chi2 = (abs(b - c) - 1)**2 / (b + c + 1e-6)
    from scipy.stats import chi2 as chi2_dist
    p_mcnemar = 1 - chi2_dist.cdf(chi2, 1)
    
    return {
        "acc_diff": acc_a - acc_b,
        "p_mcnemar": p_mcnemar,
        "contingency_table": [[int(a), int(b)], [int(c), int(d)]]
    }
