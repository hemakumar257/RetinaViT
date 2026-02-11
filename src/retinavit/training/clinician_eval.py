import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from typing import Dict, Any

from .calibration import compute_ece, get_reliability_stats

def analyze_critical_errors(y_true, y_pred):
    """
    Severe DR missed as Normal/Mild is a critical error in screening.
    """
    critical_errors = 0
    total_severe = 0
    
    # Assuming grades 0,1 are mild/normal, 3,4 are severe
    severe_mask = (y_true >= 3)
    critical_mask = (y_true >= 3) & (y_pred <= 1)
    
    critical_errors = np.sum(critical_mask)
    total_severe = np.sum(severe_mask)
    
    rate = critical_errors / (total_severe + 1e-6)
    return {
        "critical_error_count": int(critical_errors),
        "total_severe_cases": int(total_severe),
        "critical_error_rate": float(rate)
    }

def generate_clinician_report(results: Dict[str, Any], output_path: str):
    """
    Generates a clinician-facing Markdown report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    content = f"""# RetinaViT Clinical Evaluation Report

## 1. Overall Diagnostic Performance
| Metric | Value | 95% CI |
| :--- | :--- | :--- |
| Accuracy | {results.get('accuracy', 0):.4f} | {results.get('accuracy_ci', (0,0))} |
| Quadratic Kappa | {results.get('kappa', 0):.4f} | {results.get('kappa_ci', (0,0))} |
| AUC (OVR) | {results.get('auc_macro', 0):.4f} | {results.get('auc_ci', (0,0))} |

## 2. Safety & Critical Errors
> [!IMPORTANT]
> Critical errors are defined as Severe/Proliferative DR (Grade 3-4) misclassified as No DR or Mild DR (Grade 0-1).

- **Critical Underdiagnosis Rate**: {results.get('critical_error_rate', 0):.2%}
- **Severe Cases Missed**: {results.get('critical_error_count', 0)} / {results.get('total_severe_cases', 0)}

## 3. Decision Support & Calibration
- **Expected Calibration Error (ECE)**: {results.get('ece', 0):.4f}
- **Interpretation**: A lower ECE indicates that the model's confidence scores are more reliable indicators of actual accuracy.

## 4. Robustness to Image Quality
| Quality | Accuracy |
| :--- | :--- |
| Good | {results.get('quality_good', {}).get('accuracy', 0):.4f} |
| Fair | {results.get('quality_fair', {}).get('accuracy', 0):.4f} |
| Poor | {results.get('quality_poor', {}).get('accuracy', 0):.4f} |

---
*Report generated automatically by RetinaViT Evaluation Pipeline.*
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Clinical report saved to {output_path}")

def evaluate_clinical(model, dataloader, device, output_path="experiments/eval/clinical_summary.md"):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_quality = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            logits = outputs if not isinstance(outputs, dict) else outputs["logits"]
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_quality.extend(batch.get("quality_label", ["Unknown"]*len(labels)))
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    # 1. Critical Analysis
    critical = analyze_critical_errors(y_true, y_pred)
    
    # 2. Calibration
    ece = compute_ece(y_prob, y_true)
    
    # 3. Stratified
    from .metrics import get_stratified_metrics, compute_metrics
    stratified = get_stratified_metrics(y_true, y_pred, all_quality, y_prob)
    base = compute_metrics(y_true, y_pred, y_prob)
    
    results = {**base, **critical, "ece": ece}
    for k, v in stratified.items():
        results[k] = v
        
    # Generate Markdown
    generate_clinician_report(results, output_path)
    return results
