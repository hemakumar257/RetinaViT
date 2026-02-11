import torch
import numpy as np
import cv2
import os
from typing import Dict, Any, Optional, Tuple

def create_lesion_overlay(
    image: np.ndarray, 
    saliency_map: np.ndarray, 
    segmentation_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Combines saliency and segmentation into a color-coded overlay.
    - Saliency: Red/Yellow heatmap
    - Segmentation: Green contours or solid fill
    """
    # 1. Resize saliency to image size
    h, w = image.shape[:2]
    saliency_map = cv2.resize(saliency_map, (w, h))
    saliency_map = np.uint8(255 * saliency_map)
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Apply saliency overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    # 2. Add segmentation if available
    if segmentation_mask is not None:
        segmentation_mask = cv2.resize(segmentation_mask, (w, h))
        mask_binary = (segmentation_mask > 0.5).astype(np.uint8)
        
        # Color the mask green
        green_mask = np.zeros_like(image)
        green_mask[:, :, 1] = 255
        
        # Blend mask
        overlay = np.where(mask_binary[:, :, None] == 1, 
                           cv2.addWeighted(overlay, 0.6, green_mask, 0.4, 0), 
                           overlay)
        
        # Add contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
    return overlay

def generate_clinical_report(results: Dict[str, Any]) -> str:
    """
    Template-based report generator for clinical interpretability.
    """
    grade = results.get("predicted_grade", 0)
    confidence = results.get("confidence", 0.0)
    quality = results.get("quality_label", "Unknown")
    
    dr_names = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
    outcome = dr_names.get(grade, "Unknown")
    
    report = f"RetinaViT Automated Clinical Analysis Report\n"
    report += "="*40 + "\n"
    report += f"Predicted Diagnosis: {outcome} (Grade {grade})\n"
    report += f"Model Confidence: {confidence:.2%}\n"
    report += f"Image Quality: {quality}\n\n"
    
    # Saliency analysis
    has_hotspots = results.get("has_hotspots", False)
    if has_hotspots:
        report += "Evidence Analysis:\n"
        report += "- Significant attention detected in regions corresponding to potential retinal lesions.\n"
        if results.get("segmentation_detected", False):
            lesion_count = results.get("lesion_count", 0)
            report += f"- Segmentation head identified ~{lesion_count} discrete lesion candidates.\n"
    else:
        report += "Evidence Analysis: No major pathological features identified by the model.\n"
        
    if quality == "Poor":
        report += "\n[WARNING] Note: Image quality is POOR. Clinical confidence in this automated result should be reduced.\n"
        
    return report

def explain_decision_difference(
    explainer, 
    image: torch.Tensor, 
    class_a: int, 
    class_b: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates discriminative saliency maps between two classes.
    """
    map_a = explainer.compute_grad_cam(image, target_class=class_a)
    map_b = explainer.compute_grad_cam(image, target_class=class_b)
    
    # Difference map (what makes it class_a more than class_b?)
    diff_map = np.maximum(0, map_a - map_b)
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    
    return map_a, map_b, diff_map
