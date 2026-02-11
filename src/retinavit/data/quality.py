import cv2
import numpy as np
from typing import Dict, Any

def compute_blur_laplacian(image: np.ndarray) -> float:
    """Computes the variance of the Laplacian as a sharpness/blur metric."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_illumination_stats(image: np.ndarray) -> Dict[str, float]:
    """Computes basic brightness statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 5  # Ignore black borders
    if not np.any(mask):
        return {"mean_brightness": 0.0, "std_brightness": 0.0}
    
    roi = gray[mask]
    return {
        "mean_brightness": float(np.mean(roi)),
        "std_brightness": float(np.std(roi))
    }

def compute_contrast_rms(image: np.ndarray) -> float:
    """Computes RMS contrast of the luminance channel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    if not np.any(mask):
        return 0.0
    roi = gray[mask].astype(float) / 255.0
    return float(np.std(roi))

def compute_saturated_pixels(image: np.ndarray) -> Dict[str, float]:
    """Computes percentage of saturated (dark/bright) pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    if not np.any(mask):
        return {"perc_dark": 0.0, "perc_bright": 0.0}
    
    roi = gray[mask]
    total_pixels = roi.size
    dark_pixels = np.sum(roi < 10)
    bright_pixels = np.sum(roi > 245)
    
    return {
        "perc_dark": float(dark_pixels / total_pixels),
        "perc_bright": float(bright_pixels / total_pixels)
    }

def compute_quality_summary(image_path: str) -> Dict[str, Any]:
    """Orchestrates all quality metrics for a single image."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}
    
    metrics = {}
    metrics["blur_score"] = compute_blur_laplacian(img)
    illum = compute_illumination_stats(img)
    metrics["illum_mean"] = illum["mean_brightness"]
    metrics["illum_std"] = illum["std_brightness"]
    metrics["contrast_score"] = compute_contrast_rms(img)
    saturated = compute_saturated_pixels(img)
    metrics["perc_dark"] = saturated["perc_dark"]
    metrics["perc_bright"] = saturated["perc_bright"]
    
    # Assign Label
    metrics["quality_label"] = assign_quality_label(
        metrics["blur_score"], 
        metrics["illum_mean"], 
        metrics["contrast_score"]
    )
        
    return metrics

def assign_quality_label(blur: float, illum: float, contrast: float) -> str:
    """Assigns Good/Fair/Poor label based on heuristic thresholds."""
    # Thresholds based on empirical observation of fundus datasets
    # Poor: heavily blurred, extreme exposure, or very low contrast
    if blur < 60 or illum < 30 or illum > 220 or contrast < 0.04:
        return "Poor"
    # Fair: partially blurred or suboptimal exposure
    elif blur < 120 or illum < 50 or illum > 200 or contrast < 0.08:
        return "Fair"
    # Good: sharp and well-exposed
    else:
        return "Good"
