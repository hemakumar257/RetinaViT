import numpy as np
import cv2
import pytest
import os
from retinavit.data.quality import compute_quality_metrics, compute_blur_laplacian

def test_blur_synthetic():
    # Create sharp image (random noise)
    sharp_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # Create blurred image
    blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)
    
    sharp_score = compute_blur_laplacian(sharp_img)
    blurred_score = compute_blur_laplacian(blurred_img)
    
    assert sharp_score > blurred_score, f"Sharp score ({sharp_score}) should be > blurred score ({blurred_score})"

def test_quality_metrics_on_file(tmp_path):
    # Save a dummy image
    img_path = str(tmp_path / "test_img.png")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)  # White circle
    cv2.imwrite(img_path, img)
    
    metrics = compute_quality_metrics(img_path)
    
    assert "blur_sharpness" in metrics
    assert "mean_brightness" in metrics
    assert "contrast_rms" in metrics
    assert "quality_label" in metrics
    assert metrics["mean_brightness"] > 0
    assert metrics["contrast_rms"] > 0

def test_invalid_path():
    metrics = compute_quality_metrics("non_existent.jpg")
    assert "error" in metrics
