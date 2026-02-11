import torch
import numpy as np
from retinavit.explainability.clinical_explanations import generate_clinical_report, create_lesion_overlay

def test_report_generation():
    results = {
        "predicted_grade": 3,
        "confidence": 0.85,
        "quality_label": "Fair",
        "has_hotspots": True
    }
    report = generate_clinical_report(results)
    assert "Severe NPDR" in report
    assert "85.00%" in report

def test_overlay_creation():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    sal = np.random.rand(10, 10)
    mask = np.zeros((100, 100))
    mask[40:60, 40:60] = 1.0
    
    overlay = create_lesion_overlay(img, sal, mask)
    assert overlay.shape == (100, 100, 3)
    # Check green color in middle
    assert np.any(overlay[50, 50, 1] > 0)

if __name__ == "__main__":
    test_report_generation()
    test_overlay_creation()
    print("Clinical explanation tests passed!")
