import pytest
from retinavit.data.quality import assign_quality_label

def test_quality_labels():
    # Good image
    assert assign_quality_label(blur=200, illum=128, contrast=0.15) == "Good"
    
    # Fair image (low-ish contrast)
    assert assign_quality_label(blur=200, illum=128, contrast=0.06) == "Fair"
    
    # Poor image (very blurry)
    assert assign_quality_label(blur=40, illum=128, contrast=0.15) == "Poor"
    
    # Poor image (extreme exposure)
    assert assign_quality_label(blur=200, illum=240, contrast=0.15) == "Poor"
    assert assign_quality_label(blur=200, illum=10, contrast=0.15) == "Poor"

def test_boundary_conditions():
    # Right on the edge
    assert assign_quality_label(blur=60, illum=30, contrast=0.04) == "Fair" # Boundary check
    assert assign_quality_label(blur=59, illum=30, contrast=0.04) == "Poor"
