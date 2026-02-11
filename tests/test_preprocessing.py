import numpy as np
import cv2
import pytest
from retinavit.data.preprocessing import FundusPreprocessor

def test_circular_crop():
    # Create a synthetic fundus-like image: white circle on black background
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(img, (250, 250), 200, (255, 255, 255), -1)
    
    preprocessor = FundusPreprocessor(apply_circular_crop=True)
    cropped = preprocessor.circular_crop(img)
    
    # The bounding box of a circle with center (250, 250) and radius 200
    # should be roughly (50, 50, 400, 400)
    assert cropped.shape[0] <= 410 and cropped.shape[0] >= 390
    assert cropped.shape[1] <= 410 and cropped.shape[1] >= 390

def test_clahe_effect():
    # Create low contrast image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.circle(img, (50, 50), 20, (135, 135, 135), -1)
    
    preprocessor = FundusPreprocessor(apply_clahe=True)
    processed = preprocessor.apply_multiscale_clahe(img)
    
    # Contrast should increase: std of luminance should increase
    orig_std = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    proc_std = np.std(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))
    
    assert proc_std > orig_std

def test_preprocessing_pipeline():
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(img, (250, 250), 200, (255, 255, 255), -1)
    
    preprocessor = FundusPreprocessor(target_size=(256, 256))
    out = preprocessor.preprocess_image(img)
    
    assert out.shape == (256, 256, 3)
