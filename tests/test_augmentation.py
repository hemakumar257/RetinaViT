import numpy as np
import pytest
from retinavit.data.augmentation import MedicalAugmentor

def test_medical_augmentor_shape():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    config = {
        "augmentation": {
            "enable_medical_aug": True,
            "simulated_exudates_prob": 1.0,
            "camera_artifacts_prob": 1.0
        }
    }
    augmentor = MedicalAugmentor(config)
    aug_img = augmentor.augment(img)
    
    assert aug_img.shape == img.shape
    assert aug_img.dtype == img.dtype

def test_medical_augmentor_disabled():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    config = {"augmentation": {"enable_medical_aug": False}}
    augmentor = MedicalAugmentor(config)
    aug_img = augmentor.augment(img)
    
    assert np.array_equal(img, aug_img)

def test_exudates_pixel_change():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    config = {
        "augmentation": {
            "enable_medical_aug": True,
            "simulated_exudates_prob": 1.0,
            "camera_artifacts_prob": 0.0
        }
    }
    augmentor = MedicalAugmentor(config)
    aug_img = augmentor.augment(img)
    
    # Pixel sum should increase because exudates are bright
    assert np.sum(aug_img) > 0
