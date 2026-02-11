import numpy as np
import pytest
from retinavit.data.smartphone_simulation import SmartphoneSimulator

def test_smartphone_simulator_shape():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    config = {
        "smartphone_simulation": {
            "enable": True,
            "motion_blur_prob": 1.0,
            "low_light_prob": 1.0,
            "jpeg_artifacts_prob": 1.0
        }
    }
    simulator = SmartphoneSimulator(config)
    sim_img = simulator.simulate(img)
    
    assert sim_img.shape == img.shape
    assert sim_img.dtype == img.dtype

def test_smartphone_simulator_disabled():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    config = {"smartphone_simulation": {"enable": False}}
    simulator = SmartphoneSimulator(config)
    sim_img = simulator.simulate(img)
    
    assert np.array_equal(img, sim_img)

def test_jpeg_artifacts_change():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    config = {
        "smartphone_simulation": {
            "enable": True,
            "motion_blur_prob": 0.0,
            "low_light_prob": 0.0,
            "jpeg_artifacts_prob": 1.0
        }
    }
    simulator = SmartphoneSimulator(config)
    sim_img = simulator.simulate(img)
    
    # JPEG compression changes pixels
    assert not np.array_equal(img, sim_img)
