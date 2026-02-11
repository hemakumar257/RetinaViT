import numpy as np
import cv2
import random
from typing import Dict, Any

class SmartphoneSimulator:
    """
    Mimics smartphone-based fundus capture degradations.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("smartphone_simulation", {})
        self.enable = self.cfg.get("enable", False)
        self.blur_prob = self.cfg.get("motion_blur_prob", 0.0)
        self.noise_prob = self.cfg.get("low_light_prob", 0.0)
        self.jpeg_prob = self.cfg.get("jpeg_artifacts_prob", 0.0)

    def apply_motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Simulates hand tremor using linear kernels."""
        if random.random() > self.blur_prob:
            return img
            
        size = random.randint(5, 12)
        kernel = np.zeros((size, size))
        angle = random.uniform(0, np.pi)
        
        # Create linear kernel
        pt1 = (int(size/2 - size/2 * np.cos(angle)), int(size/2 - size/2 * np.sin(angle)))
        pt2 = (int(size/2 + size/2 * np.cos(angle)), int(size/2 + size/2 * np.sin(angle)))
        cv2.line(kernel, pt1, pt2, 1, 1)
        kernel /= kernel.sum()
        
        return cv2.filter2D(img, -1, kernel)

    def add_low_light_noise(self, img: np.ndarray) -> np.ndarray:
        """Adds Poisson-Gaussian noise and dims the image."""
        if random.random() > self.noise_prob:
            return img
            
        # Dimming
        alpha = random.uniform(0.6, 0.9)
        dimmed = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        # Gaussian Noise
        noise = np.random.normal(0, random.uniform(5, 15), dimmed.shape).astype(np.float32)
        noisy = dimmed.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Degrades image quality via JPEG re-encoding."""
        if random.random() > self.jpeg_prob:
            return img
            
        quality = random.randint(10, 40)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

    def apply_offcenter_crop(self, img: np.ndarray) -> np.ndarray:
        """Simulates patient movement by shifting the crop window."""
        if random.random() > self.cfg.get("offcenter_crop_prob", 0.0):
            return img
        
        h, w = img.shape[:2]
        # Shift crop by up to 10%
        shift_x = random.randint(int(-w*0.1), int(w*0.1))
        shift_y = random.randint(int(-h*0.1), int(h*0.1))
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    def simulate(self, image: np.ndarray) -> np.ndarray:
        """Main simulation entry point."""
        if not self.enable:
            return image
            
        out = image.copy()
        out = self.apply_offcenter_crop(out)
        out = self.apply_motion_blur(out)
        out = self.add_low_light_noise(out)
        out = self.apply_jpeg_compression(out)
        return out
