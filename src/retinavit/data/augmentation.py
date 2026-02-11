import numpy as np
import cv2
import random
from typing import Dict, Any, Optional

class MedicalAugmentor:
    """
    Implements anatomical-aware and pathological augmentations for fundus images.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("augmentation", {})
        self.enable_medical_aug = self.cfg.get("enable_medical_aug", False)
        self.exudate_prob = self.cfg.get("simulated_exudates_prob", 0.0)
        self.artifact_prob = self.cfg.get("camera_artifacts_prob", 0.0)

    def attract_exudates(self, img: np.ndarray) -> np.ndarray:
        """Simulates bright exudate-like spots."""
        if random.random() > self.exudate_prob:
            return img
        
        out = img.copy()
        h, w = img.shape[:2]
        num_spots = random.randint(5, 15)
        
        for _ in range(num_spots):
            # Focus on central-ish region where exudates often appear
            cx = random.randint(int(w*0.2), int(w*0.8))
            cy = random.randint(int(h*0.2), int(h*0.8))
            axes = (random.randint(2, 6), random.randint(2, 6))
            angle = random.randint(0, 360)
            
            # Bright yellowish/white spots
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(150, 220))
            cv2.ellipse(out, (cx, cy), axes, angle, 0, 360, color, -1)
            # Add slight blur to make it realistic
            out[cy-10:cy+10, cx-10:cx+10] = cv2.GaussianBlur(out[cy-10:cy+10, cx-10:cx+10], (5, 5), 0)
            
        return out

    def apply_vignetting(self, img: np.ndarray) -> np.ndarray:
        """Simulates radial darkening toward corners."""
        if random.random() > self.artifact_prob:
            return img
            
        h, w = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w/2)
        kernel_y = cv2.getGaussianKernel(h, h/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        # Apply strength
        strength = random.uniform(0.5, 0.8)
        vignette = np.copy(img)
        for i in range(3):
            vignette[:,:,i] = vignette[:,:,i] * (mask * strength + (1-strength))
            
        return vignette.astype(np.uint8)

    def apply_anatomical_shift(self, img: np.ndarray) -> np.ndarray:
        """Simulates slight misalignment by shifting the fundus center."""
        if random.random() > self.cfg.get("anatomical_shift_prob", 0.0):
            return img
        
        h, w = img.shape[:2]
        shift_x = random.randint(int(-w*0.05), int(w*0.05))
        shift_y = random.randint(int(-h*0.05), int(h*0.05))
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    def apply_dust_spots(self, img: np.ndarray) -> np.ndarray:
        """Simulates small dust/spot artifacts on the camera lens."""
        if random.random() > self.artifact_prob:
            return img
            
        out = img.copy()
        num_spots = random.randint(2, 6)
        for _ in range(num_spots):
            r = random.randint(2, 5)
            # Peripheral placement
            cx = random.choice([random.randint(0, int(out.shape[1]*0.2)), random.randint(int(out.shape[1]*0.8), out.shape[1])])
            cy = random.choice([random.randint(0, int(out.shape[0]*0.2)), random.randint(int(out.shape[0]*0.8), out.shape[0])])
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            cv2.circle(out, (cx, cy), r, color, -1)
            
        return out

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Main augmentation entry point."""
        if not self.enable_medical_aug:
            return image
            
        out = image.copy()
        out = self.apply_anatomical_shift(out)
        out = self.attract_exudates(out)
        out = self.apply_vignetting(out)
        out = self.apply_dust_spots(out)
        return out
