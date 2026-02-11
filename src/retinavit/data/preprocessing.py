import cv2
import numpy as np
from typing import Dict, Any, Optional

class FundusPreprocessor:
    """
    Standardizes fundus images through circular cropping, illumination normalization (CLAHE),
    and color consistency (histogram matching).
    """
    
    def __init__(
        self,
        target_size: tuple = (512, 512),
        apply_circular_crop: bool = True,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple = (8, 8),
        apply_hist_match: bool = False,
        reference_image_path: Optional[str] = None
    ):
        self.target_size = target_size
        self.apply_circular_crop = apply_circular_crop
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        self.apply_hist_match = apply_hist_match
        self.reference_image = None
        if apply_hist_match and reference_image_path:
            self.reference_image = cv2.imread(reference_image_path)

    def circular_crop(self, img: np.ndarray) -> np.ndarray:
        """Detects the fundus circle and crops away black borders."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding to find the fundus
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
            
        # Get the largest contour (the fundus)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop
        crop = img[y:y+h, x:x+w]
        return crop

    def apply_multiscale_clahe(self, img: np.ndarray) -> np.ndarray:
        """Applies CLAHE to the Luminance channel in LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def match_histogram(self, img: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Matches histogram of source image to reference image."""
        # Simplified implementation for efficiency
        from skimage import exposure
        matched = exposure.match_histograms(img, reference, channel_axis=-1)
        return matched.astype(np.uint8)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline."""
        out = image.copy()
        
        # 1. Circular Crop
        if self.apply_circular_crop:
            out = self.circular_crop(out)
            
        # 2. Resizing
        out = cv2.resize(out, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 3. CLAHE
        if self.apply_clahe:
            out = self.apply_multiscale_clahe(out)
            
        # 4. Histogram Matching
        if self.apply_hist_match and self.reference_image is not None:
            out = self.match_histogram(out, self.reference_image)
            
        return out
