import os
import cv2
import numpy as np
from retinavit.data.augmentation import MedicalAugmentor
from retinavit.utils.config import load_config
from pathlib import Path

def main():
    config = {
        "augmentation": {
            "enable_medical_aug": True,
            "simulated_exudates_prob": 1.0,
            "camera_artifacts_prob": 1.0
        }
    }
    augmentor = MedicalAugmentor(config)
    
    # Load sample image
    data_root = "datasets"
    ds_dir = Path(data_root) / "raw" / "aptos2019"
    img_path = next(ds_dir.rglob("*.png"), None)
    
    if img_path is None:
        print("No sample image found.")
        return
        
    img = cv2.imread(str(img_path))
    output_dir = Path("experiments/augmentation/medical_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 examples
    for i in range(5):
        augmented = augmentor.augment(img)
        panel = np.hstack((img, augmented))
        cv2.imwrite(str(output_dir / f"compare_{i}.png"), panel)
        print(f"Saved medical augmentation example {i}")

if __name__ == "__main__":
    main()
