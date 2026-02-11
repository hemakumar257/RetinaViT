import os
import cv2
import numpy as np
from retinavit.data.smartphone_simulation import SmartphoneSimulator
from pathlib import Path

def main():
    config = {
        "smartphone_simulation": {
            "enable": True,
            "motion_blur_prob": 1.0,
            "low_light_prob": 1.0,
            "jpeg_artifacts_prob": 1.0
        }
    }
    simulator = SmartphoneSimulator(config)
    
    # Load sample image
    data_root = "datasets"
    ds_dir = Path(data_root) / "raw" / "aptos2019"
    img_path = next(ds_dir.rglob("*.png"), None)
    
    if img_path is None:
        print("No sample image found.")
        return
        
    img = cv2.imread(str(img_path))
    output_dir = Path("experiments/augmentation/smartphone_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate steps
    blur = simulator.apply_motion_blur(img)
    noise = simulator.add_low_light_noise(img)
    jpeg = simulator.apply_jpeg_compression(img)
    full = simulator.simulate(img)
    
    # Save panels
    cv2.imwrite(str(output_dir / "step_blur.png"), np.hstack((img, blur)))
    cv2.imwrite(str(output_dir / "step_noise.png"), np.hstack((img, noise)))
    cv2.imwrite(str(output_dir / "step_jpeg.png"), np.hstack((img, jpeg)))
    cv2.imwrite(str(output_dir / "step_full.png"), np.hstack((img, full)))
    print(f"Saved smartphone simulation examples to {output_dir}")

if __name__ == "__main__":
    main()
