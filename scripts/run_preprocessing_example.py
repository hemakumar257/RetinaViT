import os
import cv2
from retinavit.data.preprocessing import FundusPreprocessor
from retinavit.utils.config import load_config
from pathlib import Path

def main():
    try:
        cfg = load_config(dataset="aptos", model="vit_base", training="standard")
        data_root = cfg["paths"]["data_dir"]
    except Exception:
        data_root = "datasets"

    preprocessor = FundusPreprocessor(target_size=(512, 512), apply_clahe=True)
    datasets = ["aptos2019", "messidor2", "idrid"]
    
    for ds_name in datasets:
        ds_dir = Path(data_root) / "raw" / ds_name
        if not ds_dir.exists():
            continue
            
        print(f"Processing examples for {ds_name}...")
        
        # Get first 3 images
        images = []
        for ext in ["*.jpg", "*.png", "*.tif", "*.jpeg"]:
            images.extend(list(ds_dir.rglob(ext)))
            if len(images) >= 3:
                break
        
        if not images:
            continue
            
        output_dir = Path(data_root) / "processed" / ds_name / "examples"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(images[:3]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            processed = preprocessor.preprocess_image(img)
            
            # Save original and processed for comparison
            out_name = f"example_{i}_processed.png"
            cv2.imwrite(str(output_dir / out_name), processed)
            print(f"  Saved {out_name} to {output_dir}")

if __name__ == "__main__":
    main()
