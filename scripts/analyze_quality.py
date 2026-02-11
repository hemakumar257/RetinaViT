import os
import pandas as pd
from tqdm import tqdm
from retinavit.data.quality import compute_quality_metrics
from retinavit.utils.config import load_config

def main():
    try:
        cfg = load_config(dataset="aptos", model="vit_base", training="standard")
        data_root = cfg["paths"]["data_dir"]
    except Exception:
        data_root = "datasets"

    datasets = ["aptos2019", "messidor2", "idrid"]
    output_dir = "experiments/eda/quality"
    os.makedirs(output_dir, exist_ok=True)

    # We limit images for quick analysis if dataset is huge
    MAX_IMAGES_PER_DS = 500

    for ds_name in datasets:
        print(f"Analyzing image quality for {ds_name} (max {MAX_IMAGES_PER_DS} images)...")
        ds_dir = os.path.join(data_root, "raw", ds_name)
        
        if not os.path.exists(ds_dir):
            print(f"Directory {ds_dir} missing. Skipping.")
            continue

        image_files = []
        for root, _, files in os.walk(ds_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, f))
        
        if not image_files:
            print(f"No images found in {ds_dir}. Skipping.")
            continue

        # Sample subset
        subset = image_files[:MAX_IMAGES_PER_DS]
        
        results = []
        for img_path in tqdm(subset):
            metrics = compute_quality_metrics(img_path)
            metrics["filename"] = os.path.basename(img_path)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, f"{ds_name}_quality.csv"), index=False)
        print(f"Quality metrics saved for {ds_name}.")

if __name__ == "__main__":
    main()
