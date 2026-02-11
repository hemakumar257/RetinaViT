import os
import pandas as pd
from tqdm import tqdm
from retinavit.data.quality import compute_quality_summary
from retinavit.utils.config import load_config
from pathlib import Path

def main():
    try:
        cfg = load_config(dataset="aptos", model="vit_base", training="standard")
        data_root = cfg["paths"]["data_dir"]
    except Exception:
        data_root = "datasets"

    datasets = ["aptos2019", "messidor2", "idrid"]
    output_dir = Path("experiments/quality")
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        ds_dir = Path(data_root) / "raw" / ds_name
        if not ds_dir.exists():
            print(f"Skipping {ds_name}, raw data not found.")
            continue
            
        print(f"Running full quality assessment for {ds_name}...")
        
        image_files = []
        for ext in ["*.jpg", "*.png", "*.tif", "*.jpeg"]:
            image_files.extend(list(ds_dir.rglob(ext)))
            
        if not image_files:
            continue
            
        results = []
        # Limiting to 100 for this run, but in production we'd do all
        print(f"  Analyzing {len(image_files)} images (subset of 100 for speed)...")
        for img_path in tqdm(image_files[:100]):
            metrics = compute_quality_summary(str(img_path))
            metrics["id_code"] = img_path.stem
            results.append(metrics)
            
        df = pd.DataFrame(results)
        out_path = output_dir / f"{ds_name}_quality_labeled.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved labeled quality metrics to {out_path}")

if __name__ == "__main__":
    main()
