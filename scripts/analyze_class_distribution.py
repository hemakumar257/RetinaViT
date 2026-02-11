import os
import matplotlib.pyplot as plt
import seaborn as sns
from retinavit.data.analysis import compute_class_distribution
from retinavit.utils.config import load_config

def main():
    # Attempt to load configs to get paths
    try:
        cfg = load_config(dataset="aptos", model="vit_base", training="standard")
        data_root = cfg["paths"]["data_dir"]
    except Exception:
        data_root = "datasets"

    datasets = ["aptos2019", "messidor2", "idrid"]
    output_dir = "experiments/eda/class_distribution"
    os.makedirs(output_dir, exist_ok=True)

    for ds_name in datasets:
        print(f"Analyzing class distribution for {ds_name}...")
        ds_dir = os.path.join(data_root, "raw", ds_name)
        dist = compute_class_distribution(ds_name, ds_dir)
        
        if dist is not None:
            # Save CSV
            dist.to_csv(os.path.join(output_dir, f"{ds_name}_dist.csv"), index=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=dist, x="class", y="count")
            plt.title(f"Class Distribution: {ds_name}")
            plt.ylabel("Number of Images")
            plt.xlabel("Class / Grade")
            plt.savefig(os.path.join(output_dir, f"{ds_name}_dist.png"))
            plt.close()
            print(f"Successfully analyzed {ds_name}.")
        else:
            print(f"Could not load labels for {ds_name}. Ensure raw data and CSVs are present.")

if __name__ == "__main__":
    main()
