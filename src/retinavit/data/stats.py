import os
import pandas as pd
from pathlib import Path
from PIL import Image

def get_dataset_stats(dataset_name, data_dir):
    """Computes basic statistics for a dataset."""
    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        return None

    stats = {
        "name": dataset_name,
        "num_images": 0,
        "resolutions": [],
        "classes": {}
    }

    # Count images and get resolutions
    for file in data_path.rglob("*"):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            stats["num_images"] += 1
            with Image.open(file) as img:
                stats["resolutions"].append(img.size)

    # Note: Class counts would typically come from a CSV file.
    # This is a simplified version.
    
    return stats

def format_stats_markdown(stats_list):
    """Formats dataset statistics into a Markdown table."""
    markdown = "# Dataset Statistics Summary\n\n"
    
    if not stats_list:
        return markdown + "No datasets found. Please download data first.\n"

    for stats in stats_list:
        markdown += f"## {stats['name']}\n"
        markdown += "| Metric | Value |\n"
        markdown += "| --- | --- |\n"
        markdown += f"| Number of Images | {stats['num_images']} |\n"
        
        if stats["resolutions"]:
            widths = [r[0] for r in stats["resolutions"]]
            heights = [r[1] for r in stats["resolutions"]]
            markdown += f"| Min Resolution | {min(widths)}x{min(heights)} |\n"
            markdown += f"| Max Resolution | {max(widths)}x{max(heights)} |\n"
            markdown += f"| Median Resolution | {sorted(widths)[len(widths)//2]}x{sorted(heights)[len(heights)//2]} |\n"
        
        markdown += "\n"
    
    return markdown

if __name__ == "__main__":
    datasets = ["aptos2019", "messidor2", "idrid"]
    stats_list = []
    
    for ds in datasets:
        stats = get_dataset_stats(ds, f"datasets/raw/{ds}")
        if stats:
            stats_list.append(stats)
    
    summary = format_stats_markdown(stats_list)
    with open("docs/datasets/stats_summary.md", "w") as f:
        f.write(summary)
    print("Dataset statistics summary generated in docs/datasets/stats_summary.md")
