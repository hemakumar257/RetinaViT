import streamlit as st
import pandas as pd
import os
from PIL import Image
from retinavit.data.quality import compute_quality_metrics
from retinavit.utils.config import load_config

st.set_page_config(page_title="RetinaViT Dataset Explorer", layout="wide")

st.title("👁️ RetinaViT Dataset Explorer")
st.markdown("Explore class distributions and image quality metrics across datasets.")

# Load config to get data paths
try:
    cfg = load_config(dataset="aptos", model="vit_base", training="standard")
    data_root = cfg["paths"]["data_dir"]
except Exception:
    data_root = "datasets"

# Sidebar selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["aptos2019", "messidor2", "idrid"]
)

st.sidebar.divider()

# Load Class Distribution
dist_path = f"experiments/eda/class_distribution/{dataset_name}_dist.csv"
if os.path.exists(dist_path):
    st.subheader(f"Class Distribution: {dataset_name}")
    dist_df = pd.read_csv(dist_path)
    st.bar_chart(dist_df.set_index("class")["count"])
else:
    st.info(f"Class distribution data not found. Run `python scripts/analyze_class_distribution.py` first.")

# Load Quality Metrics
quality_path = f"experiments/eda/quality/{dataset_name}_quality.csv"
if os.path.exists(quality_path):
    st.subheader(f"Quality Metrics: {dataset_name}")
    q_df = pd.read_csv(quality_path)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Blur/Sharpness**")
        st.line_chart(q_df["blur_sharpness"])
    with col2:
        st.write("**Mean Brightness**")
        st.line_chart(q_df["mean_brightness"])
    with col3:
        st.write("**Contrast (RMS)**")
        st.line_chart(q_df["contrast_rms"])

    st.write("### Data Table")
    st.dataframe(q_df)
else:
    st.info(f"Quality metrics not found. Run `python scripts/analyze_quality.py` first.")

# Image Explorer
st.divider()
st.subheader("🖼️ Image Gallery & Live Analysis")

ds_dir = os.path.join(data_root, "raw", dataset_name)
if os.path.exists(ds_dir):
    image_files = []
    for root, _, files in os.walk(ds_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    if image_files:
        sample_img_path = st.selectbox("Select Image to Analyze", image_files[:20])
        img = Image.open(sample_img_path)
        
        col_img, col_metrics = st.columns([2, 1])
        with col_img:
            st.image(img, use_column_width=True, caption=os.path.basename(sample_img_path))
        
        with col_metrics:
            st.write("**Live Quality Check**")
            with st.spinner("Computing metrics..."):
                metrics = compute_quality_metrics(sample_img_path)
                st.json(metrics)
    else:
        st.warning("No images found in the raw data directory.")
else:
    st.error(f"Directory {ds_dir} not found. Please download the dataset.")
