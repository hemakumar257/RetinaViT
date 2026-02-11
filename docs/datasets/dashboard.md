# Dataset Dashboard Documentation

The project includes a Streamlit dashboard for interactive exploration of dataset statistics and image quality.

## Setup
Ensure you have installed the requirements:
```bash
pip install streamlit
```

## Running the Dashboard
Run the following command from the project root:
```bash
streamlit run tools/dataset_dashboard/app.py
```

## Features
- **Dataset Selection**: Switch between APTOS, MESSIDOR-2, and IDRiD via the sidebar.
- **Distribution Plots**: View class imbalance charts derived from `experiments/eda/class_distribution/`.
- **Quality Stats**: Visualize distributions of blur, illumination, and contrast metrics.
- **Live Analysis**: Select an image from the raw data to view it alongside its real-time quality metrics.

## Troubleshooting
- If no charts appear, ensure you have run the analysis scripts first:
  ```bash
  python scripts/analyze_class_distribution.py
  python scripts/analyze_quality.py
  ```
