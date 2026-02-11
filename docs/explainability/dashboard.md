# RetinaViT Explainer Dashboard

The interactive dashboard allows clinicians and researchers to interpret model decisions visually and conduct counterfactual reasoning.

## How to Run
From the project root:
```bash
streamlit run tools/explainer_dashboard/app.py
```

## Features

### 1. Unified Saliency Visualization
- **Attention Rollout**: Visualizes how information flows across Transformer layers, highlighting global areas of focus.
- **Grad-CAM**: Highlights discriminative regions using class-specific gradients, ideal for identifying specific lesions.

### 2. Clinical Overlays
- Combines saliency heatmaps with green segmentation contours (from the Multitask head) for high-precision diagnostic evidence.

### 3. ProtoViT Similarity Retrieval
- View the most similar training/support cases that influenced the model's decision.
- Understand "Why this diagnosis?" by comparing against historical cases.

### 4. Counterfactual 'What-If' Scenarios
- **Lesion Reduction**: Observe how model confidence drops as simulated lesions are erased.
- **Quality Improvement**: Estimate the impact of image denoising on diagnostic reliability.

## Deployment Notes
- The dashboard is compatible with classification (ViT), multitask, and few-shot (ProtoViT) models.
- Ensure the appropriate model checkpoint is loaded in the lateral configuration panel.
