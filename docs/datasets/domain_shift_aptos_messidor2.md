# Domain Shift Analysis: APTOS 2019 vs MESSIDOR-2

This report quantifies and discusses the domain shift between the Indian (APTOS) and French (MESSIDOR-2) fundus image datasets.

## Population & Geography Shift
- **APTOS**: South Asian population, India. Diverse clinical environments.
- **MESSIDOR-2**: European population, France. Highly standardized screening environments.

## Visual & Quality Comparison
*Note: Plots will be generated into `experiments/eda/domain_shift/`.*

### Resolution & Field-of-View
- APTOS images vary significantly in aspect ratio and padding.
- MESSIDOR-2 images are generally 2240x1488 (though often resized in preprocessing) and have consistent circular masking.

### Color & Illumination
- Indian fundus images (APTOS) often show darker pigmentation due to higher melanin levels in the Retinal Pigment Epithelium (RPE).
- European fundus images (MESSIDOR-2) typically exhibit lighter, more orange/red pigmentation.
- Mean RGB distributions show a shift in the G and B channels between the two domains.

### Quality Distribution
- Preliminary analysis suggests that APTOS has a larger tail of "Poor" quality images (blur and artifacts) compared to the well-controlled MESSIDOR-2 set.

## Implications for RetinaViT

1. **Protocol Generalization**: A model optimized for the high-quality, high-contrast lesions in MESSIDOR-2 may fail on the noisier APTOS samples.
2. **Feature Robustness**: By using quality-aware training, RetinaViT aims to "ignore" or weigh less the samples where the domain shift is dominated by sensor noise or poor lighting.
3. **Prototype Stability**: Few-shot prototypes must be learned from diverse samples to capture the "canonical" appearance of lesions across different retinal pigmentations.

## Reference Plots
- `res_comparison.png`: Histogram of image dimensions.
- `mean_brightness_comparison.png`: Density plot of mean pixel intensities.
- `quality_distribution_comparison.png`: Stacked bar chart of heuristic quality labels.
