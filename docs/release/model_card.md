# RetinaViT Model Card

## Model Details
- **Developed by:** RetinaViT Team
- **Model date:** 2026-02
- **Model type:** Vision Transformer (ViT-Base-16) with Quality conditioning.
- **Task:** Diabetic Retinopathy Grading (5-class), Lesion Segmentation, and Few-shot quality adaptation.

## Intended Use
- **Primary intended uses:** Clinical decision support for ophthalmologists in resource-constrained environments.
- **Primary intended users:** Trained healthcare professionals.
- **Out-of-scope use cases:** Autonomous diagnosis without human validation; non-retinal image classification.

## Factors
- **Image Quality:** Model performance is heavily dependent on fundus image clarity.
- **Demographics:** Potential variations in fundus pigmentation across ethnicities.

## Metrics
- **Classification:** Quadratic Weighted Kappa (QWK), Accuracy, Macro-F1.
- **Segmentation:** IoU, Dice Score.
- **Fairness:** Impact Ratio per Subgroup.

## Training Data
- Aggregated data from **APTOS 2019**, **IDRiD**, and **MESSIDOR-2**.
- Synthetic augmentations simulating mobile fundus cameras.

## Quantitative Analyses
| Metric | Full Test Set | Good Quality | Poor Quality |
| :--- | :--- | :--- | :--- |
| **QWK** | 0.89 | 0.94 | 0.72 |
| **Accuracy** | 86.5% | 91.2% | 75.1% |

## Ethical Considerations
- The model is trained on publicly available datasets that may have intrinsic biases. 
- Clinical use requires confirmation of system calibration in the local environment.

## Caveats and Recommendations
- **Always review the saliency maps (Attention Rollout/Grad-CAM).**
- Re-acquire images labeled as "Poor" quality.
- System is not yet approved as a standalone diagnostic medical device.
