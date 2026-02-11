# RetinaViT: A Robust Vision Transformer for Quality-Aware Diabetic Retinopathy Grading

## Abstract
Diabetic Retinopathy (DR) remains a leading cause of blindness globally. While deep learning has shown promise in automated DR grading, existing models often fail in real-world clinical settings where image quality is inconsistent due to the use of handheld/smartphone fundus cameras. We present **RetinaViT**, a novel Vision Transformer architecture that utilizes multi-task learning and quality-conditioning to provide robust diagnostic performance. Our model integrates a QualityAware encoder, a ProtoViT module for few-shot adaptation to new camera hardware, and a comprehensive explainability suite.

## 1. Methodology
- **Architecture:** ViT-Base with quality embedding injection in the patch-embedding layer.
- **Multi-task Learning:** Joint training for DR classification and lesion segmentation (Microaneurysms, Exudates).
- **Quality Adaptation:** Episodic training using ProtoViT to adapt to low-shot data from novel domains.

## 2. Experimental Setup
We evaluated RetinaViT on a unified benchmark consisting of APTOS 2019, IDRiD, and MESSIDOR-2 datasets, alongside a custom `SmartphoneSimulator` to bridge the gap between high-end table-top cameras and mobile devices.

## 3. Results
RetinaViT achieved a **QWK of 0.89** across the unified test set, outperforming several CNN-based baselines. Notably, the quality-conditioning module improved accuracy on low-light and blurred images by 12% compared to standard ViT.

## 4. Conclusion
RetinaViT demonstrates the feasibility of deploying transformer-based screening tools in resource-limited environments. Future work will focus on longitudinal modeling with our `TemporalRetinaModule`.

---
*Authors: RetinaViT Project Team*
*Keywords: Vision Transformer, Diabetic Retinopathy, Medical AI, Few-shot Learning.*
