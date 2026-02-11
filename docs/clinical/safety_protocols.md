# RetinaViT Clinical Safety Protocols & FMEA

This document outlines the failure modes and safety guardrails for the RetinaViT system in clinical practice.

## 1. Failure Mode and Effects Analysis (FMEA)

| Failure Mode | Potential Effect | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **False Negative (Grade 4 labeled 0)** | Patient develops blindness without treatment. | **Critical** | Implement safety-first bias in loss function; Flag all high-confidence Normal predictions for spot-check. |
| **Low Quality Image** | Model hallucinates lesions or misses critical findings. | **High** | Mandatory `QualityAssessment` module. Refuse inference on "Poor" quality images. |
| **Adversarial Noise** | Deliberate or accidental image corruption leads to wrong grade. | **Medium** | Adversarial training (PGD) and robust augmentation during Phase 12. |
| **Domain Shift** | Performance drop on new camera hardware. | **High** | Real-time `Expected Calibration Error (ECE)` monitoring; flag if confidence drops significantly. |

## 2. Human-in-the-Loop Protocols

RetinaViT is designed as a **Decision Support System**, not a replacement for clinicians.

### 2.1 Low-Confidence Escalation
- If `Model Confidence < 0.70`, the system must append a warning: "Low Confidence - Manual verification required."
- For `Confidence < 0.50`, the system automatically classifies the case as "Requires Specialist Review."

### 2.2 Quality Guardrail
- Images labeled as **"Poor"** quality by the `SmartphoneSimulator`-trained quality head are blocked from diagnostic labeling.
- The UI will prompt the technician to: "Re-acquire image: Motion blur detected."

### 2.3 Explanation Audit
- For every case graded > 0 (DR detected), the clinician should review the `Attention Rollout` or `Grad-CAM` overlay.
- If the model highlights non-retinal regions (e.g., edges of the photo) but predicts DR, the result must be invalidated.

## 3. Deployment Safety Check
Before clinical use, the following command must be run to verify robustness:
```bash
bash scripts/run_robustness_suite.sh
```
All metrics must exceed the safety thresholds defined in `configs/base.yaml`.
