# RetinaViT Android Deployment Guide

This document outlines how to integrate the optimized RetinaViT models into an Android application using PyTorch Mobile.

## Project Structure
- `mobile/android/`: Root of the Android Studio project.
- `mobile/android/app/src/main/assets/`: Place your exported `.pt` (TorchScript) models here.

## How to Build
1. Open the `mobile/android/` directory in **Android Studio**.
2. Ensure you have the **PyTorch Mobile Lite** dependency in `build.gradle`:
   ```gradle
   implementation 'org.pytorch:pytorch_android_lite:1.13.1'
   ```
3. Sync Gradle and build the project.

## Model Export (Python side)
Before building, export your trained model to TorchScript:
```bash
python -c "
from retinavit.deployment.optimize import export_torchscript
# ... load your model ...
export_torchscript(model, 'mobile/android/app/src/main/assets/retinavit_mobile.pt')
"
```

## Running Inference
The `MainActivity.kt` provides a template for loading the model using `LiteModuleLoader`.
You must ensure the input image is preprocessed (224x224, normalized) before calling `model.forward()`.
