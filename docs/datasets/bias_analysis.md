# Dataset Bias Analysis

This document identifies potential sources of bias in the datasets used for the RetinaViT project and discusses their implications for model generalization.

## Dataset Profiles

### APTOS 2019
- **Geography**: India (Aravind Eye Hospital).
- **Population**: Primarily South Asian patients in a clinical setting related to blindness detection.
- **Device**: Multiple camera types, including portable and specialized fundus cameras.
- **Conditions**: Wide variety of quality, reflecting real-world clinical conditions in India.

### MESSIDOR-2
- **Geography**: France (three different ophthalmology departments).
- **Population**: European patients, often part of routine screening programs.
- **Device**: Specialized fundus cameras (e.g., Topcon TRC NW6).
- **Conditions**: generally high quality and standardized capture protocols.

### IDRiD
- **Geography**: India.
- **Population**: Indian patients visiting a retinal specialty clinic.
- **Device**: Forus 3Nethra fundus camera.
- **Conditions**: High-resolution images (4288x2848), focused on diabetic retinopathy grading and lesion segmentation.

## Identified Bias Sources

1. **Ethnicity & Pigmentation**: 
   - APTOS and IDRiD are dominated by South Asian populations, whereas MESSIDOR-2 is primarily European. 
   - Fundus pigmentation varies by ethnicity, affecting illumination patterns and lesion visibility.

2. **Capture Hardware**: 
   - Cross-dataset performance may be hindered by "camera fingerprints" (resolution, aspect ratio, FOV, and sensor noise patterns).
   - MESSIDOR-2 is highly standardized; APTOS is heterogeneous.

3. **Clinical Context**:
   - MESSIDOR-2 represents broad screening (many healthy/mild cases), while IDRiD and APTOS often come from specialized clinics where severe cases are more prevalent.

## Impact on Generalization
- A model trained on APTOS/IDRiD might overfit to "India-specific" image characteristics.
- Domain shift is expected when testing an "India-trained" model on the "France-derived" MESSIDOR-2 dataset.
- RetinaViT's quality-aware and prototype-based approach aims to mitigate these shifts by focusing on robust features and quality assessment.
