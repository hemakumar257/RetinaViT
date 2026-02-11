# RetinaViT API Documentation

## 1. Python API
The core logic resides in `src/retinavit`.

### Preprocessing
```python
from retinavit.data.preprocessing import FundusPreprocessor
preprocessor = FundusPreprocessor(cfg)
processed_img = preprocessor.preprocess("path/to/image.jpg")
```

### Inference
```python
from retinavit.models.vit import build_vit_base
model = build_vit_base(cfg)
logits = model(input_tensor)
```

## 2. FastAPI Clinical Service
The backend service provides high-performance inference via HTTP.

### Run Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

#### `POST /predict`
- **Description:** Submit a fundus image for DR grading.
- **Request:** Multipart form-data with `file`.
- **Response:**
  ```json
  {
    "grade": 2,
    "confidence": 0.94,
    "quality_label": "Good"
  }
  ```

#### `GET /health`
- **Description:** System health check.
- **Response:** `{"status": "healthy"}`

## 3. Mobile Inference
Models exported via `src/retinavit/deployment/optimize.py` are compatible with `org.pytorch.pytorch_android_lite`.
Refer to `docs/deployment/android.md` for integration details.
