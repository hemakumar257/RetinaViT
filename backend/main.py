from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import io
from PIL import Image
from typing import List, Optional

# Mock import of retinavit components
from retinavit.models.mobile_student import MobileRetinaNet

app = FastAPI(title="RetinaViT Clinical API")

# Global model state
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionResponse(BaseModel):
    grade: int
    confidence: float
    quality_label: str
    explanation_hotspots: Optional[List[List[float]]] = None

@app.on_event("startup")
def load_model():
    global model
    # Load optimized model (INT8 quantized or distilled)
    # Using small student for fast startup in this demo
    cfg = {"model": {"num_classes": 5}}
    model = MobileRetinaNet(cfg).to(device)
    model.eval()
    print("Optimization model loaded and ready.")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Preprocessing
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
    return {
        "grade": int(pred.item()),
        "confidence": float(conf.item()),
        "quality_label": "Good", # Simplified for demo
        "explanation_hotspots": None
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "RetinaViT-Mobile-v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
