from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import sys
from pathlib import Path

# Add project root to path for dafegate module
ROOT = Path(__file__).resolve().parent.parent

# Handle Docker environment vs Local environment
if Path("/app/dafegate").exists():
    sys.path.append("/app")
else:
    sys.path.append(str(ROOT))

from dafegate.modules.dafe import DAFEGate
import ultralytics.nn.tasks as ultralytics_tasks

app = FastAPI(title="DigiSteel YOLO Defect Detection API with DAFEGate")

# Inject custom DAFEGate module
ultralytics_tasks.DAFEGate = DAFEGate

# Load YOLO model
try:
    if Path("/app/models/best.pt").exists():
        MODEL_PATH = Path("/app/models/best.pt")
    else:
        MODEL_PATH = ROOT / "huggingface_space" / "models" / "best.pt" 
    model = YOLO(str(MODEL_PATH))
except Exception as e:
    print(f"Warning: Failed to load YOLO model: {e}")
    model = None

# Initialize CLIP for semantic zero-shot gating
try:
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(CLIP_DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
except Exception as e:
    print(f"Warning: Failed to load CLIP model: {e}")
    clip_model = None
    clip_processor = None

POSITIVE_PROMPTS = [
    "close-up industrial steel plate surface",
    "hot-rolled steel sheet texture",
    "steel surface with defects"
]
NEGATIVE_PROMPTS = [
    "human face",
    "photo of a person",
    "website screenshot",
    "outdoor natural landscape",
    "graphic design banner",
    "colorful abstract image",
    "text document"
]
ALL_PROMPTS = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

class DefectBox(BaseModel):
    defect_name: str
    confidence: float
    box: List[float]

class DefectDetectionResponse(BaseModel):
    message: str
    defects: List[DefectBox]

def is_steel_surface(image: Image.Image) -> tuple[bool, str]:
    if min(image.size) < 96:
        return False, "Input image is too small for reliable steel-surface inspection."

    if clip_model is None or clip_processor is None:
        return True, "Input passed (CLIP disabled)."

    try:
        inputs = clip_processor(
            text=ALL_PROMPTS, 
            images=image.convert("RGB"), 
            return_tensors="pt", 
            padding=True
        ).to(CLIP_DEVICE)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        pos_prob = sum(probs[:len(POSITIVE_PROMPTS)])
        neg_prob = sum(probs[len(POSITIVE_PROMPTS):])
        
        if neg_prob > pos_prob:
            highest_neg_idx = np.argmax(probs[len(POSITIVE_PROMPTS):])
            detected_concept = NEGATIVE_PROMPTS[highest_neg_idx]
            return False, f"Input appears to be a '{detected_concept}', not a steel surface."
            
        return True, "Input passed the semantic steel-surface domain check."
    except Exception as e:
        print(f"Error during CLIP domain check: {e}")
        return True, "Input passed (CLIP error)."

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "DigiSteel-YOLO Defect Detection API",
        "semantic_guard": "CLIP (openai/clip-vit-base-patch32)",
        "model": "DAFEGate v4 YOLOv11n",
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=DefectDetectionResponse)
async def predict_defect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="YOLO model is not loaded properly."
        )

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file.")

    # Apply Semantic Gate
    is_valid, message = is_steel_surface(image)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail=f"Domain Rejection: {message}"
        )
    
    # Run YOLO Prediction
    results = model.predict(image, conf=0.45, iou=0.70, verbose=False)
    
    formatted_defects = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
           
            defect_name = model.names[class_id]
            formatted_defects.append(
                DefectBox(
                    defect_name=defect_name,
                    confidence=round(confidence, 4),
                    box=[round(c, 1) for c in coords]
                )
            )

    return DefectDetectionResponse(
        message=f"{message} Detected {len(formatted_defects)} defects.",
        defects=formatted_defects
    )