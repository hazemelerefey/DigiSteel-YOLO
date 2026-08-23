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

app = FastAPI(title="DAFEGate-YOLO — Steel Surface Defect Detection API")

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

# Initialize CLIP for semantic zero-shot domain guard
try:
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(CLIP_DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
except Exception as e:
    print(f"Warning: Failed to load CLIP model: {e}")
    clip_model = None
    clip_processor = None

# ─────────────────────────────────────────────────────────────────────────────
# CLIP Domain Guard Design
#
# PROBLEM 1 — False rejections of valid steel defect images (e.g. "patches"):
#   The root cause is that CLIP's embedding space places some grainy/textured
#   steel defect images ambiguously between positive and negative clusters.
#   A bare neg > pos decision (0.0pp margin) triggers too easily.
#
#   FIX:
#   (a) Expanded POSITIVE_PROMPTS to cover all 6 NEU-DET defect morphologies,
#       giving CLIP stronger in-domain anchors for every class.
#   (b) Reduced NEGATIVE_PROMPTS to only unambiguously out-of-domain concepts
#       that cannot appear in an industrial inspection pipeline.
#   (c) Introduced REJECTION_MARGIN: neg_prob must exceed pos_prob by at least
#       this margin to trigger a rejection. This eliminates border-line cases
#       while keeping the guard robust against clearly off-domain inputs.
#
# PROBLEM 2 — Rejection message exposes internal CLIP label categories:
#   Showing "Input appears to be a 'graphic design banner'" leaks internal
#   implementation details and is confusing to end-users.
#
#   FIX: Return a single professional rejection message with no category.
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_PROMPTS = [
    # General steel surface concepts
    "close-up industrial steel plate surface",
    "hot-rolled flat steel sheet texture",
    "steel surface with defects",
    "metallic surface with surface anomalies",
    # Morphology-specific — covers all 6 NEU-DET classes
    "grayscale steel surface with crazing cracks",       # crazing
    "gray steel sheet with inclusion defects",           # inclusion
    "steel surface with patch-like surface anomalies",   # patches
    "metallic surface with pitting defects",             # pitted_surface
    "industrial steel with rolled-in scale markings",    # rolled-in_scale
    "steel surface with linear scratch defects",         # scratches
]

NEGATIVE_PROMPTS = [
    # Unambiguously non-industrial, non-metallic categories only
    "photograph of a human face or person",
    "outdoor natural landscape with trees or sky",
    "screenshot of a website or user interface",
    "colorful poster or graphic design artwork",
    "printed text document or page of writing",
    "food photograph or meal image",
]

ALL_PROMPTS = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

# Rejection margin: neg_prob must exceed pos_prob by at least this fraction
# to reject an image. Prevents borderline steel-texture images from being
# rejected due to CLIP's distributional uncertainty.
REJECTION_MARGIN = 0.15  # 15 percentage-point safety buffer


class DefectBox(BaseModel):
    defect_name: str
    confidence: float
    box: List[float]


class DefectDetectionResponse(BaseModel):
    message: str
    defects: List[DefectBox]


def is_steel_surface(image: Image.Image) -> tuple[bool, str]:
    """
    Zero-shot CLIP semantic domain guard.

    Returns (True, pass_message) if the image is a plausible steel surface,
    or (False, rejection_message) if it is clearly out-of-domain.

    The rejection message never discloses internal CLIP prompt categories.
    """
    if min(image.size) < 96:
        return False, "Input image is too small for reliable steel-surface inspection (minimum 96×96 px)."

    if clip_model is None or clip_processor is None:
        return True, "Input passed domain check (CLIP guard disabled)."

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

        pos_prob = float(sum(probs[:len(POSITIVE_PROMPTS)]))
        neg_prob = float(sum(probs[len(POSITIVE_PROMPTS):]))

        # Require neg_prob to exceed pos_prob by REJECTION_MARGIN before rejecting.
        # This prevents ambiguous steel-texture images from being incorrectly rejected.
        if neg_prob > pos_prob + REJECTION_MARGIN:
            return (
                False,
                "The uploaded image does not appear to be a steel surface. "
                "Please upload a grayscale or industrial image of a hot-rolled "
                "flat steel sheet for defect inspection."
            )

        return True, "Input passed the semantic steel-surface domain check."

    except Exception as e:
        print(f"Error during CLIP domain check: {e}")
        return True, "Input passed domain check (CLIP error fallback)."


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "DAFEGate-YOLO Steel Surface Defect Detection API",
        "model": "DAFEGate v4 — YOLOv11n (mAP@0.5 = 81.98%)",
        "domain": "Hot-rolled flat steel sheet — NEU-DET (6 classes)",
        "semantic_guard": "CLIP openai/clip-vit-base-patch32",
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=DefectDetectionResponse)
async def predict_defect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="YOLO model is not loaded. Please check the server configuration."
        )

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file. Please upload a valid PNG or JPEG image."
        )

    # Apply Semantic Domain Guard
    is_valid, message = is_steel_surface(image)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message
        )

    # Run YOLO Defect Detection
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
        message=f"{message} Detected {len(formatted_defects)} defect(s).",
        defects=formatted_defects
    )