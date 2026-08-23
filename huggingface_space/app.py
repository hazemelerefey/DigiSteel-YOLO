from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
import ultralytics.nn.tasks as ultralytics_tasks
from transformers import CLIPProcessor, CLIPModel
import torch

try:
    import spaces
    def gpu_decorator(func):
        return spaces.GPU(func)
except ImportError:
    def gpu_decorator(func):
        return func

from dafegate.modules.dafe import DAFEGate


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.45
DEFAULT_IOU = 0.70
DOMAIN_CHECK_SIZE = 256


ultralytics_tasks.DAFEGate = DAFEGate
MODEL = YOLO(str(MODEL_PATH))

# Initialize CLIP for semantic zero-shot gating
try:
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(CLIP_DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
except Exception as e:
    print(f"Failed to load CLIP model: {e}")
    clip_model = None
    clip_processor = None

# Semantic prompts for domain classification
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


def _steel_surface_gate(image: Image.Image) -> tuple[bool, str]:
    """Reject obvious out-of-domain images before defect detection using zero-shot CLIP.

    This semantic gate catches natural, colorful, non-industrial images and screenshots
    with high reliability compared to brittle heuristic pixel checks.
    """
    if min(image.size) < 96:
        return False, "Input image is too small for reliable steel-surface inspection."

    if clip_model is None or clip_processor is None:
        # Fallback if CLIP fails to load (allow inference)
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
        
        # Calculate total probability for positive vs negative classes
        pos_prob = sum(probs[:len(POSITIVE_PROMPTS)])
        neg_prob = sum(probs[len(POSITIVE_PROMPTS):])
        
        if neg_prob > pos_prob:
            # Find the highest scoring negative prompt for a descriptive message
            highest_neg_idx = np.argmax(probs[len(POSITIVE_PROMPTS):])
            detected_concept = NEGATIVE_PROMPTS[highest_neg_idx]
            return False, f"Input appears to be a '{detected_concept}', not a steel surface."
            
        return True, "Input passed the semantic steel-surface domain check."
    except Exception as e:
        print(f"Error during CLIP domain check: {e}")
        return True, "Input passed (CLIP error)."


def _prediction_rows(result: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    boxes = result.boxes
    if boxes is None:
        return rows

    for box in boxes:
        class_id = int(box.cls[0])
        rows.append(
            [
                MODEL.names[class_id],
                round(float(box.conf[0]), 4),
                round(float(box.xyxy[0][0]), 1),
                round(float(box.xyxy[0][1]), 1),
                round(float(box.xyxy[0][2]), 1),
                round(float(box.xyxy[0][3]), 1),
            ]
        )
    return rows


@gpu_decorator
def detect_defects(
    image: Image.Image | None,
    confidence: float,
    iou: float,
) -> tuple[Image.Image | None, list[list[Any]], str]:
    if image is None:
        return None, [], "Upload a steel surface image to run detection."

    rgb_image = image.convert("RGB")
    is_steel_surface, domain_message = _steel_surface_gate(rgb_image)
    if not is_steel_surface:
        return (
            rgb_image,
            [],
            f"Rejected: {domain_message} No defect detection was performed.",
        )

    results = MODEL.predict(
        source=rgb_image,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=iou,
        verbose=False,
    )
    result = results[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    rows = _prediction_rows(result)

    if not rows:
        return annotated, rows, f"{domain_message} No defects detected at the current threshold."

    return annotated, rows, f"{domain_message} Detected {len(rows)} defect candidate(s)."


with gr.Blocks(title="DigiSteel YOLO DAFEGate v4") as demo:
    gr.Markdown(
        """
        # DigiSteel YOLO: DAFEGate v4

        Steel surface defect detection for the NEU-DET classes:
        crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.
        Non-steel images are rejected by a lightweight domain guard before inference.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Steel surface image")
            confidence_input = gr.Slider(
                minimum=0.25,
                maximum=0.90,
                value=DEFAULT_CONFIDENCE,
                step=0.05,
                label="Confidence threshold",
            )
            iou_input = gr.Slider(
                minimum=0.30,
                maximum=0.90,
                value=DEFAULT_IOU,
                step=0.05,
                label="IoU threshold",
            )
            run_button = gr.Button("Detect defects", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="Annotated result")
            status_output = gr.Textbox(label="Status", interactive=False)
            table_output = gr.Dataframe(
                headers=["defect", "confidence", "x1", "y1", "x2", "y2"],
                datatype=["str", "number", "number", "number", "number", "number"],
                label="Detections",
                interactive=False,
            )

    gr.Markdown(
        """
        Model: DAFEGate v4, YOLOv11n-based P3 enhancement. Test mAP@0.5: 81.98%;
        mAP@0.5:0.95: 46.80%; precision: 72.55%; recall: 79.79%.
        """
    )

    run_button.click(
        detect_defects,
        inputs=[image_input, confidence_input, iou_input],
        outputs=[image_output, table_output, status_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
