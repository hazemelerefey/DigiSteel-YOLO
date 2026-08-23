FROM python:3.11-slim-bookworm

WORKDIR /app

# 1. Install Python dependencies
COPY Deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Fix OpenCV dependencies without apt-get (bypasses corporate firewalls)
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless

# 3. Copy custom neural network modules
COPY dafegate/ ./dafegate/

# 4. Copy model weights
COPY huggingface_space/models/best.pt ./models/best.pt

# 5. Pre-download Hugging Face CLIP model weights into the Docker image cache
# This ensures the container starts instantly offline without downloading 600MB on every boot
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# 6. Copy API server code
COPY Deployment/Main.py ./main.py

# Expose API port
EXPOSE 8000

# Run FastAPI app via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]