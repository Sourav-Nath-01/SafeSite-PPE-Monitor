# CPU image for the PPE detection service.
# For GPU inference, swap the base for nvidia/cuda:12.4.1-runtime-ubuntu22.04,
# install the matching torch wheel, and run with `--gpus all`.
FROM python:3.10-slim

# OpenCV needs these; the slim base ships neither.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch wheels are far smaller than the default CUDA ones.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY Model-Deployment/ ./Model-Deployment/
COPY Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt \
     ./Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt

WORKDIR /app/Model-Deployment

ENV PPE_SECRET_KEY=change-me-in-production \
    PYTHONUNBUFFERED=1

EXPOSE 3333

CMD ["python", "app.py"]
