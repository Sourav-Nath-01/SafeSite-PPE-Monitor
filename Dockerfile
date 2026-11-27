# CPU image for the SafeSite monitoring service.
# Deployed to Azure Container Apps by .github/workflows/deploy.yml (see infra/).
# For GPU inference, swap the base for nvidia/cuda:12.4.1-runtime-ubuntu22.04,
# install the matching torch wheel, and run with `--gpus all`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV needs these; the slim base ships neither.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch from the PyTorch index. --index-url (not --extra-index-url)
# so pip cannot pick the PyPI build, which drags in ~2.5 GB of CUDA libraries.
# Own layer: it changes far less often than requirements.txt.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt \
     ./Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt
COPY Model-Deployment/ ./Model-Deployment/

# Run as an unprivileged user; /data holds the SQLite log and snapshots.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data \
    && chown app:app /data
USER app
WORKDIR /app/Model-Deployment

# Fail the build, not the first request, if the weights or tracker deps are
# broken. Also turns off Ultralytics' anonymous usage telemetry.
RUN python -c "import lap, gunicorn, simple_websocket, psycopg, psycopg_pool, azure.identity, azure.storage.blob; \
from ultralytics import YOLO, settings; settings.update({'sync': False}); \
YOLO('../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt')"

ENV PORT=3333 \
    PPE_DATA_DIR=/data

EXPOSE 3333

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3333') + '/healthz', timeout=4)"

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
