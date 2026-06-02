# ─── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.12.7-slim AS builder

WORKDIR /build

COPY conda.yaml /tmp/conda.yaml

# Install Python dependencies without conda (lighter Docker image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        mlflow==2.19.0 \
        scikit-learn==1.5.2 \
        pandas==2.2.3 \
        numpy==1.26.4 \
        matplotlib==3.9.2 \
        seaborn==0.13.2 \
        dagshub==0.3.45 \
        flask==3.0.3 \
        prometheus-client==0.20.0 \
        psutil==5.9.8 \
        requests==2.32.3

# ─── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.12.7-slim

LABEL maintainer="Timotius Kristafael Harjanto"
LABEL description="Adult Income RF Model — SMSML Dicoding"
LABEL version="1.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model artefacts
COPY modelling.py    ./
COPY conda.yaml      ./
COPY MLproject       ./

# Model directory is copied in by CI after training
COPY model/          ./model/

# Dataset is needed only for re-training; comment out for serving-only images
# COPY dataset_preprocessed/ ./dataset_preprocessed/

# Expose the mlflow models serve port
EXPOSE 5001

# Environment defaults — override at runtime
ENV MODEL_PATH=/app/model
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV PORT=5001

# Health-check every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:5001/ping',timeout=5); r.raise_for_status()"

# Serve the model via MLflow REST API (no conda inside container)
CMD ["mlflow", "models", "serve", \
     "--model-uri", "/app/model", \
     "--host",      "0.0.0.0", \
     "--port",      "5001", \
     "--no-conda"]
