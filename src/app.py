"""
app.py — M2: FastAPI inference service  |  M5: logging + in-app metrics.

Endpoints:
    GET  /health   -> {"status": "ok", "model_loaded": true}
    POST /predict  -> multipart image upload -> {"label", "probabilities"}
    GET  /metrics  -> request counters + latency stats (M5 monitoring)

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000   (from src/, model at ../models)
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from predict import load_model, predict_image

# ---------------------------------------------------------------- logging (M5)
# Request/response logging WITHOUT sensitive data: we log filenames, sizes,
# latencies and predicted labels — never the image content itself.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inference-service")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pt"))

app = FastAPI(title="Cats vs Dogs Inference Service", version="1.0.0")
@app.get("/")
def root() -> dict:
    return {
        "service": "Cats vs Dogs Inference API",
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }

# ---------------------------------------------------------------- metrics (M5)
METRICS = {
    "requests_total": 0,
    "predictions_total": 0,
    "predictions_by_label": {"cat": 0, "dog": 0},
    "errors_total": 0,
    "latency_ms_sum": 0.0,
}

model = None
classes = None


@app.on_event("startup")
def startup() -> None:
    global model, classes
    if MODEL_PATH.exists():
        model, classes = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model not found at %s — running without model", MODEL_PATH)
        model, classes = None, None


@app.middleware("http")
async def log_and_count(request: Request, call_next):
    """M5: log every request and track latency, excluding sensitive payloads."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    METRICS["requests_total"] += 1
    METRICS["latency_ms_sum"] += elapsed_ms
    logger.info("%s %s -> %s in %.1f ms",
                request.method, request.url.path,
                response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health() -> dict:
    """Health check endpoint (required by M2, used by M4 smoke tests)."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Prediction endpoint: accepts an image, returns label + probabilities."""
    if model is None:
        METRICS["errors_total"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        METRICS["errors_total"] += 1
        raise HTTPException(status_code=400,
                            detail=f"Expected an image upload, got {content_type}")

    image_bytes = await file.read()
    try:
        result = predict_image(model, classes, image_bytes)
    except Exception as exc:  # unreadable/corrupted image
        METRICS["errors_total"] += 1
        logger.warning("Prediction failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=400, detail="Could not read image") from exc

    METRICS["predictions_total"] += 1
    METRICS["predictions_by_label"][result["label"]] += 1
    logger.info("Predicted '%s' for file '%s' (%d bytes)",
                result["label"], file.filename, len(image_bytes))
    return result


@app.get("/metrics")
def metrics() -> dict:
    """M5: simple in-app counters — request count, error count, avg latency."""
    n = METRICS["requests_total"]
    return {
        **METRICS,
        "avg_latency_ms": round(METRICS["latency_ms_sum"] / n, 2) if n else 0.0,
    }
