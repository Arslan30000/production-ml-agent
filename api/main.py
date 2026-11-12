"""Minimal inference service using the JSON model trained by this project."""

import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    temperature_c: float = Field(ge=-100, le=300)
    vibration_mm_s: float = Field(ge=0, le=100)
    load_pct: float = Field(ge=0, le=100)


app = FastAPI(title="Equipment Anomaly API")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/model.json"))


def load_model() -> dict:
    try:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if len(model["features"]) != 3 or len(model["weights"]) != 3:
            raise ValueError("unexpected model shape")
        return model
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Model unavailable") from exc


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    model = load_model()
    return {"status": "ready", "dataset_sha256": model["dataset_sha256"]}


@app.get("/version")
def version():
    model = load_model()
    return {"app_version": os.environ.get("APP_VERSION", "dev"),
            "dataset_sha256": model["dataset_sha256"]}


@app.post("/predict")
def predict(payload: PredictionInput):
    model = load_model()
    values = payload.model_dump()
    if not all(math.isfinite(value) for value in values.values()):
        raise HTTPException(status_code=422, detail="Non-finite input")
    z = model["bias"] + sum(
        model["weights"][i] * (values[name] - model["means"][i]) / model["scales"][i]
        for i, name in enumerate(model["features"])
    )
    probability = 1 / (1 + math.exp(-max(-35, min(35, z))))
    return {"anomaly": probability >= model["threshold"],
            "anomaly_probability": probability,
            "model_dataset_sha256": model["dataset_sha256"]}
