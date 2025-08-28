# src/serve/app.py
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import uvicorn
# ------------------------------ Config ------------------------------
MLFLOW_TRACKING_URI =  "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

MODEL_NAME =  "BestClassifierModel_RandomUnderSampler_RandomForest"
mvs = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(mvs, key=lambda mv: int(mv.version))  # raises if none exist
MODEL_VERSION = latest.version
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(MODEL_URI)

TOP_N_FEATURES = int(os.getenv("TOP_N_FEATURES", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("serve")

app = FastAPI(
    title="Model Serving API",
    version="1.0.0",
    description="Serves the champion model from MLflow with schema-aware validation.",
)

STATE: Dict[str, Any] = {
    "model": None,            # mlflow.pyfunc.PyFuncModel
    "model_uri": None,        # str
    "model_version": None    # str
}


class InputData(BaseModel):
    Age: float
    Course: int   # 1=STEM, 0=Non-STEM
    
    Hours_of_Sleep: float
    In_a_Relationship: float  # 0, 0.5, 1
    Hours_Reviewing: float

# ---------------------------- Startup -------------------------------
@app.on_event("startup")
def startup() -> None:
    # Resolve and load model
    if MODEL_URI:
        try:
            model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as e:
            logger.exception("Failed to load model.")
    
    else:
        logger.exception("Failed to find model.")
        raise RuntimeError(f"Could not load model from MLflow: {e}") from e

    logger.info(
            "Model ready. name=%s version=%s",
            MODEL_NAME, MODEL_VERSION,
        )

    STATE.update(
        model = model,
        model_uri = MODEL_URI,
        model_version = MODEL_VERSION
    )


# ----------------------------- Routes -------------------------------

@app.get("/")
def read_root():
    return {"message": "FastAPI is running. Go to /docs to test the model."}


@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    preds = STATE["model"].predict(df)
    return {"predictions": preds.tolist()}


# ---------------------- Local dev entrypoint ------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.serve.app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD", "1") == "1"),
    )
