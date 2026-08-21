from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

from .prediction import UnsupportedVehicleError, predict
from .schemas import PredictionInput, TrainingRequest
from .training import activate_model, list_models, load_active_bundle, rollback_model, train_from_records


load_dotenv()
app = FastAPI(title="Executive Cars Valuation Service", version="1.0.0", docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None)
state_lock = threading.Lock()
active_bundle = load_active_bundle()


def require_service_key(x_service_key: Annotated[str | None, Header()] = None) -> None:
    configured = os.getenv("ML_SERVICE_KEY", "").strip()
    if configured and x_service_key != configured:
        raise HTTPException(status_code=401, detail="Invalid service key")


def reload_bundle() -> None:
    global active_bundle
    with state_lock:
        active_bundle = load_active_bundle()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "modelLoaded": active_bundle is not None,
        "modelVersion": active_bundle.get("modelVersion") if active_bundle else None,
        "modelName": active_bundle.get("modelName") if active_bundle else None,
        "trainedRows": active_bundle.get("trainedRows", 0) if active_bundle else 0,
        "datasetSize": active_bundle.get("datasetSize", 0) if active_bundle else 0,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metadata", dependencies=[Depends(require_service_key)])
def metadata() -> dict:
    if active_bundle is None:
        raise HTTPException(status_code=503, detail="No trained model is active")
    return {
        "modelName": active_bundle.get("modelName"),
        "modelVersion": active_bundle.get("modelVersion"),
        "trainedRows": active_bundle.get("trainedRows", 0),
        "datasetSize": active_bundle.get("datasetSize", 0),
        "datasetReferenceYear": active_bundle.get("datasetReferenceYear"),
        "modelMetrics": active_bundle.get("metrics", {}),
        "rangeMethod": active_bundle.get("rangeCalibration", {}),
        "inputCatalog": active_bundle.get("inputCatalog", {}),
        "limitations": active_bundle.get("limitations", []),
    }


@app.post("/predict")
def prediction(payload: PredictionInput) -> dict:
    if active_bundle is None:
        raise HTTPException(status_code=503, detail="No trained model is active")
    try:
        return predict(active_bundle, payload)
    except UnsupportedVehicleError as error:
        raise HTTPException(
            status_code=422,
            detail={"message": str(error), "errors": error.errors},
        ) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed") from error


@app.post("/train", dependencies=[Depends(require_service_key)])
async def train(payload: TrainingRequest) -> dict:
    try:
        result = await run_in_threadpool(train_from_records, payload.records)
        reload_bundle()
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/models", dependencies=[Depends(require_service_key)])
def models() -> dict:
    return list_models()


@app.post("/models/{version}/activate", dependencies=[Depends(require_service_key)])
def activate(version: str) -> dict:
    try:
        registry = activate_model(version)
        reload_bundle()
        return registry
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/models/rollback", dependencies=[Depends(require_service_key)])
def rollback() -> dict:
    try:
        registry = rollback_model()
        reload_bundle()
        return registry
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
