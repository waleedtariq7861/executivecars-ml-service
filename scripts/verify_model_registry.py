"""Load and exercise every rollback-capable model artifact in the registry."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.prediction import predict
from app.schemas import PredictionInput
from app.training import model_root


def main() -> None:
    root = model_root()
    registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    versions = registry.get("versions", [])
    if not versions:
        raise RuntimeError("Model registry is empty")

    for entry in versions:
        version = entry["version"]
        artifact = root / version / "model.joblib"
        if not artifact.exists():
            raise FileNotFoundError(f"Missing artifact for {version}")
        bundle = joblib.load(artifact)
        comparables = bundle.get("comparables")
        if comparables is None or comparables.empty:
            raise RuntimeError(f"{version} has no comparable input fixture")
        row = comparables.iloc[0]
        result = predict(bundle, PredictionInput(
            make=str(row["make"]), model=str(row["model"]), year=int(row["modelYear"]),
            mileage=float(row["mileage"]), engineCapacity=float(row["engineCapacity"]),
            transmission=str(row.get("transmission", "")), fuelType=str(row.get("fuelType", "")),
            city=str(row.get("listingCity", "")), registrationCity=str(row.get("registrationCity", "")),
            bodyType=str(row.get("bodyType", "")), assemblyType=str(row.get("assemblyType", "Unknown")),
        ))
        if result.get("estimatedPrice", 0) <= 0:
            raise RuntimeError(f"{version} returned an invalid prediction")
        print(f"verified {version}: PKR {result['estimatedPrice']:,}")
        del bundle, comparables, result
        gc.collect()

    print(f"Verified {len(versions)} registered model artifacts.")


if __name__ == "__main__":
    main()
