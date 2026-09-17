import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app import training
from app.prediction import predict
from app.schemas import PredictionInput


def records(count=45):
    makes = [("Toyota", "Corolla", 4_800_000), ("Honda", "City", 4_200_000), ("Suzuki", "Swift", 3_300_000)]
    rows = []
    for index in range(count):
        make, model, base = makes[index % len(makes)]
        year = 2016 + index % 8
        mileage = 25_000 + (index % 12) * 7_000
        rows.append({
            "make": make, "model": model, "variant": "Test Variant", "modelYear": year,
            "mileage": mileage, "engineCapacity": 1300 + (index % 2) * 200,
            "transmission": "Auto" if index % 2 else "Manual", "fuelType": "Petrol",
            "listingCity": "Rawalpindi", "condition": "Good", "assemblyType": "Local",
            "listingPrice": base + (year - 2016) * 180_000 - mileage * 4 + index * 1_000,
            "listingDate": f"2025-{(index % 12) + 1:02d}-01", "source": "Synthetic unit-test fixture",
        })
    return rows


def test_prediction_schema_rejects_negative_mileage():
    with pytest.raises(ValidationError):
        PredictionInput(make="Toyota", model="Corolla", year=2020, mileage=-1)


def test_training_saves_loadable_model_and_predicts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(training, "candidates", lambda: {
        "median_baseline": Pipeline([("preprocess", training._one_hot_preprocessor()), ("model", DummyRegressor(strategy="median"))]),
        "ridge_regression": Pipeline([("preprocess", training._one_hot_preprocessor()), ("model", Ridge(alpha=10.0))]),
    })
    provenance = {
        "datasetName": "Synthetic training records",
        "sourceCategory": "Unit-test fixture",
        "sources": [{"source": "Synthetic unit-test fixture", "rows": 45}],
    }
    result = training.train_from_frame(pd.DataFrame.from_records(records()), provenance=provenance)
    assert result["metrics"]["mae"] >= 0
    bundle = training.load_active_bundle()
    assert bundle is not None
    response = predict(bundle, PredictionInput(
        make="Toyota", model="Corolla", variant="Test Variant", year=2021, mileage=45_000,
        engineCapacity=1300, transmission="Auto", fuelType="Petrol", city="Rawalpindi",
        condition="Good", assemblyType="Local",
    ))
    assert response["estimatedMarketPrice"] > 0
    assert response["predicted_price"] == response["estimatedPrice"]
    assert response["currency"] == "PKR"
    assert response["method"] == "trained_ml"
    assert response["lowerRange"] <= response["estimatedPrice"] <= response["upperRange"]
    assert response["rangeMethod"]["method"] == "mondrian_split_conformal_absolute_residual"
    assert "confidence" not in response
    assert response["modelMetrics"]["mae"] >= 0
    assert response["modelVersion"] == result["version"]
    schema_path = tmp_path / result["version"] / "feature_schema.json"
    assert schema_path.exists()
    assert result["splitSummary"]["crossValidationFolds"] == 3
    assert len((tmp_path / "reports" / "model_evaluation.json").read_text()) > 0
    registered = training.list_models()["versions"][0]
    assert registered["provenance"]["datasetName"] == "Synthetic training records"
    assert registered["provenance"]["sourceCategory"] == "Unit-test fixture"
    assert registered["provenance"]["datasetFingerprint"] == bundle["datasetFingerprint"]


def test_list_models_does_not_attach_root_metadata_to_versionless_entry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    (tmp_path / "registry.json").write_text(json.dumps({
        "activeVersion": None,
        "previousVersion": None,
        "versions": [{"modelName": "legacy", "datasetSize": 12}],
    }), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "datasetFile": "unrelated-root-data.csv",
        "datasetSize": 999,
    }), encoding="utf-8")

    registered = training.list_models()["versions"][0]

    assert registered["provenance"]["datasetName"] == "Legacy metadata unavailable"
    assert registered["provenance"]["datasetRows"] == 12


def test_activation_and_rollback_update_only_a_temporary_registry(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    for version in ("model-v1", "model-v2"):
        version_dir = tmp_path / version
        version_dir.mkdir()
        (version_dir / "model.joblib").write_bytes(b"synthetic registry fixture")
    initial = {
        "activeVersion": "model-v1",
        "previousVersion": None,
        "versions": [{"version": "model-v2"}, {"version": "model-v1"}],
    }
    (tmp_path / "registry.json").write_text(json.dumps(initial), encoding="utf-8")

    activated = training.activate_model("model-v2")
    assert activated["activeVersion"] == "model-v2"
    assert activated["previousVersion"] == "model-v1"
    assert training.activate_model("model-v2") == activated

    rolled_back = training.rollback_model()
    assert rolled_back["activeVersion"] == "model-v1"
    assert rolled_back["previousVersion"] == "model-v2"
    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8")) == rolled_back


def test_invalid_activation_and_unavailable_rollback_leave_registry_unchanged(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    initial = {"activeVersion": "model-v1", "previousVersion": None, "versions": [{"version": "model-v1"}]}
    (tmp_path / "registry.json").write_text(json.dumps(initial), encoding="utf-8")

    with pytest.raises(ValueError, match="Model version does not exist"):
        training.activate_model("missing-model")
    with pytest.raises(ValueError, match="No previous model version"):
        training.rollback_model()

    assert json.loads((tmp_path / "registry.json").read_text(encoding="utf-8")) == initial
