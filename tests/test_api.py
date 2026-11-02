import asyncio

import httpx
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from app import main, training
from test_training_prediction import records


class ASGITestClient:
    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


client = ASGITestClient()


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    """A small model trained on synthetic records in a temporary folder: the API checks need no real model file."""
    folder = tmp_path_factory.mktemp("api-models")
    patch = pytest.MonkeyPatch()
    try:
        patch.setenv("MODEL_DIR", str(folder))
        patch.setenv("REPORT_DIR", str(folder / "reports"))
        patch.setattr(training, "candidates", lambda: {
            "median_baseline": Pipeline([("preprocess", training._one_hot_preprocessor()), ("model", DummyRegressor(strategy="median"))]),
            "ridge_regression": Pipeline([("preprocess", training._one_hot_preprocessor()), ("model", Ridge(alpha=10.0))]),
        })
        training.train_from_frame(pd.DataFrame.from_records(records()), provenance={"datasetName": "Synthetic training records", "sourceCategory": "Unit-test fixture", "sources": [{"source": "Synthetic unit-test fixture", "rows": 45}]})
        bundle = training.load_active_bundle()
    finally:
        patch.undo()
    assert bundle is not None
    return bundle


def test_health_reports_model_state():
    response = client.get('/health')
    assert response.status_code == 200
    assert 'modelLoaded' in response.json()


def test_predict_requires_an_active_model(monkeypatch):
    monkeypatch.setattr(main, 'active_bundle', None)
    response = client.post('/predict', json={"make": "Toyota", "model": "Corolla", "year": 2020, "mileage": 45000, "engineCapacity": 1300})
    assert response.status_code == 503


def test_predict_rejects_invalid_schema():
    response = client.post('/predict', json={"make": "", "model": "", "year": 1900, "mileage": -1})
    assert response.status_code == 422


def test_predict_returns_real_pkr_contract_and_handles_unseen_categories(monkeypatch, trained_bundle):
    monkeypatch.setattr(main, 'active_bundle', trained_bundle)
    valid = client.post('/predict', json={
        "make": "Toyota", "model": "Corolla", "year": 2020, "mileage": 45000,
        "engineCapacity": 1300, "transmission": "Auto", "fuelType": "Petrol",
        "city": "Lahore", "registrationCity": "Lahore", "bodyType": "Sedan", "assemblyType": "Local",
    })
    assert valid.status_code == 200
    body = valid.json()
    assert body["predicted_price"] > 0
    assert body["currency"] == "PKR"
    assert body["modelVersion"]

    unseen = client.post('/predict', json={
        "make": "Unseen Brand", "model": "Imaginary", "year": 2020, "mileage": 45000,
        "engineCapacity": 1300,
    })
    assert unseen.status_code == 200
    assert unseen.json()["predicted_price"] > 0
    assert unseen.json()["extrapolationWarnings"]


def test_active_model_predicts_every_vehicle_it_was_trained_on(monkeypatch, trained_bundle):
    monkeypatch.setattr(main, 'active_bundle', trained_bundle)
    examples = trained_bundle['comparables'].drop_duplicates(['make', 'model'])
    assert len(examples) == 3
    for row in examples.itertuples():
        response = client.post('/predict', json={
            'make': row.make,
            'model': row.model,
            'year': int(row.modelYear),
            'mileage': float(row.mileage),
            'engineCapacity': float(row.engineCapacity),
            'transmission': row.transmission,
            'fuelType': row.fuelType,
            'city': row.listingCity,
            'registrationCity': row.registrationCity,
            'bodyType': row.bodyType,
            'assemblyType': row.assemblyType,
        })
        assert response.status_code == 200
        assert response.json()['predicted_price'] > 0
