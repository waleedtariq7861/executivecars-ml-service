from fastapi.testclient import TestClient

from app import main
from app.training import load_active_bundle


client = TestClient(main.app)


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


def test_predict_returns_real_pkr_contract_and_handles_unseen_categories(monkeypatch):
    bundle = load_active_bundle()
    assert bundle is not None, "The repository must include a trained active model for API checks."
    monkeypatch.setattr(main, 'active_bundle', bundle)
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


def test_active_model_predicts_ten_representative_dataset_vehicles(monkeypatch):
    bundle = load_active_bundle()
    assert bundle is not None
    monkeypatch.setattr(main, 'active_bundle', bundle)
    examples = bundle['comparables'].drop_duplicates(['make', 'model']).head(10)
    assert len(examples) == 10
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
