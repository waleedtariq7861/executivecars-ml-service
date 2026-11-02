"""Access control of the valuation service. Every test runs against a temporary model folder: nothing here reads or writes a real model."""
import asyncio
import json

import httpx
import pytest

from app import main, training

KEY = "test-service-key-0123456789"
VALID = {"make": "Toyota", "model": "Corolla", "year": 2020, "mileage": 45000, "engineCapacity": 1300}


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(main, "active_bundle", None)
    return tmp_path / "models"


def call(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def with_key(monkeypatch, value=KEY):
    monkeypatch.setenv("ML_SERVICE_KEY", value)


def test_production_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ML_SERVICE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ML_SERVICE_KEY is required"):
        main.ensure_service_key_configured()
    monkeypatch.setenv("ML_SERVICE_KEY", "   ")
    with pytest.raises(RuntimeError):
        main.ensure_service_key_configured()
    monkeypatch.setenv("ML_SERVICE_KEY", KEY)
    main.ensure_service_key_configured()
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ML_SERVICE_KEY", raising=False)
    main.ensure_service_key_configured()


@pytest.mark.parametrize("method,path", [
    ("get", "/metadata"), ("get", "/models"), ("post", "/predict"), ("post", "/train"),
    ("post", "/models/rollback"), ("post", "/models/some-version/activate"),
])
def test_every_endpoint_but_health_needs_the_key_once_one_is_set(monkeypatch, method, path):
    with_key(monkeypatch)
    body = {"json": VALID} if path == "/predict" else ({"json": {"records": [{"make": "Toyota"}]}} if path == "/train" else {})
    assert call(method, path, **body).status_code == 401
    assert call(method, path, headers={"X-Service-Key": "wrong"}, **body).status_code == 401
    assert call(method, path, headers={"X-Service-Key": KEY[:-1]}, **body).status_code == 401
    assert call(method, path, headers={"X-Service-Key": ""}, **body).status_code == 401


def test_the_right_key_gets_through_to_the_endpoint_itself(monkeypatch):
    with_key(monkeypatch)
    headers = {"X-Service-Key": KEY}
    assert call("post", "/predict", json=VALID, headers=headers).status_code == 503  # authorised, but no model is active
    assert call("get", "/models", headers=headers).status_code == 200
    assert call("post", "/models/rollback", headers=headers).status_code == 409


def test_health_stays_open_for_monitoring(monkeypatch):
    with_key(monkeypatch)
    assert call("get", "/health").status_code == 200


def test_the_key_is_compared_in_constant_time(monkeypatch):
    seen = []
    real = main.hmac.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(main.hmac, "compare_digest", spy)
    with_key(monkeypatch)
    call("get", "/models", headers={"X-Service-Key": "guess"})
    assert seen == [(b"guess", KEY.encode())]


@pytest.mark.parametrize("version", ["..", ".", "../models", r"..\models", "a/b", "/etc/passwd", "", " ", ".hidden", "-x", "a" * 101, "v1\x00", "v 1"])
def test_a_model_version_name_cannot_point_outside_the_model_folder(version, isolated_models, monkeypatch):
    (isolated_models.parent / "model.joblib").write_bytes(b"outside")
    isolated_models.mkdir(parents=True, exist_ok=True)
    (isolated_models / "model.joblib").write_bytes(b"in the root itself")
    assert training._version_directory(version) is None
    with pytest.raises(ValueError, match="does not exist"):
        training.activate_model(version)
    assert not (isolated_models / "registry.json").exists()


def test_a_real_version_folder_is_accepted(isolated_models):
    folder = isolated_models / "ec-20260101-000000-abcd1234"
    folder.mkdir(parents=True)
    (folder / "model.joblib").write_bytes(b"model")
    assert training._version_directory("ec-20260101-000000-abcd1234") == folder.resolve()
    registry = training.activate_model("ec-20260101-000000-abcd1234")
    assert registry["activeVersion"] == "ec-20260101-000000-abcd1234"


def test_a_tampered_registry_cannot_make_the_service_load_a_file_outside_the_folder(isolated_models):
    isolated_models.mkdir(parents=True, exist_ok=True)
    (isolated_models.parent / "model.joblib").write_bytes(b"not a model")
    (isolated_models / "registry.json").write_text(json.dumps({"activeVersion": "..", "previousVersion": None, "versions": []}), encoding="utf-8")
    assert training.load_active_bundle() is None
