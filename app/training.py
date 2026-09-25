from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import (
    add_features,
    build_input_catalog,
    clean_dataframe,
    dataframe_fingerprint,
    determine_reference_year,
    inspect_dataframe,
)


SEED = 42
MIN_TRAINING_ROWS = 30
INTERVAL_COVERAGE = 0.90
INTERVAL_COVERAGES = (0.70, 0.80, 0.85, 0.90)
SELECTED_INTERVAL_COVERAGE = 0.85
MIN_GROUP_CALIBRATION_ROWS = 40
CV_FOLDS = 3

NUMERIC_FEATURES = [
    "modelYear",
    "mileage",
    "engineCapacity",
    "vehicleAge",
    "mileagePerYear",
]
CATEGORICAL_FEATURES = [
    "make",
    "model",
    "makeModel",
    "registrationCity",
    "listingCity",
    "fuelType",
    "transmission",
    "bodyType",
    "assemblyType",
    "engineCategory",
    "ageCategory",
    "mileageCategory",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "modelYear": "Model year",
    "mileage": "Mileage",
    "engineCapacity": "Engine size",
    "vehicleAge": "Vehicle age",
    "mileagePerYear": "Mileage per year",
    "make": "Make",
    "model": "Model",
    "makeModel": "Make and model",
    "registrationCity": "Registration city",
    "listingCity": "Listing city",
    "fuelType": "Fuel type",
    "transmission": "Transmission",
    "bodyType": "Body type",
    "assemblyType": "Assembly",
    "engineCategory": "Engine category",
    "ageCategory": "Age category",
    "mileageCategory": "Mileage category",
}

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parents[1] / "reports"


def model_root() -> Path:
    root = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_ROOT))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def report_root() -> Path:
    root = Path(os.getenv("REPORT_DIR", str(DEFAULT_REPORT_ROOT))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _one_hot_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def _ordinal_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def _catboost_model(
    *,
    iterations: int = 1_200,
    depth: int = 8,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 6,
    random_strength: float = 0.5,
    bagging_temperature: float = 0.4,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        eval_metric="MAE",
        l2_leaf_reg=l2_leaf_reg,
        random_seed=SEED,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def candidates() -> dict[str, Any]:
    all_candidates = {
        "median_baseline": {
            "family": "sklearn",
            "configuration": {"strategy": "median"},
            "model": Pipeline([
                ("preprocess", _one_hot_preprocessor()),
                ("model", DummyRegressor(strategy="median")),
            ]),
        },
        "extra_trees_220": {
            "family": "sklearn",
            "configuration": {
                "n_estimators": 220,
                "max_features": 0.75,
                "min_samples_leaf": 1,
                "max_depth": None,
            },
            "model": Pipeline([
                ("preprocess", _ordinal_preprocessor()),
                ("model", ExtraTreesRegressor(
                    n_estimators=220,
                    max_features=0.75,
                    min_samples_leaf=1,
                    max_depth=None,
                    n_jobs=1,
                    random_state=SEED,
                )),
            ]),
        },
        "extra_trees_320_regularized": {
            "family": "sklearn",
            "configuration": {
                "n_estimators": 320,
                "max_features": 1.0,
                "min_samples_leaf": 2,
                "max_depth": None,
            },
            "model": Pipeline([
                ("preprocess", _ordinal_preprocessor()),
                ("model", ExtraTreesRegressor(
                    n_estimators=320,
                    max_features=1.0,
                    min_samples_leaf=2,
                    max_depth=None,
                    n_jobs=1,
                    random_state=SEED,
                )),
            ]),
        },
        "random_forest_180_leaf_2": {
            "family": "sklearn",
            "configuration": {
                "n_estimators": 180,
                "max_features": 0.8,
                "min_samples_leaf": 2,
                "max_depth": None,
            },
            "model": Pipeline([
                ("preprocess", _ordinal_preprocessor()),
                ("model", RandomForestRegressor(
                    n_estimators=180,
                    max_features=0.8,
                    min_samples_leaf=2,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=SEED,
                )),
            ]),
        },
        "hist_gradient_320": {
            "family": "sklearn",
            "configuration": {
                "max_iter": 320,
                "learning_rate": 0.06,
                "max_leaf_nodes": 40,
                "min_samples_leaf": 25,
                "l2_regularization": 1.5,
            },
            "model": Pipeline([
                ("preprocess", _ordinal_preprocessor()),
                ("model", HistGradientBoostingRegressor(
                    max_iter=320,
                    learning_rate=0.06,
                    max_leaf_nodes=40,
                    min_samples_leaf=25,
                    l2_regularization=1.5,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=30,
                    random_state=SEED,
                )),
            ]),
        },
        "hist_gradient_450": {
            "family": "sklearn",
            "configuration": {
                "max_iter": 450,
                "learning_rate": 0.045,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 18,
                "l2_regularization": 1.0,
            },
            "model": Pipeline([
                ("preprocess", _ordinal_preprocessor()),
                ("model", HistGradientBoostingRegressor(
                    max_iter=450,
                    learning_rate=0.045,
                    max_leaf_nodes=63,
                    min_samples_leaf=18,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=35,
                    random_state=SEED,
                )),
            ]),
        },
        "catboost_depth_7": {
            "family": "catboost",
            "configuration": {
                "iterations": 1_100,
                "depth": 7,
                "learning_rate": 0.065,
                "l2_leaf_reg": 4,
                "random_strength": 0.7,
                "bagging_temperature": 0.2,
            },
            "model": _catboost_model(
                iterations=1_100,
                depth=7,
                learning_rate=0.065,
                l2_leaf_reg=4,
                random_strength=0.7,
                bagging_temperature=0.2,
            ),
        },
        "catboost_depth_8": {
            "family": "catboost",
            "configuration": {
                "iterations": 1_300,
                "depth": 8,
                "learning_rate": 0.05,
                "l2_leaf_reg": 6,
                "random_strength": 0.5,
                "bagging_temperature": 0.4,
            },
            "model": _catboost_model(iterations=1_300),
        },
        "catboost_depth_9": {
            "family": "catboost",
            "configuration": {
                "iterations": 1_500,
                "depth": 9,
                "learning_rate": 0.035,
                "l2_leaf_reg": 8,
                "random_strength": 0.35,
                "bagging_temperature": 0.6,
            },
            "model": _catboost_model(
                iterations=1_500,
                depth=9,
                learning_rate=0.035,
                l2_leaf_reg=8,
                random_strength=0.35,
                bagging_temperature=0.6,
            ),
        },
    }
    # Keep selection bounded, but include the strongest available categorical
    # learner.  CatBoost uses early stopping; models are selected on
    # development CV, never on the final test partition.
    selected = (
        "median_baseline", "extra_trees_220", "extra_trees_320_regularized",
        "random_forest_180_leaf_2", "hist_gradient_320", "catboost_depth_7",
    )
    return {name: all_candidates[name] for name in selected}


def regression_metrics(y_true: pd.Series | np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    safe_denominator = np.maximum(np.abs(actual), 100_000)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "r2": round(float(r2_score(actual, predicted)), 4) if len(actual) > 1 else 0.0,
        "mape": round(float(np.mean(np.abs((actual - predicted) / safe_denominator)) * 100), 3),
        "medianAbsoluteError": round(float(median_absolute_error(actual, predicted)), 2),
    }


def _round_price(value: float) -> int:
    return max(0, int(round(float(value) / 1_000) * 1_000))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def _feature_schema(catalog: dict[str, Any], numeric_ranges: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Public request contract saved beside every versioned model artifact."""
    return {
        "version": 1,
        "required": ["make", "model", "year", "mileage", "engineCapacity"],
        "fields": {
            "make": {"type": "string", "minLength": 1, "acceptedValuesSource": "inputCatalog.makes"},
            "model": {"type": "string", "minLength": 1, "acceptedValuesSource": "inputCatalog.makes[].models"},
            "year": {"type": "integer", "minimum": int(numeric_ranges["modelYear"]["min"]), "maximum": int(numeric_ranges["modelYear"]["max"])},
            "mileage": {"type": "number", "minimum": int(numeric_ranges["mileage"]["min"]), "maximum": int(numeric_ranges["mileage"]["max"])},
            "engineCapacity": {"type": "number", "minimum": int(numeric_ranges["engineCapacity"]["min"]), "maximum": int(numeric_ranges["engineCapacity"]["max"])},
            "transmission": {"type": "string", "acceptedValuesSource": "inputCatalog.transmissions"},
            "fuelType": {"type": "string", "acceptedValuesSource": "inputCatalog.fuelTypes"},
            "city": {"type": "string", "acceptedValuesSource": "inputCatalog.listingCities"},
            "registrationCity": {"type": "string", "acceptedValuesSource": "inputCatalog.registrationCities"},
            "bodyType": {"type": "string", "acceptedValuesSource": "inputCatalog.bodyTypes"},
            "assemblyType": {"type": "string", "acceptedValuesSource": "inputCatalog.assemblyTypes"},
            "variant": {"type": "string", "required": False},
            "condition": {"type": "string", "required": False, "note": "Accepted by the API but not used by this dataset-trained model."},
        },
        "unseenCategoryPolicy": "Unseen categories are accepted by the prediction pipeline. They use the model's unknown-category handling and the broader global conformal range.",
        "catalog": catalog,
    }


def _registry() -> dict[str, Any]:
    path = model_root() / "registry.json"
    if not path.exists():
        return {"activeVersion": None, "previousVersion": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(registry: dict[str, Any]) -> None:
    _write_json(model_root() / "registry.json", registry)


def list_models() -> dict[str, Any]:
    registry = _registry()
    versions = []
    for entry in registry.get("versions", []):
        enriched = dict(entry)
        metadata: dict[str, Any] = {}
        version = entry.get("version")
        if isinstance(version, str) and version.strip():
            metadata_path = model_root() / version / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
        recorded = metadata.get("datasetProvenance") or {}
        enriched["provenance"] = {
            "datasetName": recorded.get("datasetName") or metadata.get("datasetFile") or "Legacy metadata unavailable",
            "sourceCategory": recorded.get("sourceCategory") or "Legacy metadata unavailable",
            "datasetFingerprint": metadata.get("datasetFingerprint") or "Legacy metadata unavailable",
            "datasetRows": metadata.get("datasetSize", entry.get("datasetSize")),
            "trainedRows": metadata.get("trainedRows", entry.get("trainedRows")),
            "trainingDate": metadata.get("trainingDate", entry.get("trainingDate")),
            "sources": recorded.get("sources", []),
        }
        versions.append(enriched)
    return {**registry, "versions": versions}


def activate_model(version: str) -> dict[str, Any]:
    registry = _registry()
    if not (model_root() / version / "model.joblib").exists():
        raise ValueError("Model version does not exist")
    if registry.get("activeVersion") != version:
        registry["previousVersion"] = registry.get("activeVersion")
        registry["activeVersion"] = version
        _write_registry(registry)
    return registry


def rollback_model() -> dict[str, Any]:
    registry = _registry()
    previous = registry.get("previousVersion")
    if not previous:
        raise ValueError("No previous model version is available")
    active = registry.get("activeVersion")
    registry["activeVersion"] = previous
    registry["previousVersion"] = active
    _write_registry(registry)
    return registry


def load_active_bundle() -> dict[str, Any] | None:
    version = _registry().get("activeVersion")
    if not version:
        return None
    path = model_root() / version / "model.joblib"
    return joblib.load(path) if path.exists() else None


def _normalise_candidate(spec: Any) -> tuple[str, Any]:
    if isinstance(spec, dict) and "model" in spec:
        return str(spec.get("family", "sklearn")), spec["model"]
    return "sklearn", spec


def _fit_candidate(
    family: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Any:
    if family == "catboost":
        model.fit(
            x_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(x_validation, y_validation),
            early_stopping_rounds=75,
            verbose=False,
        )
        return model
    model.fit(x_train, y_train)
    return model


def _predict(family: str, model: Any, frame: pd.DataFrame) -> np.ndarray:
    if family == "catboost":
        return np.asarray(model.predict(frame), dtype=float)
    return np.asarray(model.predict(frame), dtype=float)


def _refit_selected(
    family: str,
    model: Any,
    x_train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
) -> Any:
    if family == "catboost":
        best_iteration = model.get_best_iteration()
        parameters = model.get_params()
        configured_iterations = int(parameters.get("iterations", 1_200))
        iterations = max(80, int(best_iteration) + 1) if best_iteration is not None and best_iteration >= 0 else configured_iterations
        parameters["iterations"] = iterations
        final = CatBoostRegressor(**parameters)
        final.fit(
            x_train_validation,
            y_train_validation,
            cat_features=CATEGORICAL_FEATURES,
            verbose=False,
        )
        return final
    final = clone(model)
    final.fit(x_train_validation, y_train_validation)
    return final


def _conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    values = np.asarray(residuals, dtype=float)
    if not len(values):
        raise ValueError("Calibration residuals are empty")
    alpha = 1.0 - coverage
    corrected = min(1.0, math.ceil((len(values) + 1) * (1 - alpha)) / len(values))
    return float(np.quantile(values, corrected, method="higher"))


def _interval_calibration(
    calibration_features: pd.DataFrame,
    residuals: np.ndarray,
) -> dict[str, Any]:
    """Mondrian split-conformal intervals, falling back to global residuals.

    Make/model groups with sufficient held-out calibration examples receive
    their own finite-sample residual quantile. Rare and unseen vehicles use
    the global quantile, so the system never invents narrow certainty.
    """
    data = calibration_features[["makeModel"]].copy()
    data["absoluteResidual"] = np.asarray(residuals, dtype=float)
    calibration_by_coverage: dict[str, Any] = {}
    for coverage in INTERVAL_COVERAGES:
        global_q = _conformal_quantile(data["absoluteResidual"].to_numpy(), coverage)
        group_quantiles: dict[str, float] = {}
        for name, group in data.groupby("makeModel", dropna=False):
            if len(group) >= MIN_GROUP_CALIBRATION_ROWS:
                group_quantiles[str(name)] = round(
                    _conformal_quantile(group["absoluteResidual"].to_numpy(), coverage), 2
                )
        widths = np.array([group_quantiles.get(str(name), global_q) * 2 for name in data["makeModel"]])
        calibration_by_coverage[str(coverage)] = {
            "globalAbsoluteResidualQuantilePkr": round(float(global_q), 2),
            "makeModelQuantilesPkr": group_quantiles,
            "groupsWithDedicatedCalibration": len(group_quantiles),
            "medianIntervalWidthPkr": round(float(np.median(widths)), 2),
            "averageIntervalWidthPkr": round(float(np.mean(widths)), 2),
        }
    chosen = calibration_by_coverage[str(SELECTED_INTERVAL_COVERAGE)]
    return {
        "method": "mondrian_split_conformal_absolute_residual",
        "targetCoverage": SELECTED_INTERVAL_COVERAGE,
        "minimumMakeModelCalibrationRows": MIN_GROUP_CALIBRATION_ROWS,
        "calibrationRows": int(len(data)),
        "coverageOptions": calibration_by_coverage,
        **chosen,
        "note": "Uses held-out absolute residuals. Common make/model groups use their own calibration quantile; rare or unseen groups use the global quantile.",
    }


def _interval_errors(frame: pd.DataFrame, calibration: dict[str, Any], coverage: float) -> np.ndarray:
    option = calibration["coverageOptions"][str(coverage)]
    global_q = float(option["globalAbsoluteResidualQuantilePkr"])
    group_q = option.get("makeModelQuantilesPkr", {})
    return frame["makeModel"].map(lambda value: group_q.get(str(value), global_q)).to_numpy(dtype=float)


def _feature_importance(
    family: str,
    model: Any,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> list[dict[str, Any]]:
    if family == "catboost":
        values = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        sample_size = min(2_500, len(x_validation))
        sample = x_validation.sample(sample_size, random_state=SEED)
        target = y_validation.loc[sample.index]
        result = permutation_importance(
            model,
            sample,
            target,
            scoring="neg_mean_absolute_error",
            n_repeats=2,
            random_state=SEED,
            n_jobs=1,
        )
        values = np.maximum(0, np.asarray(result.importances_mean, dtype=float))
    total = float(values.sum()) or 1.0
    items = [
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "importance": round(float(value / total), 6),
        }
        for feature, value in zip(FEATURES, values)
    ]
    return sorted(items, key=lambda item: item["importance"], reverse=True)


def _exploratory_report(data: pd.DataFrame, cleaning_report: dict[str, Any]) -> dict[str, Any]:
    def distribution(column: str) -> dict[str, float | None]:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            return {"min": None, "p01": None, "p25": None, "median": None, "p75": None, "p99": None, "max": None}
        return {
            name: round(float(value), 2)
            for name, value in zip(
                ("min", "p01", "p25", "median", "p75", "p99", "max"),
                values.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).tolist(),
            )
        }

    prices = pd.to_numeric(data["targetPrice"], errors="coerce").dropna()
    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = int(((prices < q1 - 1.5 * iqr) | (prices > q3 + 1.5 * iqr)).sum())
    return {
        "datasetSize": len(data),
        "featureCoveragePercent": {
            column: round(float(data[column].notna().mean() * 100), 2)
            for column in FEATURES
        },
        "missingValuePercentagesBeforeCleaning": cleaning_report.get("missingValuePercentagesBeforeCleaning", {}),
        "priceDistribution": distribution("targetPrice"),
        "mileageDistribution": distribution("mileage"),
        "modelYearDistribution": distribution("modelYear"),
        "engineSizeDistribution": distribution("engineCapacity"),
        "commonMakes": {str(key): int(value) for key, value in data["make"].value_counts().head(25).items()},
        "commonModels": {str(key): int(value) for key, value in data["model"].value_counts().head(40).items()},
        "outlierSummary": {
            "iqrPriceOutliersRetained": outliers,
            "note": "IQR outliers inside documented hard bounds are retained because the dataset legitimately includes premium vehicles.",
        },
    }


def _metrics_by_group(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum_rows: int = 2,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    counts = frame[column].value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    for value, count in counts.items():
        if int(count) < minimum_rows:
            continue
        mask = frame[column].eq(value)
        if pd.isna(value):
            mask = frame[column].isna()
        subset = frame.loc[mask]
        groups.append({
            "group": "Unknown" if pd.isna(value) else str(value),
            "rows": int(len(subset)),
            **regression_metrics(subset["actualPrice"], subset["predictedPrice"].to_numpy()),
        })
    return groups


def _segmented_error_report(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    development_data: pd.DataFrame,
) -> dict[str, Any]:
    analysis = x_test[["make", "makeModel", "vehicleAge"]].copy()
    analysis["actualPrice"] = y_test.to_numpy(dtype=float)
    analysis["predictedPrice"] = np.asarray(predictions, dtype=float)
    analysis["priceBand"] = pd.cut(
        analysis["actualPrice"],
        bins=[0, 1_000_000, 2_500_000, 5_000_000, 10_000_000, np.inf],
        labels=["Under PKR 1m", "PKR 1m–2.5m", "PKR 2.5m–5m", "PKR 5m–10m", "PKR 10m+"],
        include_lowest=True,
    ).astype(str)
    analysis["vehicleAgeBand"] = pd.cut(
        analysis["vehicleAge"],
        bins=[-1, 2, 5, 10, 15, np.inf],
        labels=["0–2 years", "3–5 years", "6–10 years", "11–15 years", "16+ years"],
    ).astype(str)

    make_counts = development_data["make"].value_counts()
    model_counts = development_data["makeModel"].value_counts()
    analysis["makeCoverage"] = np.where(
        analysis["make"].map(make_counts).fillna(0) < 100,
        "Rare make (<100 development rows)",
        "Common make",
    )
    analysis["modelCoverage"] = np.where(
        analysis["makeModel"].map(model_counts).fillna(0) < 30,
        "Rare make-model (<30 development rows)",
        "Common make-model",
    )
    return {
        "overall": regression_metrics(analysis["actualPrice"], analysis["predictedPrice"].to_numpy()),
        "majorMakes": _metrics_by_group(analysis, "make", minimum_rows=20, limit=12),
        "commonModels": _metrics_by_group(analysis, "makeModel", minimum_rows=15, limit=20),
        "priceBands": _metrics_by_group(analysis, "priceBand", minimum_rows=2),
        "vehicleAgeBands": _metrics_by_group(analysis, "vehicleAgeBand", minimum_rows=2),
        "makeCoverage": _metrics_by_group(analysis, "makeCoverage", minimum_rows=2),
        "modelCoverage": _metrics_by_group(analysis, "modelCoverage", minimum_rows=2),
        "coverageDefinitions": {
            "rareMake": "Fewer than 100 rows in the development data.",
            "rareMakeModel": "Fewer than 30 rows in the development data.",
        },
    }


def train_from_frame(raw_frame: pd.DataFrame, file_name: str = "", provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    inspection_report = inspect_dataframe(raw_frame, file_name=file_name)
    cleaned, cleaning_report = clean_dataframe(raw_frame)
    if len(cleaned) < MIN_TRAINING_ROWS:
        raise ValueError(f"At least {MIN_TRAINING_ROWS} usable records are required; received {len(cleaned)}")

    reference_year = determine_reference_year(cleaned)
    data = add_features(cleaned, reference_year=reference_year)
    x = data[FEATURES].copy()
    y = data["targetPrice"].astype(float)

    # Reserve the final test set before any model/preprocessor is fitted.  The
    # development set is used only for cross-validated model selection; a
    # separate calibration set is retained for prediction intervals.
    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.10, random_state=SEED,
    )
    x_train_validation, x_calibration, y_train_validation, y_calibration = train_test_split(
        x_development,
        y_development,
        test_size=1 / 9,
        random_state=SEED,
    )
    fold_count = min(CV_FOLDS, len(x_train_validation))
    if fold_count < 2:
        raise ValueError("At least two development records are required for cross-validation")
    folds = KFold(n_splits=fold_count, shuffle=True, random_state=SEED)

    candidate_specs = candidates()
    comparisons: dict[str, Any] = {}
    best_candidate: tuple[str, Any] | None = None
    best_candidate_key: tuple[float, float, float, float] | None = None
    best_candidate_name: str | None = None
    successful_trials = 0
    failed_trials = 0
    for trial_number, (name, specification) in enumerate(candidate_specs.items(), start=1):
        family, model = _normalise_candidate(specification)
        started = time.perf_counter()
        try:
            fold_metrics: list[dict[str, float]] = []
            fitted_model = None
            for train_indices, validation_indices in folds.split(x_train_validation):
                x_train = x_train_validation.iloc[train_indices]
                y_train = y_train_validation.iloc[train_indices]
                x_validation = x_train_validation.iloc[validation_indices]
                y_validation = y_train_validation.iloc[validation_indices]
                # clone ensures imputation/encoding are fitted inside each fold,
                # rather than leaking category frequencies or medians across folds.
                fitted_model = _fit_candidate(
                    family,
                    clone(model),
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                )
                fold_metrics.append(regression_metrics(y_validation, _predict(family, fitted_model, x_validation)))
            if fitted_model is None:
                raise RuntimeError("Cross-validation did not fit a model")
            training_seconds = time.perf_counter() - started
            inference_sample = x_train_validation.iloc[: min(1_000, len(x_train_validation))]
            inference_started = time.perf_counter()
            _predict(family, fitted_model, inference_sample)
            inference_ms = (time.perf_counter() - inference_started) * 1_000
            validation_metrics = {
                metric: round(float(np.mean([item[metric] for item in fold_metrics])), 4)
                for metric in fold_metrics[0]
            }
            validation_std = {
                metric: round(float(np.std([item[metric] for item in fold_metrics], ddof=0)), 4)
                for metric in fold_metrics[0]
            }
            comparisons[name] = {
                "trialNumber": trial_number,
                "status": "completed",
                "family": family,
                "configuration": specification.get("configuration", {}) if isinstance(specification, dict) else {"estimator": type(model).__name__},
                "validation": validation_metrics,
                "crossValidation": {"folds": fold_count, "mean": validation_metrics, "standardDeviation": validation_std},
                "trainingSeconds": round(training_seconds, 3),
                "inferenceMillisecondsPer1000Rows": round(inference_ms, 3),
            }
            if family == "catboost":
                comparisons[name]["bestIteration"] = int(fitted_model.get_best_iteration())
            candidate_key = (
                validation_metrics["mae"],
                validation_metrics["rmse"],
                validation_std["mae"],
                comparisons[name]["inferenceMillisecondsPer1000Rows"],
            )
            if best_candidate_key is None or candidate_key < best_candidate_key:
                best_candidate_key = candidate_key
                best_candidate = (family, fitted_model)
                best_candidate_name = name
            successful_trials += 1
        except Exception as error:
            failed_trials += 1
            comparisons[name] = {
                "trialNumber": trial_number,
                "status": "failed",
                "family": family,
                "configuration": specification.get("configuration", {}) if isinstance(specification, dict) else {"estimator": type(model).__name__},
                "trainingSeconds": round(time.perf_counter() - started, 3),
                "errorType": type(error).__name__,
                "error": str(error)[:500],
            }

    if best_candidate is None or best_candidate_name is None:
        raise RuntimeError("Every configured model-training trial failed")
    selected_name = best_candidate_name
    selected_family, selected_candidate = best_candidate
    selected = _refit_selected(
        selected_family,
        selected_candidate,
        x_train_validation,
        y_train_validation,
    )

    calibration_prediction = _predict(selected_family, selected, x_calibration)
    calibration_residuals = np.abs(y_calibration.to_numpy(dtype=float) - calibration_prediction)
    range_calibration = _interval_calibration(x_calibration, calibration_residuals)

    test_prediction = _predict(selected_family, selected, x_test)
    final_metrics = regression_metrics(y_test, test_prediction)
    error_analysis = _segmented_error_report(
        x_test,
        y_test,
        test_prediction,
        data.loc[x_train_validation.index],
    )
    interval_evaluation = {}
    for coverage in INTERVAL_COVERAGES:
        errors = _interval_errors(x_test, range_calibration, coverage)
        lower = np.maximum(0, test_prediction - errors)
        upper = test_prediction + errors
        covered = (y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)
        interval_evaluation[str(coverage)] = {
            "targetCoverage": coverage,
            "empiricalTestCoverage": round(float(np.mean(covered)), 4),
            "averageIntervalWidthPkr": round(float(np.mean(upper - lower)), 2),
            "medianIntervalWidthPkr": round(float(np.median(upper - lower)), 2),
        }
    range_calibration["testEvaluationByCoverage"] = interval_evaluation
    range_calibration["empiricalTestCoverage"] = interval_evaluation[str(SELECTED_INTERVAL_COVERAGE)]["empiricalTestCoverage"]
    range_calibration["testRows"] = int(len(x_test))
    importance = _feature_importance(
        selected_family,
        selected,
        x_calibration,
        y_calibration,
    )

    timestamp = datetime.now(timezone.utc)
    fingerprint = dataframe_fingerprint(cleaned)
    version = f"ec-{timestamp.strftime('%Y%m%d-%H%M%S')}-{fingerprint[:8]}"
    model_path = model_root() / version
    model_path.mkdir(parents=True, exist_ok=False)
    catalog = build_input_catalog(cleaned)
    exploratory = _exploratory_report(data, cleaning_report)
    comparable_columns = [
        "make", "model", "modelYear", "mileage", "engineCapacity",
        "transmission", "fuelType", "listingCity", "registrationCity",
        "bodyType", "assemblyType", "targetPrice",
    ]
    comparables = cleaned[comparable_columns].copy()
    numeric_ranges = {
        column: {
            "min": float(data[column].min()),
            "max": float(data[column].max()),
        }
        for column in ("modelYear", "mileage", "engineCapacity")
    }
    split_summary = {
        "seed": SEED,
        "developmentRows": int(len(x_train_validation)),
        "crossValidationFolds": fold_count,
        "calibrationRows": int(len(x_calibration)),
        "testRows": int(len(x_test)),
        "selectionRule": "Lowest cross-validation MAE, then RMSE, MAE stability, then inference time as tie-breakers.",
    }
    tuning_summary = {
        "method": "Controlled three-fold cross-validation search with a fixed candidate budget and CatBoost early stopping.",
        "trialsPlanned": int(len(candidate_specs)),
        "trialsCompleted": int(successful_trials),
        "trialsFailed": int(failed_trials),
        "randomSeed": SEED,
        "selectedConfiguration": comparisons[selected_name].get("configuration", {}),
        "selectionRule": split_summary["selectionRule"],
    }
    bundle = {
        "model": selected,
        "pipeline": selected,
        "modelFamily": selected_family,
        "modelVersion": version,
        "modelName": selected_name,
        "trainingDate": timestamp.isoformat(),
        "trainedRows": int(len(x_train_validation)),
        "datasetSize": int(len(cleaned)),
        "datasetFile": file_name,
        "datasetReferenceYear": reference_year,
        "dataFreshnessDate": None,
        "features": FEATURES,
        "numericFeatures": NUMERIC_FEATURES,
        "categoricalFeatures": CATEGORICAL_FEATURES,
        "metrics": final_metrics,
        "modelComparisons": comparisons,
        "tuningSummary": tuning_summary,
        "splitSummary": split_summary,
        "errorAnalysis": error_analysis,
        "rangeCalibration": range_calibration,
        "featureImportance": importance,
        "cleaningReport": cleaning_report,
        "datasetInspectionReport": inspection_report,
        "exploratoryReport": exploratory,
        "numericRanges": numeric_ranges,
        "inputCatalog": catalog,
        "comparables": comparables,
        "datasetFingerprint": fingerprint,
        "datasetProvenance": provenance or {
            "datasetName": file_name or "Legacy metadata unavailable",
            "sourceCategory": "Legacy metadata unavailable",
            "sources": [],
        },
        "limitations": [
            "The source contains listing prices rather than confirmed transaction prices.",
            "The source has no listing date and its newest model year is used as a fixed feature-engineering reference year.",
            "The model does not account for inflation or market movement after the dataset snapshot.",
            "Condition and inspection features are excluded because this dataset does not supply them reliably.",
            "Colour is excluded because it is high-cardinality and was not required for the core model.",
        ],
    }
    # Loading happens once at FastAPI startup.  Avoid CPU-heavy compression so
    # retraining finishes reliably on ordinary student-development machines.
    joblib.dump(bundle, model_path / "model.joblib", compress=0)
    artifact_size_bytes = int((model_path / "model.joblib").stat().st_size)
    if selected_family == "catboost":
        selected.save_model(str(model_path / "model.cbm"))

    metadata = {
        key: value for key, value in bundle.items()
        if key not in {"model", "pipeline", "comparables"}
    }
    metadata["artifactSizeBytes"] = artifact_size_bytes
    schema = _feature_schema(catalog, numeric_ranges)
    _write_json(model_path / "feature_schema.json", schema)
    metadata["featureSchemaFile"] = "feature_schema.json"
    _write_json(model_path / "metadata.json", metadata)

    evaluation_report = {
        "modelVersion": version,
        "selectedModel": selected_name,
        "selectedFamily": selected_family,
        "selectionMetrics": comparisons[selected_name]["validation"],
        "testMetrics": final_metrics,
        "modelComparisons": comparisons,
        "tuningSummary": tuning_summary,
        "splitSummary": split_summary,
        "errorAnalysis": error_analysis,
        "predictionRange": range_calibration,
        "featureImportance": importance,
        "featuresUsed": FEATURES,
        "actualVsPredictedSamples": [
            {
                "make": str(data.loc[index, "make"]),
                "model": str(data.loc[index, "model"]),
                "year": int(data.loc[index, "modelYear"]),
                "mileage": int(data.loc[index, "mileage"]),
                "actualPrice": int(actual),
                "predictedPrice": _round_price(predicted),
                "absoluteError": _round_price(abs(float(actual) - float(predicted))),
            }
            for index, actual, predicted in zip(x_test.index[:10], y_test.iloc[:10], test_prediction[:10])
        ],
        "featuresExcluded": {
            "addref": "Identifier; excluded to prevent leakage.",
            "colour": "High-cardinality optional field; excluded from the core maintained model.",
            "condition": "Not present in the attached source.",
            "inspectionScore": "Not present in the attached source.",
        },
        "limitations": bundle["limitations"],
        "artifactSizeBytes": artifact_size_bytes,
    }
    reports = report_root()
    _write_json(reports / "dataset_inspection_report.json", inspection_report)
    _write_json(reports / "data_cleaning_report.json", cleaning_report)
    _write_json(reports / "model_evaluation.json", evaluation_report)
    _write_json(reports / "vehicle_catalog.json", catalog)
    _write_json(reports / "feature_schema.json", schema)

    registry = _registry()
    registry["previousVersion"] = registry.get("activeVersion")
    registry["activeVersion"] = version
    registry.setdefault("versions", []).insert(0, {
        "version": version,
        "modelName": selected_name,
        "modelFamily": selected_family,
        "trainingDate": timestamp.isoformat(),
        "datasetSize": len(cleaned),
        "trainedRows": len(x_train_validation),
        "metrics": final_metrics,
        "rangeCoverage": range_calibration,
        "artifactSizeBytes": artifact_size_bytes,
        "provenance": bundle["datasetProvenance"],
    })
    _write_registry(registry)
    return _json_safe({
        "version": version,
        "selectedModel": selected_name,
        "selectedFamily": selected_family,
        "metrics": final_metrics,
        "modelComparisons": comparisons,
        "tuningSummary": tuning_summary,
        "cleaningReport": cleaning_report,
        "datasetInspectionReport": inspection_report,
        "exploratoryReport": exploratory,
        "predictionRange": range_calibration,
        "errorAnalysis": error_analysis,
        "splitSummary": split_summary,
    })


def train_from_records(records: list[dict[str, Any]], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    return train_from_frame(pd.DataFrame.from_records(records), file_name="database-export", provenance=provenance)
