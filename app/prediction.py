from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from .data import add_features
from .schemas import PredictionInput
from .training import FEATURE_LABELS


class UnsupportedVehicleError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("The selected vehicle is not supported by the active dataset.")
        self.errors = errors


def _round_price(value: float) -> int:
    return max(0, int(round(float(value) / 1_000) * 1_000))


def _input_frame(payload: PredictionInput, reference_year: int) -> pd.DataFrame:
    row = payload.model_dump()
    row.update({
        "modelYear": row.pop("year"),
        "listingCity": row.pop("city") or "Unknown",
        "registrationCity": row.get("registrationCity") or "Unknown",
        "bodyType": row.get("bodyType") or "Unknown",
        "assemblyType": row.get("assemblyType") or "Unknown",
        "fuelType": row.get("fuelType") or "Unknown",
        "transmission": row.get("transmission") or "Unknown",
        "listingDate": datetime.now(timezone.utc),
    })
    return add_features(pd.DataFrame([row]), reference_year=reference_year)


def _catalog_warnings(payload: PredictionInput, catalog: dict[str, Any]) -> list[str]:
    make_map = {
        str(item.get("make", "")).casefold(): item
        for item in catalog.get("makes", [])
    }
    selected_make = make_map.get(payload.make.casefold())
    warnings: list[str] = []
    if not selected_make:
        warnings.append("This make is not represented in the active dataset, so the market range is intentionally broader.")
    else:
        supported_models = {
            str(item.get("model", "")).casefold()
            for item in selected_make.get("models", [])
        }
        if payload.model.casefold() not in supported_models:
            warnings.append("This model is not represented for the selected make, so the market range is intentionally broader.")
    return warnings


def _comparable_rows(payload: PredictionInput, records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    data = records.copy()
    same_make = data["make"].astype(str).str.casefold() == payload.make.casefold()
    candidates = data.loc[same_make].copy()
    if candidates.empty:
        return candidates
    candidates["distance"] = (
        (candidates["model"].astype(str).str.casefold() != payload.model.casefold()).astype(float) * 0.75
        + (pd.to_numeric(candidates["modelYear"], errors="coerce") - payload.year).abs().clip(upper=10) / 10 * 0.45
        + (pd.to_numeric(candidates["mileage"], errors="coerce") - payload.mileage).abs().clip(upper=250_000) / 250_000 * 0.25
        + (
            (candidates["transmission"].astype(str).str.casefold() != payload.transmission.casefold())
            & bool(payload.transmission)
        ).astype(float) * 0.12
    )
    return candidates.sort_values("distance").head(25)


def _display_value(feature: str, row: pd.Series) -> str:
    value = row.get(feature, "")
    if feature in {"mileage", "mileagePerYear"} and pd.notna(value):
        return f"{float(value):,.0f} km"
    if feature == "engineCapacity" and pd.notna(value):
        return f"{float(value):,.0f} cc"
    if feature == "modelYear" and pd.notna(value):
        return str(int(round(float(value))))
    if feature == "vehicleAge" and pd.notna(value):
        years = int(round(float(value)))
        return f"{years} year{'s' if years != 1 else ''}"
    return str(value or "Unknown")


def _important_factors(bundle: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    features = bundle["features"]
    row = frame.iloc[0]
    family = bundle.get("modelFamily", "sklearn")
    model = bundle.get("model")
    if model is None:
        model = bundle.get("pipeline")
    if family == "catboost":
        pool = Pool(
            frame[features],
            cat_features=bundle.get("categoricalFeatures", []),
        )
        shap_values = np.asarray(
            model.get_feature_importance(pool, type="ShapValues"),
            dtype=float,
        )[0][:-1]
        ordered = np.argsort(np.abs(shap_values))[::-1]
        factors = []
        for index in ordered:
            feature = features[int(index)]
            contribution = float(shap_values[int(index)])
            if abs(contribution) < 1:
                continue
            factors.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": _display_value(feature, row),
                "direction": "higher" if contribution > 0 else "lower",
                "impactPkr": _round_price(abs(contribution)),
                "method": "local_shap",
            })
            if len(factors) == 5:
                break
        return factors

    return [
        {
            "feature": item["feature"],
            "label": item["label"],
            "value": _display_value(item["feature"], row),
            "direction": "model influence",
            "importance": item["importance"],
            "method": "held_out_permutation_importance",
        }
        for item in bundle.get("featureImportance", [])[:5]
    ]


def _warnings(payload: PredictionInput, bundle: dict[str, Any]) -> list[str]:
    ranges = bundle.get("numericRanges", {})
    warnings = []
    checks = {
        "modelYear": (payload.year, "Model year"),
        "mileage": (payload.mileage, "Mileage"),
        "engineCapacity": (payload.engineCapacity, "Engine capacity"),
    }
    for field, (value, label) in checks.items():
        bounds = ranges.get(field)
        if value is None or not bounds:
            continue
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{label} is outside the active dataset range ({bounds['min']:,.0f}–{bounds['max']:,.0f}); the estimate is an extrapolation."
            )
    return warnings


def predict(bundle: dict[str, Any], payload: PredictionInput) -> dict[str, Any]:
    catalog_warnings = _catalog_warnings(payload, bundle.get("inputCatalog", {}))
    reference_year = int(bundle.get("datasetReferenceYear") or datetime.now(timezone.utc).year)
    frame = _input_frame(payload, reference_year=reference_year)
    model = bundle.get("model")
    if model is None:
        model = bundle.get("pipeline")
    estimate_raw = max(0.0, float(model.predict(frame[bundle["features"]])[0]))
    estimate = _round_price(estimate_raw)

    calibration = bundle.get("rangeCalibration", {})
    if calibration.get("method") == "mondrian_split_conformal_absolute_residual":
        group_quantiles = calibration.get("makeModelQuantilesPkr", {})
        interval_error = float(group_quantiles.get(str(frame.iloc[0].get("makeModel")), calibration.get("globalAbsoluteResidualQuantilePkr", 0)))
    else:
        interval_error = float(calibration.get("absoluteResidualQuantilePkr", 0))
    low = _round_price(max(0, estimate_raw - interval_error))
    high = _round_price(estimate_raw + interval_error)
    comparables = _comparable_rows(payload, bundle.get("comparables", pd.DataFrame()))
    metrics = bundle.get("metrics", {})
    important_factors = _important_factors(bundle, frame)
    warnings = catalog_warnings + _warnings(payload, bundle)
    disclaimer = "Estimated from historical Pakistani used-car listing data; it is not a guaranteed sale price or offer."

    return {
        # Snake-case keys are the stable service contract.  The existing
        # camel-case aliases remain during the frontend transition.
        "predicted_price": estimate,
        "estimatedPrice": estimate,
        "estimatedMarketPrice": estimate,
        "predicted": estimate,
        "lowerRange": low,
        "upperRange": high,
        "recommendedRange": {"low": low, "high": high},
        "range": {"low": low, "high": high},
        "currency": "PKR",
        "modelName": bundle.get("modelName"),
        "modelVersion": bundle.get("modelVersion"),
        "modelMetrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2"),
            "medianAbsoluteError": metrics.get("medianAbsoluteError"),
        },
        "rangeMethod": {
            "method": calibration.get("method"),
            "targetCoverage": calibration.get("targetCoverage"),
            "empiricalTestCoverage": calibration.get("empiricalTestCoverage"),
            "calibrationRows": calibration.get("calibrationRows"),
        },
        "importantFactors": important_factors,
        "mainPricingFactors": important_factors,
        "comparableVehicleCount": int(len(comparables)),
        "datasetReferenceYear": reference_year,
        "datasetSize": bundle.get("datasetSize"),
        "dataFreshnessDate": bundle.get("dataFreshnessDate"),
        "extrapolationWarnings": warnings,
        "method": "trained_ml",
        "isFallback": False,
        "disclaimer": disclaimer,
        "note": disclaimer,
    }
