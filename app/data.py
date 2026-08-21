from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


# Hard validity limits are deliberately broad enough to retain premium vehicles.
MIN_MODEL_YEAR = 1980
MIN_PRICE_PKR = 100_000
MAX_PRICE_PKR = 500_000_000
MIN_ENGINE_CC = 300
MAX_ENGINE_CC = 10_000
MAX_MILEAGE_KM = 1_000_000

ALIASES = {
    "make": ("make", "brand", "manufacturer"),
    "model": ("model", "carmodel"),
    "variant": ("variant", "trim"),
    "modelYear": ("modelyear", "year"),
    "registrationCity": ("registrationcity", "registeredcity", "registered", "registration"),
    "listingCity": ("listingcity", "city", "location"),
    "engineCapacity": ("enginecapacity", "enginecc", "engine", "cc"),
    "fuelType": ("fueltype", "fuel"),
    "transmission": ("transmission", "gearbox"),
    "mileage": ("mileage", "km", "kilometers", "kilometres"),
    "bodyType": ("bodytype", "body"),
    "assemblyType": ("assemblytype", "assembly"),
    "colour": ("colour", "color"),
    "condition": ("condition", "vehiclecondition"),
    "numberOfOwners": ("numberofowners", "owners", "ownercount"),
    "inspectionScore": ("inspectionscore", "score"),
    "listingPrice": ("listingprice", "price", "askingprice"),
    "finalSalePrice": ("finalsaleprice", "saleprice", "soldprice"),
    "listingDate": ("listingdate", "date", "posteddate"),
    "source": ("source",),
    "sourceUrl": ("sourceurl", "url", "listingurl"),
}

NUMERIC_COLUMNS = (
    "modelYear",
    "engineCapacity",
    "mileage",
    "numberOfOwners",
    "inspectionScore",
    "listingPrice",
    "finalSalePrice",
)
CATEGORICAL_COLUMNS = (
    "make",
    "model",
    "variant",
    "registrationCity",
    "listingCity",
    "fuelType",
    "transmission",
    "bodyType",
    "assemblyType",
    "colour",
    "condition",
)


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def parse_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raw = re.sub(r"pkr|rs\.?|,", "", str(value).strip().lower())
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan
    parsed = float(match.group(0))
    if re.search(r"crore|\bcr\b", raw):
        parsed *= 10_000_000
    elif re.search(r"lakh|lac|\blacs?\b", raw):
        parsed *= 100_000
    elif re.search(r"\d(?:\.\d+)?\s*k\b", raw):
        parsed *= 1_000
    return parsed


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _title(value: Any) -> str:
    return _clean_text(value).title()


def _map_enum(value: Any, groups: dict[str, set[str]], fallback: str = "Unknown") -> str:
    normalized = _clean_text(value).casefold()
    for canonical, variants in groups.items():
        if normalized in variants:
            return canonical
    return _title(normalized) or fallback


def _standardize_city(value: Any) -> str:
    cleaned = _title(value)
    aliases = {
        "Islambad": "Islamabad",
        "Rwp": "Rawalpindi",
        "Pindi": "Rawalpindi",
        "Un-Registered": "Unregistered",
        "Un Registered": "Unregistered",
        "Not Registered": "Unregistered",
    }
    return aliases.get(cleaned, cleaned or "Unknown")


def _standardize_make(value: Any) -> str:
    cleaned = _title(value)
    aliases = {
        "Kia": "Kia",
        "Mg": "MG",
        "Bmw": "BMW",
        "Dfsk": "DFSK",
        "Faw": "FAW",
        "Baic": "BAIC",
        "Jac": "JAC",
    }
    return aliases.get(cleaned, cleaned)


def _standardize_body(value: Any) -> str:
    cleaned = _title(value)
    aliases = {
        "Pick Up": "Pickup",
        "Mini Van": "Minivan",
        "Micro Van": "Microvan",
        "Compact Sedan": "Compact Sedan",
        "Compact Hatchback": "Compact Hatchback",
        "Off-Road Vehicles": "Off-Road",
    }
    return aliases.get(cleaned, cleaned or "Unknown")


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    key_to_column = {_key(column): column for column in frame.columns}
    result = pd.DataFrame(index=frame.index)
    for canonical, aliases in ALIASES.items():
        source = next((key_to_column[alias] for alias in aliases if alias in key_to_column), None)
        result[canonical] = frame[source] if source else np.nan
    return result


def _distribution(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"valid": 0, "min": None, "p01": None, "p25": None, "median": None, "p75": None, "p99": None, "max": None}
    quantiles = values.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).tolist()
    return {
        "valid": int(len(values)),
        **{
            name: round(float(value), 2)
            for name, value in zip(("min", "p01", "p25", "median", "p75", "p99", "max"), quantiles)
        },
    }


def inspect_dataframe(frame: pd.DataFrame, file_name: str = "") -> dict[str, Any]:
    numeric = {column: pd.to_numeric(frame[column], errors="coerce") for column in ("year", "engine", "mileage", "price") if column in frame}
    price = numeric.get("price", pd.Series(dtype=float))
    q1 = float(price.quantile(0.25)) if not price.dropna().empty else 0.0
    q3 = float(price.quantile(0.75)) if not price.dropna().empty else 0.0
    iqr = q3 - q1
    duplicate_columns = [column for column in frame.columns if _key(column) not in {"addref", "id", "identifier"}]
    important_categories = [
        column for column in ("city", "assembly", "body", "make", "model", "transmission", "fuel", "registered", "color")
        if column in frame
    ]
    return {
        "fileName": file_name,
        "totalRows": int(len(frame)),
        "totalColumns": int(len(frame.columns)),
        "columnNames": [str(column) for column in frame.columns],
        "dataTypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missingValues": {str(column): int(value) for column, value in frame.isna().sum().items()},
        "exactDuplicateRows": int(frame.duplicated().sum()),
        "duplicateRowsIgnoringIdentifier": int(frame.duplicated(subset=duplicate_columns).sum()) if duplicate_columns else 0,
        "uniqueValues": {column: int(frame[column].nunique(dropna=True)) for column in important_categories},
        "categoryCounts": {
            column: {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}
            for column in important_categories
        },
        "distributions": {column: _distribution(series) for column, series in numeric.items()},
        "invalidValues": {
            "missingOrInvalidPrice": int(price.isna().sum()) if "price" in numeric else None,
            "priceOutsideHardBounds": int((~price.between(MIN_PRICE_PKR, MAX_PRICE_PKR) & price.notna()).sum()) if "price" in numeric else None,
            "yearOutsideHardBounds": int((~numeric["year"].between(MIN_MODEL_YEAR, datetime.now(timezone.utc).year + 1) & numeric["year"].notna()).sum()) if "year" in numeric else None,
            "mileageOutsideHardBounds": int((~numeric["mileage"].between(0, MAX_MILEAGE_KM) & numeric["mileage"].notna()).sum()) if "mileage" in numeric else None,
            "engineOutsideHardBounds": int((~numeric["engine"].between(MIN_ENGINE_CC, MAX_ENGINE_CC) & numeric["engine"].notna()).sum()) if "engine" in numeric else None,
        },
        "priceOutlierSummary": {
            "method": "1.5 × IQR diagnostic only; plausible premium vehicles are retained inside hard bounds",
            "lowerFence": round(q1 - 1.5 * iqr, 2),
            "upperFence": round(q3 + 1.5 * iqr, 2),
            "outlierCount": int(((price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)).sum()),
        },
        "topMakes": {str(key): int(value) for key, value in frame.get("make", pd.Series(dtype=str)).value_counts().head(25).items()},
        "topModels": {str(key): int(value) for key, value in frame.get("model", pd.Series(dtype=str)).value_counts().head(40).items()},
        "topMakeModels": {
            f"{key[0]} {key[1]}": int(value)
            for key, value in (
                frame.groupby(["make", "model"], dropna=False).size().sort_values(ascending=False).head(40).items()
                if {"make", "model"}.issubset(frame.columns) else []
            )
        },
    }


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = canonicalize_frame(frame.copy())
    received = len(data)
    missing_percentages = {
        column: round(float(data[column].isna().mean() * 100), 2)
        for column in data.columns
    }

    for column in NUMERIC_COLUMNS:
        data[column] = data[column].map(parse_number)

    data["make"] = data["make"].map(_standardize_make)
    data["model"] = data["model"].map(_title)
    data["variant"] = data["variant"].map(_title).replace("", "Unknown")
    data["registrationCity"] = data["registrationCity"].map(_standardize_city)
    data["listingCity"] = data["listingCity"].map(_standardize_city)
    data["bodyType"] = data["bodyType"].map(_standardize_body)
    data["colour"] = data["colour"].map(_title).replace("", "Unknown")
    data["transmission"] = data["transmission"].map(lambda value: _map_enum(value, {
        "Auto": {"auto", "automatic", "cvt", "dct"},
        "Manual": {"manual", "mt"},
    }))
    data["fuelType"] = data["fuelType"].map(lambda value: _map_enum(value, {
        "Petrol": {"petrol", "gasoline"},
        "Diesel": {"diesel"},
        "CNG": {"cng"},
        "Hybrid": {"hybrid", "hev"},
        "Electric": {"electric", "ev"},
    }))
    data["assemblyType"] = data["assemblyType"].map(lambda value: _map_enum(value, {
        "Local": {"local", "pakistan", "ckd"},
        "Imported": {"imported", "import", "cbu"},
        "Unknown": {"unknown", "n/a", ""},
    }))
    data["condition"] = data["condition"].map(lambda value: _map_enum(value, {
        "Excellent": {"excellent", "like new"},
        "Good": {"good", "used"},
        "Fair": {"fair", "average"},
        "Needs Work": {"needs work", "poor", "damaged"},
        "Unknown": {"unknown", "n/a", ""},
    }))
    data["listingDate"] = pd.to_datetime(data["listingDate"], errors="coerce", utc=True)
    data["source"] = data["source"].map(_title).replace("", "Attached Pakistani Used-Car Dataset")
    data["sourceUrl"] = data["sourceUrl"].fillna("").astype(str)

    removal_steps: list[dict[str, Any]] = []

    def remove(mask: pd.Series, reason: str, explanation: str) -> None:
        nonlocal data
        count = int(mask.fillna(False).sum())
        removal_steps.append({"reason": reason, "removedRows": count, "explanation": explanation})
        if count:
            data = data.loc[~mask.fillna(False)].copy()

    remove(
        data["listingPrice"].isna(),
        "missing_or_non_numeric_price",
        "Price is the target and cannot be imputed without fabricating labels.",
    )
    remove(
        ~data["listingPrice"].between(MIN_PRICE_PKR, MAX_PRICE_PKR),
        "price_outside_hard_bounds",
        f"Retained only PKR {MIN_PRICE_PKR:,}–{MAX_PRICE_PKR:,}; IQR-only premium outliers inside this range remain.",
    )
    remove(
        data["modelYear"].isna(),
        "missing_or_non_numeric_year",
        "Model year is a core age feature and is not imputed.",
    )
    remove(
        ~data["modelYear"].between(MIN_MODEL_YEAR, datetime.now(timezone.utc).year + 1),
        "invalid_model_year",
        f"Accepted model years from {MIN_MODEL_YEAR} through next calendar year.",
    )
    remove(
        data["mileage"].isna(),
        "missing_or_non_numeric_mileage",
        "Mileage is a core depreciation feature and is not imputed.",
    )
    remove(
        ~data["mileage"].between(0, MAX_MILEAGE_KM),
        "invalid_mileage",
        f"Accepted mileage from 0 through {MAX_MILEAGE_KM:,} km.",
    )
    remove(
        data["engineCapacity"].isna(),
        "missing_or_non_numeric_engine",
        "Engine capacity is retained as a core numeric feature and is not imputed.",
    )
    remove(
        ~data["engineCapacity"].between(MIN_ENGINE_CC, MAX_ENGINE_CC),
        "invalid_engine_capacity",
        f"Accepted engine sizes from {MIN_ENGINE_CC:,} through {MAX_ENGINE_CC:,} cc.",
    )
    remove(
        data["make"].eq("") | data["model"].eq(""),
        "missing_make_or_model",
        "Both make and model are required to prevent unsupported anonymous categories.",
    )

    data["modelYear"] = data["modelYear"].round().astype(int)
    data["mileage"] = data["mileage"].round().astype(int)
    data["engineCapacity"] = data["engineCapacity"].round().astype(int)
    data["listingPrice"] = data["listingPrice"].round().astype(int)
    duplicate_columns = [
        "make", "model", "variant", "modelYear", "engineCapacity", "transmission",
        "fuelType", "mileage", "listingCity", "registrationCity", "bodyType",
        "assemblyType", "colour", "listingPrice",
    ]
    duplicate_mask = data.duplicated(subset=duplicate_columns, keep="first")
    remove(
        duplicate_mask,
        "duplicate_vehicle_record",
        "Removed canonical duplicates after excluding source identifiers such as addref.",
    )

    data["targetPrice"] = data["finalSalePrice"].where(
        data["finalSalePrice"].between(MIN_PRICE_PKR, MAX_PRICE_PKR),
        data["listingPrice"],
    )
    data["targetPrice"] = data["targetPrice"].round().astype(int)
    removed = received - len(data)
    report = {
        "receivedRecords": received,
        "removedRecords": removed,
        "rejectedRecords": removed,
        "duplicateRecords": next((step["removedRows"] for step in removal_steps if step["reason"] == "duplicate_vehicle_record"), 0),
        "usableRecords": len(data),
        "missingValuePercentagesBeforeCleaning": missing_percentages,
        "removalSteps": removal_steps,
        "rejectionReasons": {step["reason"]: step["removedRows"] for step in removal_steps if step["removedRows"]},
        "hardThresholds": {
            "modelYear": [MIN_MODEL_YEAR, datetime.now(timezone.utc).year + 1],
            "mileageKm": [0, MAX_MILEAGE_KM],
            "engineCc": [MIN_ENGINE_CC, MAX_ENGINE_CC],
            "pricePkr": [MIN_PRICE_PKR, MAX_PRICE_PKR],
        },
        "categoricalMissingPolicy": "Missing categorical values are represented as Unknown; core numeric fields are never blindly imputed.",
    }
    return data.reset_index(drop=True), report


def determine_reference_year(frame: pd.DataFrame) -> int:
    years = pd.to_numeric(frame.get("modelYear"), errors="coerce").dropna()
    return int(years.max()) if not years.empty else datetime.now(timezone.utc).year


def add_features(frame: pd.DataFrame, reference_year: int | None = None) -> pd.DataFrame:
    enriched = frame.copy()
    year = int(reference_year or determine_reference_year(enriched))
    enriched["vehicleAge"] = (year - pd.to_numeric(enriched["modelYear"], errors="coerce")).clip(lower=0)
    denominator = enriched["vehicleAge"].fillna(0).clip(lower=1)
    enriched["mileagePerYear"] = pd.to_numeric(enriched["mileage"], errors="coerce") / denominator
    enriched["makeModel"] = (
        enriched["make"].fillna("Unknown").astype(str)
        + " "
        + enriched["model"].fillna("Unknown").astype(str)
    )
    enriched["engineCategory"] = pd.cut(
        pd.to_numeric(enriched["engineCapacity"], errors="coerce"),
        bins=[0, 800, 1000, 1300, 1600, 2000, 3000, np.inf],
        labels=["Micro", "Compact", "Standard", "Mid", "Large", "Performance", "Ultra"],
        include_lowest=True,
    ).astype(str).replace("nan", "Unknown")
    enriched["ageCategory"] = pd.cut(
        enriched["vehicleAge"],
        bins=[-1, 2, 5, 10, 15, 25, np.inf],
        labels=["Nearly New", "Recent", "Established", "Mature", "Classic", "Vintage"],
    ).astype(str).replace("nan", "Unknown")
    enriched["mileageCategory"] = pd.cut(
        pd.to_numeric(enriched["mileage"], errors="coerce"),
        bins=[-1, 20_000, 50_000, 100_000, 150_000, 250_000, np.inf],
        labels=["Very Low", "Low", "Moderate", "High", "Very High", "Extreme"],
    ).astype(str).replace("nan", "Unknown")
    return enriched


def build_input_catalog(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.groupby(["make", "model"]).size().rename("count").reset_index()
    counts = counts.sort_values(["make", "count", "model"], ascending=[True, False, True])
    makes = []
    for make, group in counts.groupby("make", sort=True):
        makes.append({
            "make": str(make),
            "count": int(group["count"].sum()),
            "models": [{"model": str(row.model), "count": int(row.count)} for row in group.itertuples()],
        })
    options = lambda column: [
        {"value": str(value), "count": int(count)}
        for value, count in frame[column].value_counts().items()
        if str(value).strip()
    ]
    return {
        "makes": sorted(makes, key=lambda item: (-item["count"], item["make"])),
        "listingCities": options("listingCity"),
        "registrationCities": options("registrationCity"),
        "bodyTypes": options("bodyType"),
        "fuelTypes": options("fuelType"),
        "transmissions": options("transmission"),
        "assemblyTypes": options("assemblyType"),
        "yearRange": {
            "min": int(frame["modelYear"].min()),
            "max": int(frame["modelYear"].max()),
        },
        "engineRange": {
            "min": int(frame["engineCapacity"].min()),
            "max": int(frame["engineCapacity"].max()),
        },
        "mileageRange": {
            "min": int(frame["mileage"].min()),
            "max": int(frame["mileage"].max()),
        },
    }


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()[:16]
