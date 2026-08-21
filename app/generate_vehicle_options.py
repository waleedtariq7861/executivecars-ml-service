"""Build the runtime vehicle-options index from the same cleaned data used for ML.

This does not train or change the valuation model.  It creates a compact lookup
asset so dropdowns never need to read the source CSV at runtime.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import clean_dataframe


MIN_MODEL_ROWS = 3
MIN_SPEC_ROWS = 2
SPEC_COLUMNS = {
    "engines": "engineCapacity",
    "transmissions": "transmission",
    "fuels": "fuelType",
    "bodyTypes": "bodyType",
    "assemblies": "assemblyType",
}


def options(series: pd.Series, numeric: bool = False) -> list[dict]:
    counts = series.dropna().value_counts()
    result = []
    for value, count in counts.items():
        if int(count) < MIN_SPEC_ROWS:
            continue
        value = int(value) if numeric else str(value)
        if value in ("", "Unknown"):
            continue
        result.append({"value": value, "count": int(count)})
    return result


def build(frame: pd.DataFrame) -> dict:
    models = []
    for (make, model), group in frame.groupby(["make", "model"], sort=True):
        if len(group) < MIN_MODEL_ROWS:
            continue
        years = []
        for year, year_group in group.groupby("modelYear", sort=False):
            if not pd.notna(year):
                continue
            item = {"year": int(year), "count": int(len(year_group))}
            for name, column in SPEC_COLUMNS.items():
                item[name] = options(year_group[column], numeric=column == "engineCapacity")
            years.append(item)
        if not years:
            continue
        models.append({"make": str(make), "model": str(model), "count": int(len(group)), "years": sorted(years, key=lambda item: item["year"], reverse=True)})

    makes = []
    for make, group in pd.DataFrame(models).groupby("make", sort=True):
        listed = group.drop(columns="make").to_dict("records")
        makes.append({"make": str(make), "count": sum(item["count"] for item in listed), "models": sorted(listed, key=lambda item: (-item["count"], item["model"]))})
    return {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "cleaned_ml_training_dataset",
        "filtering": {"minimumModelRows": MIN_MODEL_ROWS, "minimumSpecificationRowsPerModelYear": MIN_SPEC_ROWS, "excludedValues": ["Unknown", ""]},
        "makes": sorted(makes, key=lambda item: (-item["count"], item["make"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "pakwheels_used_car_data_v02.csv"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "reports" / "vehicle_options.json"))
    args = parser.parse_args()
    raw = pd.read_csv(args.dataset)
    cleaned, _ = clean_dataframe(raw)
    result = build(cleaned)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}: {len(result['makes'])} makes")


if __name__ == "__main__":
    main()
