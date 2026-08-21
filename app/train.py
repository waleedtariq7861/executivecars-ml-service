from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .training import train_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Executive Cars valuation model from a legal CSV or JSON dataset.")
    parser.add_argument("--input", required=True, help="Path to a CSV or JSON dataset")
    parser.add_argument("--model-dir", help="Optional model registry directory (defaults to ml-service/models)")
    parser.add_argument("--report-dir", help="Optional report directory (defaults to ml-service/reports)")
    args = parser.parse_args()
    if args.model_dir:
        os.environ["MODEL_DIR"] = str(Path(args.model_dir).expanduser().resolve())
    if args.report_dir:
        os.environ["REPORT_DIR"] = str(Path(args.report_dir).expanduser().resolve())
    path = Path(args.input).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame.from_records(payload if isinstance(payload, list) else payload.get("records", []))
    else:
        raise SystemExit("Input must be CSV or JSON")
    print(json.dumps(train_from_frame(frame, file_name=path.name), indent=2))


if __name__ == "__main__":
    main()
