# Executive Cars ML service

This FastAPI service trains and serves the existing Executive Cars price predictor. It loads one versioned model at startup and never retrains during a prediction request.

## Structure

```text
app/data.py          schema inspection, canonicalisation, cleaning, features
app/training.py      splits, model comparison, evaluation, intervals, artifacts
app/prediction.py    request normalisation, inference, explanations
app/main.py          FastAPI endpoints
app/train.py         command-line training entry point
models/              local versioned model bundles and registry
reports/             reproducible JSON reports
tests/               unit/API tests
```

## Cleaning policy

The original CSV is read-only. Sequential removal rules are:

- missing/non-numeric price;
- price outside PKR 100,000–500,000,000;
- missing/non-numeric or impossible model year (1980 through next calendar year);
- missing/non-numeric or impossible mileage (0–1,000,000 km);
- missing/non-numeric or impossible engine size (300–10,000 cc);
- missing make/model;
- canonical duplicates excluding identifiers such as `addref`.

Categorical whitespace, case, common city/make/body spellings, transmission, fuel, and assembly values are standardised. Missing categorical fields become `Unknown`. Core numeric fields are not blindly imputed. IQR price outliers inside the broad hard bounds are retained because premium vehicles are plausible.

Current sequential removals:

| Reason | Rows |
|---|---:|
| Missing/non-numeric target price | 583 |
| Price outside hard bounds | 1 |
| Missing/non-numeric year after prior removals | 4,623 |
| Missing/non-numeric engine after prior removals | 2 |
| Invalid engine capacity after prior removals | 152 |
| Canonical duplicates after prior removals | 338 |
| Total removed | 5,699 |

No rows were removed for mileage bounds or model-year bounds beyond missing values in this dataset.

## Features

Numeric:

- model year, mileage, engine capacity;
- vehicle age;
- mileage per year.

Categorical:

- make, model, make-model combination;
- registration and listing city;
- fuel, transmission, body, assembly;
- engine, age, and mileage categories.

`addref` is excluded as an identifier. Colour is excluded because it was not justified as a stable valuation feature. With no listing date in the source CSV, the fixed feature reference year is the cleaned dataset maximum model year (2022), stored in model metadata.

## Training and evaluation

Seed: `42`. Split: 80% development, 10% conformal calibration, 10% untouched test. Candidate selection uses three-fold cross-validation within the development partition; preprocessing is fitted inside each fold to prevent leakage.

The bounded model-selection run completed four trials:

- one median baseline;
- two `ExtraTreesRegressor` configurations;
- one `HistGradientBoostingRegressor` configuration.

This is a controlled cross-validation search, not an unbounded grid search.

The selected model is the lowest cross-validation MAE, with cross-validation RMSE, MAE stability and inference time as tie-breakers. Current final test metrics:

| Metric | Value |
|---|---:|
| MAE | PKR 328,377.64 |
| RMSE | PKR 952,957.28 |
| R² | 0.9548 |
| MAPE | 10.324% |
| Median absolute error | PKR 152,900.99 |

The selected model is `extra_trees_320_regularized` with 320 trees, all available features considered per split, and a minimum of two samples per leaf. All four trials completed; Extra Trees achieved the lowest cross-validation MAE (PKR 366,825.41). The saved joblib bundle is stored uncompressed so it can be written reliably during retraining and loaded once at service startup.

The evaluation report also contains errors by major make, common make-model, price band, vehicle-age band, and category coverage. Rare categories remain the largest limitation: rare make-models in the untouched test set had MAE PKR 1,417,626.34 versus PKR 289,782.83 for common make-models.

## Prediction interval and explanations

The service reserves a calibration split and calculates the finite-sample corrected 90th percentile of absolute calibration residuals. The current error radius is PKR 632,402.86. It is subtracted from and added to the point prediction, with the lower bound clipped at zero. Untouched-test empirical coverage is 89.97%.

For the selected scikit-learn model, important factors come from held-out permutation importance. If CatBoost is selected in a future training run, local SHAP contributions are returned where available. No random explanation or invented confidence percentage is generated. Detailed metrics, configurations, segmented errors, and range calibration remain in `reports/model_evaluation.json`, not the customer-facing interface.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.train --input ../pakwheels_used_car_data_v02.csv
python run.py
python -m pytest tests -q
```

## API

- `GET /health` — status, model load state, model name, version, trained rows
- `GET /metadata` — model-supported input catalogue and public metadata
- `POST /predict` — validated valuation response
- `GET /models` — version registry
- `POST /train` — protected administrative training from supplied records
- `POST /models/{version}/activate` — activate a stored model
- `POST /models/rollback` — roll back to the previous version

Set the same `ML_SERVICE_KEY` in FastAPI and Express when service-key protection is required.
