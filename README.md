



# Machine Learning Portfolio

Comprehensive, production-ready machine learning pipelines covering regression, gradient boosting, and NLP clustering. This repository also includes a FastAPI service for real-time inference, training utilities, tests, and supporting datasets.

---

## Contents

- [Overview](#overview)
- [Project Layout](#project-layout)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Training Pipelines](#training-pipelines)
- [API Service](#api-service)
- [Utilities & Scripts](#utilities--scripts)
- [Testing & Quality](#testing--quality)
- [Deployment Notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)

---

## Overview

- Regression pipeline using the scikit-learn diabetes dataset with automated export of trained artifacts.
- XGBoost pipeline for tabular data, defaulting to a bundled mock dataset and optionally switching to the Titanic dataset loader.
- NLP text clustering pipeline built on TF-IDF + KMeans, exposing helper methods for cluster analysis.
- FastAPI application serving predictions for regression, XGBoost, and NLP clustering.
- CLI scripts for training, reproducible configuration via `pyproject.toml`, and a curated Poetry environment.

---

## Project Layout

| Path | Description |
| --- | --- |
| `src/data/` | Data loaders (`load_csv_dataset()`, `load_sql_dataset()`, `load_text_corpus()`, `load_titanic_for_xgboost()`). |
| `src/models/` | Model implementations for regression, XGBoost, NLP, and shared persistence utilities. |
| `src/api/` | FastAPI app (`app.py`) plus helper scripts (`call_end_point.sh`). |
| `scripts/` | CLI entry points for training regression and XGBoost models. |
| `data/` | Mock datasets (`xgboost_dataset.csv`, `text_corpus.txt`). |
| `artifacts/` | Exported `.pkl` model artifacts (gitignored by default; generated after training). |
| `tests/` | Pytest suite validating preprocessing, regression, and NLP behavior. |
| `requirements/` | `prod.txt` and `dev.txt` split for runtime vs. development dependencies. |

---

## Environment Setup

This project is managed with Poetry.

```bash
poetry install
poetry shell  # optional, spawns an interactive shell in the venv
```

To export dependencies without Poetry (e.g., for Docker):

```bash
poetry export --without-hashes -f requirements.txt -o requirements.txt
```

---

## Datasets

- `data/xgboost_dataset.csv`: Mock tabular dataset with three numerical features and a binary target. Used by default for XGBoost training and inference.
- `data/text_corpus.txt`: Sample corpus for the text clustering pipeline and tests.
- `load_titanic_for_xgboost()` (in `src/data/loaders.py`): Downloads and caches the Titanic dataset, performing minimal preprocessing and feature engineering when no local CSV is available.

Feel free to replace these files with domain-specific data; the loaders and trainers accept custom paths/configurations.

---

## Training Pipelines

Run from the repository root (`Machine-Learning/`). Commands below leverage the Poetry environment.

```bash
poetry run python scripts/train_regression.py
poetry run python scripts/train_xgboost.py
```

- Regression script trains a scikit-learn `LinearRegression`, logs MSE/R², and exports `artifacts/linear_regression.pkl`.
- XGBoost script trains via `GridSearchCV`, reports ROC-AUC & KS, and exports `artifacts/xgboost_model.pkl`. If `data/xgboost_dataset.csv` is missing, it automatically loads the Titanic dataset.

Artifacts are stored under `artifacts/` and consumed by the FastAPI service.

---

## API Service

Launch the service:

```bash
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method & Path | Description | Request Body | Response |
| --- | --- | --- | --- |
| `POST /predict/regression` | Predict target value using the linear regression model. | JSON `{"features": [float, ...]}` (length 10). | `{"prediction": float}` |
| `POST /predict/xgboost` | Predict probability for binary classification using XGBoost. | JSON `{"features": [f1, f2, f3]}` by default (or 6 Titanic features if retrained). | `{"probability": float}` |
| `POST /predict/text-clusters` | Cluster new documents based on TF-IDF + KMeans. | JSON `{"documents": ["text", ...]}` | `{"clusters": [int, ...]}` |

Validation ensures payloads match the input dimensionality expected by the trained artifact. The helper script `src/api/call_end_point.sh` provides ready-to-run `curl` commands.

---

## Utilities & Scripts

- `scripts/train_regression.py`: Trains regression model and exports artifact.
- `scripts/train_xgboost.py`: Trains XGBoost model, with automatic dataset fallback via `load_titanic_for_xgboost()`.
- `src/api/call_end_point.sh`: Quick manual tests for regression and XGBoost endpoints.
- `src/utils/logger.py`: Basic structured logging helper used across modules.

---

## Testing & Quality

```bash
poetry run pytest -v
poetry run pytest tests/test_nlp.py
poetry run black src tests
poetry run mypy src
```

- `tests/test_preprocessing.py`: Validates preprocessing strategies and scaling.
- `tests/test_regression.py`: Confirms regression trainer outputs metrics consistent with scikit-learn’s diabetes dataset.
- `tests/test_nlp.py`: Exercises the text clustering pipeline.

Coverage reports are generated automatically when running `pytest` with the default configuration.

---

## Deployment Notes

- Artifacts are excluded from source control; regenerate as needed via the training scripts.
- For containerization, export dependencies with `poetry export` or install Poetry inside the Docker image.
- FastAPI can be served behind a production ASGI server such as Gunicorn + Uvicorn workers (`uvicorn.workers.UvicornWorker`).
- Configure environment variables (e.g., dataset paths, cache directories) via `.env` and load with your preferred settings manager if required.

---

## Troubleshooting

- **Feature length mismatch (XGBoost)**: Ensure the request vector length matches the trained artifact. Retrain after changing datasets.
- **Artifact not found**: Run the training script to regenerate `.pkl` files before invoking the API.
- **Poetry not installed**: Install via `curl -sSL https://install.python-poetry.org | python3 -` and ensure `$HOME/.local/bin` is on `PATH`.
- **Port already in use**: Stop existing services using port 8000 or start Uvicorn on an alternate port.

---

*Last updated: September 2025*
