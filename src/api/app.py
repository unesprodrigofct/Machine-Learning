"""FastAPI service exposing predictions for trained models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.base import ModelMetadata, PersistableModel
from src.core.pipelines import TextClusteringConfig, TextClusteringPipeline
from src.core.trainers import LinearRegressionTrainer, RegressionConfig, XGBoostConfig, XGBoostTrainer
from src.infra.logging import configure_logging, get_logger
from src.infra.settings import AppSettings, get_settings

configure_logging()
logger = get_logger(__name__)
settings: AppSettings = get_settings()

ARTIFACTS_DIR: Path = settings.artifacts_path
TEXT_CORPUS_PATH: Path = settings.text_corpus_path

app = FastAPI(title="ML Portfolio API", version="1.0.0")


class RegressionRequest(BaseModel):
    features: List[float]


class RegressionResponse(BaseModel):
    prediction: float


class XGBoostRequest(BaseModel):
    features: List[float]


class XGBoostResponse(BaseModel):
    probability: float


class TextClusteringRequest(BaseModel):
    documents: List[str]


class TextClusteringResponse(BaseModel):
    clusters: List[int]


@lru_cache(maxsize=1)
def _load_regression_model():
    artifact_path = ARTIFACTS_DIR / "linear_regression.pkl"
    if artifact_path.exists():
        logger.info("model.artifact_found", model="linear_regression", path=str(artifact_path))
        return PersistableModel.load(artifact_path)

    logger.warning("model.artifact_missing", model="linear_regression", path=str(artifact_path))
    trainer = LinearRegressionTrainer(
        RegressionConfig(test_size=0.2, random_state=42),
        metadata=ModelMetadata(name="linear_regression", artifact_path=ARTIFACTS_DIR),
    )
    mse, r2 = trainer.train()
    trainer.export()
    logger.info("model.trained", model="linear_regression", mse=mse, r2=r2)
    return PersistableModel.load(artifact_path)


@lru_cache(maxsize=1)
def _load_xgboost_model():
    artifact_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    if artifact_path.exists():
        logger.info("model.artifact_found", model="xgboost_classifier", path=str(artifact_path))
        return PersistableModel.load(artifact_path)

    logger.warning("model.artifact_missing", model="xgboost_classifier", path=str(artifact_path))
    dataset_path = Path("data/xgboost_dataset.csv")
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
    else:
        from src.data.loaders import load_titanic_for_xgboost

        logger.info("dataset.fallback", source="titanic")
        df = load_titanic_for_xgboost(cache_dir=Path("data"))

    trainer = XGBoostTrainer(
        config=XGBoostConfig(),
        metadata=ModelMetadata(name="xgboost_model", artifact_path=ARTIFACTS_DIR),
    )
    auc, ks_stat = trainer.train(df, target_column="TARGET")
    trainer.export()
    logger.info("model.trained", model="xgboost_classifier", auc=auc, ks=ks_stat)
    return PersistableModel.load(artifact_path)


@app.post("/predict/regression", response_model=RegressionResponse)
async def predict_regression(payload: RegressionRequest) -> RegressionResponse:
    model = _load_regression_model()
    features = np.array(payload.features).reshape(1, -1)
    prediction = float(model.predict(features)[0])
    logger.info("prediction.regression", features=len(payload.features), prediction=prediction)
    return RegressionResponse(prediction=prediction)


@app.post("/predict/xgboost", response_model=XGBoostResponse)
async def predict_xgboost(payload: XGBoostRequest) -> XGBoostResponse:
    model = _load_xgboost_model()
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and len(payload.features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Feature length mismatch. Expected {expected_features} values, received {len(payload.features)}.",
        )
    features = np.array(payload.features).reshape(1, -1)
    probability = float(model.predict_proba(features)[0, 1])
    logger.info(
        "prediction.xgboost",
        features=len(payload.features),
        probability=probability,
        expected_features=expected_features,
    )
    return XGBoostResponse(probability=probability)


@app.post("/predict/text-clusters", response_model=TextClusteringResponse)
async def predict_text_clusters(payload: TextClusteringRequest) -> TextClusteringResponse:
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    pipeline = TextClusteringPipeline(TextClusteringConfig(n_clusters=2))
    if TEXT_CORPUS_PATH.exists():
        corpus = TEXT_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        pipeline.fit(corpus)
    else:
        pipeline.fit(payload.documents)

    clusters = pipeline.predict(payload.documents).tolist()
    logger.info("prediction.text_clusters", documents=len(payload.documents))
    return TextClusteringResponse(clusters=clusters)
