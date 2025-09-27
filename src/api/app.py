"""FastAPI service exposing predictions for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.base import PersistableModel
from src.models.nlp import TextClusteringConfig, TextClusteringPipeline
from src.models.regression import LinearRegressionTrainer, RegressionConfig
from src.models.xgboost_model import XGBoostConfig, XGBoostTrainer

ARTIFACTS_DIR = Path("artifacts")
TEXT_CORPUS_PATH = Path("data/text_corpus.txt")

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


def _load_regression_model():
    artifact_path = ARTIFACTS_DIR / "linear_regression.pkl"
    if not artifact_path.exists():
        trainer = LinearRegressionTrainer(RegressionConfig(test_size=0.2, random_state=42))
        trainer.train()
        trainer.metadata.artifact_path = ARTIFACTS_DIR
        trainer.export()
    return PersistableModel.load(artifact_path)


def _load_xgboost_model():
    artifact_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    if not artifact_path.exists():
        dataset_path = Path("data/xgboost_dataset.csv")
        df = pd.read_csv(dataset_path)
        trainer = XGBoostTrainer(config=XGBoostConfig())
        trainer.train(df, target_column="TARGET")
        trainer.metadata.artifact_path = ARTIFACTS_DIR
        trainer.export()
    return PersistableModel.load(artifact_path)


@app.post("/predict/regression", response_model=RegressionResponse)
async def predict_regression(payload: RegressionRequest) -> RegressionResponse:
    model = _load_regression_model()
    features = np.array(payload.features).reshape(1, -1)
    prediction = float(model.predict(features)[0])
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
    return TextClusteringResponse(clusters=clusters)
