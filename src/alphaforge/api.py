"""Local FastAPI inference. Start with ALPHAFORGE_MODEL pointing to a model bundle."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .data import HOUR, validate_candles
from .deep import Predictor


class Candle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp: datetime
    symbol: Literal["BTC-USD", "ETH-USD"]
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candles: list[Candle] = Field(min_length=2, max_length=2000)


def create_app(model_path=None, *, clock=None):
    path = model_path or os.environ.get("ALPHAFORGE_MODEL")
    if not path:
        raise ValueError("Set ALPHAFORGE_MODEL to a saved model directory")
    predictor = Predictor(Path(path))
    now = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    app = FastAPI(
        title="AlphaForge",
        version=__version__,
        description="Local hourly crypto return inference; no order execution.",
    )

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "symbol": predictor.metadata["symbol"],
            "model": predictor.metadata["kind"],
        }

    @app.get("/model")
    def model():
        return {
            k: predictor.metadata[k]
            for k in ("schema_version", "kind", "symbol", "features", "training", "trained_through")
        }

    @app.post("/predict")
    def predict(request: PredictionRequest):
        try:
            frame = validate_candles(pd.DataFrame([c.model_dump() for c in request.candles]))
            current = pd.Timestamp(now())
            newest_close = frame.timestamp.max() + HOUR
            if newest_close > current:
                raise ValueError("Input contains an unfinished or future candle")
            if current - newest_close > HOUR:
                raise ValueError("Latest candle is stale by more than one hour")
            return predictor.predict_candles(frame)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
