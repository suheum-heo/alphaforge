"""Small model interface; future torch adapters can implement the same contract."""

from typing import Protocol

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


def baselines(alpha: float) -> dict[str, Regressor]:
    return {
        "zero_return": DummyRegressor(strategy="constant", constant=0.0),
        "training_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
    }
