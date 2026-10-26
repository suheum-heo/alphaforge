"""Forecast metrics, deliberately separate from execution/backtest assumptions."""

import numpy as np


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Expected matching, nonempty one-dimensional vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite inputs")
    error = actual - predicted
    nonzero = actual != 0
    correlation = None
    if actual.size > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": correlation,
        # A zero prediction is an abstention and counts as incorrect on nonzero returns.
        "direction_accuracy": float(
            np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero]))
        )
        if nonzero.any()
        else None,
        "direction_coverage": float(np.mean(predicted != 0)),
    }
