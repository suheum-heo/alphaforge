"""Causal feature contract shared by baseline and future sequence models."""

import numpy as np
import pandas as pd

from .config import Config
from .data import HOUR, validate_candles


def build_features(
    candles: pd.DataFrame, config: Config, *, include_targets: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """At decision_time=t+1h, forecast log(close[t+h]/close[t]).

    Reindex to a complete hourly grid before any shifts. Never forward-fill gaps.
    A missing bar invalidates every sample whose lookback or target span touches it.
    """
    candles = validate_candles(candles)
    outputs = []
    feature_names = ["return_1", "range", "body", "log_volume", "hour_sin", "hour_cos"]
    for window in config.windows:
        feature_names += [f"momentum_{window}", f"volatility_{window}", f"volume_ratio_{window}"]
    for symbol, group in candles.groupby("symbol", sort=True):
        group = group.set_index("timestamp").sort_index()
        grid = pd.date_range(group.index.min(), group.index.max(), freq="h")
        group = group.reindex(grid)
        out = pd.DataFrame(index=grid)
        returns = np.log(group.close).diff()
        out["return_1"] = returns
        out["range"] = (group.high - group.low) / group.close
        out["body"] = np.log(group.close / group.open)
        out["log_volume"] = np.log1p(group.volume)
        decision = grid + HOUR
        out["hour_sin"] = np.sin(2 * np.pi * decision.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * decision.hour / 24)
        for window in config.windows:
            out[f"momentum_{window}"] = np.log(group.close / group.close.shift(window))
            out[f"volatility_{window}"] = returns.rolling(window).std()
            # log ratio stays finite on valid zero-volume candles.
            out[f"volume_ratio_{window}"] = np.log1p(group.volume) - np.log1p(
                group.volume.rolling(window).mean()
            )
        out["target"] = np.log(group.close.shift(-config.horizon) / group.close)
        out["timestamp"] = grid
        out["decision_time"] = decision
        out["target_time"] = decision + config.horizon * HOUR
        out["symbol"] = symbol
        lookback = max(config.windows) + 1
        history_ok = group.close.notna().rolling(lookback).sum().eq(lookback)
        future_ok = (
            group.close.notna()
            .rolling(config.horizon + 1)
            .sum()
            .shift(-config.horizon)
            .eq(config.horizon + 1)
        )
        valid = history_ok & future_ok if include_targets else history_ok
        required = feature_names + (["target"] if include_targets else [])
        out = out.loc[valid].dropna(subset=required)
        outputs.append(out)
    result = pd.concat(outputs, ignore_index=True).sort_values(["decision_time", "symbol"])
    if result.empty:
        raise ValueError("No usable samples; increase history or check gaps/windows")
    if not np.isfinite(
        result[feature_names + (["target"] if include_targets else [])].to_numpy()
    ).all():
        raise ValueError("Nonfinite features or targets")
    return result.reset_index(drop=True), feature_names
