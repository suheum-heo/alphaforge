"""Auditable experiment outputs with train-only fitting and validation-only selection."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from .config import Config, utc
from .data import quality, validate_candles
from .evaluation import metrics
from .features import build_features
from .models import baselines
from .splits import chronological_split


def run(candles: pd.DataFrame, config: Config, output: Path, source: str) -> dict:
    candles = validate_candles(candles)
    candles = candles.loc[
        candles.symbol.isin(config.symbols)
        & (candles.timestamp >= utc(config.start))
        & (candles.timestamp < utc(config.end))
    ].reset_index(drop=True)
    if set(candles.symbol) != set(config.symbols):
        raise ValueError("Missing configured symbols in input")
    frame, names = build_features(candles, config)
    splits = chronological_split(frame, config)
    # Each asset gets its own scaler and model, avoiding implicit asset mixing.
    report = {
        "source": source,
        "config": config.as_dict(),
        "features": names,
        "data_quality": quality(candles),
        "purged_rows": splits.purged_rows,
        "boundaries": {
            "validation_start": str(splits.validation_start),
            "test_start": str(splits.test_start),
        },
        "results": {},
    }
    predictions = []
    split_frames = {name: getattr(splits, name) for name in ("train", "validation", "test")}
    report["splits"] = {
        name: {
            "rows": len(part),
            "first_decision": str(part.decision_time.min()),
            "last_decision": str(part.decision_time.max()),
            "last_target": str(part.target_time.max()),
        }
        for name, part in split_frames.items()
    }
    for symbol in config.symbols:
        train = splits.train.loc[splits.train.symbol == symbol]
        models = baselines(config.ridge_alpha)
        result = {}
        for name, model in models.items():
            model.fit(train[names].to_numpy(), train.target.to_numpy())
            result[name] = {}
            for split in ("validation", "test"):
                part = split_frames[split].loc[split_frames[split].symbol == symbol]
                pred = model.predict(part[names].to_numpy())
                result[name][split] = metrics(part.target, pred)
                rows = part[["symbol", "decision_time", "target_time", "target"]].copy()
                rows["prediction"], rows["model"], rows["split"] = pred, name, split
                predictions.append(rows)
        winner = min(result, key=lambda name: result[name]["validation"]["rmse"])
        report["results"][symbol] = {"models": result, "selected_by_validation_rmse": winner}
    raw_csv = candles.to_csv(index=False)
    report["input_sha256"] = hashlib.sha256(raw_csv.encode()).hexdigest()
    report["environment"] = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
    }
    # Reserve a new directory; never silently replace a previous experiment.
    output.mkdir(parents=True, exist_ok=False)
    (output / "candles.csv").write_text(raw_csv)
    frame.to_csv(output / "features.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
