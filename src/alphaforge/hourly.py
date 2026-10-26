"""Atomic hourly forecast export from a versioned, frozen model registry."""

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from .data import HOUR, ingest, validate_candles
from .deep import Predictor


def load_registry(path):
    path = Path(path)
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported registry schema")
    entries = registry["models"]
    if len(entries) != 2 or {e["symbol"] for e in entries} != {"BTC-USD", "ETH-USD"}:
        raise ValueError("Registry must contain exactly BTC and ETH")
    loaded = []
    for entry in entries:
        bundle = (path.parent / entry["path"]).resolve()
        if not bundle.is_relative_to(path.parent.resolve()):
            raise ValueError("Bundle must be inside registry directory")
        digest = hashlib.sha256((bundle / "model.json").read_bytes()).hexdigest()
        if digest != entry["metadata_sha256"]:
            raise ValueError("Model metadata checksum mismatch")
        predictor = Predictor(bundle)
        if predictor.metadata["symbol"] != entry["symbol"] or predictor.config.horizon != 1:
            raise ValueError("Registry symbol/horizon mismatch")
        loaded.append((entry, predictor))
    return registry, loaded


def refresh(registry_path, output, *, now=None, loader=ingest):
    """Publish both assets together or leave the prior artifact untouched on failure."""
    torch.set_num_threads(1)
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None or pd.isna(current):
        raise ValueError("now must be timezone-aware")
    current = current.tz_convert("UTC")
    boundary = current.floor("h")
    registry, models = load_registry(registry_path)
    forecasts = []
    for entry, predictor in models:
        history = max(168, max(predictor.config.windows) + predictor.settings.sequence_length + 24)
        config = replace(
            predictor.config,
            symbols=(entry["symbol"],),
            start=(boundary - history * HOUR).isoformat(),
            end=boundary.isoformat(),
        )
        candles = validate_candles(loader(config, now=current))
        if set(candles.symbol) != {entry["symbol"]}:
            raise ValueError("Provider returned the wrong asset")
        if candles.timestamp.max() + HOUR != boundary:
            raise ValueError("Latest closed hour is missing or contains an unfinished candle")
        result = predictor.predict_candles(candles)
        if pd.Timestamp(result["decision_time"]) != boundary:
            raise ValueError("Forecast must use the latest closed candle")
        result.update(
            last_close_usd=float(candles.sort_values("timestamp").iloc[-1].close),
            candle_open_time=candles.timestamp.max().isoformat(),
            model_version=entry["version"],
            metadata_sha256=entry["metadata_sha256"],
            weights_sha256=predictor.metadata.get("weights_sha256"),
            trained_through=predictor.metadata["trained_through"],
            validation_through=registry["validation_through"],
        )
        forecasts.append(result)
    generated = pd.Timestamp.now(tz="UTC") if now is None else current
    expires = boundary + HOUR
    if generated >= expires:
        raise ValueError("Forecast expired during generation; retry with the new hour")
    payload = {
        "schema_version": 1,
        "generated_at": generated.isoformat(),
        "forecast_start": boundary.isoformat(),
        "expires_at": expires.isoformat(),
        "source": "Coinbase Exchange public hourly candles",
        "registry_version": registry["version"],
        "forecasts": forecasts,
        "schedule": "Around minute 17 each hour UTC; scheduling delays are possible",
        "purpose": "Forecasting demonstration only; no trade execution",
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=".hourly-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("models/hourly-v1/registry.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/latest-forecasts.json"))
    args = parser.parse_args()
    payload = refresh(args.registry, args.output)
    print(
        json.dumps(
            {
                "generated_at": payload["generated_at"],
                "expires_at": payload["expires_at"],
                "models": [f["model_version"] for f in payload["forecasts"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
