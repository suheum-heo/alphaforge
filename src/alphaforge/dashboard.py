"""Export a dependency-free, read-only research dashboard from local experiment artifacts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

WEB = Path(__file__).with_name("web")


def points(path, limit=300):
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    indices = np.unique(np.linspace(0, len(frame) - 1, min(limit, len(frame)), dtype=int))
    return [
        {"time": str(frame.iloc[i].timestamp), "value": float(frame.iloc[i].equity)}
        for i in indices
    ]


def export_dashboard(runs, output):
    """Publish only selected market metrics; omit local paths, model weights and raw manifests."""
    runs, output = Path(runs), Path(output)
    source = runs / "deep-001"
    if not (source / "report.json").exists():
        raise ValueError("Dashboard requires runs/deep-001/report.json; train the example first")
    report = json.loads((source / "report.json").read_text())
    candle_path = source / "candles.csv"
    candles = pd.read_csv(candle_path)
    data = {
        "schema": 1,
        "title": "AlphaForge Research",
        "dataset": {
            "source": "Coinbase Exchange",
            "rows": len(candles),
            "start": str(candles.timestamp.min()),
            "end": str(candles.timestamp.max()),
            "symbols": sorted(candles.symbol.unique().tolist()),
        },
        "comparison": report["results"],
        "forecasts": {},
        "backtests": {},
        "study": None,
    }
    forecasts = pd.read_csv(source / "predictions.csv")
    for (symbol, model), group in forecasts.loc[forecasts.split == "test"].groupby(
        ["symbol", "model"]
    ):
        data["forecasts"].setdefault(symbol, {})[model] = [
            {"time": row.decision_time, "actual": row.target, "prediction": row.prediction}
            for row in group.sort_values("decision_time").tail(96).itertuples()
        ]
    for kind in ("ridge", "lstm", "transformer"):
        folder = runs / f"walk-{kind}-001"
        if not (folder / "report.json").exists():
            continue
        backtest = json.loads((folder / "report.json").read_text())
        data["backtests"][kind] = {
            "stats": backtest["strategy"],
            "benchmark": backtest["long_only_benchmark"],
            "folds": len(backtest["folds"]),
            "equity": points(folder / "equity.csv"),
        }
    folder = runs / "study-001"
    if (folder / "report.json").exists():
        study = json.loads((folder / "report.json").read_text())
        data["study"] = {
            key: study[key]
            for key in (
                "protocol",
                "data_quality",
                "selection",
                "selected_portfolio",
                "selected_holdout_return",
                "results",
                "long_only_benchmark",
            )
        }
        data["study"]["equity"] = {
            kind: points(folder / f"equity-{kind}.csv") for kind in study["results"]
        }
        data["study"]["equity"]["buy_hold"] = points(folder / "equity-buy-hold.csv")
    # Roundtrip rejects NaN before touching an existing output.
    encoded = json.dumps(data, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css", "PROTOCOL.md"):
        shutil.copyfile(WEB / name, output / name)
    (output / "dashboard-data.json").write_text(encoded + "\n")
    return {
        "output": str(output),
        "rows": len(candles),
        "extended_study": data["study"] is not None,
    }
