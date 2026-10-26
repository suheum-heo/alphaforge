"""Date-bounded research protocol with validation-only trading-rule selection."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import utc
from .data import HOUR, quality
from .deep import fit_model, predict_model, save_bundle
from .evaluation import metrics
from .research import KINDS, BacktestConfig, prepare, provenance, simulate, summarize_trades


def evaluate_signals(candles, predictions, settings, symbols):
    trades = simulate(candles, predictions, settings)
    stats, equity = summarize_trades(trades, symbols)
    stats["position_changes"] = int((trades.turnover > 0).sum())
    stats["cost_bps_sum"] = float(trades.cost.sum() * 10000)
    return stats, trades, equity


def choose_threshold(candles, predictions, settings, symbols, thresholds):
    """Only validation predictions belong here; ties favor less turnover then larger threshold."""
    candidates = []
    for threshold in thresholds:
        stats, _, _ = evaluate_signals(
            candles, predictions, replace(settings, threshold=threshold), symbols
        )
        candidates.append({"threshold": threshold, **stats})
    best = max(candidates, key=lambda c: (c["total_return"], -c["turnover"], c["threshold"]))
    return best["threshold"], candidates


def study(
    candles,
    config,
    training,
    output,
    *,
    validation_start="2026-01-01T00:00:00Z",
    holdout_start="2026-07-01T00:00:00Z",
    thresholds=(0.0, 0.0005, 0.001, 0.002, 0.003),
    kinds=KINDS,
    execution=None,
):
    """Fit once, freeze validation-selected rules, then evaluate the designated holdout."""
    if config.horizon != 1:
        raise ValueError("Study execution requires horizon=1")
    vs, hs = utc(validation_start), utc(holdout_start)
    if not utc(config.start) < vs < hs < utc(config.end):
        raise ValueError("Require start < validation_start < holdout_start < end")
    if not thresholds or any(not np.isfinite(t) or t < 0 for t in thresholds):
        raise ValueError("Thresholds must be nonnegative and finite")
    if not kinds or not set(kinds).issubset(KINDS):
        raise ValueError("Unknown or empty model candidates")
    execution = execution or BacktestConfig()
    candles, names, x, rows = prepare(candles, config, training)
    # Purge both forecast-label availability and simulated execution exit time.
    masks = {
        "train": rows.target_time < vs,
        "validation": (rows.decision_time >= vs)
        & (rows.target_time < hs)
        & (rows.decision_time + 2 * HOUR < hs),
        "holdout": (rows.decision_time >= hs) & (rows.decision_time + 2 * HOUR < utc(config.end)),
    }
    for symbol in config.symbols:
        if any(not (mask & (rows.symbol == symbol)).any() for mask in masks.values()):
            raise ValueError("Each study period must contain every configured asset")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = provenance(candles, config, training)
    report.update(
        protocol={
            "validation_start": str(vs),
            "holdout_start": str(hs),
            "end_exclusive": str(utc(config.end)),
            "thresholds": list(thresholds),
            "fee_bps": execution.fee_bps,
            "slippage_bps": execution.slippage_bps,
            "selection": "Validation net portfolio return; cash if no positive edge",
            "candidates": list(kinds),
        },
        features=names,
        data_quality=quality(candles),
        results={},
        selection={},
    )
    report["splits"] = {
        name: {
            "rows": int(mask.sum()),
            "first_decision": str(rows.loc[mask, "decision_time"].min()),
            "last_target": str(rows.loc[mask, "target_time"].max()),
        }
        for name, mask in masks.items()
    }
    (output / "protocol.json").write_text(json.dumps(report["protocol"], indent=2) + "\n")
    # Keep holdout predictions deferred until model and threshold choices are frozen.
    models, validation = {}, {}
    for kind in kinds:
        models[kind] = {}
        parts = []
        for symbol in config.symbols:
            ti = rows.index[masks["train"] & (rows.symbol == symbol)].to_numpy()
            vi = rows.index[masks["validation"] & (rows.symbol == symbol)].to_numpy()
            model = fit_model(
                kind,
                training,
                config,
                x[ti],
                rows.loc[ti, "target"].to_numpy(),
                x[vi],
                rows.loc[vi, "target"].to_numpy(),
            )
            models[kind][symbol] = model
            part = rows.loc[vi, ["symbol", "decision_time", "target"]].copy()
            part["prediction"] = predict_model(model, x[vi])
            part["fold"] = 0
            parts.append(part)
            save_bundle(
                output / "models" / symbol / kind,
                model,
                kind,
                symbol,
                names,
                config,
                training,
                rows.loc[ti, "target_time"].max(),
            )
        validation[kind] = pd.concat(parts, ignore_index=True)
        threshold, candidates = choose_threshold(
            candles, validation[kind], execution, config.symbols, thresholds
        )
        chosen = next(c for c in candidates if c["threshold"] == threshold)
        report["selection"][kind] = {
            "threshold": threshold,
            "validation": chosen,
            "validation_grid": candidates,
        }
    winner = max(kinds, key=lambda kind: report["selection"][kind]["validation"]["total_return"])
    if report["selection"][winner]["validation"]["total_return"] <= 0:
        winner = "cash"
    report["selected_portfolio"] = winner
    frozen = {
        "selected_portfolio": winner,
        "selection": report["selection"],
        "protocol": report["protocol"],
    }
    # Explicit artifact written before any holdout model predictions or metrics.
    (output / "validation-selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    all_predictions = []
    for kind in kinds:
        parts = []
        for symbol in config.symbols:
            ei = rows.index[masks["holdout"] & (rows.symbol == symbol)].to_numpy()
            part = rows.loc[ei, ["symbol", "decision_time", "target"]].copy()
            part["prediction"] = predict_model(models[kind][symbol], x[ei])
            part["fold"] = 0
            parts.append(part)
        holdout = pd.concat(parts, ignore_index=True)
        threshold = report["selection"][kind]["threshold"]
        stats, trades, equity = evaluate_signals(
            candles, holdout, replace(execution, threshold=threshold), config.symbols
        )
        zero_stats, _, _ = evaluate_signals(
            candles, holdout, replace(execution, threshold=0), config.symbols
        )
        report["results"][kind] = {
            "holdout_strategy": stats,
            "zero_threshold_strategy": zero_stats,
            "forecast": {
                symbol: metrics(group.target, group.prediction)
                for symbol, group in holdout.groupby("symbol")
            },
        }
        trades.to_csv(output / f"trades-{kind}.csv", index=False)
        equity.to_csv(output / f"equity-{kind}.csv", index=False)
        for split, part in [("validation", validation[kind]), ("holdout", holdout)]:
            part = part.copy()
            part["model"], part["split"] = kind, split
            all_predictions.append(part)
    report["selected_holdout_return"] = (
        0.0 if winner == "cash" else report["results"][winner]["holdout_strategy"]["total_return"]
    )
    benchmark = holdout.copy()
    benchmark["prediction"] = 1.0
    stats, _, eq = evaluate_signals(
        candles, benchmark, replace(execution, threshold=0), config.symbols
    )
    report["long_only_benchmark"] = stats
    eq.to_csv(output / "equity-buy-hold.csv", index=False)
    pd.concat(all_predictions).to_csv(output / "predictions.csv", index=False)
    candles.to_csv(output / "candles.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
