"""Training comparisons and expanding-window, out-of-sample execution simulations."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import utc
from .data import HOUR, validate_candles
from .deep import fit_model, predict_model, save_bundle, sequences
from .evaluation import metrics
from .features import build_features
from .splits import chronological_split

KINDS = ("zero_return", "training_mean", "ridge", "lstm", "transformer")


def prepare(candles, config, settings):
    candles = validate_candles(candles)
    candles = candles.loc[
        candles.symbol.isin(config.symbols)
        & (candles.timestamp >= utc(config.start))
        & (candles.timestamp < utc(config.end))
    ].reset_index(drop=True)
    if set(candles.symbol) != set(config.symbols):
        raise ValueError("Input is missing configured symbols")
    frame, names = build_features(candles, config)
    x, rows = sequences(frame, names, settings.sequence_length)
    return candles, names, x, rows


def provenance(candles, config, settings):
    return {
        "config": config.as_dict(),
        "training": asdict(settings),
        "input_sha256": hashlib.sha256(candles.to_csv(index=False).encode()).hexdigest(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("alphaforge", "numpy", "pandas", "scikit-learn", "torch")
        },
    }


def train(candles, config, settings, output, kinds=KINDS):
    if not kinds or not set(kinds).issubset(KINDS):
        raise ValueError("Unknown or empty model selection")
    candles, names, x, rows = prepare(candles, config, settings)
    splits = chronological_split(rows, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = provenance(candles, config, settings)
    report.update(
        features=names,
        results={},
        purged_rows=splits.purged_rows,
        boundaries={"validation": str(splits.validation_start), "test": str(splits.test_start)},
    )
    predictions = []
    for symbol in config.symbols:
        indices = {
            s: getattr(splits, s).loc[getattr(splits, s).symbol == symbol].index.to_numpy()
            for s in ("train", "validation", "test")
        }
        ti, vi = indices["train"], indices["validation"]
        result = {}
        for kind in kinds:
            model = fit_model(
                kind,
                settings,
                config,
                x[ti],
                rows.loc[ti, "target"].to_numpy(),
                x[vi],
                rows.loc[vi, "target"].to_numpy(),
            )
            result[kind] = {}
            for split in ("validation", "test"):
                idx = indices[split]
                pred = predict_model(model, x[idx])
                result[kind][split] = metrics(rows.loc[idx, "target"], pred)
                part = rows.loc[idx, ["symbol", "decision_time", "target_time", "target"]].copy()
                part["prediction"], part["model"], part["split"] = pred, kind, split
                predictions.append(part)
            save_bundle(
                output / "models" / symbol / kind,
                model,
                kind,
                symbol,
                names,
                config,
                settings,
                rows.loc[ti, "target_time"].max(),
            )
        winner = min(result, key=lambda name: result[name]["validation"]["rmse"])
        report["results"][symbol] = {"models": result, "selected_by_validation_rmse": winner}
    candles.to_csv(output / "candles.csv", index=False)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@dataclass(frozen=True)
class BacktestConfig:
    initial_train_hours: int = 2160
    validation_hours: int = 336
    test_hours: int = 336
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    threshold: float = 0.0

    def __post_init__(self):
        for name in ("initial_train_hours", "validation_hours", "test_hours"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fee_bps", "slippage_bps", "threshold"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.fee_bps + self.slippage_bps >= 5000:
            raise ValueError("Round-trip costs must be below 100%")


def simulate(candles, predictions, settings):
    """Forecast at d, execute at open[d+1h], mark/exit at open[d+2h].

    Long/flat, unlevered, full asset allocation. Costs are basis points of starting
    bar equity. Consecutive longs carry; gaps and the terminal bar force liquidation.
    """
    if predictions.duplicated(["symbol", "decision_time"]).any():
        raise ValueError("Duplicate trading signals")
    trades = []
    for symbol, signals in predictions.groupby("symbol"):
        prices = candles.loc[candles.symbol == symbol].set_index("timestamp").open
        for row in signals.sort_values("decision_time").itertuples():
            entry, exit_time = row.decision_time + HOUR, row.decision_time + 2 * HOUR
            if entry not in prices.index or exit_time not in prices.index:
                continue
            trades.append(
                {
                    "symbol": symbol,
                    "decision_time": row.decision_time,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "prediction": row.prediction,
                    "fold": row.fold,
                    "position": float(row.prediction > settings.threshold),
                    "asset_return": float(prices.loc[exit_time] / prices.loc[entry] - 1),
                }
            )
    if not trades:
        raise ValueError("No executable signals with observed entry and exit prices")
    result = pd.DataFrame(trades).sort_values(["symbol", "entry_time"]).reset_index(drop=True)
    cost = (settings.fee_bps + settings.slippage_bps) / 10000
    result["turnover"] = 0.0
    for _, group in result.groupby("symbol"):
        previous, previous_exit, previous_index = 0.0, None, None
        for index, row in group.iterrows():
            if previous_exit != row.entry_time:
                if previous_index is not None:
                    result.loc[previous_index, "turnover"] += previous
                previous = 0.0
            result.loc[index, "turnover"] += abs(row.position - previous)
            previous, previous_exit, previous_index = row.position, row.exit_time, index
        result.loc[previous_index, "turnover"] += previous
    result["gross_return"] = result.position * result.asset_return
    result["cost"] = result.turnover * cost
    result["net_return"] = result.gross_return - result.cost
    if (result.net_return <= -1).any():
        raise ValueError("Simulation equity exhausted under configured costs")
    return result


def summarize_trades(trades, symbols):
    grid = pd.date_range(trades.entry_time.min(), trades.entry_time.max(), freq="h")
    matrix = trades.pivot(index="entry_time", columns="symbol", values="net_return")
    matrix = matrix.reindex(index=grid, columns=symbols).fillna(0)
    # Equal initial subaccounts compound independently; no implicit free rebalancing.
    equity = (1 + matrix).cumprod().mean(axis=1)
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] - 1
    peaks = equity.cummax().clip(lower=1.0)
    std = float(returns.std(ddof=1))
    stats = {
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / peaks - 1).min()),
        "annualized_hourly_sharpe": float(returns.mean() / std * np.sqrt(24 * 365))
        if len(returns) > 1 and np.isfinite(std) and std > 0
        else None,
        "execution_rows": len(trades),
        "turnover": float(trades.turnover.sum()),
        "average_exposure": float(trades.position.mean()),
    }
    return stats, pd.DataFrame(
        {"timestamp": grid, "net_return": returns.to_numpy(), "equity": equity.to_numpy()}
    )


def walk_forward(candles, config, training, settings, output, kind="ridge"):
    if config.horizon != 1:
        raise ValueError(
            "Execution backtests require horizon=1; overlapping horizons are unsupported"
        )
    if kind not in KINDS:
        raise ValueError("Unknown model")
    candles, names, x, rows = prepare(candles, config, training)
    start = rows.decision_time.min()
    last = rows.decision_time.max()
    validation_start = start + settings.initial_train_hours * HOUR
    predictions, folds = [], []
    while validation_start + settings.validation_hours * HOUR <= last:
        test_start = validation_start + settings.validation_hours * HOUR
        test_end = test_start + settings.test_hours * HOUR
        fold_id = len(folds)
        fold = {
            "fold": fold_id,
            "validation_start": str(validation_start),
            "test_start": str(test_start),
            "test_end_exclusive": str(test_end),
            "assets": {},
        }
        for symbol in config.symbols:
            same = rows.symbol == symbol
            ti = rows.index[same & (rows.target_time < validation_start)].to_numpy()
            vi = rows.index[
                same & (rows.decision_time >= validation_start) & (rows.target_time < test_start)
            ].to_numpy()
            ei = rows.index[
                same & (rows.decision_time >= test_start) & (rows.decision_time < test_end)
            ].to_numpy()
            if not len(ti) or not len(vi) or not len(ei):
                raise ValueError("A fold has insufficient history for a configured asset")
            model = fit_model(
                kind,
                training,
                config,
                x[ti],
                rows.loc[ti, "target"].to_numpy(),
                x[vi],
                rows.loc[vi, "target"].to_numpy(),
            )
            part = rows.loc[ei, ["symbol", "decision_time", "target"]].copy()
            part["prediction"], part["fold"] = predict_model(model, x[ei]), fold_id
            predictions.append(part)
            fold["assets"][symbol] = {
                "train_rows": len(ti),
                "validation_rows": len(vi),
                "test_rows": len(ei),
                "last_training_target": str(rows.loc[ti, "target_time"].max()),
                "last_validation_target": str(rows.loc[vi, "target_time"].max()),
                "metrics": metrics(part.target, part.prediction),
            }
        folds.append(fold)
        validation_start += settings.test_hours * HOUR
    if not predictions:
        raise ValueError("History is too short for requested walk-forward windows")
    predictions = pd.concat(predictions, ignore_index=True)
    trades = simulate(candles, predictions, settings)
    stats, equity = summarize_trades(trades, config.symbols)
    reference = predictions.copy()
    reference["prediction"] = settings.threshold + 1
    buy_hold = simulate(candles, reference, settings)
    benchmark, _ = summarize_trades(buy_hold, config.symbols)
    report = provenance(candles, config, training)
    report.update(
        model=kind,
        backtest=asdict(settings),
        folds=folds,
        strategy=stats,
        long_only_benchmark=benchmark,
        skipped_execution_rows=len(predictions) - len(trades),
        assumptions="Long/flat; one-hour execution delay; open-to-open; constant bps "
        "cost per turnover; equal initial asset subaccounts; no market impact or tax.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    candles.to_csv(output / "candles.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)
    equity.to_csv(output / "equity.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
