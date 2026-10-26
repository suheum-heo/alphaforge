"""Persistent, prospective forecast ledger and closed-candle scoring."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .config import Config
from .data import HOUR, ingest, validate_candles
from .hourly import refresh


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(state, feed, now):
    """First forecast per asset/hour wins, including after model changes or reruns."""
    now = pd.Timestamp(now)
    if not pd.Timestamp(feed["generated_at"]) <= now < pd.Timestamp(feed["expires_at"]):
        raise ValueError("Cannot record an expired or future forecast")
    existing = {(r["symbol"], r["decision_time"]) for r in state["forecasts"]}
    for forecast in feed["forecasts"]:
        key = (forecast["symbol"], forecast["decision_time"])
        if key not in existing:
            state["forecasts"].append(
                {
                    **forecast,
                    "generated_at": feed["generated_at"],
                    "recorded_at": now.isoformat(),
                    "status": "pending",
                }
            )
            existing.add(key)


def score(state, now, loader=ingest):
    """Never use open candles or substitute a nearby candle for a missing target."""
    now = pd.Timestamp(now)
    for symbol in ("BTC-USD", "ETH-USD"):
        pending = [
            r
            for r in state["forecasts"]
            if r["symbol"] == symbol
            and r["status"] != "scored"
            and pd.Timestamp(r["target_time"]) <= now.floor("h")
        ]
        if not pending:
            continue
        # Fetch only target candles, in bounded chunks, even after a long outage.
        starts = sorted({pd.Timestamp(r["target_time"]) - HOUR for r in pending})
        closes = {}
        for offset in range(0, len(starts), 168):
            batch = starts[offset : offset + 168]
            config = Config(
                symbols=(symbol,), start=batch[0].isoformat(), end=(batch[-1] + HOUR).isoformat()
            )
            frame = validate_candles(loader(config, now=now))
            if not frame.empty and set(frame.symbol) != {symbol}:
                raise ValueError("Scoring provider returned the wrong asset")
            closes.update(
                {
                    row.timestamp: float(row.close)
                    for row in frame.itertuples()
                    if row.timestamp + HOUR <= now
                }
            )
        for row in pending:
            close = closes.get(pd.Timestamp(row["target_time"]) - HOUR)
            if close is None:
                row["status"] = "missing_candle"
                continue
            row.update(
                status="scored",
                target_close_usd=close,
                realized_log_return=math.log(close / row["last_close_usd"]),
                scored_at=now.isoformat(),
            )


def report(state, now):
    groups = {}
    for row in state["forecasts"]:
        key = (row["symbol"], row["model_version"])
        groups.setdefault(key, []).append(row)
    metrics = []
    for (symbol, version), rows in sorted(groups.items()):
        scored = [r for r in rows if r["status"] == "scored"]
        result = {
            "symbol": symbol,
            "model_version": version,
            "recorded": len(rows),
            "scored": len(scored),
            "pending": sum(r["status"] == "pending" for r in rows),
            "missing_candles": sum(r["status"] == "missing_candle" for r in rows),
        }
        if scored:
            errors = [r["predicted_log_return"] - r["realized_log_return"] for r in scored]
            actual = [r["realized_log_return"] for r in scored]
            result.update(
                rmse_bps=math.sqrt(sum(e * e for e in errors) / len(errors)) * 10000,
                baseline_rmse_bps=math.sqrt(sum(y * y for y in actual) / len(actual)) * 10000,
                mae_bps=sum(abs(e) for e in errors) / len(errors) * 10000,
                baseline_mae_bps=sum(abs(y) for y in actual) / len(actual) * 10000,
            )
        metrics.append(result)
    first = min((pd.Timestamp(r["decision_time"]) for r in state["forecasts"]), default=None)
    expected = int((pd.Timestamp(now).floor("h") - first) / HOUR) + 1 if first is not None else 0
    return {
        "schema_version": 1,
        "updated_at": pd.Timestamp(now).isoformat(),
        "started_at": first.isoformat() if first is not None else None,
        "missing_forecast_slots": max(0, expected * 2 - len(state["forecasts"])),
        "metrics": metrics,
        "last_attempt": state["attempts"][-1] if state["attempts"] else None,
        "recent_attempts": state["attempts"][-48:],
        "recent_forecasts": state["forecasts"][-96:],
    }


def run(registry, ledger, output, *, now=None, loader=ingest, phase="all"):
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    ledger, output = Path(ledger), Path(output)
    state = (
        json.loads(ledger.read_text())
        if ledger.exists()
        else {"schema_version": 1, "forecasts": [], "attempts": []}
    )
    if state["schema_version"] != 1:
        raise ValueError("Unsupported ledger schema")
    attempt = {"at": current.isoformat(), "forecast": "ok", "scoring": "ok"}
    if phase == "score":
        attempt = state["attempts"].pop()
        attempt["scoring"] = "ok"
    else:
        try:
            feed = refresh(registry, output / "latest-forecasts.json", now=now, loader=loader)
            recorded = pd.Timestamp.now(tz="UTC") if now is None else current
            record(state, feed, recorded)
            state["latest_feed"] = feed
        except Exception as exc:
            attempt.update(forecast="failed", forecast_error=f"{type(exc).__name__}: {exc}")
    # Save the forecast before attempting outcome retrieval. No historical backfill.
    write_json(ledger, state)
    try:
        if phase != "record":
            score(state, current, loader)
        else:
            attempt["scoring"] = "pending"
    except Exception as exc:
        attempt.update(scoring="failed", scoring_error=f"{type(exc).__name__}: {exc}")
    state["attempts"].append(attempt)
    write_json(ledger, state)
    if "latest_feed" in state:
        write_json(output / "latest-forecasts.json", state["latest_feed"])
    write_json(output / "live-performance.json", report(state, current))
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "record", "score"], default="all")
    parser.add_argument("--registry", type=Path, default=Path("models/hourly-v1/registry.json"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    print(json.dumps(run(args.registry, args.ledger, args.output, phase=args.phase), indent=2))


if __name__ == "__main__":
    main()
