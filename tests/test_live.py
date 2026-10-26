import copy
import math

import pandas as pd
import pytest

from alphaforge.live import record, report, score

NOW = pd.Timestamp("2026-10-26T05:17:00Z")


def fixture():
    state = {"schema_version": 1, "forecasts": [], "attempts": []}
    feed = {
        "generated_at": NOW.isoformat(),
        "expires_at": "2026-10-26T06:00:00Z",
        "forecasts": [
            {
                "symbol": "BTC-USD",
                "decision_time": "2026-10-26T05:00:00Z",
                "target_time": "2026-10-26T06:00:00Z",
                "last_close_usd": 100,
                "model_version": "v1",
                "predicted_log_return": 0.01,
            }
        ],
    }
    record(state, feed, NOW)
    return state, feed


def candles(config, now):
    return pd.DataFrame(
        [
            dict(
                symbol="BTC-USD",
                timestamp=pd.Timestamp(config.start),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1.0,
            )
        ]
    )


def test_first_prediction_is_immutable_and_expired_rejected():
    state, feed = fixture()
    original = copy.deepcopy(state)
    feed["forecasts"][0]["predicted_log_return"] = 99
    record(state, feed, NOW)
    assert state == original
    with pytest.raises(ValueError):
        record(state, feed, NOW + pd.Timedelta(hours=1))


def test_only_closed_target_scored_and_baseline_matches():
    state, _ = fixture()
    score(state, NOW, lambda *a, **k: pytest.fail("must not fetch future outcome"))
    assert state["forecasts"][0]["status"] == "pending"
    score(state, NOW + pd.Timedelta(hours=1), candles)
    actual = math.log(1.01)
    assert state["forecasts"][0]["realized_log_return"] == pytest.approx(actual)
    result = report(state, NOW + pd.Timedelta(hours=1))["metrics"][0]
    assert result["rmse_bps"] == pytest.approx(abs(0.01 - actual) * 10000)
    assert result["baseline_rmse_bps"] == pytest.approx(actual * 10000)
    score(state, NOW + pd.Timedelta(hours=2), lambda *a, **k: pytest.fail("immutable score"))


def test_missing_target_is_not_substituted_and_retried():
    state, _ = fixture()

    def wrong_hour(config, now):
        frame = candles(config, now)
        frame["timestamp"] -= pd.Timedelta(hours=1)
        return frame

    score(state, NOW + pd.Timedelta(hours=1), wrong_hour)
    assert state["forecasts"][0]["status"] == "missing_candle"
    score(state, NOW + pd.Timedelta(hours=2), candles)
    assert state["forecasts"][0]["status"] == "scored"


def test_model_versions_are_separate_and_missing_slots_counted():
    state, _ = fixture()
    row = dict(
        state["forecasts"][0],
        model_version="v2",
        decision_time="2026-10-26T06:00:00Z",
        target_time="2026-10-26T07:00:00Z",
    )
    state["forecasts"].append(row)
    result = report(state, NOW + pd.Timedelta(hours=2))
    assert len(result["metrics"]) == 2
    assert result["missing_forecast_slots"] == 4


def test_failed_refresh_preserves_history_and_reports_failure(tmp_path):
    import json

    from alphaforge.live import run, write_json

    state, _ = fixture()
    ledger = tmp_path / "state.json"
    write_json(ledger, state)

    def unavailable(*args, **kwargs):
        raise OSError("provider unavailable")

    registry = __import__("pathlib").Path(__file__).parents[1] / "models/hourly-v1/registry.json"
    attempt = run(registry, ledger, tmp_path / "docs", now=NOW, loader=unavailable)
    assert attempt["forecast"] == "failed"
    saved = json.loads(ledger.read_text())
    assert saved["forecasts"] == state["forecasts"]
    assert (
        json.loads((tmp_path / "docs/live-performance.json").read_text())["last_attempt"] == attempt
    )


def test_two_phase_record_then_score(tmp_path):
    import json
    from pathlib import Path

    from alphaforge.data import synthetic
    from alphaforge.live import run

    registry = Path(__file__).parents[1] / "models/hourly-v1/registry.json"
    ledger = tmp_path / "state.json"

    def loader(config, now):
        return synthetic(config)

    run(registry, ledger, tmp_path / "docs", now=NOW, loader=loader, phase="record")
    recorded = json.loads(ledger.read_text())
    assert len(recorded["forecasts"]) == 2
    assert recorded["attempts"][-1]["scoring"] == "pending"
    run(
        registry,
        ledger,
        tmp_path / "docs",
        now=NOW + pd.Timedelta(hours=1),
        loader=loader,
        phase="score",
    )
    saved = json.loads(ledger.read_text())
    assert len(saved["attempts"]) == 1
    assert saved["attempts"][-1]["scoring"] == "ok"
    assert all(row["status"] == "scored" for row in saved["forecasts"])
