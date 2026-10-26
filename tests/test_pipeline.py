import json
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from alphaforge.config import Config
from alphaforge.data import ingest, request_candles, synthetic, validate_candles
from alphaforge.evaluation import metrics
from alphaforge.features import build_features
from alphaforge.models import baselines
from alphaforge.pipeline import run
from alphaforge.splits import chronological_split


@pytest.fixture
def config():
    return Config(
        start="2024-01-01T00:00:00Z", end="2024-02-01T00:00:00Z", windows=(6, 24), horizon=3
    )


@pytest.fixture
def candles(config):
    return synthetic(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 1.5},
        {"windows": (1,)},
        {"windows": (6, 6)},
        {"train_fraction": 0.9, "validation_fraction": 0.2},
        {"start": "2024-01-01"},
        {"end": "2020-01-01T00:00:00Z"},
        {"symbols": ("BTC-USD", "BTC-USD")},
        {"ridge_alpha": float("nan")},
        {"seed": -1},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_features_are_causal(config, candles):
    first, names = build_features(candles, config)
    cutoff = pd.Timestamp("2024-01-15T00:00:00Z")
    modified = candles.copy()
    modified.loc[modified.timestamp > cutoff, ["open", "high", "low", "close"]] *= 2
    second, _ = build_features(modified, config)
    mask = first.timestamp <= cutoff
    pd.testing.assert_frame_equal(first.loc[mask, names], second.loc[mask, names])
    row = first.iloc[10]
    prices = candles.loc[candles.symbol == row.symbol].set_index("timestamp").close
    assert row.target == pytest.approx(
        np.log(prices.loc[row.timestamp + pd.Timedelta(hours=3)] / prices.loc[row.timestamp])
    )
    assert row.decision_time == row.timestamp + pd.Timedelta(hours=1)
    assert row.target_time == row.decision_time + pd.Timedelta(hours=3)


def test_missing_hour(config, candles):
    missing = pd.Timestamp("2024-01-15T00:00:00Z")
    candles = candles.loc[~((candles.symbol == "BTC-USD") & (candles.timestamp == missing))]
    frame, _ = build_features(candles, config)
    btc = frame.loc[frame.symbol == "BTC-USD"]
    assert not btc.timestamp.between(
        missing - pd.Timedelta(hours=config.horizon),
        missing + pd.Timedelta(hours=max(config.windows)),
    ).any()
    assert (frame.loc[frame.symbol == "ETH-USD"].timestamp == missing).any()


def test_global_split_and_purging(config, candles):
    frame, _ = build_features(candles, config)
    parts = chronological_split(frame.sample(frac=1, random_state=4), config)
    assert parts.train.target_time.max() < parts.validation.decision_time.min()
    assert parts.validation.target_time.max() < parts.test.decision_time.min()
    assert parts.purged_rows == 2 * len(config.symbols) * config.horizon
    for part in [parts.train, parts.validation, parts.test]:
        assert set(part.symbol) == set(config.symbols)
    assert set(parts.train.decision_time).isdisjoint(parts.validation.decision_time)


def test_scaler_training_only():
    model = baselines(10)["ridge"]
    model.fit(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), np.array([1.0, 0.0, -1.0]))
    model.predict(np.array([[1e9, 1e9]]))
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, [3.0, 4.0])


def test_ingestion_pagination_bounds_and_open_candle(config):
    calls = []
    now = pd.Timestamp("2024-01-30T05:45:00Z")

    def fetch(symbol, start, end):
        calls.append((symbol, start, end))
        timestamps = pd.date_range(start - pd.Timedelta(hours=1), end, freq="h")
        rows = [[int(t.timestamp()), 90, 110, 100, 101, 12] for t in timestamps]
        return list(reversed(rows)) + [rows[1]]

    frame = ingest(config, fetch=fetch, now=now)
    assert len(calls) == 6
    assert all(end - start <= pd.Timedelta(hours=299) for _, start, end in calls)
    assert frame.timestamp.min() == pd.Timestamp(config.start)
    assert frame.timestamp.max() == now.floor("h") - pd.Timedelta(hours=1)
    assert not frame.duplicated(["symbol", "timestamp"]).any()


def test_conflicting_duplicates(config):
    def fetch(symbol, start, end):
        return [
            [int(start.timestamp()), 90, 110, 100, 101, 12],
            [int(start.timestamp()), 90, 110, 100, 102, 12],
        ]

    with pytest.raises(ValueError, match="Duplicate"):
        ingest(config, fetch=fetch)


@pytest.mark.parametrize("status, expected", [(429, 4), (503, 4), (400, 1)])
def test_retry_policy(monkeypatch, status, expected):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.com", status, "failure", {}, None)

    monkeypatch.setattr("alphaforge.data.urlopen", fail)
    with pytest.raises(HTTPError):
        request_candles(
            "BTC-USD",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            sleep=lambda _: None,
        )
    assert len(calls) == expected


@pytest.mark.parametrize(
    "column,value", [("volume", -1), ("close", np.inf), ("low", 1e9), ("open", 0)]
)
def test_invalid_candles(candles, column, value):
    candles.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_candles(candles)


def test_metrics():
    result = metrics([1, -1], [0, 0])
    assert result["rmse"] == 1
    assert result["correlation"] is None
    assert result["direction_accuracy"] == 0
    assert result["direction_coverage"] == 0
    with pytest.raises(ValueError):
        metrics([np.nan], [1])


def test_end_to_end(config, candles, tmp_path):
    report = run(candles, config, tmp_path / "first", "synthetic_demo")
    assert report == run(candles, config, tmp_path / "second", "synthetic_demo")
    saved = json.loads((tmp_path / "first/report.json").read_text())
    assert saved["source"] == "synthetic_demo"
    for result in saved["results"].values():
        winner = min(result["models"], key=lambda m: result["models"][m]["validation"]["rmse"])
        assert result["selected_by_validation_rmse"] == winner
    predictions = pd.read_csv(tmp_path / "first/predictions.csv")
    assert set(predictions.split) == {"validation", "test"}
    assert np.isfinite(predictions.prediction).all()
    with pytest.raises(FileExistsError):
        run(candles, config, tmp_path / "first", "synthetic_demo")


def test_test_values_do_not_affect_validation(config, candles, tmp_path):
    frame, _ = build_features(candles, config)
    parts = chronological_split(frame, config)
    modified = candles.copy()
    modified.loc[modified.timestamp >= parts.test_start, ["open", "high", "low", "close"]] *= 1.2
    a = run(candles, config, tmp_path / "a", "demo")
    b = run(modified, config, tmp_path / "b", "demo")
    for symbol in config.symbols:
        assert (
            a["results"][symbol]["selected_by_validation_rmse"]
            == b["results"][symbol]["selected_by_validation_rmse"]
        )
        for model in a["results"][symbol]["models"]:
            assert (
                a["results"][symbol]["models"][model]["validation"]
                == b["results"][symbol]["models"][model]["validation"]
            )


def test_insufficient_data(config, candles):
    with pytest.raises(ValueError, match="No usable"):
        build_features(candles.groupby("symbol").head(5), config)
    with pytest.raises(ValueError, match="every configured symbol"):
        frame, _ = build_features(candles.loc[candles.symbol == "BTC-USD"], config)
        chronological_split(frame, config)


def test_cli_demo(tmp_path, capsys):
    from alphaforge.cli import main

    path = tmp_path / "config.toml"
    path.write_text('start = "2024-01-01T00:00:00Z"\nend = "2024-01-15T00:00:00Z"\n')
    main(["demo", "--config", str(path), "--output", str(tmp_path / "demo")])
    assert "synthetic_demo" in capsys.readouterr().out


def test_zero_volume(config, candles):
    candles["volume"] = 0
    frame, names = build_features(candles, config)
    assert np.isfinite(frame[names].to_numpy()).all()


def test_constant_prediction_correlation_is_undefined():
    result = metrics(np.arange(1000), np.full(1000, 0.000123456789))
    assert result["correlation"] is None
