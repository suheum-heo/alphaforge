import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from alphaforge.api import create_app
from alphaforge.config import Config
from alphaforge.data import HOUR, synthetic
from alphaforge.deep import (
    Predictor,
    TrainingConfig,
    fit_model,
    predict_model,
    save_bundle,
    sequences,
)
from alphaforge.features import build_features
from alphaforge.research import BacktestConfig, simulate, train, walk_forward


@pytest.fixture
def config():
    return Config(start="2024-01-01T00:00:00Z", end="2024-01-11T00:00:00Z", windows=(6,))


@pytest.fixture
def settings():
    return TrainingConfig(sequence_length=8, hidden_size=8, epochs=2, patience=1)


@pytest.fixture
def candles(config):
    return synthetic(config)


@pytest.mark.parametrize("kind", ["zero_return", "training_mean", "ridge", "lstm", "transformer"])
def test_bundle_roundtrip_and_causal_inference(config, settings, candles, tmp_path, kind):
    asset = candles.loc[candles.symbol == "BTC-USD"]
    frame, names = build_features(asset, config, include_targets=False)
    x, rows = sequences(frame, names, settings.sequence_length)
    model = fit_model(
        kind,
        settings,
        config,
        x[:80],
        rows.target.iloc[:80].to_numpy(),
        x[80:100],
        rows.target.iloc[80:100].to_numpy(),
    )
    path = tmp_path / kind
    save_bundle(path, model, kind, "BTC-USD", names, config, settings, rows.target_time.iloc[79])
    predictor = Predictor(path)
    np.testing.assert_allclose(
        predictor.predict_array(x[-1:]), predict_model(model, x[-1:]), atol=1e-8, rtol=1e-5
    )
    prediction = predictor.predict_candles(asset)
    assert pd.Timestamp(prediction["decision_time"]) == asset.timestamp.max() + HOUR
    assert np.isfinite(prediction["predicted_log_return"])
    if kind in ("lstm", "transformer"):
        assert len(model.history) <= settings.epochs
        weights = path / "weights.pt"
        weights.write_bytes(weights.read_bytes() + b"corruption")
        with pytest.raises(ValueError, match="checksum"):
            Predictor(path)


def test_sequences_do_not_cross_gaps_or_assets(config, settings, candles):
    frame, names = build_features(candles, config)
    missing = pd.Timestamp("2024-01-05T00:00:00Z")
    frame = frame.loc[frame.decision_time != missing]
    x, rows = sequences(frame, names, settings.sequence_length)
    assert not rows.decision_time.between(missing, missing + 7 * HOUR).any()
    assert x.shape[1:] == (8, len(names))
    for i, row in rows.iterrows():
        history = frame.loc[
            (frame.symbol == row.symbol) & (frame.decision_time <= row.decision_time)
        ].tail(8)
        np.testing.assert_allclose(x[i], history[names], rtol=1e-6)


def test_inference_features_do_not_need_future(config, candles):
    labelled, names = build_features(candles, config)
    causal, names2 = build_features(candles, config, include_targets=False)
    assert names == names2
    joined = labelled.merge(causal, on=["symbol", "timestamp"], suffixes=("_a", "_b"))
    for name in names:
        np.testing.assert_array_equal(joined[name + "_a"], joined[name + "_b"])
    assert causal.timestamp.max() == candles.timestamp.max()
    assert causal.loc[causal.timestamp == candles.timestamp.max(), "target"].isna().all()


def test_training_integration(config, settings, candles, tmp_path):
    report = train(candles, config, settings, tmp_path / "training")
    assert set(report["results"]) == set(config.symbols)
    for symbol, result in report["results"].items():
        assert len(result["models"]) == 5
        best = min(result["models"], key=lambda k: result["models"][k]["validation"]["rmse"])
        assert result["selected_by_validation_rmse"] == best
        assert (tmp_path / "training/models" / symbol / best / "model.json").exists()


@pytest.fixture
def bundle(config, settings, candles, tmp_path):
    train(candles, config, settings, tmp_path / "training", kinds=("ridge",))
    return tmp_path / "training/models/BTC-USD/ridge"


def payload(candles):
    return {
        "candles": json.loads(
            candles.loc[candles.symbol == "BTC-USD"]
            .tail(40)
            .to_json(orient="records", date_format="iso")
        )
    }


def test_api_matches_offline_and_rejects_invalid(bundle, candles):
    current = candles.timestamp.max() + HOUR + pd.Timedelta(minutes=10)
    with TestClient(create_app(bundle, clock=lambda: current)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/model").json()["kind"] == "ridge"
        response = client.post("/predict", json=payload(candles))
        assert response.status_code == 200, response.text
        expected = Predictor(bundle).predict_candles(candles.loc[candles.symbol == "BTC-USD"])
        assert response.json()["predicted_log_return"] == pytest.approx(
            expected["predicted_log_return"]
        )
        malformed = payload(candles)
        malformed["candles"][-1]["volume"] = -1
        assert client.post("/predict", json=malformed).status_code == 422
        malformed = payload(candles)
        malformed["candles"][-1]["symbol"] = "ETH-USD"
        assert client.post("/predict", json=malformed).status_code == 422
        malformed = payload(candles)
        malformed["candles"][-2] = malformed["candles"][-1]
        assert client.post("/predict", json=malformed).status_code == 422
        malformed = payload(candles)
        malformed["candles"][-1]["timestamp"] = "2024-01-10T23:00:00"
        assert client.post("/predict", json=malformed).status_code == 422


@pytest.mark.parametrize("offset", [0, 3])
def test_api_rejects_open_or_stale(bundle, candles, offset):
    with TestClient(
        create_app(bundle, clock=lambda: candles.timestamp.max() + offset * HOUR)
    ) as client:
        assert client.post("/predict", json=payload(candles)).status_code == 422


def test_inference_rejects_recent_gap(bundle, candles):
    asset = candles.loc[candles.symbol == "BTC-USD"].copy()
    asset = asset.loc[asset.timestamp != asset.timestamp.max() - HOUR]
    with pytest.raises(ValueError, match="Latest candle"):
        Predictor(bundle).predict_candles(asset)


def test_execution_delay_and_costs(config, candles):
    stamp = candles.timestamp.min() + 24 * HOUR
    signals = pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * 3,
            "decision_time": [stamp, stamp + HOUR, stamp + 2 * HOUR],
            "prediction": [1.0, 1.0, -1.0],
            "fold": [0, 0, 0],
        }
    )
    trades = simulate(candles, signals, BacktestConfig(fee_bps=10, slippage_bps=5))
    assert trades.entry_time.iloc[0] == stamp + HOUR
    assert trades.exit_time.iloc[0] == stamp + 2 * HOUR
    np.testing.assert_array_equal(trades.turnover, [1, 0, 1])
    np.testing.assert_allclose(trades.cost, [0.0015, 0, 0.0015])
    prices = candles.loc[candles.symbol == "BTC-USD"].set_index("timestamp").open
    assert trades.asset_return.iloc[0] == pytest.approx(
        prices[stamp + 2 * HOUR] / prices[stamp + HOUR] - 1
    )
    free = simulate(candles, signals, BacktestConfig(fee_bps=0, slippage_bps=0))
    assert free.net_return.sum() > trades.net_return.sum()


def test_terminal_and_gap_liquidation(candles):
    stamp = candles.timestamp.min() + 24 * HOUR
    signals = pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * 2,
            "decision_time": [stamp, stamp + 3 * HOUR],
            "prediction": [1.0, 1.0],
            "fold": [0, 0],
        }
    )
    trades = simulate(candles, signals, BacktestConfig())
    np.testing.assert_array_equal(trades.turnover, [2, 2])


@pytest.mark.parametrize("kind", ["ridge", "lstm", "transformer"])
def test_walk_forward_purging_and_execution(config, settings, candles, tmp_path, kind):
    backtest = BacktestConfig(initial_train_hours=96, validation_hours=24, test_hours=48)
    report = walk_forward(candles, config, settings, backtest, tmp_path / kind, kind)
    assert len(report["folds"]) >= 2
    for fold in report["folds"]:
        for asset in fold["assets"].values():
            assert pd.Timestamp(asset["last_training_target"]) < pd.Timestamp(
                fold["validation_start"]
            )
            assert pd.Timestamp(asset["last_validation_target"]) < pd.Timestamp(fold["test_start"])
    pred = pd.read_csv(tmp_path / kind / "predictions.csv")
    assert not pred.duplicated(["symbol", "decision_time"]).any()
    assert report["strategy"]["execution_rows"] > 0
    assert (tmp_path / kind / "equity.csv").exists()


def test_multihorizon_execution_rejected(config, settings, candles, tmp_path):
    with pytest.raises(ValueError, match="horizon=1"):
        walk_forward(
            candles, replace(config, horizon=3), settings, BacktestConfig(), tmp_path / "out"
        )


def test_future_data_does_not_change_earlier_fold(config, settings, candles, tmp_path):
    backtest = BacktestConfig(initial_train_hours=96, validation_hours=24, test_hours=24)
    walk_forward(candles, config, settings, backtest, tmp_path / "a")
    changed = candles.copy()
    changed.loc[
        changed.timestamp >= pd.Timestamp("2024-01-09T00:00:00Z"), ["open", "high", "low", "close"]
    ] *= 1.5
    walk_forward(changed, config, settings, backtest, tmp_path / "b")
    a = pd.read_csv(tmp_path / "a/predictions.csv")
    b = pd.read_csv(tmp_path / "b/predictions.csv")
    pd.testing.assert_frame_equal(a.loc[a.fold == 0], b.loc[b.fold == 0])


@pytest.mark.parametrize(
    "kwargs",
    [{"epochs": 0}, {"sequence_length": 0}, {"hidden_size": 7}, {"learning_rate": float("nan")}],
)
def test_invalid_training_settings(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_portfolio_has_no_free_rebalancing():
    from alphaforge.research import summarize_trades

    grid = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    trades = pd.DataFrame(
        {
            "entry_time": list(grid) * 2,
            "symbol": ["BTC-USD"] * 2 + ["ETH-USD"] * 2,
            "net_return": [0.1, -0.1, 0.0, 0.0],
            "turnover": [1.0, 1.0, 0.0, 0.0],
            "position": [1.0, 1.0, 0.0, 0.0],
        }
    )
    report, equity = summarize_trades(trades, ["BTC-USD", "ETH-USD"])
    assert equity.equity.iloc[-1] == pytest.approx((1.1 * 0.9 + 1.0) / 2)
    assert report["total_return"] == pytest.approx(-0.005)
