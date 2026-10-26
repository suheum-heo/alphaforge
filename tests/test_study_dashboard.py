import json

import pandas as pd
import pytest

pytest.importorskip("torch")
from alphaforge.config import Config
from alphaforge.dashboard import export_dashboard
from alphaforge.data import synthetic
from alphaforge.deep import TrainingConfig
from alphaforge.experiment import study
from alphaforge.research import train


@pytest.fixture
def config():
    return Config(start="2024-01-01T00:00:00Z", end="2024-01-21T00:00:00Z", windows=(6,))


@pytest.fixture
def settings():
    return TrainingConfig(sequence_length=4, hidden_size=8, epochs=1)


def test_holdout_cannot_change_selection(config, settings, tmp_path):
    candles = synthetic(config)
    args = dict(
        validation_start="2024-01-11T00:00:00Z",
        holdout_start="2024-01-16T00:00:00Z",
        kinds=("zero_return", "ridge"),
        thresholds=(0.0, 0.001),
    )
    first = study(candles, config, settings, tmp_path / "a", **args)
    changed = candles.copy()
    mask = changed.timestamp >= pd.Timestamp(args["holdout_start"])
    changed.loc[mask, ["open", "high", "low", "close"]] *= 1.8
    second = study(changed, config, settings, tmp_path / "b", **args)
    assert first["selection"] == second["selection"]
    assert first["selected_portfolio"] == second["selected_portfolio"]
    assert (tmp_path / "a/validation-selection.json").read_bytes() == (
        tmp_path / "b/validation-selection.json"
    ).read_bytes()
    assert pd.Timestamp(first["splits"]["train"]["last_target"]) < pd.Timestamp(
        args["validation_start"]
    )
    pred = pd.read_csv(tmp_path / "a/predictions.csv")
    val = pd.to_datetime(pred.loc[pred.split == "validation", "decision_time"], utc=True)
    assert (val + pd.Timedelta(hours=2)).max() < pd.Timestamp(args["holdout_start"])


def test_cash_when_validation_has_no_edge(config, settings, tmp_path):
    report = study(
        synthetic(config),
        config,
        settings,
        tmp_path / "study",
        validation_start="2024-01-11T00:00:00Z",
        holdout_start="2024-01-16T00:00:00Z",
        kinds=("zero_return",),
        thresholds=(0.0, 0.003),
    )
    assert report["selected_portfolio"] == "cash"
    assert report["selected_holdout_return"] == 0
    assert report["selection"]["zero_return"]["threshold"] == 0.003


def test_dashboard_exports_only_public_summaries(config, settings, tmp_path):
    runs = tmp_path / "runs"
    train(synthetic(config), config, settings, runs / "deep-001", kinds=("zero_return", "ridge"))
    output = tmp_path / "web"
    result = export_dashboard(runs, output)
    assert result["extended_study"] is False
    data = json.loads((output / "dashboard-data.json").read_text())
    assert data["dataset"]["rows"] == 960
    assert data["forecasts"]["BTC-USD"]["ridge"]
    text = (output / "dashboard-data.json").read_text()
    assert str(tmp_path) not in text
    assert "input_sha256" not in text
    assert (output / "index.html").exists()
    assert not any(output.glob("*.pt"))


def test_dashboard_missing_run_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="train the example"):
        export_dashboard(tmp_path / "runs", tmp_path / "web")
