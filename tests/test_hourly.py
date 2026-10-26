import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
from alphaforge.data import HOUR, synthetic
from alphaforge.hourly import load_registry, refresh

REGISTRY = Path(__file__).parents[1] / "models/hourly-v1/registry.json"
NOW = pd.Timestamp("2026-10-26T05:17:00Z")


def loader(config, now):
    return synthetic(config)


def test_fresh_hourly_forecasts_have_provenance(tmp_path):
    output = tmp_path / "latest.json"
    result = refresh(REGISTRY, output, now=NOW, loader=loader)
    assert result == json.loads(output.read_text())
    assert result["forecast_start"] == "2026-10-26T05:00:00+00:00"
    assert result["expires_at"] == "2026-10-26T06:00:00+00:00"
    assert {f["symbol"] for f in result["forecasts"]} == {"BTC-USD", "ETH-USD"}
    assert all(f["model_version"] and f["weights_sha256"] for f in result["forecasts"])
    assert not list(tmp_path.glob(".hourly-*"))


@pytest.mark.parametrize("mode", ["stale", "open", "gap", "failure"])
def test_bad_input_never_replaces_last_good_file(tmp_path, mode):
    output = tmp_path / "latest.json"
    output.write_text("previous successful forecast")

    def bad(config, now):
        if mode == "failure":
            raise OSError("provider unavailable")
        frame = synthetic(config)
        if mode == "stale":
            return frame.iloc[:-1]
        if mode == "open":
            frame["timestamp"] += HOUR
        if mode == "gap":
            frame = frame.loc[frame.timestamp != frame.timestamp.max() - HOUR]
        return frame

    with pytest.raises((ValueError, OSError)):
        refresh(REGISTRY, output, now=NOW, loader=bad)
    assert output.read_text() == "previous successful forecast"


def test_second_asset_failure_is_atomic(tmp_path):
    output = tmp_path / "latest.json"

    def fail_second(config, now):
        if config.symbols == ("ETH-USD",):
            raise OSError("ETH download failed")
        return synthetic(config)

    with pytest.raises(OSError):
        refresh(REGISTRY, output, now=NOW, loader=fail_second)
    assert not output.exists()


def test_registry_metadata_is_verified(tmp_path):
    shutil.copytree(REGISTRY.parent, tmp_path / "registry")
    registry = tmp_path / "registry/registry.json"
    entry = json.loads(registry.read_text())["models"][0]
    model = registry.parent / entry["path"] / "model.json"
    model.write_text(model.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        load_registry(registry)


def test_registry_rejects_path_escape(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    registry["models"][0]["path"] = "../../outside"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="inside registry"):
        load_registry(path)


def test_naive_clock_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        refresh(REGISTRY, tmp_path / "latest.json", now="2026-10-26T05:17:00", loader=loader)
