"""Small deterministic CPU sequence models and portable, non-pickle model bundles."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import Config
from .data import HOUR
from .features import build_features
from .models import baselines


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 24
    hidden_size: int = 32
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 0.001
    patience: int = 3

    def __post_init__(self):
        for name in ("sequence_length", "hidden_size", "epochs", "batch_size", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")


def sequences(frame, names, length):
    """Return contiguous sequences and their last-row metadata; never mix assets."""
    if length < 1:
        raise ValueError("Sequence length must be positive")
    xs, rows = [], []
    for _, group in frame.groupby("symbol", sort=True):
        group = group.sort_values("decision_time")
        values = group[names].to_numpy(dtype=np.float32)
        times = group.decision_time
        for end in range(length - 1, len(group)):
            start = end - length + 1
            if times.iloc[end] - times.iloc[start] != (length - 1) * HOUR:
                continue
            xs.append(values[start : end + 1])
            rows.append(group.iloc[end])
    if not xs:
        raise ValueError("Insufficient contiguous history for sequences")
    return np.stack(xs), pd.DataFrame(rows).reset_index(drop=True)


class Network(nn.Module):
    def __init__(self, kind, features, settings):
        super().__init__()
        self.kind = kind
        width = settings.hidden_size
        if kind == "lstm":
            self.encoder = nn.LSTM(features, width, batch_first=True)
        elif kind == "transformer":
            self.project = nn.Linear(features, width)
            self.position = nn.Parameter(torch.zeros(1, settings.sequence_length, width))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        else:
            raise ValueError(f"Unknown neural architecture: {kind}")
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        if self.kind == "lstm":
            x, _ = self.encoder(x)
        else:
            # All positions are historical at the last-position prediction time.
            x = self.encoder(self.project(x) + self.position[:, : x.shape[1]])
        return self.head(x[:, -1]).squeeze(-1)


class TorchRegressor:
    def __init__(self, kind, settings, seed=42):
        self.kind, self.settings, self.seed = kind, settings, seed
        self.history = []

    def fit(self, x, y, validation_x, validation_y):
        if not len(x) or not len(validation_x):
            raise ValueError("Training and validation sequences are required")
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
        self.y_mean = float(np.mean(y))
        self.y_scale = max(float(np.std(y)), 1e-8)
        tx = self._tensor(x)
        ty = torch.tensor((np.asarray(y) - self.y_mean) / self.y_scale, dtype=torch.float32)
        vx = self._tensor(validation_x)
        vy = torch.tensor(
            (np.asarray(validation_y) - self.y_mean) / self.y_scale, dtype=torch.float32
        )
        self.network = Network(self.kind, x.shape[-1], self.settings)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.settings.learning_rate)
        loss_fn = nn.MSELoss()
        best, stale, best_state = float("inf"), 0, None
        self.history = []
        for epoch in range(self.settings.epochs):
            self.network.train()
            # Deterministic chronological batches; sequences already encode their history.
            for start in range(0, len(tx), self.settings.batch_size):
                optimizer.zero_grad()
                loss = loss_fn(
                    self.network(tx[start : start + self.settings.batch_size]),
                    ty[start : start + self.settings.batch_size],
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
            self.network.eval()
            with torch.no_grad():
                losses = []
                for start in range(0, len(vx), self.settings.batch_size):
                    pred = self.network(vx[start : start + self.settings.batch_size])
                    losses.extend(
                        (pred - vy[start : start + self.settings.batch_size]).square().tolist()
                    )
                validation_loss = float(np.mean(losses))
            self.history.append({"epoch": epoch + 1, "validation_mse_scaled": validation_loss})
            if validation_loss < best:
                best, stale = validation_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.settings.patience:
                    break
        if best_state is None:
            raise ValueError("Training produced no finite checkpoint")
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def _tensor(self, x):
        scaled = self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return torch.tensor(scaled, dtype=torch.float32)

    def predict(self, x):
        self.network.eval()
        with torch.no_grad():
            values = []
            for start in range(0, len(x), self.settings.batch_size):
                values.extend(
                    self.network(self._tensor(x[start : start + self.settings.batch_size]))
                    .numpy()
                    .tolist()
                )
        return np.asarray(values) * self.y_scale + self.y_mean


def fit_model(kind, settings, config, x, y, vx, vy):
    if kind in ("lstm", "transformer"):
        return TorchRegressor(kind, settings, config.seed).fit(x, y, vx, vy)
    model = baselines(config.ridge_alpha)[kind]
    return model.fit(x[:, -1], y)


def predict_model(model, x):
    return model.predict(x if isinstance(model, TorchRegressor) else x[:, -1])


def save_bundle(path, model, kind, symbol, names, config, settings, trained_through):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "symbol": symbol,
        "features": names,
        "config": config.as_dict(),
        "training": asdict(settings),
        "trained_through": str(trained_through),
    }
    if isinstance(model, TorchRegressor):
        metadata.update(
            mean=model.scaler.mean_.tolist(),
            scale=model.scaler.scale_.tolist(),
            y_mean=model.y_mean,
            y_scale=model.y_scale,
            history=model.history,
        )
        weights = path / "weights.pt"
        torch.save(model.network.state_dict(), weights)
        metadata["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    elif kind == "ridge":
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        metadata.update(
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            coef=ridge.coef_.tolist(),
            intercept=float(ridge.intercept_),
        )
    else:
        metadata["constant"] = float(np.ravel(model.constant_)[0])
    (path / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


class Predictor:
    """Loads a local bundle and computes forecasts without needing future labels."""

    def __init__(self, path):
        path = Path(path)
        self.metadata = m = json.loads((path / "model.json").read_text())
        if m["schema_version"] != 1:
            raise ValueError("Unsupported model schema")
        self.config = Config(**m["config"])
        self.settings = TrainingConfig(**m["training"])
        self.network = None
        if m["kind"] in ("lstm", "transformer"):
            weights = path / "weights.pt"
            if hashlib.sha256(weights.read_bytes()).hexdigest() != m["weights_sha256"]:
                raise ValueError("Model checksum mismatch")
            self.network = Network(m["kind"], len(m["features"]), self.settings)
            self.network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.network.eval()

    def predict_array(self, x):
        m = self.metadata
        if "constant" in m:
            return np.full(len(x), m["constant"])
        scaled = (x - np.asarray(m["mean"])) / np.asarray(m["scale"])
        if self.network is None:
            return scaled[:, -1] @ np.asarray(m["coef"]) + m["intercept"]
        with torch.no_grad():
            return (
                self.network(torch.tensor(scaled, dtype=torch.float32)).numpy() * m["y_scale"]
                + m["y_mean"]
            )

    def predict_candles(self, candles):
        if set(candles.symbol) != {self.metadata["symbol"]}:
            raise ValueError("Input must contain only the model symbol")
        frame, names = build_features(candles, self.config, include_targets=False)
        if names != self.metadata["features"]:
            raise ValueError("Feature contract mismatch")
        x, rows = sequences(frame, names, self.settings.sequence_length)
        row = rows.iloc[-1]
        if row.timestamp != pd.to_datetime(candles.timestamp, utc=True).max():
            raise ValueError("Latest candle lacks contiguous feature/sequence history")
        prediction = float(self.predict_array(x[-1:])[0])
        if not np.isfinite(prediction):
            raise ValueError("Nonfinite prediction")
        return {
            "symbol": row.symbol,
            "model": self.metadata["kind"],
            "decision_time": row.decision_time.isoformat(),
            "target_time": row.target_time.isoformat(),
            "predicted_log_return": prediction,
        }
