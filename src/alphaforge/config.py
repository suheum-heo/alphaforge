"""Validated experiment configuration. All timestamps must explicitly include a timezone."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp != timestamp.floor("h"):
        raise ValueError("Timestamp must be aligned to an hour")
    return timestamp


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("BTC-USD", "ETH-USD")
    start: str = "2024-01-01T00:00:00Z"
    end: str = "2024-07-01T00:00:00Z"
    horizon: int = 1
    windows: tuple[int, ...] = (6, 24, 72)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    ridge_alpha: float = 10.0
    seed: int = 42

    def __post_init__(self):
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Symbols must be nonempty and unique")
        if any(s not in {"BTC-USD", "ETH-USD"} for s in self.symbols):
            raise ValueError("Phase 1 supports BTC-USD and ETH-USD")
        if utc(self.start) >= utc(self.end):
            raise ValueError("start must precede end")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not self.windows or any(type(w) is not int or w < 2 for w in self.windows):
            raise ValueError("windows must contain integers >= 2")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be unique")
        if not (
            0 < self.train_fraction < 1
            and 0 < self.validation_fraction < 1
            and self.train_fraction + self.validation_fraction < 1
        ):
            raise ValueError("Split fractions must leave nonempty train/validation/test periods")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as handle:
            values = tomllib.load(handle)
        for key in ("symbols", "windows"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def as_dict(self):
        return asdict(self)
