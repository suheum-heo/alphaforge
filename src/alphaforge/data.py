"""Public Coinbase ingestion, strict OHLCV validation and deterministic demo data."""

import json
import logging
import os
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import numpy as np
import pandas as pd

from .config import Config, utc

LOG = logging.getLogger(__name__)
COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
HOUR = pd.Timedelta(hours=1)


def request_candles(symbol, start, end, *, attempts=4, sleep=time.sleep):
    """Bounded retries for transient failures; timeout and conservative request pacing."""
    params = urlencode({"granularity": 3600, "start": start.isoformat(), "end": end.isoformat()})
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?{params}"
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "AlphaForge/0.1"})
            context = ssl.create_default_context(
                cafile=os.environ.get("SSL_CERT_FILE") or certifi.where()
            )
            with urlopen(request, timeout=30, context=context) as response:
                payload = json.load(response)
            if not isinstance(payload, list):
                raise ValueError("Unexpected Coinbase response")
            sleep(0.15)
            return payload
        except (HTTPError, URLError, TimeoutError) as exc:
            if isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            if isinstance(exc, HTTPError) and exc.code != 429 and exc.code < 500:
                raise
            if attempt == attempts - 1:
                raise
            LOG.warning("Transient data request failure; retry %s", attempt + 1)
            sleep(min(2**attempt, 16))
    raise ValueError("attempts must be positive")


def validate_candles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(COLUMNS).issubset(frame):
        raise ValueError("Empty candles or missing required columns")
    frame = frame[COLUMNS].copy()
    # Reject naive CSV timestamps instead of silently assigning a timezone.
    if any(pd.Timestamp(v).tzinfo is None for v in frame.timestamp):
        raise ValueError("Candle timestamps must include a timezone")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.timestamp.isna().any() or (frame.timestamp != frame.timestamp.dt.floor("h")).any():
        raise ValueError("Invalid or non-hourly timestamp")
    if frame.symbol.isna().any() or not frame.symbol.isin(["BTC-USD", "ETH-USD"]).all():
        raise ValueError("Invalid symbol")
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("OHLCV must be finite")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame.volume < 0).any():
        raise ValueError("Volume must be nonnegative")
    if (
        (frame.low > frame[["open", "close"]].min(axis=1))
        | (frame.high < frame[["open", "close"]].max(axis=1))
    ).any():
        raise ValueError("Inconsistent OHLC range")
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate candle keys")
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def ingest(config: Config, fetch=request_candles, now=None) -> pd.DataFrame:
    """Fetch [start, end), excluding the currently open candle. Gaps remain explicit."""
    start, end = utc(config.start), utc(config.end)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    end = min(end, now.tz_convert("UTC").floor("h"))
    if start >= end:
        raise ValueError("No closed candles in requested interval")
    records = []
    for symbol in config.symbols:
        cursor = start
        while cursor < end:
            # 299-hour span also fits providers that count both endpoints.
            stop = min(cursor + 299 * HOUR, end)
            for row in fetch(symbol, cursor, stop):
                if not isinstance(row, (list, tuple)) or len(row) != 6:
                    raise ValueError("Malformed candle response")
                stamp, low, high, opening, close, volume = row
                stamp = pd.to_datetime(stamp, unit="s", utc=True)
                if cursor <= stamp < stop:
                    records.append([stamp, symbol, opening, high, low, close, volume])
            cursor = stop
    frame = pd.DataFrame(records, columns=COLUMNS)
    # Exact repeats are harmless; conflicting duplicate candles fail validation.
    frame = validate_candles(frame.drop_duplicates())
    if set(frame.symbol) != set(config.symbols):
        raise ValueError("Provider returned no candles for a requested symbol")
    return frame


def synthetic(config: Config) -> pd.DataFrame:
    """Synthetic smoke-test fixture; never a source of market performance evidence."""
    rng = np.random.default_rng(config.seed)
    timestamps = pd.date_range(utc(config.start), utc(config.end), freq="h", inclusive="left")
    records = []
    for symbol in config.symbols:
        previous = 30000.0 if symbol == "BTC-USD" else 2000.0
        for stamp in timestamps:
            close = previous * np.exp(rng.normal(0, 0.006))
            high = max(previous, close) * (1 + rng.uniform(0, 0.003))
            low = min(previous, close) * (1 - rng.uniform(0, 0.003))
            records.append([stamp, symbol, previous, high, low, close, rng.lognormal(5, 1)])
            previous = close
    return validate_candles(pd.DataFrame(records, columns=COLUMNS))


def quality(frame: pd.DataFrame) -> dict:
    report = {}
    for symbol, group in frame.groupby("symbol"):
        expected = int((group.timestamp.max() - group.timestamp.min()) / HOUR) + 1
        report[symbol] = {
            "rows": len(group),
            "missing_hours_inside_range": expected - len(group),
            "start": group.timestamp.min().isoformat(),
            "last_candle_start": group.timestamp.max().isoformat(),
        }
    return report
