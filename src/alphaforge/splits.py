"""Global time boundaries with label purging across all symbols."""

from dataclasses import dataclass

import pandas as pd

from .config import Config


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    purged_rows: int


def chronological_split(frame: pd.DataFrame, config: Config) -> Splits:
    times = sorted(frame.decision_time.unique())
    first = int(len(times) * config.train_fraction)
    second = int(len(times) * (config.train_fraction + config.validation_fraction))
    if not 0 < first < second < len(times):
        raise ValueError("Insufficient timestamps for three splits")
    val_start, test_start = times[first], times[second]
    train = frame.loc[(frame.decision_time < val_start) & (frame.target_time < val_start)]
    validation = frame.loc[
        (frame.decision_time >= val_start)
        & (frame.decision_time < test_start)
        & (frame.target_time < test_start)
    ]
    test = frame.loc[frame.decision_time >= test_start]
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty or set(part.symbol) != set(config.symbols):
            raise ValueError(f"{name} must contain samples for every configured symbol")
    return Splits(
        train.copy(),
        validation.copy(),
        test.copy(),
        val_start,
        test_start,
        len(frame) - len(train) - len(validation) - len(test),
    )
