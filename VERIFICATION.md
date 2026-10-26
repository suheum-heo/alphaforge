# Local verification — October 26, 2026

Implemented and executed the full local research/inference workflow on Python 3.13.

## Checks

- 58 tests passed, including LSTM/Transformer training, model reloads, API validation,
  chronological purging, future-data invariance, execution timing, and transaction costs.
- Ruff lint and formatting passed; dependency consistency check passed.
- Wheel and source package built successfully.
- Real localhost HTTP GET /health and POST /predict returned 200 using 168 fresh BTC candles.
- HTTP prediction matched offline inference within 1e-8. Server shut down after the check.
- One upstream Starlette/httpx deprecation warning; no test failures.

## Historical model comparison

8,736 Coinbase candles covering January–June 2024. Neural runs used up to 12 epochs
with validation early stopping. All five candidates share sequence-end evaluation rows.

| Asset | Validation selection | Zero test RMSE | Mean | Ridge | LSTM | Transformer |
|---|---|---:|---:|---:|---:|---:|
| BTC-USD | transformer | 0.003906 | 0.003912 | 0.003937 | 0.003953 | 0.003999 |
| ETH-USD | training_mean | 0.004598 | 0.004602 | 0.004626 | 0.004591 | 0.004660 |

Validation-selected BTC Transformer did not outperform zero return on test RMSE.
ETH LSTM slightly improved test RMSE but was not selected by validation; this is not
a basis for selecting it after looking at test data.

## Walk-forward execution

Six expanding-window folds per family, both assets, one-hour execution delay,
10 bps fees plus 5 bps slippage per turnover, long/flat, equal initial asset subaccounts.
Neural folds used up to six epochs with validation early stopping.

| Model | Net return | Max drawdown | Always-long reference | Executions |
|---|---:|---:|---:|---:|
| ridge | -37.60% | -41.44% | 8.27% | 3548 |
| lstm | -12.10% | -20.31% | 8.27% | 3548 |
| transformer | -36.51% | -40.96% | 8.27% | 3548 |

All tested strategies lost money after the configured costs. These are research
benchmarks, not evidence of profitable live trading. No orders were placed.

## Evidence

- `runs/deep-001/report.json` and saved model directories
- `runs/walk-*/report.json`, `trades.csv`, and `equity.csv`
- `runs/api-smoke.json`
- `requirements-lock.txt`

Remaining operational work is deployment-specific: external hosting, authentication,
monitoring, retraining schedules, and exchange order routing are not enabled. The local
API and all requested research components are implemented and tested.

## Public dashboard and extended study

- Static interactive dashboard in `docs/`, generated from stored experiment artifacts.
- Asset/period/model filters, historical prediction replay, walk-forward charts, and holdout comparison.
- Narrow-screen layout and interactive controls checked in a real browser.
- Extended dataset: 65,686 candles, 2023-01-01 through 2026-09-30; 13 missing hours per asset.
- Training: 52,014 sequence-end samples; validation: 8,482; holdout: 4,412.
- Threshold choices are written before holdout inference; a future-data perturbation test verifies
  that changing holdout candles cannot change model/threshold selection.
- Validation chose Transformer at 10 bps. Holdout turnover was zero, net return 0%.
- LSTM at its validation-selected 10 bps returned 1.76%, versus -35.46% at zero threshold;
  it was not the selected portfolio. Always-long reference: 56.40%.
- Evidence: `runs/study-001/` (local), `docs/dashboard-data.json` (public summary), and `docs/PROTOCOL.md`.

## Hourly inference pipeline

- 67 Python tests and 3 JavaScript freshness tests passed locally.
- Fixed versioned BTC/ETH Transformer checkpoints, verified metadata and tensor hashes.
- Atomic current-hour generation; missing, open, stale, or discontinuous data blocks publication.
- Browser expiry logic tested at the exact target-time boundary; no stale forecast is labeled current.
- Scheduled Pages artifact deployment preserves the last successful site when generation fails.

## Prospective forecast tracking

- 73 Python tests passed; four JavaScript tests passed.
- Ruff lint/format checks and package build passed.
- Added checks for immutable first predictions, expiry rejection, closed-target-only
  scoring, matched zero baseline calculations, missing-candle retries, per-version
  metrics, provider failure preservation, and separate record/score phases.
- Production history starts empty on the `forecast-history` branch; no retrospective
  results are presented as prospective observations.
