# AlphaForge

An end-to-end Python research project for hourly BTC/USD and ETH/USD forecasting:
public historical ingestion, leakage-safe features, chronological evaluation,
PyTorch LSTM and Transformer training, portable models, walk-forward backtesting,
and a local FastAPI prediction service.


[Interactive dashboard](https://suheum-heo.github.io/alphaforge/) · [Experiment protocol](docs/PROTOCOL.md) · [Verified results](VERIFICATION.md)

The dashboard combines a public historical research snapshot with separately timestamped hourly forecasts. You can explore
asset/model comparisons, forecast replay, cost-adjusted walk-forward equity, and a
fresh 2026 holdout without installing Python. It places negative results alongside
successful engineering checks; this project does not claim a profitable strategy.

## Dashboard locally

The checked-in `docs/` snapshot works immediately after cloning:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8766`. After rerunning local experiments, rebuild the snapshot:

```sh
alphaforge dashboard --runs runs --output docs
```

Only selected aggregate market metrics, equity samples, and historical forecasts are
exported. Environment files, machine paths, and full raw datasets stay local. Only the explicitly promoted hourly model bundles are published in the repository.

## Start here

Python 3.11+; verified locally on Python 3.13. From this project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,deep,api]'
pytest -q
```

The existing `.venv` is already installed. Real data and completed experiments are
under `data/` and `runs/`. These large, reproducible artifacts are ignored by Git.
Source: [github.com/suheum-heo/alphaforge](https://github.com/suheum-heo/alphaforge).

## Run the complete workflow

Each command requires a new output path to preserve previous experiments.

```sh
# Public BTC-USD and ETH-USD candles, January–June 2024.
alphaforge ingest --config config/default.toml --output data/history.csv

# Three ordinary baselines, or an offline synthetic smoke test.
alphaforge run --input data/history.csv --output runs/baselines
alphaforge demo --output runs/synthetic

# All five models, each trained separately for BTC and ETH.
alphaforge train --input data/history.csv --output runs/training --epochs 12

# Expanding training windows; repeat for lstm and transformer.
alphaforge backtest --input data/history.csv --output runs/walk-ridge --model ridge
alphaforge backtest --input data/history.csv --output runs/walk-lstm --model lstm --epochs 6
alphaforge backtest --input data/history.csv --output runs/walk-transformer --model transformer --epochs 6

# Offline inference (the CLI selects the model's asset from the input CSV).
alphaforge predict --input data/history.csv --bundle runs/training/models/BTC-USD/lstm

# Local HTTP server and interactive API documentation.
alphaforge serve --bundle runs/training/models/BTC-USD/lstm --port 8000
```

Open http://127.0.0.1:8000/docs while the server is running. Serving binds to localhost.
To use an already trained model immediately, replace `runs/training` above with
`runs/deep-001`. The server loads one asset/model bundle at startup.

## Data and configuration

Edit `config/default.toml`, then pass `--config` to commands. Start is inclusive,
end exclusive; timezone-aware hour boundaries normalize to UTC. Symbols, prediction
horizon, feature windows, split fractions, ridge regularization, and seed are configured
there. Training CLI options include `--epochs`, `--sequence-length`, `--hidden-size`,
and `--patience`; resolved settings are saved with every run and model.

CSV columns: `timestamp,symbol,open,high,low,close,volume`. Timestamp is the candle's
**opening** time. Prices must be finite and positive; volume nonnegative; high/low
must contain open/close. Imported CSVs must contain completed candles. Ingestion
excludes currently open candles automatically; offline CSV commands trust the data source.

The [Coinbase Exchange candle API](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
is public and requires no key. Ingestion uses 299-hour chunks, bounded transient-error
retries, verified TLS, exact-duplicate removal, conflicting-duplicate rejection,
and no gap filling. Downloads write only after all requests succeed; they restart
rather than resume after a failure. `SSL_CERT_FILE` can specify a custom trusted CA
bundle; otherwise certifi supplies trusted roots.

## Leakage and model contract

For a candle opening at `t`, features are available at `decision_time=t+1h`. A horizon
`h` target is `log(close[t+h]/close[t])`, available at `target_time=t+(h+1)h`.
Features include one-hour log return, candle range/body, log volume, cyclic UTC hour,
and momentum, volatility, and volume ratios at 6/24/72 hours by default.

Each asset is reindexed to an hourly grid before shifts or rolling calculations.
A missing candle invalidates every history or target span touching it. No filling,
interpolation, random time splits, or whole-dataset normalization is used. Sequence
windows stay inside one asset and must be contiguous. Inference uses the same causal
features but does not require target labels, so it can predict from the latest candle.

Global timestamp cutoffs split train/validation/test at 70/15/15 by default. Any label
available at or beyond the next split's first decision is purged. Historical feature
context may cross a split boundary because it was already observable. Every split
must contain each configured asset. Deep-model comparisons use identical sequence-end
rows for all five models; Phase 1 tabular runs have additional early rows.

Models:

- Zero-return and training-mean baselines.
- Ridge regression with train-only feature scaling.
- One-layer LSTM with a regression head.
- Two-layer Transformer encoder with learned position embeddings and a regression head.

Neural models use train-only feature and target scaling, deterministic CPU execution,
chronological minibatches, AdamW, gradient clipping, validation early stopping, and
restoration of the best validation checkpoint. The Transformer consumes only past
observations and emits a forecast at the final position. No future mask is needed
within this historical input window. Reproducibility is within the recorded software
and hardware environment, not a promise of identical results across versions/devices.

Validation RMSE selects a model per asset. The untouched test period evaluates each
predeclared candidate; do not retune against it. MAE/RMSE are log-return units. Direction
accuracy treats zero predictions as abstentions, incorrect on nonzero targets;
coverage is reported separately. Constant-series correlations are null.

## Walk-forward execution

`--initial-train-hours 2160 --validation-hours 336 --test-hours 336` are defaults.
Each fold expands the training window, reserves trailing validation for neural early
stopping, purges labels at boundaries, and predicts the next disjoint test block.
Prior out-of-sample observations may enter later training when their labels are known.
The model family is fixed by `--model`, not selected using fold test results.

Execution requires `horizon=1`. Larger horizons are supported for forecasts but
explicitly rejected for execution to avoid silently overlapping positions.
A signal at decision time `d` executes at the observed open at `d+1h` and earns the
open-to-open return through `d+2h`. This deliberately leaves a full hour for publication
and execution. Missing entry/exit prices skip that execution and are counted.

Positive forecasts above `--threshold` (default zero) hold one unlevered long position;
otherwise the asset stays in cash. Consecutive long positions carry across bars/folds.
Gaps and the last bar force liquidation. Defaults charge 10 bps fees plus 5 bps
slippage per unit turnover, with explicit entry/exit charges. Costs are a fixed
fraction of starting bar equity: a simple approximation, not an exchange fill model.
Assets receive equal initial subaccounts which compound independently, avoiding
implicit cost-free portfolio rebalancing. No shorts, leverage, funding, taxes,
order-book impact, liquidity constraints, or actual orders are implemented.

Reports include net total return, drawdown, hourly annualized Sharpe, exposure,
turnover, and a cost-adjusted always-long benchmark over the same executable rows.
Sharpe is descriptive and does not adjust for serial dependence. Historical simulation
and prediction accuracy do not establish profitable live trading.

## Saved models and API

Training saves each candidate under `models/<symbol>/<model>/`:

- `model.json`: architecture, feature order, scaling, configuration, training cutoff,
  and neural validation history; numerical parameters for ordinary baselines.
- `weights.pt`: neural tensor state only, loaded with `weights_only=True` and checked
  against the manifest's SHA-256. No pickled sklearn estimator is loaded.

`GET /health` returns readiness. `GET /model` describes the loaded contract.
`POST /predict` accepts `{"candles": [...]}` with the CSV fields as JSON objects.
Supply at least `max(windows)+sequence_length` contiguous closed candles (96 by default)
for the model's symbol. The limit is 2,000 candles per request. The service rejects
invalid/duplicate timestamps, wrong symbols, unfinished/future candles, recent gaps,
and a latest candle close older than one hour. Use ISO timestamps with offsets.
API inference is for current data; `alphaforge predict` deliberately supports historical
replay without a freshness check. The service has no external authentication, TLS
termination, rate limiting, or order execution; deploy infrastructure separately if needed.

## Artifacts and verification

- `runs/real-001`: verified Phase 1 baseline evaluation.
- `runs/deep-001`: all five models and their validation/test forecasts.
- `runs/walk-{ridge,lstm,transformer}-001`: fold manifests, forecasts, executed trades,
  equity CSVs, and cost-adjusted summary reports.
- `VERIFICATION.md`: results and checks from the completed local run.

Reports record configuration, dependency versions, and input hashes; raw candles are
copied into every experiment. A failed training run may leave a partial directory;
choose a new output path when retrying. Synthetic results are smoke tests, not market evidence.

```sh
ruff check .
ruff format --check .
pytest -q
python -m build
```

CI runs the full suite and package build on Python 3.11/3.13. `requirements-lock.txt`
records the verified local environment; project metadata uses compatible dependency
ranges for other platforms. Optional dependency groups retain a lightweight baseline
installation; the full suite requires `.[dev,deep,api]`.

Source modules separate ingestion, causal features, splits, baselines, neural models,
research workflows, evaluation, API, and CLI. Tests exercise future-data perturbations,
sequence continuity, label purging, serialization, all model families, execution delay,
turnover/fees, fold independence, and API validation.

## Extended study: stronger signals, fresh holdout

```sh
alphaforge ingest --config config/extended.toml --output data/extended.csv
alphaforge study --config config/extended.toml --input data/extended.csv --output runs/study-001 --epochs 6
alphaforge dashboard --output docs
```

The study trains on 2023–2025, validates on January–June 2026, and evaluates frozen
choices on July–September 2026. Threshold candidates are 0/5/10/20/30 bps. Each model's
threshold and the portfolio model are selected solely on validation net return;
`validation-selection.json` is written before holdout predictions. Cash is selected
if no validation candidate has a positive return. Both forecast-label times and
simulated execution exits are purged at the validation/holdout boundary.

The verified dataset has 65,686 candles and 13 missing hours per asset, handled by
excluding affected samples. Validation chose Transformer at 10 bps. It generated no
holdout positions, returning 0%, while the always-long reference returned 56.40%.
The LSTM candidate returned 1.76% with its frozen 10-bps threshold, but was not the
validation-selected portfolio. Its zero-threshold counterpart lost 35.46%. Sparse
trading reduced costs; these results do not establish a reliable forecasting edge.

The snapshot includes each predeclared candidate for transparency, not post-hoc model
selection. Further experimentation must reserve a new holdout period.

## CI and publication

GitHub Actions runs lint, formatting, tests, and a package build on Python 3.11/3.13.
The dashboard deploys through a GitHub Actions Pages workflow. There are no cloud API
keys or live exchange credentials. Run data, models, and virtual environments are
excluded from the repository; a fresh clone includes the public dashboard snapshot,
source code, protocol, and reproducible commands.

## Latest hourly forecasts

The public dashboard now includes an automated hourly forecast section, separate from
all frozen historical experiments. `.github/workflows/hourly.yml` runs at minute 17
of each hour UTC, on relevant main-branch pushes, and through **Run workflow**. It
fetches the latest closed BTC/ETH candles, loads the fixed `models/hourly-v1` Transformer
bundles, and deploys the dashboard plus `latest-forecasts.json` as a Pages artifact.
It does not retrain, place trades, or run a permanent server. The two small tensor
bundles are intentionally version-controlled for reproducible public inference.

The display includes the source candle close, forecast interval, generation time,
model version, training cutoff, and validation cutoff. Predictions expire at their
target time. The browser checks once per minute and updates the expiry state every
15 seconds. The interval before the next scheduled update is explicitly marked
expired; this is hourly forecasting, not a continuously live price feed.

Generation is atomic: both assets must have the latest closed candle and valid
contiguous model history. Missing/stale data or either asset's failure prevents
publication. Failed workflows leave the prior successful site deployment in place;
its forecasts age into the expired state. Inspect the linked workflow status for
failures, or manually rerun the workflow after a transient provider failure. GitHub
Actions schedules can be delayed, and public-repository schedules can be disabled
after 60 days without repository activity. See
[GitHub scheduling behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Run the same generator locally from the repository root:

```sh
python -m alphaforge.hourly
python3 -m http.server 8766 --bind 127.0.0.1 --directory docs
```

Publishing now uses GitHub Actions Pages artifacts instead of `main:/docs` branch
builds. The source-controlled `docs/` still contains the historical snapshot and UI;
`docs/latest-forecasts.json` is generated, ignored by Git, and retained in each
workflow's evidence artifact for seven days. The job has read-only repository access;
only its deployment job receives Pages/OIDC publication permissions. No exchange keys
or repository-write token are needed. To pause automatic updates, disable **Hourly
forecasts and Pages** in Actions. Resume by enabling it and choosing **Run workflow**.

The frozen Transformer family was selected using the extended study's validation
period. Training ended in December 2025 and validation in June 2026. There is no
claim that these forecasts have a reliable trading edge. Updating weights requires
an explicit new model version, provenance checks, and validation before promotion.
