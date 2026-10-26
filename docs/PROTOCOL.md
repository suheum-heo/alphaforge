# Extended experiment protocol

This protocol is specified before the extended holdout is evaluated.

- Source: public Coinbase BTC-USD and ETH-USD hourly OHLCV.
- History: January 1, 2023 through September 30, 2026 (UTC).
- Training: before January 1, 2026; purge labels at the boundary.
- Validation: January–June 2026; purge execution exits at the holdout boundary.
- Holdout: July–September 2026, evaluated after choices are frozen.
- Features: unchanged 6/24/72-hour causal features; 24-hour sequences.
- Candidates: zero return, training mean, Ridge, LSTM, Transformer.
- Neural fit: hidden size 32, at most 6 epochs, patience 3, seed 42; CPU.
- Threshold grid: 0, 5, 10, 20, 30 bps of predicted hourly log return.
- Costs: 10 bps fee plus 5 bps slippage per turnover, one-hour execution delay.
- Selection: per-family threshold maximizing validation net portfolio return;
  ties favor lower turnover, then higher threshold. Portfolio model chosen by
  validation net return; cash wins if all candidates are nonpositive.
- Holdout comparisons: each family's frozen threshold, its predeclared zero-threshold
  counterpart, and an always-long reference over the same dates. No test-based tuning.
- Output: protocol, validation selection written before holdout inference, forecasts,
  model bundles, per-family trades/equity, version and input-hash provenance.

The previous January–June 2024 results remain an exploratory benchmark. The new
holdout is fresh to this project's evaluation, not proof of prospective live returns.
Repeated experimentation on it would turn it into development data; future research
must use a newly reserved period. Cost and threshold magnitudes are research settings,
not calibrated exchange execution estimates.
