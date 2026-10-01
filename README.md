# CTA Position Monitoring

A dashboard that estimates how trend-following CTAs are positioned across 16 futures markets and what they would have to trade if prices move.

**Live demo: https://hobinkwak.github.io/CTA-Position-Monitoring/**

The positioning model follows Kestner (2020), ["Replicating CTA Positioning: An Improved Method"](https://ssrn.com/abstract=3674828): volatility-normalised momentum, capped and inverse-volatility weighted, from an ensemble of three parameter sets (momentum lookback in weeks / cap / volatility lookback in days: 16/1/60, 32/1/60, 52/1/60). The ensemble is scaled to match the SG Trend Index volatility. Everything outside the paper (scenario engine, trade pressure, flip prices, CFTC comparison) is an extension and is labelled as such in the dashboard.

## Dashboard tabs

| Tab | Content |
|---|---|
| Main | Position overview (net notional, largest long/short, 4-week change, percentile and flow tags per market), price chart with 1-week and 4-week flip prices |
| Exposure | Current position (notional or risk basis), exposure history by market and asset class, momentum signals, realised volatility |
| Flow | Cumulative net trades by market and by period (percentile, factor decomposition) |
| Scenario | Required trades and projected positions for five driver scenarios (Strong down to Strong up), certain trades, trade pressure, position age and flip proximity, flip triggers |
| Model monitoring | Replication R-squared, tracking error, rolling fit, turnover, ensemble disagreement, parameter sensitivity (125 combinations) |
| Data & method | Data checks, scenario assumptions, universe, model settings |
| CFTC positioning | COT crowding percentiles and model versus COT comparison (needs the optional COT file) |

The dashboard has a light and dark theme and an asset-class filter in the sidebar.

## Quick start

```bash
pip install -r requirements.txt

# build output/dashboard.html and open it
python main.py

# faster / offline variants
python main.py --no-sweep                 # skip the 125-combination sweep
python main.py --standalone --no-open     # embed plotly.js (about +5 MB)
python main.py --check                    # show which input files were found
```

Python 3.10+ with pandas, numpy, scipy, plotly and jinja2. Tests: `python tests/test_model.py`, `tests/test_scenario.py`, `tests/test_pressure.py`.

Useful options: `--aum 500` (AUM assumption in $BN, default 300), `--horizon 4` (shock horizon in weeks), `--driver TY` (default driver index), `--scaling expanding` (no look-ahead leverage), `--theme dark`, `--csv` (also export result tables), `--universe recommended` (24 markets instead of the 16 in the paper).

## Data

The repository ships the input files under `data/` (history up to 2026-09-30). To refresh or replace them, keep the same layout: wide tables with first column `date` and one column per market code (`ES, Z, VG, NK, TY, G, RX, JB, EC, BP, JY, AD, CL, GC, HG, S`). CSV, parquet, feather and xlsx are accepted.

| File | Required | Content |
|---|---|---|
| `data/raw/futures_rolled.csv` | one of these two | Ratio-adjusted (back-adjusted) continuous futures prices. Returns and volatility must come from these, never from raw generic prices |
| `data/raw/futures_c1.csv` + `futures_c2.csv` | one of these two (shipped) | Raw front-month and second-month prices. The code builds the rolled series using an exchange expiry calendar (optional `roll_dates.csv`). `futures_c1.csv` is also used for the price chart and contract counts |
| `data/raw/benchmark.csv` | yes (shipped) | SG Trend Index level (column `SG_TREND` or first column) |
| `data/raw/fx.csv` | optional | FX rates for USD conversion (`EURUSD, GBPUSD, USDJPY, AUDUSD`) |
| `data/external/cftc_positions.csv` | optional (shipped) | Weekly net positions of the CTA-proxy bucket per market code (Managed Money for commodities, Leveraged Funds for financials). Without it the CFTC tab is skipped |

Data rules the model relies on: weekend forward-fills removed, prices must be positive, stale series are flagged by the data checks. See the Data & method tab after a run.

## Layout

```
main.py                 entry point (CLI and `run()` API)
ctarep/                 model, scenario engine and dashboard package
  config.py             all parameters (paper grid, run settings)
  universe.py           the 16 markets (single source of truth for codes)
  rolling.py, expiry.py continuous futures construction
  signals.py, volatility.py, portfolio.py, ensemble.py   replication model
  exposure.py, scenario.py, pressure.py                   exposure, scenarios, trade pressure
  cftc.py               COT comparison (reads the local file only)
  pipeline.py           orchestration
  dashboard/            builder, figures, theme, HTML template
tests/
```

## Notes

- Positions are estimates of an index-level replication, not reported CTA holdings.
- The colour palette is defined in `ctarep/dashboard/theme.py` (`series`, `pos_pole`, `neg_pole`); edit it to re-skin the dashboard.
- Licensed under the MIT License (see `LICENSE`).
