"""
Project-wide settings.

All model parameters live here; no magic numbers are hard-coded in other modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Project directory layout."""

    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    output: Path = ROOT / "output"

    @property
    def dashboard_html(self) -> Path:
        return self.output / "dashboard.html"

    def ensure(self) -> "Paths":
        """Create output directories (data/raw is user-owned and left untouched)."""
        for p in (self.data, self.raw, self.processed, self.output):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


# --------------------------------------------------------------------------- #
# Annualization constants
# --------------------------------------------------------------------------- #

TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52
DAYS_PER_WEEK = 5  # converts daily volatility to weekly volatility (sqrt(5))


# --------------------------------------------------------------------------- #
# Parameter grid (125 = 5 x 5 x 5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParameterGrid:
    """Parameter list from the paper (page 4)."""

    momentum_lookbacks: Sequence[int] = (4, 8, 16, 32, 52)  # weeks
    momentum_caps: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0)  # normalized momentum
    volatility_lookbacks: Sequence[int] = (20, 30, 60, 90, 180)  # days

    def combinations(self) -> list[tuple[int, float, int]]:
        """All 125 (momentum_lookback, cap, vol_lookback) combinations."""
        return [
            (lb, cap, vl)
            for lb in self.momentum_lookbacks
            for cap in self.momentum_caps
            for vl in self.volatility_lookbacks
        ]

    @staticmethod
    def label(lookback: int, cap: float, vol_lookback: int) -> str:
        """Paper notation: "8/1/90" = 8 weeks / cap 1.0 / 90-day volatility."""
        cap_txt = f"{cap:g}"
        return f"{lookback}/{cap_txt}/{vol_lookback}"


PARAM_GRID = ParameterGrid()


# --------------------------------------------------------------------------- #
# Model parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelParams:
    """Parameter set of a single replication model."""

    momentum_lookback: int = 32  # weeks
    momentum_cap: float = 1.0  # normalized momentum (t-stat) cap
    volatility_lookback: int = 90  # days
    vol_method: str = "simple"  # 'simple' | 'ewma'
    vol_min_periods: int | None = None  # None -> 60% of volatility_lookback

    #: Whether to fix the risk budget equally across sectors.
    #:
    #: False (default, as in the paper): every market gets the same risk weight. The
    #:   paper universe is balanced 4/4/4/4, so sector risk is equal automatically.
    #: True: each market weight is divided by the number of active markets in its
    #:   sector. Needed when the universe is extended: adding 10 commodities alone
    #:   would raise commodity sector risk 2.5x and distort the sector mix.
    sector_risk_parity: bool = False

    def __post_init__(self) -> None:
        if self.momentum_lookback < 1:
            raise ValueError("momentum_lookback must be >= 1 week")
        if self.volatility_lookback < 2:
            raise ValueError("volatility_lookback must be >= 2 days")
        if self.momentum_cap < 0:
            raise ValueError("momentum_cap must be >= 0 (0 means 'no cap')")
        if self.vol_method not in ("simple", "ewma"):
            raise ValueError("vol_method must be 'simple' or 'ewma'")

    @property
    def label(self) -> str:
        return ParameterGrid.label(
            self.momentum_lookback, self.momentum_cap, self.volatility_lookback
        )

    @property
    def effective_cap(self) -> float | None:
        """A cap of 0.0 is interpreted as 'no cap'."""
        if CAP_ZERO_MEANS_NO_CAP and self.momentum_cap == 0.0:
            return None
        return self.momentum_cap

    @property
    def effective_vol_min_periods(self) -> int:
        if self.vol_min_periods is not None:
            return self.vol_min_periods
        return max(2, int(round(self.volatility_lookback * 0.6)))

    def with_(self, **kwargs) -> "ModelParams":
        return replace(self, **kwargs)


#: Treat a cap of 0.0 as "unlimited". Clipping literally at 0 would zero every
#: position and leave R-squared undefined. The paper's sensitivity chart shows all
#: five cap values, so "unlimited" is the consistent reading.
CAP_ZERO_MEANS_NO_CAP = True

#: Paper's final choice (page 7: "momentum lookbacks around 32 weeks, caps around 1.0,
#: volatility lookbacks of 90 days"). See ENSEMBLE_VOL_LOOKBACK for the volatility lookback.
BEST_SINGLE_PARAMS = ModelParams(momentum_lookback=32, momentum_cap=1.0, volatility_lookback=60)

#: Paper's ensemble (page 8): 16 / 32 / 52 weeks, cap 1.0, 90-day volatility -> R-squared > 75%
ENSEMBLE_LOOKBACKS: Sequence[int] = (16, 32, 52)
ENSEMBLE_CAP = 1.0

#: Volatility lookback. The paper uses 90 days, but on 2000-2026 data 60 days is better.
#:
#: Comparison over 28 combinations (session filter applied):
#:     lookback  20      30      60      90(paper) 180     252
#:     R-sq      0.635   0.651   0.650   0.629     0.589   0.564
#: 30-60 is a flat plateau and R-squared drops at 90. Why 60:
#:   * R-sq 0.6500, 0.16pp below the maximum of 0.6516 (noise)
#:   * Best in the recent regime: R-sq (2023+) 0.565, CAGR gap (2023+) -6.49%
#:   * Smallest full-period CAGR gap (+0.11%)
#:   * Healthier positions than the 30-day group: max single-market notional 3.3 vs 5.8
#:     (-43%), weekly turnover 0.749 vs 0.904 (-17%), position persistence 0.957 vs 0.942
#: To reproduce the paper exactly, set this back to 90 (costs 2.1pp of R-squared).
ENSEMBLE_VOL_LOOKBACK = 60


def ensemble_params() -> list[ModelParams]:
    """The three members of the paper's final ensemble."""
    return [
        ModelParams(
            momentum_lookback=lb,
            momentum_cap=ENSEMBLE_CAP,
            volatility_lookback=ENSEMBLE_VOL_LOOKBACK,
        )
        for lb in ENSEMBLE_LOOKBACKS
    ]


# --------------------------------------------------------------------------- #
# Run settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    """Pipeline run options."""

    # Leverage scaling: 'in_sample' (as in the paper) | 'expanding' (true OOS)
    leverage_scaling: str = "in_sample"
    #: Minimum observed weeks for expanding-window scaling
    expanding_min_weeks: int = 104
    #: Whether to rescale the ensemble to benchmark volatility
    rescale_ensemble: bool = True

    #: Assumed total trend-following AUM. The paper cites $300BN (WSJ).
    aum_usd: float = 300e9

    #: Rolling window for the regression comparison (paper page 10: "20 week rolling window")
    regression_window_weeks: int = 20
    #: Rolling R-squared window
    rolling_r2_window_weeks: int = 104

    #: Analysis period (None means the full data)
    start: str | None = None
    end: str | None = None

    #: Whether to run the 125-combination parameter sweep (can be disabled if slow)
    run_parameter_sweep: bool = True

    #: Whether to run the CFTC COT cross-check (needs data/external/cftc_positions.csv).
    #: It is an independent validation that does not depend on the AUM assumption.
    run_cftc_validation: bool = True

    #: Flow sensitivity scenarios (price shock, %); extension beyond the paper
    flow_shocks: Sequence[float] = (-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05)

    # ---- Scenario positioning (core dashboard feature) --------------------- #
    #: Weeks over which the shock is realized (for the trade-flow question)
    scenario_horizon_weeks: int = 1

    #: **Horizon for flip triggers only (weeks).**
    #:
    #: Two opposite traps:
    #:   * Too short: momentum is a 16-52 week average, so one week moves that
    #:     average by only 1/L, and a flip needs an unrealistically large shock.
    #:   * Too long: the zero returns of the synthetic stretch fill the volatility
    #:     window and lower sigma, so `m = rbar/sigma_w*sqrt(L)` grows, whether the
    #:     cap (+-1.0) binds changes, and the flip point itself moves. If H*5 >
    #:     lookback, sigma = 0 and all positions collapse to 0.
    #:
    #: Measured sweep (2026-08-05, independent mode, L_min = 16 weeks, 60-day vol lookback).
    #: Reach probability uses the empirical standardized-return distribution, not a
    #: normal assumption:
    #:     H                  1      2      3      4      5      6
    #:     median |sigma|     3.43   2.49   2.04   1.52   1.28   1.02
    #:     # |sigma| > 3      11     5      0      0      0      0
    #:     median reach prob  0.7%   2.6%   5.5%   12.7%  19.6%  28.8%
    #:     # prob >= 10%      1      5      6      9      11     13
    #:     sigma re-est. bias 0.03   0.23   2.09   2.86   3.83   4.50 pp
    #:     sum |flat pos.|    +7%    +10%   +13%   +25%   +25%   +52%   (vs 318% now)
    #:
    #: The valid range is 3-5 weeks (no |sigma| > 3, bias under 4pp, inflation within 25%).
    #: H=4, the middle, is fixed: 9 of 16 markets have reach probability of 10% or more,
    #: so a watch list actually forms. (H=1 has a median reach probability of 0.7%, an
    #: artifact of the short horizon rather than market information, so it is excluded.)
    trigger_horizon_weeks: int = 4
    #: Horizon candidates for the dashboard trigger table. 4 weeks is the middle of the
    #: valid range, so it is fixed to a single value. To widen, stay within 3-5.
    trigger_horizon_choices: tuple[int, ...] = (4,)
    #: Default driver market for index propagation mode
    scenario_driver: str = "ES"
    #: Driver candidates shown in the dashboard dropdown.
    #: None = the whole universe (default); any market can be a driver.
    #: Each driver adds 5 charts x 2 themes plus reaction curve/trigger recomputation,
    #: so for a slow run or a large file pass a tuple subset, e.g. ("ES", "TY", "CL")
    scenario_driver_choices: Sequence[str] | None = None
    #: Beta estimation window for index propagation (weeks)
    scenario_beta_window_weeks: int = 156
    #: Whether to also update realized volatility after the shock
    scenario_update_vol: bool = True

    # ---- Trade pressure (extension beyond the paper) ----------------------- #
    #: Own-sigma range and grid size for the independent-mode trade curve. Tail mass
    #: outside the range is pushed onto both end points (density is not discarded).
    pressure_sigma_range: float = 5.0
    pressure_n_points: int = 201
    #: Buy probability at or above this value (at or below 1 - value) is 'certain' /
    #: 'leaning'. In between is 'neutral'.
    pressure_sure_prob: float = 0.95
    pressure_lean_prob: float = 0.75
    #: Whether to subtract each market's mean (sample-period drift) from standardized
    #: returns before fitting the KDE
    pressure_demean: bool = True
    #: Minimum observed weeks to use a per-market KDE. Below it, a normal distribution is used.
    pressure_min_obs: int = 104

    #: Price history length (years) shown by the 'Price and flip price' card
    price_chart_years: int = 2
    #: Flip price horizons (weeks) drawn as dashed lines on the same card. 1 week =
    #: next week's close, 4 weeks = close 4 weeks ahead. The 1-week move needed is
    #: often large, so that line is frequently outside the plotted range.
    price_flip_horizons: tuple[int, ...] = (1, 4)

    # ---- Position overview (regime chips) ---------------------------------- #
    #: First chip: percentile of current notional within its own history over this many
    #: years (0% = max short, 100% = max long). 5 years is about 260 weeks. 3 years holds
    #: only 1-2 trend cycles so the percentile swings, and 10+ years mixes volatility
    #: regimes and blurs the 'recent' baseline. It also matches the default CFTC
    #: crowding chart window (5 years).
    overview_pctile_years: int = 5
    #: Weeks over which change is measured. A market is tagged 'expanding/shrinking'
    #: only when its |N-week change| is at or above the `regime_flow_quantile`
    #: percentile of its own last `overview_pctile_years` years (per-market threshold).
    #: A fixed pp threshold would never light the chip for structurally small-notional
    #: CL/HG and would always light it for JB.
    regime_flow_weeks: int = 4
    regime_flow_quantile: float = 0.75

    def __post_init__(self) -> None:
        if self.leverage_scaling not in ("in_sample", "expanding"):
            raise ValueError("leverage_scaling must be 'in_sample' or 'expanding'")
        if self.scenario_horizon_weeks < 1:
            raise ValueError("scenario_horizon_weeks must be >= 1")


RUN_CONFIG = RunConfig()


# --------------------------------------------------------------------------- #
# Data settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataConfig:
    """File naming convention under data/ and roll rules."""

    rolled_file: str = "futures_rolled"  # extension is searched by DataLoader
    front_file: str = "futures_c1"
    second_file: str = "futures_c2"
    roll_dates_file: str = "roll_dates"
    benchmark_file: str = "benchmark"
    fx_file: str = "fx"

    benchmark_column: str = "SG_TREND"

    #: Continuous-futures adjustment. 'relative' = ratio adjustment (the only choice that fits this model)
    roll_method: str = "relative"
    roll_backward: bool = True
    #: Days before expiry to roll
    roll_days_before_expiry: int = 5

    #: Roll-gap diagnostic threshold (z-score of daily return)
    roll_jump_zscore: float = 8.0
    #: A value unchanged for at least this many business days is flagged as missing data
    max_stale_days: int = 10

    #: Whether to keep only each market's exchange sessions when computing daily returns.
    #:
    #: Raw data is forward-filled over exchange holidays as well as weekends, which
    #: gives zero returns. Those zeros understate realized volatility, and since the
    #: number of holidays differs by market (JB 8.8% / NK 8.0% vs VG 1.7%) the relative
    #: weights are distorted. If True, only per-market sessions are kept using
    #: `pandas_market_calendars`. Measured effect: sigma up 3.2% for NK, 3.0% for JB,
    #: 1.8% for S; CME contracts are almost unchanged.
    use_exchange_sessions: bool = True

    #: Quality gate: a market is excluded automatically if at least this share of days
    #: equal the previous day. Catches a mis-mapped ticker. A stale series understates
    #: volatility and inflates risk-parity weights, which would contaminate the whole model.
    max_stale_ratio: float = 0.25
    #: Markets with fewer observations than this are excluded
    min_observations: int = 260

    def __post_init__(self) -> None:
        if self.roll_method not in ("relative", "absolute"):
            raise ValueError("roll_method must be 'relative' or 'absolute'")


DATA_CONFIG = DataConfig()


#: Benchmark = SG Trend Index.
BENCHMARK_TICKER = "NEIXCTAT Index"
BENCHMARK_NAME = "SG Trend Index"


__all__ = [
    "ROOT",
    "PATHS",
    "Paths",
    "TRADING_DAYS_PER_YEAR",
    "WEEKS_PER_YEAR",
    "DAYS_PER_WEEK",
    "ParameterGrid",
    "PARAM_GRID",
    "ModelParams",
    "CAP_ZERO_MEANS_NO_CAP",
    "BEST_SINGLE_PARAMS",
    "ENSEMBLE_LOOKBACKS",
    "ENSEMBLE_CAP",
    "ENSEMBLE_VOL_LOOKBACK",
    "ensemble_params",
    "RunConfig",
    "RUN_CONFIG",
    "DataConfig",
    "DATA_CONFIG",
    "BENCHMARK_TICKER",
    "BENCHMARK_NAME",
]
