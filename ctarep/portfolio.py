"""
Replication portfolio: naive risk parity plus leverage scaling.

Each market gets an equal weight once adjusted for volatility, with no adjustment for
correlations among markets (naive risk parity). Leverage is scaled to match the
benchmark volatility using the in-sample results of the replication portfolio.

Formulas:
    raw_i,t = m~_i,t / sigma^ann_i,t         (volatility adjustment = naive risk parity)
    w_i,t   = k * raw_i,t                    (k = global leverage scalar)
    R_t     = sum_i w_i,t-1 * r^w_i,t        (filled at t-1 close -> return of week t)

`w_i,t` is the **notional exposure ratio per $1 of benchmark**. Per-market target
volatility constants are absorbed into k, so the formula above is sufficient.

Note: k is a scalar, so it has **no effect on R-squared** (R-squared is scale invariant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import WEEKS_PER_YEAR, ModelParams
from .signals import NormalizedMomentum, SignalPanel
from .volatility import VolatilityPanel


@dataclass(frozen=True)
class FitResult:
    """Leverage scaling result."""

    leverage: float | pd.Series  # k (scalar if in_sample, time series if expanding)
    method: str
    target_vol: float  # benchmark annualised volatility
    unlevered_vol: float  # unlevered replication portfolio annualised volatility
    n_weeks: int

    def summary(self) -> str:
        k = self.leverage
        k_txt = f"{k:.3f}" if np.isscalar(k) else f"{float(pd.Series(k).iloc[-1]):.3f} (latest)"
        return (
            f"leverage k={k_txt} [{self.method}]  "
            f"target_vol={self.target_vol:.2%}  unlevered_vol={self.unlevered_vol:.2%}  "
            f"n={self.n_weeks}w"
        )


def sector_scaling(
    positions: pd.DataFrame, sector_map: dict[str, str]
) -> pd.DataFrame:
    """Equalise the risk budget across sectors.

    Each market weight is divided by the **number of markets active in its sector at
    that date**, so total risk per sector is equal regardless of market count.

    In the base universe (4/4/4/4) every sector has 4 markets, so everything is divided
    uniformly by 1/4, which is absorbed into the global constant (leverage k). **The
    base results therefore do not change.** It only matters when the universe is
    expanded asymmetrically.
    """
    sectors = pd.Series(sector_map).reindex(positions.columns)
    active = positions.notna()
    scaled = positions.copy()

    for sector in sectors.dropna().unique():
        cols = sectors.index[sectors == sector]
        cols = [c for c in cols if c in positions.columns]
        if not cols:
            continue
        n_active = active[cols].sum(axis=1).replace(0, np.nan)
        scaled[cols] = positions[cols].div(n_active, axis=0)
    return scaled


class ReplicationModel:
    """Replication model for a single parameter set (L, cap, V).

    Usage::

        model = ReplicationModel(params)
        model.fit(weekly_returns, daily_returns, benchmark_weekly_returns)
        model.positions   # notional weights (per $1 of benchmark)
        model.returns     # replication portfolio weekly returns

    Notes
    -----
    Calling only `build_positions()` without `fit()` gives unlevered (k=1) positions.
    When only R-squared is needed (e.g. parameter sweeps), leverage is irrelevant, so
    this path is faster.
    """

    def __init__(self, params: ModelParams, sector_map: dict[str, str] | None = None) -> None:
        self.params = params
        self.signal_engine = NormalizedMomentum(params)
        #: used only when sector_risk_parity=True
        self.sector_map = sector_map

        # fit outputs
        self.signals: SignalPanel | None = None
        self.raw_positions: pd.DataFrame | None = None  # before k
        self.positions: pd.DataFrame | None = None  # after k = final notional weights
        self.returns: pd.Series | None = None  # replication weekly returns
        self.fit_result: FitResult | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.positions is not None else "unfitted"
        return f"<ReplicationModel {self.params.label} [{state}]>"

    @property
    def label(self) -> str:
        return self.params.label

    # ------------------------------------------------------------------ #
    # Position construction
    # ------------------------------------------------------------------ #

    def build_positions(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        vol_panel: VolatilityPanel | None = None,
    ) -> pd.DataFrame:
        """Build unlevered positions raw_i,t = m~_i,t / sigma^ann_i,t."""
        signals = self.signal_engine.compute(weekly_returns, daily_returns, vol_panel)
        self.signals = signals

        raw = signals.momentum.divide(signals.vol_annualized)
        raw = raw.replace([np.inf, -np.inf], np.nan)

        if self.params.sector_risk_parity:
            if self.sector_map is None:
                raise ValueError(
                    "sector_risk_parity=True requires ReplicationModel(sector_map=...)."
                )
            raw = sector_scaling(raw, self.sector_map)

        self.raw_positions = raw
        return raw

    # ------------------------------------------------------------------ #
    # Returns
    # ------------------------------------------------------------------ #

    @staticmethod
    def portfolio_returns(positions: pd.DataFrame, weekly_returns: pd.DataFrame) -> pd.Series:
        """R_t = sum_i w_i,t-1 * r^w_i,t.

        Positions are taken at the t-1 close and earn the week-t return. No look-ahead.
        Markets with NaN weights (signal not yet mature) count as 0.
        """
        aligned = positions.reindex(columns=weekly_returns.columns)
        contrib = aligned.shift(1).fillna(0.0) * weekly_returns.fillna(0.0)
        gross = contrib.sum(axis=1)
        # Early period with no signal in any market yet is NaN
        active = aligned.shift(1).notna().any(axis=1)
        return gross.where(active).rename("replication")

    @staticmethod
    def contributions(positions: pd.DataFrame, weekly_returns: pd.DataFrame) -> pd.DataFrame:
        """Return contribution per market."""
        aligned = positions.reindex(columns=weekly_returns.columns)
        return aligned.shift(1).fillna(0.0) * weekly_returns.fillna(0.0)

    # ------------------------------------------------------------------ #
    # Leverage scaling
    # ------------------------------------------------------------------ #

    def fit(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        method: str = "in_sample",
        expanding_min_weeks: int = 104,
        vol_panel: VolatilityPanel | None = None,
    ) -> "ReplicationModel":
        """Build positions and set leverage to match benchmark volatility.

        Parameters
        ----------
        method : 'in_sample' - a single k from the full sample (as in the source model)
                 'expanding' - k re-estimated weekly on an expanding window (no look-ahead)
        """
        raw = self.build_positions(weekly_returns, daily_returns, vol_panel)
        unlevered = self.portfolio_returns(raw, weekly_returns)

        common = unlevered.dropna().index.intersection(benchmark_returns.dropna().index)
        if len(common) < 8:
            raise ValueError(
                f"{self.label}: only {len(common)} common weeks of replication/benchmark "
                "observations. Check the data period."
            )

        rep_r = unlevered.loc[common]
        bm_r = benchmark_returns.loc[common]

        target_vol = float(bm_r.std(ddof=1)) * np.sqrt(WEEKS_PER_YEAR)
        unlevered_vol = float(rep_r.std(ddof=1)) * np.sqrt(WEEKS_PER_YEAR)

        if method == "in_sample":
            k: float | pd.Series = (
                target_vol / unlevered_vol if unlevered_vol > 0 else 1.0
            )
        elif method == "expanding":
            k = self._expanding_leverage(rep_r, bm_r, expanding_min_weeks)
            k = k.reindex(raw.index).ffill()
        else:
            raise ValueError("method must be 'in_sample' or 'expanding'")

        self.fit_result = FitResult(
            leverage=k,
            method=method,
            target_vol=target_vol,
            unlevered_vol=unlevered_vol,
            n_weeks=len(common),
        )

        self.positions = raw.multiply(k, axis=0) if isinstance(k, pd.Series) else raw * k
        self.returns = self.portfolio_returns(self.positions, weekly_returns).rename(
            f"rep_{self.label}"
        )
        return self

    @staticmethod
    def _expanding_leverage(
        rep_returns: pd.Series, bm_returns: pd.Series, min_weeks: int
    ) -> pd.Series:
        """Expanding-window k. k at t uses only information up to t (and applies at t+1)."""
        rep_vol = rep_returns.expanding(min_periods=min_weeks).std(ddof=1)
        bm_vol = bm_returns.expanding(min_periods=min_weeks).std(ddof=1)
        k = (bm_vol / rep_vol).replace([np.inf, -np.inf], np.nan)
        return k.shift(1).ffill().bfill()

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def require_fitted(self) -> None:
        if self.positions is None:
            raise RuntimeError(f"{self.label}: call fit() first.")

    def latest_positions(self) -> pd.Series:
        """Notional weight per market at the latest weekly date."""
        self.require_fitted()
        valid = self.positions.dropna(how="all")
        return valid.iloc[-1].rename(valid.index[-1])

    def gross_exposure(self) -> pd.Series:
        """Sum |w_i|: total gross notional."""
        self.require_fitted()
        return self.positions.abs().sum(axis=1)

    def net_exposure(self) -> pd.Series:
        """Sum w_i: net notional (sectors are mixed, so interpretation is limited)."""
        self.require_fitted()
        return self.positions.sum(axis=1)

    def turnover(self) -> pd.Series:
        """Weekly notional turnover, sum |delta w_i|."""
        self.require_fitted()
        return self.positions.diff().abs().sum(axis=1)


__all__ = ["ReplicationModel", "FitResult"]
