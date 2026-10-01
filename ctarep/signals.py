"""
Core formula of the model: normalized momentum.

The average weekly return over a lookback window is computed first (positions are larger
when momentum intensity is higher). It is then normalised by a measure of realised return
volatility and multiplied by the square root of the number of weeks in the lookback
window, giving the relative position size for a market in a given week. Exposures are
capped at defined thresholds so that most of the portfolio variance does not come from a
few outlier market moves.

Formulas:
    r_bar_i,t = mean of the last L weekly returns r^w_i
    sigma^w_i,t = sigma^d_i,t * sqrt(5)           (daily volatility -> weekly scale)
    m_i,t  = (r_bar_i,t / sigma^w_i,t) * sqrt(L)  <- normalized momentum
    m~_i,t = clip(m_i,t, -cap, +cap)

`m` is the **t-statistic** of the mean weekly return:
    t = r_bar / SE(r_bar) = r_bar / (sigma^w/sqrt(L)) = (r_bar/sigma^w) * sqrt(L) = m
Multiplying numerator and denominator by L also gives
`m = (L-week cumulative return) / (L-week volatility)`, a z-score of the trend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ModelParams
from .volatility import RealizedVolatility, VolatilityPanel


@dataclass(frozen=True)
class SignalPanel:
    """Signal outputs for one parameter set. All: index=weekly grid, columns=market code."""

    mean_weekly_return: pd.DataFrame  # r_bar
    raw_momentum: pd.DataFrame  # m (before cap)
    momentum: pd.DataFrame  # m~ (after cap)
    vol_weekly: pd.DataFrame  # σ_w
    vol_annualized: pd.DataFrame  # σ_ann
    params: ModelParams

    @property
    def label(self) -> str:
        return self.params.label

    @property
    def cap_hit_rate(self) -> pd.Series:
        """Share of weeks the cap binds, per market. Diagnoses how often the cap is hit."""
        cap = self.params.effective_cap
        if cap is None:
            return pd.Series(0.0, index=self.raw_momentum.columns)
        return (self.raw_momentum.abs() >= cap).mean()

    def latest(self) -> pd.DataFrame:
        """Latest signal snapshot (one row per market)."""
        dt = self.momentum.dropna(how="all").index[-1]
        return pd.DataFrame(
            {
                "mean_weekly_return": self.mean_weekly_return.loc[dt],
                "raw_momentum": self.raw_momentum.loc[dt],
                "momentum": self.momentum.loc[dt],
                "vol_annualized": self.vol_annualized.loc[dt],
            }
        ).assign(date=dt)


class NormalizedMomentum:
    """Normalized momentum calculator.

    Parameters
    ----------
    params : ModelParams (momentum lookback / cap / volatility lookback)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from ctarep.config import ModelParams
    >>> # with constant weekly return c and volatility s, m should equal (c/s)*sqrt(L)
    >>> sig = NormalizedMomentum(ModelParams(momentum_lookback=4, momentum_cap=0.0))
    >>> float(np.sqrt(4))
    2.0
    """

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        self.vol_estimator = RealizedVolatility.from_params(params)

    def __repr__(self) -> str:
        return f"<NormalizedMomentum {self.params.label}>"

    # ------------------------------------------------------------------ #

    @property
    def lookback(self) -> int:
        return self.params.momentum_lookback

    @property
    def cap(self) -> float | None:
        return self.params.effective_cap

    # ------------------------------------------------------------------ #

    def compute(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        vol_panel: VolatilityPanel | None = None,
    ) -> SignalPanel:
        """Compute normalized momentum.

        Parameters
        ----------
        weekly_returns : simple returns on the weekly grid (index=weekly grid)
        daily_returns : daily simple returns (for volatility estimation)
        vol_panel : precomputed volatility panel. Pass it in parameter sweeps that share
                    the same volatility lookback to avoid recomputation. If None, it is
                    computed internally.

        Returns
        -------
        SignalPanel
        """
        if vol_panel is None:
            vol_panel = self.vol_estimator.compute(daily_returns)

        # Volatility is computed on the daily grid, then sampled at weekly-grid dates
        vol_at_grid = vol_panel.at(weekly_returns.index)
        sigma_w = vol_at_grid.weekly.reindex(columns=weekly_returns.columns)
        sigma_ann = vol_at_grid.annualized.reindex(columns=weekly_returns.columns)

        L = self.lookback
        # min_periods=L: no signal until the lookback is fully populated (avoids look-ahead / instability)
        r_bar = weekly_returns.rolling(L, min_periods=L).mean()

        raw = r_bar.divide(sigma_w).multiply(np.sqrt(L))
        raw = raw.replace([np.inf, -np.inf], np.nan)

        capped = raw if self.cap is None else raw.clip(lower=-self.cap, upper=self.cap)

        return SignalPanel(
            mean_weekly_return=r_bar,
            raw_momentum=raw,
            momentum=capped,
            vol_weekly=sigma_w,
            vol_annualized=sigma_ann,
            params=self.params,
        )

    # ------------------------------------------------------------------ #

    def compute_shocked(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        shock: float,
        vol_panel: VolatilityPanel | None = None,
    ) -> pd.DataFrame:
        """Normalized momentum if next week's price moves by `shock`.

        Extension for flow estimation. The same shock is assumed **independently** in
        each market, then signals are recomputed. Volatility is held at its pre-shock
        value (the effect of one week's shock on 90-day volatility is second order).

        Returns
        -------
        Capped momentum after the shock at the last weekly date (a Series as a one-row DataFrame).
        """
        if vol_panel is None:
            vol_panel = self.vol_estimator.compute(daily_returns)

        extended = weekly_returns.copy()
        last_dt = extended.index[-1]
        next_dt = last_dt + pd.Timedelta(days=7)
        extended.loc[next_dt] = shock

        vol_at_grid = vol_panel.at(pd.DatetimeIndex([last_dt]))
        sigma_w = vol_at_grid.weekly.iloc[0].reindex(extended.columns)

        L = self.lookback
        r_bar = extended.iloc[-L:].mean()
        raw = (r_bar / sigma_w) * np.sqrt(L)
        raw = raw.replace([np.inf, -np.inf], np.nan)
        capped = raw if self.cap is None else raw.clip(lower=-self.cap, upper=self.cap)
        return capped.to_frame(name=next_dt).T


__all__ = ["NormalizedMomentum", "SignalPanel"]
