"""
Realised volatility estimation.

The volatility lookback is given in **days** (20/30/60/90/180). Momentum is in **weeks**,
so daily volatility must be converted to the weekly scale for the units of the signal
`m = r_bar/sigma_w * sqrt(L)` to match.

    sigma_w   = sigma_d * sqrt(5)      (DAYS_PER_WEEK)
    sigma_ann = sigma_d * sqrt(252)    (TRADING_DAYS_PER_YEAR)

The base model uses plain realised return volatility with a day-count parameter (no
EWMA); EWMA and other options are provided for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS_PER_WEEK, TRADING_DAYS_PER_YEAR, ModelParams


class RealizedVolatility:
    """Compute realised volatility from daily returns.

    Parameters
    ----------
    lookback : lookback in days (20/30/60/90/180)
    method : 'simple' (rolling std, ddof=1) | 'ewma'
    min_periods : minimum number of valid values. None means 60% of lookback.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range('2020-01-01', periods=300)
    >>> r = pd.DataFrame({'A': np.zeros(300)}, index=idx)
    >>> rv = RealizedVolatility(lookback=90)
    >>> bool(rv.daily(r).iloc[-1, 0] == 0.0)
    True
    """

    #: Supported estimation methods
    METHODS: tuple[str, ...] = ("simple", "rv", "ewma", "ewma_rv")

    def __init__(
        self,
        lookback: int = 90,
        method: str = "simple",
        min_periods: int | None = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.lookback = int(lookback)
        self.method = method
        self.min_periods = (
            int(min_periods) if min_periods is not None else max(2, int(round(lookback * 0.6)))
        )

    # -- factory ----------------------------------------------------------- #

    @classmethod
    def from_params(cls, params: ModelParams) -> "RealizedVolatility":
        return cls(
            lookback=params.volatility_lookback,
            method=params.vol_method,
            min_periods=params.effective_vol_min_periods,
        )

    def __repr__(self) -> str:
        return f"<RealizedVolatility lookback={self.lookback}d method={self.method}>"

    # -- computation ------------------------------------------------------- #

    def daily(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """Daily volatility sigma_d.

        Definitions by method
        ---------------------
        ``simple``   sample standard deviation (mean removed):  sigma = std(r, ddof=1)
        ``rv``       realised volatility, **mean not removed**: sigma = sqrt(mean(r^2))
                     Standard definition in the RV literature. Nearly identical to
                     ``simple`` at high frequency where mean return is near 0, but
                     ``rv`` is larger in strongly trending periods.
        ``ewma``     exponentially weighted standard deviation (mean removed)
        ``ewma_rv``  RiskMetrics form: sigma^2_t = lambda*sigma^2_{t-1} + (1-lambda)*r^2_{t-1}  (no mean removal)
        ``har``      HAR-RV expanding-window forecast (handled by `HARVolatility`)
        """
        if self.method == "simple":
            vol = daily_returns.rolling(self.lookback, min_periods=self.min_periods).std(ddof=1)

        elif self.method == "rv":
            var = (daily_returns**2).rolling(
                self.lookback, min_periods=self.min_periods
            ).mean()
            vol = np.sqrt(var)

        elif self.method == "ewma":
            # halflife = lookback/2 gives an effective sample of roughly lookback
            vol = daily_returns.ewm(
                halflife=self.lookback / 2.0, min_periods=self.min_periods
            ).std(bias=False)

        elif self.method == "ewma_rv":
            var = (daily_returns**2).ewm(
                halflife=self.lookback / 2.0, min_periods=self.min_periods
            ).mean()
            vol = np.sqrt(var)

        else:  # pragma: no cover - filtered out in from_params
            raise ValueError(f"unsupported method {self.method!r}")

        return vol.replace(0.0, np.nan)

    def weekly(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """Weekly-scale volatility sigma_w = sigma_d * sqrt(5)."""
        return self.daily(daily_returns) * np.sqrt(DAYS_PER_WEEK)

    def annualized(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """Annualised volatility sigma_ann = sigma_d * sqrt(252)."""
        return self.daily(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

    def compute(self, daily_returns: pd.DataFrame) -> "VolatilityPanel":
        """Daily, weekly and annualised in one call."""
        sigma_d = self.daily(daily_returns)
        return VolatilityPanel(
            daily=sigma_d,
            weekly=sigma_d * np.sqrt(DAYS_PER_WEEK),
            annualized=sigma_d * np.sqrt(TRADING_DAYS_PER_YEAR),
            lookback=self.lookback,
            method=self.method,
        )


class VolatilityPanel:
    """Container holding the three scales sigma_d / sigma_w / sigma_ann together."""

    __slots__ = ("daily", "weekly", "annualized", "lookback", "method")

    def __init__(
        self,
        daily: pd.DataFrame,
        weekly: pd.DataFrame,
        annualized: pd.DataFrame,
        lookback: int,
        method: str,
    ) -> None:
        self.daily = daily
        self.weekly = weekly
        self.annualized = annualized
        self.lookback = lookback
        self.method = method

    def __repr__(self) -> str:
        return (
            f"<VolatilityPanel lookback={self.lookback}d method={self.method} "
            f"shape={self.daily.shape}>"
        )

    def at(self, dates: pd.DatetimeIndex) -> "VolatilityPanel":
        """Sample at given dates (usually the weekly grid); previous value if the date is missing."""
        return VolatilityPanel(
            daily=self.daily.reindex(dates, method="ffill"),
            weekly=self.weekly.reindex(dates, method="ffill"),
            annualized=self.annualized.reindex(dates, method="ffill"),
            lookback=self.lookback,
            method=self.method,
        )


class HARVolatility:
    """HAR-RV volatility forecaster (Corsi 2009), **fitted on an expanding window**.

    Regression (log transform keeps the forecast positive)::

        log sigma_{t+1:t+h} = b0 + b_d*log RV^d_t + b_w*log RV^w_t + b_m*log RV^m_t + e

    * ``RV^d`` = 1 day, ``RV^w`` = 5-day average, ``RV^m`` = 22-day average (daily/weekly/monthly components)
    * The target is the **next h days** of realised volatility, matched to the position
      holding period (1 week = 5 days)

    Two measures are supported:
        ``rv``     sigma = sqrt(mean(r^2))     no mean removal (RV standard)
        ``simple`` sigma = std(r, ddof=1)      mean removed

    Look-ahead protection
    ---------------------
    When fitting at time t, only targets **already realised** at t are used, so the last
    usable observation is ``t - h``. The forecast looks beyond t+1 using t's regressors.
    The model is refit every ``refit_every`` days and the last coefficients are reused in
    between (for speed; coefficients rarely change much in a day, so the impact is negligible).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.bdate_range('2015-01-01', periods=1200)
    >>> r = pd.DataFrame({'A': rng.normal(0, 0.01, 1200)}, index=idx)
    >>> har = HARVolatility(horizon=5, min_train=500, refit_every=250)
    >>> s = har.daily(r)['A'].dropna()
    >>> bool(s.gt(0).all())
    True
    """

    METHODS: tuple[str, ...] = ("rv", "simple")

    def __init__(
        self,
        horizon: int = DAYS_PER_WEEK,
        measure: str = "rv",
        measure_window: int = 1,
        lags: tuple[int, int, int] = (1, 5, 22),
        min_train: int = 756,
        refit_every: int = 63,
    ) -> None:
        if measure not in self.METHODS:
            raise ValueError(f"measure must be one of {self.METHODS}")
        self.horizon = int(horizon)
        self.measure = measure
        self.measure_window = int(measure_window)
        self.lags = tuple(int(x) for x in lags)
        self.min_train = int(min_train)
        self.refit_every = int(refit_every)
        #: interface compatibility (same attribute name as RealizedVolatility)
        self.lookback = max(self.lags)
        self.method = f"har_{measure}"

    def __repr__(self) -> str:
        return (
            f"<HARVolatility h={self.horizon} measure={self.measure} "
            f"lags={self.lags} min_train={self.min_train}>"
        )

    # ------------------------------------------------------------------ #

    def _instantaneous_variance(self, returns: pd.Series) -> pd.Series:
        """Daily variance measure. r^2 as is when measure_window=1."""
        if self.measure == "rv":
            v = returns**2
            if self.measure_window > 1:
                v = v.rolling(self.measure_window, min_periods=1).mean()
        else:
            v = returns.rolling(
                max(2, self.measure_window), min_periods=2
            ).std(ddof=1) ** 2
        return v

    def daily(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """HAR-forecast daily volatility sigma_d per market."""
        out = {}
        for code in daily_returns.columns:
            out[code] = self._forecast_one(daily_returns[code].dropna())
        return pd.DataFrame(out).reindex(daily_returns.index)

    def _forecast_one(self, r: pd.Series) -> pd.Series:
        n = len(r)
        h, (l_d, l_w, l_m) = self.horizon, self.lags
        if n < self.min_train + l_m + h + 10:
            return pd.Series(np.nan, index=r.index)

        var = self._instantaneous_variance(r)
        eps = 1e-12

        # Regressors: daily/weekly/monthly components (all information up to t)
        x_d = var.rolling(l_d, min_periods=1).mean()
        x_w = var.rolling(l_w, min_periods=max(2, l_w // 2)).mean()
        x_m = var.rolling(l_m, min_periods=max(2, l_m // 2)).mean()
        # Target: mean variance over the next h days (t+1 .. t+h), not yet realised at t
        y = var.shift(-1).rolling(h, min_periods=h).mean().shift(-(h - 1))

        X = np.column_stack([
            np.ones(n),
            np.log(x_d.to_numpy() + eps),
            np.log(x_w.to_numpy() + eps),
            np.log(x_m.to_numpy() + eps),
        ])
        Y = np.log(y.to_numpy() + eps)
        valid_x = np.isfinite(X).all(axis=1)
        valid_y = np.isfinite(Y)

        pred = np.full(n, np.nan)
        beta = None
        for t in range(self.min_train, n):
            # Update coefficients only at refit dates
            if beta is None or (t - self.min_train) % self.refit_every == 0:
                # Look-ahead protection: use only observations whose target is realised at t
                usable = valid_x & valid_y
                usable[t - h + 1:] = False
                if usable.sum() >= 100:
                    try:
                        beta = np.linalg.lstsq(X[usable], Y[usable], rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            if beta is not None and valid_x[t]:
                pred[t] = X[t] @ beta

        sigma = pd.Series(np.sqrt(np.exp(pred)), index=r.index)
        return sigma.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------ #

    def weekly(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        return self.daily(daily_returns) * np.sqrt(DAYS_PER_WEEK)

    def annualized(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        return self.daily(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

    def compute(self, daily_returns: pd.DataFrame) -> "VolatilityPanel":
        sigma_d = self.daily(daily_returns)
        return VolatilityPanel(
            daily=sigma_d,
            weekly=sigma_d * np.sqrt(DAYS_PER_WEEK),
            annualized=sigma_d * np.sqrt(TRADING_DAYS_PER_YEAR),
            lookback=self.lookback,
            method=self.method,
        )


__all__ = ["RealizedVolatility", "HARVolatility", "VolatilityPanel"]
