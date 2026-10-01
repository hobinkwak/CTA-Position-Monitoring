"""
Replication quality evaluation: R-squared, 125-combination parameter sweep, regression vs replication comparison.

Paper page 4:
    "We measure the robustness of replication by calculating the coefficient of
     determination (R²) between our replication strategy's returns and that of the
     benchmark. The goal is to maximize the replication to the benchmark, not the return
     performance of our model."

R-squared = corr(R_rep, R_bm)^2. It equals the R-squared of a univariate OLS with intercept and
is scale invariant, so it does not depend on leverage k -> the sweep only needs unlevered
positions (much faster).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import cagr, calmar, mdd, sharpe, sortino, vol

from .config import PARAM_GRID, WEEKS_PER_YEAR, ModelParams, ParameterGrid
from .portfolio import ReplicationModel
from .volatility import RealizedVolatility, VolatilityPanel


# --------------------------------------------------------------------------- #
# Basic statistics
# --------------------------------------------------------------------------- #


def r_squared(a: pd.Series, b: pd.Series) -> float:
    """R-squared (= corr^2) between two return series, on the common period only."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    if joined["a"].std(ddof=1) == 0 or joined["b"].std(ddof=1) == 0:
        return np.nan
    return float(joined["a"].corr(joined["b"]) ** 2)


def tracking_error(rep: pd.Series, bm: pd.Series, periods: int = WEEKS_PER_YEAR) -> float:
    """Annualized tracking error."""
    joined = pd.concat([rep.rename("r"), bm.rename("b")], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    return float((joined["r"] - joined["b"]).std(ddof=1)) * np.sqrt(periods)


@dataclass(frozen=True)
class ReplicationStats:
    """Replication vs benchmark summary."""

    r2: float
    correlation: float
    beta: float
    alpha_annual: float
    tracking_error: float
    n_weeks: int

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "R2": self.r2,
                "Correlation": self.correlation,
                "Beta": self.beta,
                "Alpha (ann)": self.alpha_annual,
                "Tracking Error (ann)": self.tracking_error,
                "Weeks": self.n_weeks,
            }
        )


class ReplicationEvaluator:
    """Replication quality evaluator.

    Parameters
    ----------
    weekly_returns : market weekly return panel
    daily_returns : market daily return panel (for volatility)
    benchmark_returns : benchmark weekly returns
    """

    def __init__(
        self,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self.weekly_returns = weekly_returns
        self.daily_returns = daily_returns
        self.benchmark_returns = benchmark_returns
        self.sector_map = sector_map
        self._vol_cache: dict[tuple[int, str], VolatilityPanel] = {}

    # ------------------------------------------------------------------ #

    def _vol_panel(self, params: ModelParams) -> VolatilityPanel:
        key = (params.volatility_lookback, params.vol_method)
        if key not in self._vol_cache:
            self._vol_cache[key] = RealizedVolatility.from_params(params).compute(
                self.daily_returns
            )
        return self._vol_cache[key]

    # ------------------------------------------------------------------ #
    # Single evaluation
    # ------------------------------------------------------------------ #

    def stats(self, replication_returns: pd.Series) -> ReplicationStats:
        """Summary statistics for one replication return series."""
        joined = pd.concat(
            [replication_returns.rename("rep"), self.benchmark_returns.rename("bm")], axis=1
        ).dropna()
        if len(joined) < 3:
            return ReplicationStats(np.nan, np.nan, np.nan, np.nan, np.nan, len(joined))

        rep, bm = joined["rep"], joined["bm"]
        corr = float(rep.corr(bm))
        var_rep = float(rep.var(ddof=1))
        beta = float(rep.cov(bm) / var_rep) if var_rep > 0 else np.nan
        # Regress the benchmark on the replica; the intercept is excess return the replica cannot explain
        alpha_w = float(bm.mean() - beta * rep.mean()) if np.isfinite(beta) else np.nan
        return ReplicationStats(
            r2=corr**2,
            correlation=corr,
            beta=beta,
            alpha_annual=alpha_w * WEEKS_PER_YEAR,
            tracking_error=tracking_error(rep, bm),
            n_weeks=len(joined),
        )

    def performance_table(self, returns: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Performance metrics table for return columns (weekly data, so freq=52)."""
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        rows = {}
        for col in returns.columns:
            r = returns[col].dropna()
            if len(r) < 10:
                continue
            rows[col] = {
                "CAGR(%)": cagr(r) * 100,
                "Vol(%)": vol(r, freq=WEEKS_PER_YEAR) * 100,
                "Sharpe": sharpe(r, freq=WEEKS_PER_YEAR),
                "Sortino": sortino(r, freq=WEEKS_PER_YEAR),
                "Calmar": calmar(r),
                "MDD(%)": mdd(r) * 100,
                "R2 vs BM": r_squared(r, self.benchmark_returns),
            }
        return pd.DataFrame(rows).T.round(3)

    def rolling_r2(self, replication_returns: pd.Series, window: int = 104) -> pd.Series:
        """Rolling R-squared: how replication quality changes over time."""
        joined = pd.concat(
            [replication_returns.rename("rep"), self.benchmark_returns.rename("bm")], axis=1
        ).dropna()
        roll_corr = joined["rep"].rolling(window, min_periods=max(12, window // 2)).corr(
            joined["bm"]
        )
        return (roll_corr**2).rename(f"rolling_r2_{window}w")

    # ------------------------------------------------------------------ #
    # 125-combination parameter sweep
    # ------------------------------------------------------------------ #

    def parameter_sweep(
        self,
        grid: ParameterGrid = PARAM_GRID,
        vol_method: str = "simple",
        progress: bool = True,
    ) -> pd.DataFrame:
        """R-squared sweep over the 125 combinations (paper page 5).

        R-squared is scale invariant, so it is computed on unlevered positions (leverage fit skipped).

        Returns
        -------
        columns = [momentum_lookback, cap, vol_lookback, label, r2, corr, ann_vol, sharpe]
        """
        combos = grid.combinations()
        rows = []
        for n, (lookback, cap, vol_lb) in enumerate(combos, start=1):
            params = ModelParams(
                momentum_lookback=lookback,
                momentum_cap=cap,
                volatility_lookback=vol_lb,
                vol_method=vol_method,
            )
            model = ReplicationModel(params, sector_map=self.sector_map)
            raw = model.build_positions(
                self.weekly_returns, self.daily_returns, self._vol_panel(params)
            )
            rep_r = ReplicationModel.portfolio_returns(raw, self.weekly_returns)
            r = rep_r.dropna()

            rows.append(
                {
                    "momentum_lookback": lookback,
                    "cap": cap,
                    "vol_lookback": vol_lb,
                    "label": params.label,
                    "r2": r_squared(rep_r, self.benchmark_returns),
                    "corr": float(
                        pd.concat([rep_r, self.benchmark_returns], axis=1).dropna().corr().iloc[0, 1]
                    )
                    if len(r) > 3
                    else np.nan,
                    "ann_vol_unlevered": float(r.std(ddof=1)) * np.sqrt(WEEKS_PER_YEAR)
                    if len(r) > 3
                    else np.nan,
                    "sharpe": sharpe(r, freq=WEEKS_PER_YEAR) if len(r) > 10 else np.nan,
                }
            )
            if progress and n % 25 == 0:
                print(f"    parameter sweep {n}/{len(combos)}")

        return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)

    @staticmethod
    def sweep_sensitivity(
        sweep: pd.DataFrame, vary: str, hold: dict[str, float]
    ) -> pd.DataFrame:
        """Sensitivity varying one parameter with the others held fixed (paper pages 5-7 charts).

        Examples
        --------
        >>> # Paper page 5: hold cap 1.0 and 90-day volatility, vary only the momentum lookback
        >>> # sweep_sensitivity(sweep, 'momentum_lookback', {'cap': 1.0, 'vol_lookback': 90})
        """
        mask = pd.Series(True, index=sweep.index)
        for col, value in hold.items():
            mask &= sweep[col] == value
        return sweep.loc[mask].sort_values(vary)[[vary, "r2", "sharpe"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Regression approach (the conventional method the paper criticizes)
# --------------------------------------------------------------------------- #


class RollingRegression:
    """Rolling univariate regression beta, the comparison target on paper page 10.

    "We also graph betas for a univariate regression where the SG Trend Index is our
     dependent variable and returns on the S&P 500 are our independent variable. We use
     a 20 week rolling window to estimate regression betas."

    The purpose is to show the weaknesses of this method (lag + noise) next to the replica positions.
    """

    def __init__(self, window: int = 20, min_periods: int | None = None) -> None:
        self.window = int(window)
        self.min_periods = min_periods or max(8, int(window * 0.75))

    def __repr__(self) -> str:
        return f"<RollingRegression window={self.window}w>"

    def univariate_beta(self, dependent: pd.Series, independent: pd.Series) -> pd.Series:
        """Univariate rolling beta = Cov(y, x) / Var(x)."""
        joined = pd.concat(
            [dependent.rename("y"), independent.rename("x")], axis=1
        ).dropna()
        if joined.empty:
            return pd.Series(dtype=float)
        cov = joined["y"].rolling(self.window, min_periods=self.min_periods).cov(joined["x"])
        var = joined["x"].rolling(self.window, min_periods=self.min_periods).var(ddof=1)
        return (cov / var.replace(0, np.nan)).rename(f"beta_{independent.name}")

    def betas(self, dependent: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
        """Separate univariate rolling beta for each market."""
        out = {
            col: self.univariate_beta(dependent, factors[col]) for col in factors.columns
        }
        return pd.DataFrame(out)

    def multivariate_beta(
        self, dependent: pd.Series, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """Multivariate rolling OLS beta (with intercept).

        With 16 markets and a 20-week window there are almost no degrees of freedom. This
        itself shows the limit of the regression approach, so it is provided as a diagnostic.
        """
        joined = pd.concat([dependent.rename("__y"), factors], axis=1).dropna()
        if len(joined) < self.window:
            return pd.DataFrame(columns=factors.columns)

        y_all = joined["__y"].to_numpy()
        x_all = joined[factors.columns].to_numpy()
        n, k = x_all.shape
        rows: dict[pd.Timestamp, np.ndarray] = {}

        for end in range(self.window, n + 1):
            sl = slice(end - self.window, end)
            X = np.column_stack([np.ones(self.window), x_all[sl]])
            try:
                coef, *_ = np.linalg.lstsq(X, y_all[sl], rcond=None)
            except np.linalg.LinAlgError:
                continue
            rows[joined.index[end - 1]] = coef[1:]

        return pd.DataFrame(rows, index=factors.columns).T


__all__ = [
    "r_squared",
    "tracking_error",
    "ReplicationStats",
    "ReplicationEvaluator",
    "RollingRegression",
]
