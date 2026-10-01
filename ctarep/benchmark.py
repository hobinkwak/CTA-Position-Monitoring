"""
Benchmark (trend index) wrapper.

Takes a level series, converts it to weekly-grid returns, and provides the realised
volatility used for leverage scaling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BENCHMARK_NAME, WEEKS_PER_YEAR
from .timegrid import WeeklyGrid


class Benchmark:
    """Trend index (or any benchmark level series).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range('2020-01-01', periods=520)
    >>> lvl = pd.Series(100 * np.exp(np.linspace(0, 0.2, 520)), index=idx)
    >>> bm = Benchmark(lvl)
    >>> grid = WeeklyGrid.from_index(idx)
    >>> wr = bm.weekly_returns(grid)
    >>> len(wr) == len(grid)
    True
    """

    def __init__(self, level: pd.Series, name: str = BENCHMARK_NAME) -> None:
        level = pd.Series(level).dropna().sort_index()
        if level.empty:
            raise ValueError("benchmark level series is empty")
        level.index = pd.DatetimeIndex(level.index).tz_localize(None)
        self.level = level
        self.name = name
        self._weekly_cache: dict[int, pd.Series] = {}

    def __repr__(self) -> str:
        return (
            f"<Benchmark {self.name!r} n={len(self.level)} "
            f"{self.level.index[0]:%Y-%m-%d}..{self.level.index[-1]:%Y-%m-%d}>"
        )

    # ------------------------------------------------------------------ #

    @property
    def daily_returns(self) -> pd.Series:
        return self.level.pct_change().dropna()

    def weekly_returns(self, grid: WeeklyGrid) -> pd.Series:
        """Simple returns on the weekly grid."""
        key = id(grid)
        cached = self._weekly_cache.get(key)
        if cached is not None and cached.index.equals(grid.dates):
            return cached
        weekly_level = grid.align_series(self.level)
        rtn = weekly_level.pct_change().rename(self.name)
        self._weekly_cache[key] = rtn
        return rtn

    # ------------------------------------------------------------------ #

    def realized_vol(
        self,
        grid: WeeklyGrid,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        annualize: bool = True,
    ) -> float:
        """Realised volatility, the target for leverage scaling."""
        r = self.weekly_returns(grid).loc[start:end].dropna()
        if len(r) < 2:
            raise ValueError("benchmark weekly returns too short to compute volatility")
        sigma = float(r.std(ddof=1))
        return sigma * np.sqrt(WEEKS_PER_YEAR) if annualize else sigma

    def rolling_vol(self, grid: WeeklyGrid, window: int = 52) -> pd.Series:
        r = self.weekly_returns(grid)
        return r.rolling(window, min_periods=max(8, window // 2)).std(ddof=1) * np.sqrt(
            WEEKS_PER_YEAR
        )

    def cumulative(self, grid: WeeklyGrid) -> pd.Series:
        """Cumulative return on the weekly grid (starts at 1)."""
        return (1 + self.weekly_returns(grid).fillna(0)).cumprod()


__all__ = ["Benchmark"]
