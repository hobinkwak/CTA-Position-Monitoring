"""Small performance-statistics helpers (return series in, scalar out).

Weekly data should be annualised with ``freq=52``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _series(rtn) -> pd.Series:
    return rtn.squeeze() if isinstance(rtn, pd.DataFrame) else rtn


def cumrtn(rtn) -> float:
    """Total compounded return."""
    return float((_series(rtn) + 1).prod() - 1)


def drawdown(rtn) -> pd.Series:
    """Drawdown series from simple returns."""
    price = (_series(rtn) + 1).cumprod()
    return price / price.cummax() - 1


def mdd(rtn) -> float:
    """Maximum drawdown (negative number)."""
    return float(drawdown(rtn).min())


def cagr(rtn) -> float:
    """Compound annual growth rate based on calendar days."""
    r = _series(rtn)
    return (cumrtn(r) + 1) ** (365 / (r.index[-1] - r.index[0]).days) - 1


def vol(rtn, freq: int = 252) -> float:
    """Annualised volatility."""
    return float(_series(rtn).std() * np.sqrt(freq))


def sharpe(rtn, rf: float = 0.0, freq: int = 252) -> float:
    """Annualised Sharpe ratio (arithmetic mean, rf as annual rate)."""
    r = _series(rtn)
    return (r.mean() * freq - rf) / vol(r, freq)


def sortino(rtn, rf: float = 0.0, freq: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation of negative returns)."""
    r = _series(rtn)
    downside = r[r < 0].std() * np.sqrt(freq)
    return (r.mean() * freq - rf) / downside


def calmar(rtn) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    return cagr(rtn) / abs(mdd(rtn))
