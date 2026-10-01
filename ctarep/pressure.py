"""
Trade pressure: answers "is this market leaning buy or sell" as a probability.

Certain trades (sign unchanged across all five scenarios) use a strict criterion, so in
independent mode almost every market is "uncertain": the trade direction follows the
market's own price direction. What a portfolio manager wants to know is not "wherever it
goes" but "which way is it mostly leaning", so the binary test is turned into a probability.

Method
------
1. Get the trade size T(z) from the independent-mode response curve (z = the market's
   own price move, in sigma units).
2. Fit the distribution f(z) per market as a **KDE of historical standardised weekly
   returns** (r_t / sigma_{t-1}, sample drift removed; normal fallback if too few
   observations).
3. Buy probability P = integral of f(z)*1[T(z)>0] dz, expected trade E = integral of f(z)*T(z) dz.

An N(0,1) based probability is also reported for comparison. Tiers were identical in all
16 markets and probabilities differed by at most 6pp, because the central mass of the
distribution drives direction.

Note: this is an extension beyond the replication model. It is not a forecast but the
**mechanical response of the current position weighted by the price distribution**, and
is labelled as such on the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import RUN_CONFIG, RunConfig
from .scenario import ScenarioEngine

#: Tier keys, ordered sell -> buy.
TIERS = ("sell_sure", "sell_lean", "mixed", "buy_lean", "buy_sure")


@dataclass
class TradePressure:
    """Per-market trade pressure table. Trade sizes are delta position (per $1 of benchmark); $ = x AUM."""

    #: index=market code. See `TradePressureEstimator.COLUMNS` for columns.
    table: pd.DataFrame
    aum: float
    horizon_weeks: int
    sigma_range: float
    demeaned: bool
    sure_prob: float
    lean_prob: float

    def tier_counts(self) -> dict[str, int]:
        return {t: int((self.table["tier"] == t).sum()) for t in TIERS}

    def summary(self) -> str:
        t = self.table
        lines = ["Trade pressure (weighted by own price distribution, $BN)"]
        for code, r in t.iterrows():
            lines.append(
                f"  {code:<3} P(buy) {r['p_buy']:>5.0%}  expected {r['e_trade_pct'] * self.aum / 1e9:>+7.1f}"
                f"  {r['tier']}"
            )
        return "\n".join(lines)


class TradePressureEstimator:
    """Weights the `ScenarioEngine` independent-mode response curve by the historical distribution."""

    COLUMNS = (
        "p_buy", "p_buy_norm", "e_trade_pct", "sd_pct", "flat_pct",
        "flip_z", "flip_pct", "flip_above", "tier", "method", "n_obs",
    )

    def __init__(self, engine: ScenarioEngine, cfg: RunConfig = RUN_CONFIG) -> None:
        self.engine = engine
        self.cfg = cfg

    # ------------------------------------------------------------------ #

    def _trade_curve(self) -> pd.DataFrame:
        """index=z grid, columns=market code, values=delta position."""
        cfg = self.cfg
        curve = self.engine.response_curve(
            mode="independent", driver=None,
            sigma_range=cfg.pressure_sigma_range, n_points=cfg.pressure_n_points,
        ).drop(columns=["driver_move_pct"])
        cur = self.engine.current.reindex(curve.columns)
        return curve.sub(cur, axis=1)

    @staticmethod
    def _weights_kde(z: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Probability mass per grid point. Out-of-range tails are folded into the end points (sum = 1)."""
        kde = stats.gaussian_kde(z)
        dz = grid[1] - grid[0]
        w = kde(grid) * dz
        w[0] += kde.integrate_box_1d(-np.inf, grid[0] - dz / 2)
        w[-1] += kde.integrate_box_1d(grid[-1] + dz / 2, np.inf)
        # Grid density x dz only approximates cell mass, so the sum drifts slightly from 1
        return w / w.sum()

    @staticmethod
    def _weights_norm(grid: np.ndarray) -> np.ndarray:
        dz = grid[1] - grid[0]
        edges = np.concatenate([[-np.inf], grid[:-1] + dz / 2, [np.inf]])
        return np.diff(stats.norm.cdf(edges))

    @staticmethod
    def _nearest_flip(grid: np.ndarray, trade: np.ndarray) -> tuple[float, str | None]:
        """Trade sign flip nearest to z=0, and the trade direction above it."""
        pos = trade > 0   # compare 'is buy' so grid points exactly at 0 are not missed
        idx = np.where(pos[:-1] != pos[1:])[0]
        if not len(idx):
            return float("nan"), None
        zs = grid[idx] - trade[idx] * (grid[idx + 1] - grid[idx]) / (trade[idx + 1] - trade[idx])
        k = int(np.argmin(np.abs(zs)))
        above = "buy" if trade[idx[k] + 1] > 0 else "sell"
        return float(zs[k]), above

    def _tier(self, p: float) -> str:
        sure, lean = self.cfg.pressure_sure_prob, self.cfg.pressure_lean_prob
        if p >= sure:
            return "buy_sure"
        if p >= lean:
            return "buy_lean"
        if p <= 1 - sure:
            return "sell_sure"
        if p <= 1 - lean:
            return "sell_lean"
        return "mixed"

    # ------------------------------------------------------------------ #

    def run(self) -> TradePressure:
        cfg, eng = self.cfg, self.engine
        trade = self._trade_curve()
        grid = trade.index.to_numpy(dtype=float)
        zpanel = eng.standardized_panel()
        w_norm = self._weights_norm(grid)

        rows: dict[str, dict] = {}
        for code in trade.columns:
            tr = trade[code].to_numpy(dtype=float)
            if not np.isfinite(tr).all():
                continue
            z = zpanel[code].dropna().to_numpy(dtype=float) if code in zpanel else np.array([])
            if len(z) >= cfg.pressure_min_obs:
                zz = z - z.mean() if cfg.pressure_demean else z
                w, method = self._weights_kde(zz, grid), "kde"
            else:
                w, method = w_norm, "normal"

            e = float((w * tr).sum())
            sd = float(np.sqrt((w * (tr - e) ** 2).sum()))
            p_buy = float((w * (tr > 0)).sum())
            flip_z, above = self._nearest_flip(grid, tr)
            flip_pct = (
                float(eng.shocks_for(flip_z, mode="independent")[code])
                if np.isfinite(flip_z) else float("nan")
            )
            rows[code] = {
                "p_buy": p_buy,
                "p_buy_norm": float((w_norm * (tr > 0)).sum()),
                "e_trade_pct": e,
                "sd_pct": sd,
                "flat_pct": float(np.interp(0.0, grid, tr)),
                "flip_z": flip_z,
                "flip_pct": flip_pct,
                "flip_above": above,
                "tier": self._tier(p_buy),
                "method": method,
                "n_obs": int(len(z)),
            }

        table = pd.DataFrame.from_dict(rows, orient="index")[list(self.COLUMNS)]
        table = table.reindex([c for c in eng.codes if c in table.index])
        return TradePressure(
            table=table,
            aum=eng.aum,
            horizon_weeks=eng.horizon,
            sigma_range=cfg.pressure_sigma_range,
            demeaned=cfg.pressure_demean,
            sure_prob=cfg.pressure_sure_prob,
            lean_prob=cfg.pressure_lean_prob,
        )


__all__ = ["TIERS", "TradePressure", "TradePressureEstimator"]
