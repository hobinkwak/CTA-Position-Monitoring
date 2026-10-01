"""
Scenario positioning: how CTAs change positions if an index moves.

The replication model is a **deterministic function**: given prices, the position is
fully determined. Assuming future prices therefore yields the future position exactly,
which regression-based estimates cannot do (betas are only estimated ex post).

Five scenarios
--------------
    Strong up / Weak up / Flat / Weak down / Strong down

Shock size is defined in **units of each market's weekly volatility (sigma)**. A fixed
percentage would treat Bunds (weekly sigma ~0.7%) and crude oil (weekly sigma ~5%) with
the same yardstick. The model is volatility-normalised, so sigma units are natural.

Two propagation modes
---------------------
* ``index``       - **one index moves and the other markets follow via historical
                    betas.** Answers "if the S&P rallies strongly, what do CTAs buy and
                    sell across *all* markets". This is the dashboard default.
* ``independent`` - each market moves independently by its own sigma. Used for
                    per-market trigger-level analysis.

Note: this module is an **extension** beyond the replication model itself and is
labelled as such on the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DAYS_PER_WEEK, TRADING_DAYS_PER_YEAR
from .ensemble import EnsembleReplicator
from .universe import UNIVERSE, Universe


# --------------------------------------------------------------------------- #
# Scenario definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scenario:
    """A single price scenario.

    Size can be set in two ways:

    * ``target_prob`` - defined by **target frequency** (recommended). E.g. 0.05 means
      "top/bottom 5% event". The engine converts it to a sigma multiple using the
      **empirical quantile of standardised returns z = r_t/sigma_t**, with no
      normality assumption.
    * ``sigma`` - sigma multiple given directly (backward compatible / manual override).

    Why empirical quantiles of standardised returns
    -----------------------------------------------
    Weekly returns reject normality in **16/16 markets** (mean excess kurtosis 8.85,
    TY 27.8, G 42.0). Most of that thickness is **volatility clustering, not
    unconditional fat tails**. Conditioning on a 60-day rolling sigma gives:

        z quantile (pooled n=22,834)  5%: -1.67   30%: -0.43   70%: +0.54   95%: +1.65
        Normal z                      5%: -1.645  30%: -0.524  70%: +0.524  95%: +1.645

    The 5%/95% quantiles agree within 2%. The remaining excess kurtosis (7.53) sits
    **beyond 3 sigma**, which our scenarios (<= 2 sigma) do not reach. Empirical
    quantiles are still used to drop the normality assumption altogether.
    """

    key: str
    label: str
    label_ko: str  # kept as the field name; holds the main English display label
    sigma: float | None = None
    #: Target one-sided frequency. 0.05 = top/bottom 5%. Sign is set by direction.
    target_prob: float | None = None
    #: +1 = up, -1 = down, 0 = flat
    direction: int = 0

    def __post_init__(self) -> None:
        if self.sigma is None and self.target_prob is None and self.direction != 0:
            raise ValueError(f"{self.key}: either sigma or target_prob is required")

    @property
    def display(self) -> str:
        return self.label_ko

    @property
    def is_up(self) -> bool:
        return self.direction > 0 if self.direction else (self.sigma or 0) > 0

    @property
    def is_down(self) -> bool:
        return self.direction < 0 if self.direction else (self.sigma or 0) < 0


#: Default five scenarios, **defined by target frequency**. Ordered down -> up.
#:
#:   Strong = top/bottom 5%   -> about 2.6 times a year
#:   Weak   = top/bottom 30%  -> about 15.6 times a year
#:   Flat   = 0
#:
#: Sigma multiples are resolved at run time from empirical z quantiles (observed:
#: Strong about +/-1.66 sigma, Weak about +/-0.49 sigma). Earlier hard-coded values of
#: +/-2.00 / +/-0.75 sigma corresponded to observed frequencies of 2.9% / 19.7%, which
#: did not match the "5% / 30%" names.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("strong_down", "Sharp selloff", "Strong down", target_prob=0.05, direction=-1),
    Scenario("weak_down", "Mild selloff", "Weak down", target_prob=0.30, direction=-1),
    Scenario("flat", "Flat", "Flat", sigma=0.0, direction=0),
    Scenario("weak_up", "Mild rally", "Weak up", target_prob=0.30, direction=+1),
    Scenario("strong_up", "Strong rally", "Strong up", target_prob=0.05, direction=+1),
)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioProjection:
    """Per-scenario projection results. All frames: index=market code, columns=scenario key."""

    positions: pd.DataFrame  # projected position (notional per $1 of benchmark)
    trades_pct: pd.DataFrame  # delta position (= required trade, as a ratio)
    trades_usd: pd.DataFrame  # delta position x AUM
    shocks_pct: pd.DataFrame  # % price move actually assumed for each market
    current: pd.Series  # current position
    scenarios: tuple[Scenario, ...]
    mode: str
    driver: str | None
    horizon_weeks: int
    aum: float
    universe: Universe
    #: {scenario_key: sigma multiple}, resolved from empirical z quantiles
    sigma_map: dict[str, float] = field(default_factory=dict)
    #: {scenario_key: empirical one-sided frequency}
    empirical_freq: dict[str, float] = field(default_factory=dict)

    def sigma_of(self, key: str) -> float:
        return float(self.sigma_map.get(key, 0.0))

    def sigmas(self) -> list[float]:
        """Sigma multiples in scenario order (e.g. for colour scales)."""
        return [self.sigma_of(s.key) for s in self.scenarios]

    # ------------------------------------------------------------------ #

    @property
    def keys(self) -> list[str]:
        return [s.key for s in self.scenarios]

    @property
    def labels(self) -> dict[str, str]:
        return {s.key: s.display for s in self.scenarios}

    def by_sector(self, what: str = "trades_usd") -> pd.DataFrame:
        """Sector totals. index=sector, columns=scenario key."""
        frame = getattr(self, what)  # index=market, columns=scenario
        # aggregate_by_sector expects a frame whose columns are market codes
        aggregated = self.universe.aggregate_by_sector(frame.T.fillna(0.0))
        return aggregated.T[self.keys]

    def totals(self) -> pd.DataFrame:
        """Per-scenario totals."""
        return pd.DataFrame(
            {
                "Gross notional": self.positions.abs().sum(),
                "Net notional": self.positions.sum(),
                "Gross trade $BN": self.trades_usd.abs().sum() / 1e9,
                "Net trade $BN": self.trades_usd.sum() / 1e9,
            }
        )

    def driver_move(self) -> pd.Series:
        """Assumed % move of the driver index in each scenario."""
        if self.driver is None:
            return pd.Series(dtype=float)
        return self.shocks_pct.loc[self.driver]

    def summary(self) -> str:
        lines = [
            f"Scenario projection | mode={self.mode}"
            + (f" | driver={self.driver}" if self.driver else "")
            + f" | horizon={self.horizon_weeks}w | AUM ${self.aum / 1e9:,.0f}BN",
        ]
        if self.driver is not None:
            moves = self.driver_move()
            lines.append(
                f"  {self.driver} assumed move: "
                + " / ".join(f"{s.label_ko} {moves[s.key]:+.2%}" for s in self.scenarios)
            )
        sector_trades = self.by_sector("trades_usd") / 1e9
        header = "  " + " " * 14 + "".join(f"{s.label_ko:>12}" for s in self.scenarios)
        lines.append(header)
        for sector in sector_trades.index:
            row = "".join(f"{sector_trades.loc[sector, k]:>+12,.0f}" for k in self.keys)
            lines.append(f"  {sector:<14}{row}")
        total = self.trades_usd.sum() / 1e9
        lines.append(
            f"  {'TOTAL net':<14}" + "".join(f"{total[k]:>+12,.0f}" for k in self.keys)
        )
        lines.append("  (unit: $BN, + = buy / - = sell)")
        return "\n".join(lines)


#: Minimum |beta| for showing driver attribution. Below this, results such as "CL must
#: fall 66% for TY to flip" appear, which are **divide-by-near-zero artifacts** (TY
#: itself only needs to move +0.2%).
MIN_BETA_FOR_ATTRIBUTION = 0.15


@dataclass
class TriggerLevels:
    """Points where the position sign flips: where CTAs go from long to short.

    Interpretation notes
    --------------------
    **The primary values are `own_move_pct` / `own_sigma` / `flip_price`**: how much
    the market **itself** must move for the position to flip. These are always
    interpretable regardless of the driver.

    `driver_move_pct` (how much the driver must move) is meaningful **only when beta is
    large enough**. A small beta makes the ratio explode: e.g. with a TY beta to CL of
    -0.04, "CL must fall 66% to move TY +0.2%" does **not** mean crude drives the TY
    flip. Rows with `|beta| < MIN_BETA_FOR_ATTRIBUTION` are set to NaN.

    Horizon note
    ------------
    Momentum is a 16-52 week **average**, so one week's return moves it by only 1/L.
    A **1-week horizon therefore needs an unrealistic shock to flip** (observed: CL
    itself -41.6% = -5.3 sigma, a recurrence of roughly 300,000 years). Triggers should
    be read on a **quarterly (13-week) horizon**, where all 16 markets show a flip point.
    """

    table: pd.DataFrame
    driver: str | None
    mode: str
    horizon_weeks: int

    def summary(self, n: int = 8) -> str:
        col = "own_move_pct" if "own_move_pct" in self.table.columns else "flip_pct"
        df = self.table.dropna(subset=[col]).copy()
        if df.empty:
            return f"  No sign flip within the search range ({self.horizon_weeks}w horizon)"
        df["abs"] = df[col].abs()
        near = df.nsmallest(n, "abs")
        lines = [
            f"  Markets closest to a flip (own price move within {self.horizon_weeks}w):"
        ]
        for code, row in near.iterrows():
            price = f" @ {row['flip_price']:,.2f}" if pd.notna(row.get("flip_price")) else ""
            sig = row.get("own_sigma")
            sig_txt = f" ({sig:+.1f}σ)" if pd.notna(sig) else ""
            lines.append(
                f"    {code:<3} {row['current_pos']:+7.1%} -> flip at "
                f"{row[col]:+7.2%}{sig_txt}{price}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ScenarioEngine:
    """Price scenarios -> projected positions and required trades.

    Parameters
    ----------
    ensemble : an already **fitted** EnsembleReplicator
    weekly_returns, daily_returns : market return panels
    aum : assumed total trend-following AUM
    horizon_weeks : period over which the shock is realised (default 1 week)
    scenarios : the five scenarios
    beta_window_weeks : beta estimation window for index propagation
    update_vol : whether realised volatility is also updated for the shock (default True)

    Examples
    --------
    >>> engine = ScenarioEngine(ensemble, weekly, daily)      # doctest: +SKIP
    >>> proj = engine.project(mode="index", driver="ES")      # doctest: +SKIP
    >>> print(proj.summary())                                 # doctest: +SKIP
    """

    def __init__(
        self,
        ensemble: EnsembleReplicator,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        universe: Universe = UNIVERSE,
        aum: float = 300e9,
        horizon_weeks: int = 1,
        scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
        beta_window_weeks: int = 156,
        update_vol: bool = True,
    ) -> None:
        ensemble.require_fitted()
        if horizon_weeks < 1:
            raise ValueError("horizon_weeks must be >= 1")

        self.ensemble = ensemble
        self.weekly = weekly_returns
        self.daily = daily_returns
        self.universe = universe
        self.aum = float(aum)
        self.horizon = int(horizon_weeks)
        self.scenarios = tuple(scenarios)
        self.beta_window = int(beta_window_weeks)
        self.update_vol = update_vol

        self.codes = [c for c in universe.codes if c in weekly_returns.columns]
        self._max_lookback = max(m.params.momentum_lookback for m in ensemble.models)
        self._min_lookback = min(m.params.momentum_lookback for m in ensemble.models)
        self._max_vol = max(m.params.volatility_lookback for m in ensemble.models)

        # Collapse guard: if H exceeds half of the minimum lookback, most of the
        # momentum window is overwritten by the synthetic shock, so momentum sign =
        # shock sign. Flip points then collapse to 0% and triggers become meaningless
        # (observed: all 0.00% at H=13, L_min=16).
        replace_ratio = self.horizon / self._min_lookback
        if replace_ratio > 0.5:
            import warnings

            warnings.warn(
                f"horizon_weeks={self.horizon} covers {replace_ratio:.0%} of the "
                f"minimum momentum lookback ({self._min_lookback}w). "
                "The momentum window is dominated by the synthetic shock and flip "
                "points collapse to 0. "
                f"{max(1, self._min_lookback // 4)}w or less is recommended.",
                stacklevel=2,
            )
        self.window_replace_ratio = replace_ratio

        # Performance: only the last-date signal is needed, so keep just the tail
        n_new_days = self.horizon * DAYS_PER_WEEK
        self._daily_tail = self.daily.iloc[-(self._max_vol + n_new_days + 20) :]
        self._weekly_tail = self.weekly.iloc[-(self._max_lookback + self.horizon + 5) :]

        self.current = ensemble.positions.dropna(how="all").iloc[-1].reindex(self.codes)
        self.as_of = ensemble.positions.dropna(how="all").index[-1]

        self._leverage = self._member_leverages()
        self._weekly_vol = self._current_weekly_vol()
        self._betas_cache: dict[str, pd.Series] = {}

        # Target frequency -> sigma multiple (empirical standardised quantile)
        self._z = self._standardized_returns()
        self.sigma_map = self._resolve_sigmas()
        self.empirical_freq = self._empirical_frequencies()

    def __repr__(self) -> str:
        return (
            f"<ScenarioEngine as_of={self.as_of:%Y-%m-%d} horizon={self.horizon}w "
            f"n_scenarios={len(self.scenarios)}>"
        )

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _member_leverages(self) -> dict[str, float]:
        """Leverage k per member (last value if expanding)."""
        out = {}
        for model in self.ensemble.models:
            k = model.fit_result.leverage if model.fit_result else 1.0
            out[model.label] = float(k) if np.isscalar(k) else float(pd.Series(k).iloc[-1])
        return out

    def _current_weekly_vol(self) -> pd.Series:
        """Current weekly volatility at the ensemble's representative lookback; used to convert sigma shocks."""
        lookback = self.ensemble.models[0].params.volatility_lookback
        sigma_d = self.daily[self.codes].rolling(lookback, min_periods=lookback // 2).std(ddof=1)
        return sigma_d.iloc[-1] * np.sqrt(DAYS_PER_WEEK)

    def standardized_panel(self) -> pd.DataFrame:
        """z = r_t / sigma_{t-1}, index=weekly grid, columns=market code. Volatility clustering removed."""
        lookback = self.ensemble.models[0].params.volatility_lookback
        sigma_d = (
            self.daily[self.codes]
            .rolling(lookback, min_periods=max(2, lookback // 2))
            .std(ddof=1)
        )
        sw = (sigma_d * np.sqrt(DAYS_PER_WEEK)).reindex(
            self.weekly.index, method="ffill"
        )
        return (self.weekly[self.codes] / sw[self.codes].shift(1)).replace(
            [np.inf, -np.inf], np.nan
        )

    def _standardized_returns(self) -> pd.Series:
        """`standardized_panel` pooled across all markets."""
        z = self.standardized_panel()
        return pd.concat([z[c].dropna() for c in self.codes]) if self.codes else pd.Series(
            dtype=float
        )

    def _resolve_sigmas(self) -> dict[str, float]:
        """Sigma multiple per scenario; target_prob is converted via empirical z quantiles."""
        out: dict[str, float] = {}
        n = len(self._z)
        for s in self.scenarios:
            if s.sigma is not None:
                out[s.key] = float(s.sigma)
                continue
            p = s.target_prob
            if p is None:
                out[s.key] = 0.0
                continue
            # Empirical quantile if the sample is large enough, else normal quantile
            if n >= 2000:
                q = p if s.direction < 0 else 1.0 - p
                out[s.key] = float(self._z.quantile(q))
            else:
                from scipy import stats as _st

                z = _st.norm.ppf(p if s.direction < 0 else 1.0 - p)
                out[s.key] = float(z)
        return out

    def _empirical_frequencies(self) -> dict[str, float]:
        """Observed frequency (%) of each scenario's sigma multiple, shown to the user."""
        out: dict[str, float] = {}
        if self._z.empty:
            return {s.key: float("nan") for s in self.scenarios}
        for s in self.scenarios:
            sig = self.sigma_map[s.key]
            if sig == 0:
                out[s.key] = float("nan")
            elif sig < 0:
                out[s.key] = float((self._z < sig).mean())
            else:
                out[s.key] = float((self._z > sig).mean())
        return out

    def normality_summary(self) -> dict[str, float]:
        """Normality diagnostics for weekly returns and standardised returns (z); empirical values for dashboard text.

        * ``n_markets`` / ``n_rejected``: number of markets whose weekly returns reject
          normality under Jarque-Bera (5%)
        * ``excess_kurtosis_raw`` / ``excess_kurtosis_z``: mean excess kurtosis across
          markets (raw returns / volatility-standardised z)
        * ``max_prob_gap``: maximum gap between target probability and the normal CDF of
          the pooled z's 5-95% quantiles (probability units, 0.03 = 3pp)
        """
        from scipy import stats

        raw = self.weekly[self.codes]
        lookback = self.ensemble.models[0].params.volatility_lookback
        sigma_d = (
            self.daily[self.codes]
            .rolling(lookback, min_periods=max(2, lookback // 2))
            .std(ddof=1)
        )
        sw = (sigma_d * np.sqrt(DAYS_PER_WEEK)).reindex(raw.index, method="ffill")
        z = (raw / sw.shift(1)).replace([np.inf, -np.inf], np.nan)

        n_rej, k_raw, k_z = 0, [], []
        for c in self.codes:
            x = raw[c].dropna()
            if len(x) < 30:
                continue
            k_raw.append(float(stats.kurtosis(x)))
            n_rej += int(stats.jarque_bera(x).pvalue < 0.05)
            zc = z[c].dropna()
            if len(zc) >= 30:
                k_z.append(float(stats.kurtosis(zc)))

        gap = float("nan")
        if not self._z.empty:
            ps = np.linspace(0.05, 0.95, 91)
            q = np.quantile(self._z.to_numpy(), ps)
            gap = float(np.max(np.abs(stats.norm.cdf(q) - ps)))
        return {
            "n_markets": float(len(k_raw)),
            "n_rejected": float(n_rej),
            "excess_kurtosis_raw": float(np.mean(k_raw)) if k_raw else float("nan"),
            "excess_kurtosis_z": float(np.mean(k_z)) if k_z else float("nan"),
            "max_prob_gap": gap,
        }

    def scenario_table(self) -> pd.DataFrame:
        """Scenario definition summary: sigma multiple, target frequency, observed frequency, times per year."""
        rows = []
        for s in self.scenarios:
            sig = self.sigma_map[s.key]
            freq = self.empirical_freq.get(s.key, np.nan)
            rows.append({
                "scenario": s.display,
                "key": s.key,
                "target_prob": s.target_prob,
                "sigma": sig,
                "empirical_prob": freq,
                "per_year": freq * 52 if np.isfinite(freq) else np.nan,
            })
        return pd.DataFrame(rows).set_index("key")

    def betas(self, driver: str) -> pd.Series:
        """Univariate weekly beta of each market to the driver market.

        In index mode this sets "if the S&P rises 1%, how much do Bunds move". The
        replication model itself ignores correlations (naive risk parity), but
        **scenario propagation** needs them because markets do move together.
        """
        if driver in self._betas_cache:
            return self._betas_cache[driver]
        if driver not in self.weekly.columns:
            raise KeyError(f"driver {driver!r} not in weekly returns")

        window = self.weekly[self.codes].iloc[-self.beta_window :].dropna(how="all")
        x = window[driver]
        var = x.var(ddof=1)
        if not np.isfinite(var) or var == 0:
            raise ValueError(f"driver {driver!r} has zero variance over the beta window")
        betas = window.apply(lambda col: col.cov(x) / var)
        betas[driver] = 1.0
        self._betas_cache[driver] = betas
        return betas

    # ------------------------------------------------------------------ #
    # Shock -> position
    # ------------------------------------------------------------------ #

    def shocks_for(self, sigma: float, mode: str = "index", driver: str | None = "ES") -> pd.Series:
        """Scenario sigma -> % price move per market.

        Cumulative move over the whole horizon, scaled by sqrt(H) (random-walk assumption).
        """
        scale = np.sqrt(self.horizon)
        if mode == "independent":
            return (self._weekly_vol * sigma * scale).reindex(self.codes)
        if mode == "index":
            if driver is None:
                raise ValueError("mode='index' requires a driver market")
            driver_move = float(self._weekly_vol[driver]) * sigma * scale
            return (self.betas(driver) * driver_move).reindex(self.codes)
        raise ValueError("mode must be 'index' or 'independent'")

    def positions_after(self, shocks: pd.Series) -> pd.Series:
        """Ensemble position after the price shock is realised.

        Re-runs the replication model as is: an exact recomputation, not an approximation.
        """
        shocks = shocks.reindex(self.codes).fillna(0.0)
        H = self.horizon

        # Spread the shock evenly over the horizon (geometric)
        r_week = (1.0 + shocks) ** (1.0 / H) - 1.0
        r_day = (1.0 + r_week) ** (1.0 / DAYS_PER_WEEK) - 1.0

        ext_weekly = self._append_rows(self._weekly_tail[self.codes], r_week, H, freq_days=7)
        ext_daily = self._append_rows(
            self._daily_tail[self.codes], r_day, H * DAYS_PER_WEEK, freq_days=1
        )

        # Recompute positions per member, then blend with ensemble weights.
        # Follows the same path as ReplicationModel / EnsembleReplicator:
        #   member = clip(m) / σ_ann × k_member,  ensemble = Σ w·member × rescale
        blended = pd.Series(0.0, index=self.codes, dtype=float)
        vol_source = ext_daily if self.update_vol else self._daily_tail[self.codes]

        for model in self.ensemble.models:
            p = model.params
            sigma_d = (
                vol_source.rolling(
                    p.volatility_lookback, min_periods=p.effective_vol_min_periods
                )
                .std(ddof=1)
                .iloc[-1]
                .replace(0.0, np.nan)
            )
            sigma_w = sigma_d * np.sqrt(DAYS_PER_WEEK)
            sigma_ann = sigma_d * np.sqrt(TRADING_DAYS_PER_YEAR)

            L = p.momentum_lookback
            m = (ext_weekly.iloc[-L:].mean() / sigma_w) * np.sqrt(L)
            m = m.replace([np.inf, -np.inf], np.nan)
            if p.effective_cap is not None:
                m = m.clip(-p.effective_cap, p.effective_cap)

            member = (m / sigma_ann).replace([np.inf, -np.inf], np.nan)
            member = member * self._leverage[model.label]
            blended += member.reindex(self.codes).fillna(0.0) * self.ensemble.weights[model.label]

        return blended * self.ensemble.rescale_factor

    @staticmethod
    def _append_rows(
        frame: pd.DataFrame, values: pd.Series, n_rows: int, freq_days: int
    ) -> pd.DataFrame:
        """Append n_rows identical rows to the end of the frame."""
        last = frame.index[-1]
        new_index = pd.DatetimeIndex(
            [last + pd.Timedelta(days=freq_days * (i + 1)) for i in range(n_rows)]
        )
        block = pd.DataFrame(
            np.tile(values.reindex(frame.columns).to_numpy(dtype=float), (n_rows, 1)),
            index=new_index,
            columns=frame.columns,
        )
        return pd.concat([frame, block])

    # ------------------------------------------------------------------ #
    # Projection
    # ------------------------------------------------------------------ #

    def project(self, mode: str = "index", driver: str | None = "ES") -> ScenarioProjection:
        """Project all five scenarios."""
        positions, shocks = {}, {}
        for scenario in self.scenarios:
            shock = self.shocks_for(
                self.sigma_map[scenario.key], mode=mode, driver=driver
            )
            shocks[scenario.key] = shock
            positions[scenario.key] = self.positions_after(shock)

        pos = pd.DataFrame(positions)[[s.key for s in self.scenarios]]
        shk = pd.DataFrame(shocks)[[s.key for s in self.scenarios]]
        trades = pos.sub(self.current.reindex(pos.index).fillna(0.0), axis=0)

        return ScenarioProjection(
            positions=pos,
            trades_pct=trades,
            trades_usd=trades * self.aum,
            shocks_pct=shk,
            current=self.current,
            scenarios=self.scenarios,
            mode=mode,
            driver=driver if mode == "index" else None,
            horizon_weeks=self.horizon,
            aum=self.aum,
            universe=self.universe,
            sigma_map=dict(self.sigma_map),
            empirical_freq=dict(self.empirical_freq),
        )

    # ------------------------------------------------------------------ #
    # Response curve / triggers
    # ------------------------------------------------------------------ #

    def response_curve(
        self,
        mode: str = "index",
        driver: str | None = "ES",
        sigma_range: float = 3.0,
        n_points: int = 41,
    ) -> pd.DataFrame:
        """Sweep shock size continuously and show how positions respond.

        Shows at a glance where the cap flattens the response and where the sign flips.

        Returns
        -------
        index = sigma (driver sigma units), columns = market codes + 'driver_move_pct'
        """
        sigmas = np.linspace(-sigma_range, sigma_range, n_points)
        rows = {}
        driver_moves = {}
        for sigma in sigmas:
            shock = self.shocks_for(sigma, mode=mode, driver=driver)
            rows[sigma] = self.positions_after(shock)
            driver_moves[sigma] = (
                float(shock[driver]) if (mode == "index" and driver) else np.nan
            )
        curve = pd.DataFrame(rows).T
        curve.index.name = "sigma"
        curve["driver_move_pct"] = pd.Series(driver_moves)
        return curve

    def triggers(
        self,
        mode: str = "index",
        driver: str | None = "ES",
        prices: pd.DataFrame | None = None,
        sigma_range: float = 12.0,
        n_points: int = 121,
    ) -> TriggerLevels:
        """Find where each market's position sign flips.

        Parameters
        ----------
        prices : price panel used to convert to price levels. Pass **raw front-contract
                 (C1) prices** to get actual trigger prices (back-adjusted rolled
                 prices are not real prices).
        sigma_range : search range (driver sigma multiples). Flips are rare on short
                      horizons (one week is added to an L-week average), so use a wide range.

        Notes
        -----
        NaN means no flip inside the range. This is informative in itself: "the CTA
        position in this market does not flip within this horizon".
        """
        curve = self.response_curve(
            mode=mode, driver=driver, sigma_range=sigma_range, n_points=n_points
        )
        sigmas = curve.index.to_numpy(dtype=float)

        last_px = None
        if prices is not None:
            valid = prices.reindex(columns=self.codes).dropna(how="all")
            if not valid.empty:
                last_px = valid.iloc[-1]

        betas = self.betas(driver) if (mode == "index" and driver) else None

        # Position after H weeks with zero shock (flat).
        # Note: this differs from the current position because the momentum window
        # slides H steps and the oldest weeks drop out. Some markets **flip sign on a
        # flat scenario alone** (BP: +7.8% -> -3.1%), so flip direction should be read
        # against this value, not the current position.
        flat_pos = self.positions_after(pd.Series(0.0, index=self.codes))

        rows = []
        for code in self.codes:
            series = curve[code].to_numpy(dtype=float)
            flip_sigma = _first_zero_crossing(sigmas, series)

            own_move = own_sigma = driver_move = flip_price = np.nan
            if np.isfinite(flip_sigma):
                shock = self.shocks_for(flip_sigma, mode=mode, driver=driver)
                # Primary metric: how much the market itself must move (always interpretable)
                own_move = float(shock[code])
                sw = float(self._weekly_vol.get(code, np.nan))
                if np.isfinite(sw) and sw > 0:
                    own_sigma = own_move / (sw * np.sqrt(self.horizon))
                if last_px is not None and pd.notna(last_px.get(code)):
                    flip_price = float(last_px[code]) * (1.0 + own_move)

                # Secondary metric: how much the driver must move.
                # Dropped when beta is small (divide-by-near-zero artifact).
                if betas is not None:
                    b = float(betas.get(code, np.nan))
                    if np.isfinite(b) and abs(b) >= MIN_BETA_FOR_ATTRIBUTION:
                        driver_move = float(shock[driver])

            rows.append(
                {
                    "code": code,
                    "name": self.universe.get(code).name,
                    "sector": self.universe.get(code).sector,
                    "current_pos": float(self.current.get(code, np.nan)),
                    "flat_pos": float(flat_pos.get(code, np.nan)),
                    "own_move_pct": own_move,
                    "own_sigma": own_sigma,
                    "flip_price": flip_price,
                    "last_price": float(last_px[code]) if last_px is not None
                    and pd.notna(last_px.get(code)) else np.nan,
                    "driver_move_pct": driver_move,
                    "beta_vs_driver": float(betas.get(code, np.nan))
                    if betas is not None else np.nan,
                    "flip_sigma": flip_sigma,
                }
            )

        table = pd.DataFrame(rows).set_index("code")
        return TriggerLevels(
            table=table, driver=driver if mode == "index" else None,
            mode=mode, horizon_weeks=self.horizon,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _first_zero_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """Find the first sign change, starting where |x| is smallest, by linear interpolation.

    The flip point nearest to the current state (zero shock) is the practically
    meaningful one, so the search spreads outward from 0 in both directions.
    """
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    x, y = x[finite], y[finite]

    zero_idx = int(np.argmin(np.abs(x)))
    base_sign = np.sign(y[zero_idx])
    if base_sign == 0:
        return float(x[zero_idx])

    candidates = []
    # to the right
    for i in range(zero_idx, len(x) - 1):
        if np.sign(y[i + 1]) != base_sign and y[i + 1] != 0:
            candidates.append(_interp_zero(x[i], y[i], x[i + 1], y[i + 1]))
            break
    # to the left
    for i in range(zero_idx, 0, -1):
        if np.sign(y[i - 1]) != base_sign and y[i - 1] != 0:
            candidates.append(_interp_zero(x[i - 1], y[i - 1], x[i], y[i]))
            break

    if not candidates:
        return np.nan
    return float(min(candidates, key=abs))


def _interp_zero(x0: float, y0: float, x1: float, y1: float) -> float:
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


__all__ = [
    "Scenario",
    "DEFAULT_SCENARIOS",
    "ScenarioEngine",
    "ScenarioProjection",
    "TriggerLevels",
]
