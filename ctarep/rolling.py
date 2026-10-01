"""
Continuous (rolled) futures construction and diagnostics.

This is the most important data-processing step in the project.

Both the momentum signal (numerator) and realized volatility (denominator) are computed
from returns, so the price jump at a contract switch enters directly as a fake return. A
WTI contango roll gap is a few percent while the 32-week average weekly return is 0.1-0.5%,
so the signal can flip entirely.

Therefore:
    * signal / returns / volatility -> ratio (relative) back-adjusted continuous futures
    * contract-count conversion     -> raw front-month prices (not rolled)

Ratio adjustment is used because the model only uses % returns and % volatility. Absolute
(difference) adjustment can make adjusted prices negative (WTI, April 2020) and distorts
% returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import DATA_CONFIG, DataConfig
from .universe import UNIVERSE, Universe


def get_futures_rolled(df, col1, col2, roll_dates, method="relative", roll_backward=True):
    """Back-adjust a front-month series using the second-month price on roll dates.

    `roll_dates` follow the 'switch_day' convention: the adjustment factor is applied
    from the roll date itself, so pass the first day the new contract appears in the
    generic series. Returns (rolled_prices, roll_yield).
    """
    pos = [df.index.get_loc(dt) for dt in roll_dates]
    roll_yield = pd.Series(0.0, index=df.index)
    if method == "absolute":
        roll_yield.loc[roll_dates] = (df.loc[roll_dates, col2] - df.loc[roll_dates, col1]).values * (-1)
        gaps = df[col1] * 0
        gaps.iloc[pos] = (df[col2].iloc[pos] - df[col1].iloc[pos]).values
        gaps = gaps.cumsum()
        if roll_backward:
            gaps -= gaps.iloc[-1]
        return df[col1] - gaps, roll_yield
    roll_yield.loc[roll_dates] = (df.loc[roll_dates, col2] / df.loc[roll_dates, col1] - 1).values * (-1)
    gaps = df[col1] * 0 + 1
    gaps.iloc[pos] = (df[col2].iloc[pos] / df[col1].iloc[pos]).values
    gaps = gaps.cumprod()
    if roll_backward:
        gaps /= gaps.iloc[-1]
    return df[col1] / gaps, roll_yield


@dataclass
class RollDiagnostics:
    """Report of points where the roll adjustment may be missing."""

    per_market: pd.DataFrame  # index=code, columns=[n_extreme, worst_z, worst_date, worst_return]
    events: pd.DataFrame  # individual suspect events (date, code, ret, z)
    threshold: float

    @property
    def suspicious_markets(self) -> list[str]:
        return self.per_market.index[self.per_market["n_extreme"] > 0].tolist()

    def summary(self) -> str:
        bad = self.suspicious_markets
        if not bad:
            return f"[OK] no daily move with |z| > {self.threshold:g}; roll adjustment looks fine."
        lines = [
            f"[WARN] extreme daily moves with |z| > {self.threshold:g}; roll adjustment may be missing: {bad}",
            "       (could be genuine market shocks; compare the dates below with expiry dates)",
        ]
        head = self.events.head(12)
        for _, row in head.iterrows():
            lines.append(
                f"       {row['date']:%Y-%m-%d}  {row['code']:>3}  "
                f"ret={row['ret']:+.2%}  z={row['z']:.1f}"
            )
        if len(self.events) > len(head):
            lines.append(f"       ... and {len(self.events) - len(head)} more")
        return "\n".join(lines)


class FuturesRoller:
    """C1/C2 raw prices + roll dates -> continuous futures series.

    Calls `get_futures_rolled` for each market.

    Examples
    --------
    >>> roller = FuturesRoller()                                    # doctest: +SKIP
    >>> rolled, roll_yield = roller.build(c1, c2, roll_dates)       # doctest: +SKIP
    """

    def __init__(
        self,
        universe: Universe = UNIVERSE,
        data_config: DataConfig = DATA_CONFIG,
    ) -> None:
        self.universe = universe
        self.cfg = data_config

    # ------------------------------------------------------------------ #
    # Continuous futures construction
    # ------------------------------------------------------------------ #

    def build(
        self,
        front: pd.DataFrame,
        second: pd.DataFrame,
        roll_dates: Mapping[str, pd.DatetimeIndex],
        method: str | None = None,
        roll_backward: bool | None = None,
        convention: str = "last_old_day",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build continuous futures for each market.

        Parameters
        ----------
        front, second : raw front/second-month prices (columns = market code)
        roll_dates : {market_code: DatetimeIndex}
        method : 'relative' (ratio, default) | 'absolute' (difference). This model needs relative.
        roll_backward : if True, back-adjust so the latest price matches the actual price.
        convention : how roll_dates are interpreted
            'last_old_day' (default): roll date = last day the old contract is held.
                                     The generic series switches to the new contract on the next trading day.
            'switch_day'          : roll date = first day the new contract appears in the generic series.

        Returns
        -------
        (rolled_prices, roll_yields) : both with columns = market code

        Notes
        -----
        `get_futures_rolled` uses the 'switch_day' convention (the adjustment factor is applied
        from the roll date itself). With the default 'last_old_day' convention the dates are
        pushed one trading day forward here. A wrong convention leaves the roll gap in the
        adjusted series; this is covered by the tests.
        """
        method = method or self.cfg.roll_method
        roll_backward = self.cfg.roll_backward if roll_backward is None else roll_backward
        if convention not in ("last_old_day", "switch_day"):
            raise ValueError("convention must be 'last_old_day' or 'switch_day'")

        if method != "relative":
            import warnings

            warnings.warn(
                f"roll method='{method}': this model uses % returns/volatility, "
                "so 'relative' (ratio adjustment) is recommended.",
                stacklevel=2,
            )

        rolled: dict[str, pd.Series] = {}
        yields: dict[str, pd.Series] = {}

        for code in self.universe.codes:
            if code not in front.columns or code not in second.columns:
                continue
            pair = pd.concat(
                [front[code].rename("c1"), second[code].rename("c2")], axis=1
            ).dropna()
            if pair.empty:
                continue

            dates = pd.DatetimeIndex(roll_dates.get(code, pd.DatetimeIndex([])))
            # Snap only to dates present in the data (holidays etc.)
            dates = self._snap_to_index(dates, pair.index)
            if convention == "last_old_day":
                # get_futures_rolled uses the switch_day convention -> push one trading day later
                dates = self._advance(dates, pair.index, 1)
            if len(dates) == 0:
                # No roll dates means no roll -> keep C1 as is
                rolled[code] = pair["c1"]
                yields[code] = pd.Series(0.0, index=pair.index)
                continue

            px, ry = get_futures_rolled(
                pair,
                col1="c1",
                col2="c2",
                roll_dates=dates,
                method=method,
                roll_backward=roll_backward,
            )
            rolled[code] = px.rename(code)
            yields[code] = ry.rename(code)

        if not rolled:
            raise ValueError("No continuous futures could be built. Check front/second/roll_dates.")

        rolled_df = pd.DataFrame(rolled).sort_index()
        yields_df = pd.DataFrame(yields).reindex(rolled_df.index).fillna(0.0)
        ordered = [c for c in self.universe.codes if c in rolled_df.columns]
        return rolled_df[ordered], yields_df[ordered]

    @staticmethod
    def _snap_to_index(dates: pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Snap roll dates to the trading-day index; use the previous trading day if absent."""
        if len(dates) == 0 or len(index) == 0:
            return pd.DatetimeIndex([])
        positions = index.searchsorted(dates, side="right") - 1
        positions = positions[(positions >= 0) & (positions < len(index))]
        snapped = index[np.unique(positions)]
        # The first and last observation dates cannot be roll points
        return snapped[(snapped > index[0]) & (snapped < index[-1])]

    @staticmethod
    def _advance(dates: pd.DatetimeIndex, index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
        """Shift roll dates by n trading days on the trading-day index."""
        if len(dates) == 0:
            return dates
        positions = index.get_indexer(dates) + n
        positions = positions[(positions > 0) & (positions < len(index))]
        return index[np.unique(positions)]

    # ------------------------------------------------------------------ #
    # Roll date computation
    # ------------------------------------------------------------------ #

    def roll_dates_from_expiry(
        self,
        expiries: Mapping[str, pd.DatetimeIndex],
        trading_index: pd.DatetimeIndex,
        days_before: int | None = None,
        first_notice: Mapping[str, pd.DatetimeIndex] | None = None,
    ) -> dict[str, pd.DatetimeIndex]:
        """Expiry table -> roll dates (N business days before expiry).

        The paper does not specify a roll rule. Expiry-based rolling is the industry
        standard, and with a 32-week lookback the sensitivity to the rule is low.

        Parameters
        ----------
        expiries : {market_code: DatetimeIndex}
            Last tradeable dates.
        trading_index : reference trading-day index (usually the price data index)
        days_before : days before expiry to roll. Default DataConfig.roll_days_before_expiry
        first_notice : {market_code: DatetimeIndex}, optional
            First notice dates. If given, min(LTD, FND) is treated as the expiry.

        Notes
        -----
        Physically delivered contracts must be viewed against the First Notice Day.
        Physical commodities such as CL / GC / HG / S and bond futures such as TY / G / RX / JB
        create a delivery obligation if held past FND. Practitioner CTAs roll before FND.
        Cash-settled contracts (ES, VG, Z, NK, FX futures) only need LTD.

        So when `first_notice` is given, the earlier of the two dates is used as the roll
        basis, which is always the safe side without checking which comes first per contract.
        """
        days_before = (
            self.cfg.roll_days_before_expiry if days_before is None else days_before
        )
        out: dict[str, pd.DatetimeIndex] = {}
        for code, exp in expiries.items():
            dates = pd.DatetimeIndex(pd.to_datetime(exp)).dropna().sort_values()
            if first_notice and code in first_notice:
                fnd = pd.DatetimeIndex(pd.to_datetime(first_notice[code])).dropna().sort_values()
                dates = self._pairwise_min(dates, fnd)
            positions = trading_index.searchsorted(dates, side="left") - days_before
            positions = positions[(positions > 0) & (positions < len(trading_index))]
            out[code] = trading_index[np.unique(positions)]
        return out

    @staticmethod
    def _pairwise_min(ltd: pd.DatetimeIndex, fnd: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """min(last trading day, first notice day) for each contract month.

        The two series may differ in length, so they are paired by contract month (YYYY-MM).
        """
        a = pd.Series(ltd.values, index=ltd.to_period("M"))
        b = pd.Series(fnd.values, index=fnd.to_period("M"))
        joined = pd.concat([a.rename("ltd"), b.rename("fnd")], axis=1)
        merged = joined.min(axis=1, skipna=True).dropna()
        return pd.DatetimeIndex(sorted(merged.values))

    def roll_dates_from_volume(
        self,
        front_volume: pd.DataFrame,
        second_volume: pd.DataFrame,
        min_gap_days: int = 15,
    ) -> dict[str, pd.DatetimeIndex]:
        """Volume-switch roll dates (first day C2 volume exceeds C1).

        An alternative when no expiry table is available; a common practitioner rule.
        """
        out: dict[str, pd.DatetimeIndex] = {}
        for code in self.universe.codes:
            if code not in front_volume.columns or code not in second_volume.columns:
                continue
            pair = pd.concat(
                [front_volume[code].rename("v1"), second_volume[code].rename("v2")], axis=1
            ).dropna()
            if pair.empty:
                continue
            crossed = pair["v2"] > pair["v1"]
            triggers = crossed & ~crossed.shift(1, fill_value=False)
            dates = pair.index[triggers]
            out[code] = _thin_dates(dates, min_gap_days)
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def detect_roll_jumps(
        self, prices: pd.DataFrame, zscore: float | None = None
    ) -> RollDiagnostics:
        """Scan daily return z-scores for unadjusted roll gaps.

        Genuine market shocks (e.g. March 2020) are also flagged, so this is for checking
        by eye whether flagged dates are near expiry, not for automatic decisions.
        """
        threshold = self.cfg.roll_jump_zscore if zscore is None else zscore
        rtn = prices.pct_change()

        rows = []
        events = []
        for code in prices.columns:
            r = rtn[code].dropna()
            if len(r) < 30 or r.std() == 0:
                rows.append(
                    {"code": code, "n_extreme": 0, "worst_z": np.nan,
                     "worst_date": pd.NaT, "worst_return": np.nan}
                )
                continue
            z = (r - r.mean()).abs() / r.std()
            hits = z[z > threshold]
            worst_dt = z.idxmax()
            rows.append(
                {
                    "code": code,
                    "n_extreme": int(len(hits)),
                    "worst_z": float(z.max()),
                    "worst_date": worst_dt,
                    "worst_return": float(r.loc[worst_dt]),
                }
            )
            for dt in hits.index:
                events.append(
                    {"date": dt, "code": code, "ret": float(r.loc[dt]), "z": float(z.loc[dt])}
                )

        per_market = pd.DataFrame(rows).set_index("code")
        events_df = (
            pd.DataFrame(events).sort_values("z", ascending=False).reset_index(drop=True)
            if events
            else pd.DataFrame(columns=["date", "code", "ret", "z"])
        )
        return RollDiagnostics(per_market=per_market, events=events_df, threshold=threshold)

    @staticmethod
    def roll_cost_estimate(roll_yields: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
        """Annualized roll yield (carry) estimate per market: negative in contango, positive in backwardation."""
        return roll_yields.sum() / len(roll_yields) * periods_per_year


def _thin_dates(dates: pd.DatetimeIndex, min_gap_days: int) -> pd.DatetimeIndex:
    """Filter out triggers that are too close together (when volume flip-flops)."""
    kept: list[pd.Timestamp] = []
    for dt in sorted(dates):
        if not kept or (dt - kept[-1]).days >= min_gap_days:
            kept.append(dt)
    return pd.DatetimeIndex(kept)


__all__ = ["FuturesRoller", "RollDiagnostics"]
