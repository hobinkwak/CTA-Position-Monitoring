"""
Weekly grid construction.

All calculations are performed weekly on the last active trading day. The grid therefore
uses the **last day in each week with actual data**, not the calendar Friday, which
absorbs holiday and exchange-closure differences naturally.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class WeeklyGrid:
    """Build weekly rebalancing dates from a daily index.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.bdate_range('2024-01-01', '2024-01-31')
    >>> grid = WeeklyGrid.from_index(idx)
    >>> list(grid.dates[:2].strftime('%Y-%m-%d (%a)'))
    ['2024-01-05 (Fri)', '2024-01-12 (Fri)']
    """

    def __init__(self, dates: pd.DatetimeIndex) -> None:
        self.dates = pd.DatetimeIndex(dates).sort_values()

    def __len__(self) -> int:
        return len(self.dates)

    def __repr__(self) -> str:
        if not len(self.dates):
            return "<WeeklyGrid empty>"
        return (
            f"<WeeklyGrid n={len(self.dates)} "
            f"{self.dates[0]:%Y-%m-%d}..{self.dates[-1]:%Y-%m-%d}>"
        )

    # ------------------------------------------------------------------ #

    @classmethod
    def from_index(
        cls,
        index: pd.DatetimeIndex,
        anchor: str = "W-FRI",
        drop_partial_last: bool = False,
    ) -> "WeeklyGrid":
        """Pick the **last actual trading day** of each week (per anchor).

        Parameters
        ----------
        index : daily DatetimeIndex
        anchor : week boundary definition. Default 'W-FRI' (week ends Friday)
        drop_partial_last : whether to drop the last week if it is not yet complete.
                            Keep False for live use so the latest week is included.
        """
        index = pd.DatetimeIndex(index).sort_values().unique()
        if len(index) == 0:
            return cls(pd.DatetimeIndex([]))

        # Last observation date of each week
        s = pd.Series(index, index=index)
        last = s.resample(anchor).last().dropna()
        dates = pd.DatetimeIndex(last.values)

        if drop_partial_last and len(dates) > 1:
            week_end = pd.DatetimeIndex(last.index)
            if week_end[-1] > index[-1]:
                dates = dates[:-1]
        return cls(dates)

    def slice(self, start=None, end=None) -> "WeeklyGrid":
        d = self.dates
        if start is not None:
            d = d[d >= pd.Timestamp(start)]
        if end is not None:
            d = d[d <= pd.Timestamp(end)]
        return WeeklyGrid(d)

    # ------------------------------------------------------------------ #

    def resample_prices(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """Daily prices -> weekly-grid prices. Previous valid value if the date is missing."""
        return daily_prices.reindex(self.dates, method="ffill")

    def weekly_returns(
        self, daily_prices: pd.DataFrame, max_ffill_days: int | None = 10
    ) -> pd.DataFrame:
        """Weekly simple returns.

        Markets with no value for more than `max_ffill_days` get NaN for that week's
        return (prevents a trading halt or data gap from masquerading as a 0% return).
        """
        weekly_px = self.resample_prices(daily_prices)
        rtn = weekly_px.pct_change()

        if max_ffill_days is not None:
            staleness = _staleness_days(daily_prices).reindex(self.dates, method="ffill")
            rtn = rtn.mask(staleness > max_ffill_days)

        # Keep NaN before each market's first valid price
        return rtn

    def align_series(self, series: pd.Series) -> pd.Series:
        """Align a daily series (e.g. the benchmark) to the weekly grid."""
        return series.reindex(self.dates, method="ffill")


class SessionReturns:
    """Build daily returns keeping only each market's **actual exchange sessions**.

    Why it is needed
    ----------------
    Source price data is forward-filled over weekends and holidays. Filtering out
    weekends still leaves **exchange-holiday rows**, whose return is exactly 0. Those
    zeros understate realised volatility.

    Observed (2026-07-31): the share of zero returns is 8.8% for JB and 8.0% for NK but
    only 1.7% for VG. Sigma is understated by JB -4.5%, NK -4.1%, ES -2.0%, VG ~0%,
    **differing by market**. A global constant would be absorbed into leverage k, but
    differences across markets **distort the relative weights** of `w = m/sigma_ann`
    (Japanese positions about 4-5% too large).

    How it is fixed
    ---------------
    Get per-market sessions from `pandas_market_calendars`, **compute returns on the
    session-only series**, then map back to the common index. Then:
      * holiday rows = NaN  -> excluded from the rolling std (not diluted by zeros)
      * returns spanning a holiday are **preserved** (if Monday is a holiday, Tuesday
        carries a 2-day return)

    Markets without a calendar fall back to the old behaviour (all weekdays).
    """

    def __init__(
        self,
        calendars: dict[str, str] | None = None,
        secondary: dict[str, str] | None = None,
        strict: bool = True,
        use_secondary: bool = True,
    ) -> None:
        from .expiry import EXCHANGE_CALENDARS, SECONDARY_CALENDARS

        self.calendars = dict(EXCHANGE_CALENDARS if calendars is None else calendars)
        self.secondary = (
            dict(SECONDARY_CALENDARS if secondary is None else secondary)
            if use_secondary
            else {}
        )
        self._cache: dict[tuple, pd.DatetimeIndex | None] = {}
        #: calendar lookup failures are never silently skipped (a bug once hid behind this)
        self.failures: dict[str, str] = {}

        if strict:
            from .expiry import validate_calendars

            bad = validate_calendars({**self.calendars, **self.secondary})
            if bad:
                warnings.warn(
                    "Unregistered exchange calendar names found; the session filter "
                    f"is not applied to those markets: {bad}",
                    stacklevel=2,
                )

    def _one_calendar(self, name: str, start, end, code: str) -> pd.DatetimeIndex | None:
        key = ("__cal__", name, pd.Timestamp(start), pd.Timestamp(end))
        if key in self._cache:
            return self._cache[key]
        out: pd.DatetimeIndex | None = None
        try:
            import pandas_market_calendars as mcal

            sched = mcal.get_calendar(name).schedule(start_date=start, end_date=end)
            out = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
        except Exception as exc:
            # Record instead of silently falling back; surfaces in diagnostics
            self.failures[code] = f"{name}: {type(exc).__name__}"
        self._cache[key] = out
        return out

    def sessions(self, code: str, start, end) -> pd.DatetimeIndex | None:
        """Actual sessions of the market = primary calendar ∩ secondary (cash market) calendar.

        The secondary intersection is needed because CME Globex opens shortened sessions
        on US holidays but settlement prices are not updated (see
        `expiry.SECONDARY_CALENDARS`).
        """
        key = (code, pd.Timestamp(start), pd.Timestamp(end))
        if key in self._cache:
            return self._cache[key]

        primary = self.calendars.get(code)
        out = self._one_calendar(primary, start, end, code) if primary else None

        sec_name = self.secondary.get(code)
        if out is not None and sec_name:
            sec = self._one_calendar(sec_name, start, end, code)
            if sec is not None:
                out = out.intersection(sec)

        self._cache[key] = out
        return out

    # ------------------------------------------------------------------ #

    def returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Daily simple returns on session days. Non-session rows are NaN."""
        if prices.empty:
            return prices.copy()
        start, end = prices.index[0], prices.index[-1]
        out = {}
        for code in prices.columns:
            s = prices[code]
            sess = self.sessions(code, start, end)
            if sess is not None:
                keep = s.index.isin(sess)
                # Keep days where the value actually changed even outside a session (guards against calendar errors)
                changed = s.ne(s.shift(1)) & s.notna()
                s = s.where(keep | changed)
            compact = s.dropna()
            out[code] = compact.pct_change().reindex(prices.index)
        return pd.DataFrame(out)[list(prices.columns)]

    def diagnostics(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Zero-return share and sigma change before/after filtering."""
        raw = prices.pct_change()
        clean = self.returns(prices)
        rows = []
        for code in prices.columns:
            r0, r1 = raw[code].dropna(), clean[code].dropna()
            rows.append({
                "code": code,
                "calendar": self.calendars.get(code, "-"),
                "rows_raw": len(r0),
                "rows_clean": len(r1),
                "dropped": len(r0) - len(r1),
                "zero_pct_raw": float((r0 == 0).mean() * 100),
                "zero_pct_clean": float((r1 == 0).mean() * 100),
                "sigma_raw": float(r0.std(ddof=1)),
                "sigma_clean": float(r1.std(ddof=1)),
            })
        d = pd.DataFrame(rows).set_index("code")
        d["sigma_lift_pct"] = (d["sigma_clean"] / d["sigma_raw"] - 1) * 100
        # Flag markets with no calendar or a failed lookup (no silent fallback)
        d["applied"] = [
            bool(self.calendars.get(c)) and c not in self.failures for c in d.index
        ]
        return d


def _staleness_days(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Business days elapsed since the last valid observation, at each date."""
    positions = pd.DataFrame(
        np.arange(len(daily_prices))[:, None].repeat(daily_prices.shape[1], axis=1),
        index=daily_prices.index,
        columns=daily_prices.columns,
    )
    last_valid = positions.where(daily_prices.notna()).ffill()
    return positions - last_valid


__all__ = ["WeeklyGrid", "SessionReturns"]
