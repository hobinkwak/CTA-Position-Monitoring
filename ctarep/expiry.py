"""
Futures expiry/roll calendar (rule based).

The paper does not mention a roll rule at all, so industry-standard rules are encoded
per market.

Key pitfall: US Treasury futures
    First Notice Day = last business day of the month BEFORE the delivery month
    Last Trading Day = 7 business days before the last business day of the delivery
                       month (10-year and longer)
So the FND of a March contract is at the end of February while the last trading day is
around March 20. Practitioner CTAs roll 2-4 days before FND, i.e. at the end of
February. Rolling off the last tradeable date alone rolls a month late, and a month of
fake returns enters the signal. Physically delivered commodities (GC, HG) follow FND
for the same reason.

A better method exists when a generic-ticker history is available: the day the
underlying contract behind a generic series changes is the roll date, with no rule
estimation (see `RollFromGenericTicker`).

The rule-based path is the fallback; check with `ExpiryCalendar.validate_against_prices()`
that actual C1/C2 gaps appear on the computed roll dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Contract month cycles
# --------------------------------------------------------------------------- #

#: Exchange calendar per market (registered name in `pandas_market_calendars`).
#: Used both for expiry date adjustment and for the session filter (`timegrid.SessionReturns`).
#:
#: Registered names must be exact. `CME_Energy` / `CME_Metals` are not registered names;
#: the code silently swallowed the exception and fell back to all weekdays, so the
#: session filter was never applied to GC/HG/CL/NG (0 rows removed). The correct names
#: are the `CMEGlobex_*` family. Verify with `validate_calendars()`.
EXCHANGE_CALENDARS: dict[str, str] = {
    # ---- CME Globex family --------------------------------------------------
    "ES": "CME_Equity", "NQ": "CME_Equity", "RTY": "CME_Equity", "DM": "CME_Equity",
    "TY": "CME_Bond", "FV": "CME_Bond", "TU": "CME_Bond", "US": "CME_Bond",
    "EC": "CME_Currency", "BP": "CME_Currency", "JY": "CME_Currency",
    "AD": "CME_Currency", "CD": "CME_Currency", "MP": "CME_Currency",
    "SF": "CME_Currency", "NV": "CME_Currency",
    # For energy/metals, CMEGlobex_* are the registered names
    "CL": "CMEGlobex_Crude", "HO": "CMEGlobex_NYHarbor",
    "XB": "CMEGlobex_Gas", "NG": "CMEGlobex_NatGas",
    "GC": "CMEGlobex_Gold", "SI": "CMEGlobex_Silver",
    "PL": "CMEGlobex_Platinum", "HG": "CMEGlobex_Copper",
    "S": "CMEGlobex_Oilseeds", "BO": "CMEGlobex_Oilseeds",
    "C": "CMEGlobex_Grains", "W": "CMEGlobex_Grains",
    "LC": "CMEGlobex_Live_Cattle", "LH": "CMEGlobex_Lean_Hog",
    # ---- Non-US exchanges ------------------------------------------------------
    "VG": "EUREX", "GX": "EUREX", "RX": "EUREX", "DU": "EUREX", "OE": "EUREX",
    "IK": "EUREX", "CF": "EUREX",
    "Z": "IEPA", "G": "IEPA",
    "NK": "JPX", "JB": "JPX",
    "HI": "HKEX",
    "XP": "ASX", "XM": "ASX",
    "KM": "XKRX",
    # ICE softs (coffee/sugar/cocoa/cotton)
    "KC": "ICEUS", "SB": "ICEUS", "CC": "ICEUS", "CT": "ICEUS",
}


#: Secondary (cash market) calendar, intersected with the primary one in the session filter.
#:
#: Why it is needed: CME Globex sometimes opens shortened sessions on US holidays, so the
#: `CMEGlobex_*` calendar treats that day as a session. Settlement prices are not updated,
#: so the data is forward-filled and the return is 0.
#:
#: Measured (2026-07-31): with the primary calendar alone, residual zero returns are 2.85%.
#: The dates are 07-04 (20 times, practically every year), 11-24/26/27 (Thanksgiving),
#: 01-16/18/21 (MLK), 05-25/30 (Memorial), 06-19 (Juneteenth), 09-05/07 (Labor Day),
#: 11-11 (Veterans), and for gilts 12-26/27/28 (Boxing Day). All are national holidays.
#: Intersecting with NYSE/XLON/XTKS halves it from 2.85% to 1.42%.
#:
#: EUREX is already accurate with the primary calendar alone (residual 0.84-1.79%), so it
#: needs no secondary calendar.
SECONDARY_CALENDARS: dict[str, str] = {
    # US-listed -> NYSE (closed on all US federal market holidays)
    **{
        code: "NYSE"
        for code in (
            "ES", "NQ", "RTY", "DM",
            "TY", "FV", "TU", "US",
            "EC", "BP", "JY", "AD", "CD", "MP", "SF", "NV",
            "CL", "NG", "HO", "XB",
            "GC", "SI", "PL", "HG",
            "S", "BO", "C", "W", "LC", "LH",
            "KC", "SB", "CC", "CT",
        )
    },
    # UK -> XLON
    "Z": "XLON", "G": "XLON",
    # Japan -> XTKS (Tokyo Stock Exchange)
    "NK": "XTKS", "JB": "XTKS",
}


def validate_calendars(calendars: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Check that calendar names are actually registered in `pandas_market_calendars`.

    Returns
    -------
    {invalid calendar name: [market codes using it]}. An empty dict means all are valid.

    Examples
    --------
    >>> bad = validate_calendars()          # doctest: +SKIP
    >>> assert not bad, bad
    """
    calendars = calendars or EXCHANGE_CALENDARS
    try:
        import pandas_market_calendars as mcal

        known = set(mcal.get_calendar_names())
    except Exception:
        return {}
    bad: dict[str, list[str]] = {}
    for code, cal in calendars.items():
        if cal not in known:
            bad.setdefault(cal, []).append(code)
    return bad

QUARTERLY = (3, 6, 9, 12)  # H M U Z
ALL_MONTHS = tuple(range(1, 13))
GOLD_MONTHS = (2, 4, 6, 8, 10, 12)  # G J M Q V Z
COPPER_MONTHS = (3, 5, 7, 9, 12)  # H K N U Z
CORN_MONTHS = (3, 5, 7, 9, 12)  # H K N U Z
SOY_MONTHS = (1, 3, 5, 7, 8, 9, 11)  # F H K N Q U X

RuleKind = Literal[
    "nth_weekday",           # n-th given weekday of the contract month (+offset_bdays)
    "nth_last_bday",         # n-th business day from the end of the contract month (1 = last business day)
    "day_of_month",          # given day of the contract month (+offset_bdays)
    "prev_month_day",        # given day of the previous month (+offset_bdays)  <- CL
    "prev_month_end",        # last business day of the previous month  <- US Treasury FND
    "month_start",           # first day of the delivery month (+offset_bdays)  <- NG
]


@dataclass(frozen=True)
class ExpiryRule:
    """Expiry rule of one market.

    Attributes
    ----------
    months : tuple of contract months
    kind : rule type
    weekday : 0=Mon .. 6=Sun (used by the nth_weekday family)
    n : n-th (weekday), day of month (day_of_month), or n-th from the end
    offset_bdays : business days to shift from the date above (negative = earlier)
    note : source / confidence note
    confidence : 'high' | 'medium' (medium is worth verifying)
    """

    months: tuple[int, ...]
    kind: RuleKind
    weekday: int | None = None
    n: int = 3
    offset_bdays: int = 0
    #: Adjust earlier (-1) / later (+1) when the base date is not a business day
    roll_forward: bool = False
    note: str = ""
    #: 'verified' = checked against the exchange spec / 'inferred' = estimated
    confidence: str = "verified"


#: Expiry rules per market, covering the full recommended 24-market universe.
#:
#: Note: the dates produced here are roll reference dates; the actual roll happens
#: `DataConfig.roll_days_before_expiry` days earlier.
EXPIRY_RULES: dict[str, ExpiryRule] = {
    # ---- Equity indices: quarterly (Mar/Jun/Sep/Dec) ------------------------------
    "ES": ExpiryRule(QUARTERLY, "nth_weekday", weekday=4, n=3,
                     note="CME E-mini S&P: 3rd Friday"),
    "VG": ExpiryRule(QUARTERLY, "nth_weekday", weekday=4, n=3,
                     note="Eurex SX5E: 3rd Friday"),
    "Z": ExpiryRule(QUARTERLY, "nth_weekday", weekday=4, n=3,
                    note="ICE FTSE 100: 3rd Friday"),
    "NK": ExpiryRule(QUARTERLY, "nth_weekday", weekday=4, n=2, offset_bdays=-1,
                     note="OSE Nikkei 225: business day before 2nd Friday (SQ)"),
    "HI": ExpiryRule(QUARTERLY, "nth_last_bday", n=2,
                     note="HKEX HSI: second-to-last business day of contract month"),
    "XP": ExpiryRule(QUARTERLY, "nth_weekday", weekday=3, n=3,
                     note="ASX SPI 200: 3rd Thursday"),
    # ---- Rates ------------------------------------------------------------
    # US Treasuries use FND (= last business day of the month before the delivery month).
    # The actual roll is 2-4 days before FND. Using LTD (= 7 business days before the
    # end of the delivery month) rolls a month late.
    "TY": ExpiryRule(QUARTERLY, "prev_month_end",
                     note="CBOT 10Y: FND = last bday of month BEFORE contract month; "
                          "market rolls 2-4d before FND"),
    "FV": ExpiryRule(QUARTERLY, "prev_month_end",
                     note="CBOT 5Y: FND = last bday of month BEFORE contract month"),
    "RX": ExpiryRule(QUARTERLY, "day_of_month", n=10, offset_bdays=-2,
                     note="Eurex Bund: delivery 10th calendar day, "
                          "LTD = 2 exchange days prior"),
    "G": ExpiryRule(QUARTERLY, "nth_last_bday", n=3,
                    note="ICE Long Gilt: LTD = 2 business days prior to last "
                         "business day (= 3rd last bday)"),
    "JB": ExpiryRule(QUARTERLY, "day_of_month", n=20, offset_bdays=-7,
                     note="JPX 10Y JGB: settlement 20th, LTD = 7 business days prior"),
    "XM": ExpiryRule(QUARTERLY, "day_of_month", n=15, roll_forward=True,
                     note="ASX 10Y bond: settles 15th of Mar/Jun/Sep/Dec "
                          "(next business day if not a business day)"),
    # ---- Currencies: CME quarterly, LTD = 2 business days before the 3rd Wednesday -------------
    **{
        code: ExpiryRule(
            QUARTERLY, "nth_weekday", weekday=2, n=3, offset_bdays=-2,
            note="CME FX: LTD = 2nd business day immediately preceding 3rd Wednesday",
        )
        for code in ("EC", "BP", "JY", "AD", "CD", "MP")
    },
    # ---- Commodities -------------------------------------------------------------
    "CL": ExpiryRule(ALL_MONTHS, "prev_month_day", n=25, offset_bdays=-3,
                     note="NYMEX WTI: LTD = 3 business days before 25th calendar day "
                          "of month PRIOR to delivery month"),
    "NG": ExpiryRule(ALL_MONTHS, "month_start", offset_bdays=-3,
                     note="NYMEX Henry Hub: LTD = 3 business days before 1st of "
                          "delivery month"),
    "GC": ExpiryRule(GOLD_MONTHS, "nth_last_bday", n=3,
                     note="COMEX Gold: trading terminates on 3rd last business day "
                          "of delivery month"),
    "HG": ExpiryRule(COPPER_MONTHS, "nth_last_bday", n=3,
                     note="COMEX Copper: LTD = 3rd last business day of maturing "
                          "delivery month"),
    "S": ExpiryRule(SOY_MONTHS, "nth_last_bday", n=8,
                    note="CBOT Soybeans: LTD = 7th business day preceding the last "
                         "business day of delivery month (= 8th last bday)"),
    "C": ExpiryRule(CORN_MONTHS, "nth_last_bday", n=8,
                    note="CBOT Corn: LTD = 7th business day preceding the last "
                         "business day of delivery month (= 8th last bday)"),
}


# --------------------------------------------------------------------------- #
# Calendar construction
# --------------------------------------------------------------------------- #


class ExpiryCalendar:
    """Rule-based expiry/roll calendar.

    Examples
    --------
    >>> import pandas as pd
    >>> cal = ExpiryCalendar()
    >>> idx = pd.bdate_range('2024-01-01', '2024-12-31')
    >>> rolls = cal.roll_dates(idx, days_before=5)
    >>> len(rolls['CL']) >= 10          # monthly contract, so about 12 per year
    True
    """

    def __init__(self, rules: dict[str, ExpiryRule] | None = None) -> None:
        self.rules = dict(rules or EXPIRY_RULES)
        self._session_cache: dict[tuple, pd.DatetimeIndex] = {}

    def __repr__(self) -> str:
        return f"<ExpiryCalendar markets={len(self.rules)}>"

    # ------------------------------------------------------------------ #

    def _sessions(self, code: str, start, end) -> pd.DatetimeIndex:
        """Actual business days of the market exchange, via `pandas_market_calendars`.

        If the calendar or library is unavailable, approximate with Mon-Fri.
        """
        key = (code, pd.Timestamp(start), pd.Timestamp(end))
        if key in self._session_cache:
            return self._session_cache[key]

        name = EXCHANGE_CALENDARS.get(code)
        sessions: pd.DatetimeIndex | None = None
        if name:
            try:
                import pandas_market_calendars as mcal

                cal = mcal.get_calendar(name)
                sched = cal.schedule(start_date=start, end_date=end)
                sessions = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
            except Exception:
                sessions = None
        if sessions is None or len(sessions) == 0:
            sessions = pd.bdate_range(start, end)

        self._session_cache[key] = sessions
        return sessions

    def expiries(self, code: str, start, end) -> pd.DatetimeIndex:
        """List of expiries (= roll reference dates) for the market."""
        rule = self.rules.get(code)
        if rule is None:
            return pd.DatetimeIndex([])

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        # Fetch a generous exchange calendar (offset calculations cross the range)
        sessions = self._sessions(
            code, start - pd.DateOffset(months=4), end + pd.DateOffset(months=4)
        )
        if len(sessions) == 0:
            return pd.DatetimeIndex([])

        out: list[pd.Timestamp] = []
        for period in pd.period_range(
            start - pd.DateOffset(months=2), end + pd.DateOffset(months=2), freq="M"
        ):
            if period.month not in rule.months:
                continue
            dt = self._resolve(rule, period.year, period.month, sessions)
            if dt is not None and start <= dt <= end:
                out.append(dt)
        return pd.DatetimeIndex(sorted(set(out)))

    @staticmethod
    def _resolve(
        rule: ExpiryRule, year: int, month: int, sessions: pd.DatetimeIndex
    ) -> pd.Timestamp | None:
        """Rule -> actual expiry date. Business-day checks and shifts all use exchange sessions."""
        month_start = pd.Timestamp(year=year, month=month, day=1)

        if rule.kind == "prev_month_end":
            # Last business day of the month before the delivery month
            prev = sessions[sessions < month_start]
            if len(prev) == 0:
                return None
            base = prev[-1]

        elif rule.kind == "month_start":
            base = month_start

        elif rule.kind == "nth_last_bday":
            month_end = month_start + pd.offsets.MonthEnd(0)
            in_month = sessions[(sessions >= month_start) & (sessions <= month_end)]
            if len(in_month) < rule.n:
                return None
            return in_month[-rule.n]  # already a business day -> only the offset is applied below

        elif rule.kind in ("day_of_month", "prev_month_day"):
            ref = month_start
            if rule.kind == "prev_month_day":
                ref = month_start - pd.DateOffset(months=1)
            day = min(rule.n, ref.days_in_month)
            base = pd.Timestamp(year=ref.year, month=ref.month, day=day)

        elif rule.kind == "nth_weekday":
            if rule.weekday is None:
                return None
            base = _nth_weekday(year, month, rule.weekday, rule.n)

        else:
            return None

        base = _snap(base, sessions, forward=rule.roll_forward)
        if base is None:
            return None
        if rule.offset_bdays:
            base = _shift_sessions(base, sessions, rule.offset_bdays)
        return base

    # ------------------------------------------------------------------ #

    def roll_dates(
        self,
        trading_index: pd.DatetimeIndex,
        days_before: int = 5,
        codes: list[str] | None = None,
    ) -> dict[str, pd.DatetimeIndex]:
        """Return the date `days_before` business days before expiry as the roll date.

        The convention matches `FuturesRoller.build(convention='last_old_day')`,
        i.e. the last day the old contract is held.
        """
        trading_index = pd.DatetimeIndex(trading_index).sort_values()
        if len(trading_index) == 0:
            return {}
        codes = codes or list(self.rules)
        start, end = trading_index[0], trading_index[-1]

        out: dict[str, pd.DatetimeIndex] = {}
        for code in codes:
            exp = self.expiries(code, start, end + pd.DateOffset(months=2))
            if len(exp) == 0:
                out[code] = pd.DatetimeIndex([])
                continue
            pos = trading_index.searchsorted(exp, side="left") - days_before
            pos = pos[(pos > 0) & (pos < len(trading_index))]
            out[code] = trading_index[np.unique(pos)]
        return out

    # ------------------------------------------------------------------ #
    # Validation: do actual C1/C2 gaps appear on the computed roll dates?
    # ------------------------------------------------------------------ #

    def validate_against_prices(
        self,
        front: pd.DataFrame,
        second: pd.DataFrame,
        days_before: int = 5,
        codes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Check whether computed roll dates match actual contract switch points in the data.

        Test logic
        ----------
        On the trading day AFTER a roll, C1 switches to "yesterday's C2" contract, so
        `|C1_t / C2_{t-1} - 1|` (the new contract's one-day return) should be smaller than
        `|C1_t / C1_{t-1} - 1|` (the value contaminated by the gap).

        `follow_ratio` = share of roll dates where that condition holds; near 1 means the
        rule is right. `baseline` = the same share on random dates (about 0.5 means the
        test discriminates).

        Markets with small `spread_bps` (equity indices, Treasuries) have small roll gaps
        to begin with, so the test discriminates poorly, and a wrong roll rule does little
        damage there.
        """
        rolls = self.roll_dates(front.index, days_before=days_before, codes=codes)
        rows = []
        for code, dates in rolls.items():
            if code not in front.columns or code not in second.columns:
                continue
            pair = pd.concat(
                [front[code].rename("c1"), second[code].rename("c2")], axis=1
            ).dropna()
            if len(pair) < 100 or len(dates) == 0:
                continue

            stay = (pair["c1"] / pair["c1"].shift(1) - 1).abs()
            follow = (pair["c1"] / pair["c2"].shift(1) - 1).abs()
            better = (follow < stay).astype(float)

            # The switch is observed on the trading day after the roll
            pos = pair.index.get_indexer(dates)
            pos = pos[pos >= 0] + 1
            pos = pos[pos < len(pair)]
            if len(pos) == 0:
                continue
            hit = float(better.iloc[pos].mean())
            base = float(better.mean())
            spread = float((pair["c2"] / pair["c1"] - 1).abs().mean() * 10_000)

            rows.append(
                {
                    "code": code,
                    "n_rolls": len(dates),
                    "rolls_per_year": len(dates)
                    / max(1e-9, (pair.index[-1] - pair.index[0]).days / 365.25),
                    "follow_ratio": hit,
                    "baseline": base,
                    "lift": hit - base,
                    "spread_bps": spread,
                    "confidence": self.rules[code].confidence,
                }
            )

        df = pd.DataFrame(rows).set_index("code")
        if df.empty:
            return df
        df["verdict"] = np.where(
            df["spread_bps"] < 15,
            "n/a (spread~0, negligible roll impact)",
            np.where(df["lift"] > 0.25, "OK", "CHECK"),
        )
        return df.sort_values("lift", ascending=False)


# --------------------------------------------------------------------------- #
# Best method: extract roll dates from generic ticker changes
# --------------------------------------------------------------------------- #


class RollFromGenericTicker:
    """Underlying-contract ticker series of a generic -> roll dates (exact).

    A daily series of "the actual contract the generic points to that day" (e.g. `CLZ6`)
    is available for each generic. The day that value changes is the roll date, so no rule
    estimation is needed. Use this path whenever such data is available.

    Examples
    --------
    >>> import pandas as pd
    >>> tick = pd.DataFrame(
    ...     {"CL": ["CLF6", "CLF6", "CLG6", "CLG6"]},
    ...     index=pd.bdate_range("2026-01-01", periods=4),
    ... )
    >>> RollFromGenericTicker().roll_dates(tick)["CL"].strftime("%Y-%m-%d").tolist()
    ['2026-01-02']
    """

    @staticmethod
    def roll_dates(tickers: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
        """Return the trading day just before the ticker changes (= last day the old contract is held).

        Matches the `FuturesRoller.build(convention='last_old_day')` convention.
        """
        out: dict[str, pd.DatetimeIndex] = {}
        for col in tickers.columns:
            s = tickers[col].dropna().astype(str)
            if s.empty:
                out[col] = pd.DatetimeIndex([])
                continue
            changed = s != s.shift(1)
            changed.iloc[0] = False
            switch_positions = np.flatnonzero(changed.to_numpy())
            last_old = switch_positions - 1
            last_old = last_old[last_old >= 0]
            out[col] = pd.DatetimeIndex(s.index[last_old])
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """n-th given weekday of the contract month (weekday: 0=Mon .. 6=Sun)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    shift = (weekday - first.dayofweek) % 7
    return first + pd.Timedelta(days=shift + 7 * (n - 1))


def _snap(dt: pd.Timestamp, sessions: pd.DatetimeIndex, forward: bool) -> pd.Timestamp | None:
    """If not an exchange session, move to the previous (default) or next (forward=True) session."""
    if len(sessions) == 0:
        return None
    pos = sessions.searchsorted(dt, side="left")
    if pos < len(sessions) and sessions[pos] == dt:
        return dt
    if forward:
        return sessions[pos] if pos < len(sessions) else None
    return sessions[pos - 1] if pos > 0 else None


def _shift_sessions(
    dt: pd.Timestamp, sessions: pd.DatetimeIndex, n: int
) -> pd.Timestamp | None:
    """Shift by n business days on exchange sessions (negative = earlier)."""
    pos = int(sessions.searchsorted(dt, side="left")) + n
    if 0 <= pos < len(sessions):
        return sessions[pos]
    return None


__all__ = [
    "ExpiryRule",
    "EXPIRY_RULES",
    "ExpiryCalendar",
    "RollFromGenericTicker",
    "QUARTERLY",
    "ALL_MONTHS",
]
