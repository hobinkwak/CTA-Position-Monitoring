"""
CFTC Commitments of Traders, used to validate the replicated positions.

The estimated CTA positions are compared with the positions CFTC actually reports. It is
an independent check that does not depend on the AUM assumption.

Trader categories (the two reports use different schemes):

1. Disaggregated COT (commodities: CL, GC, HG, S, C, NG)
   Producer/Merchant, Swap Dealers, Managed Money, Other Reportables.
   Managed Money is the CTA/CPO proxy by definition.

2. Traders in Financial Futures (TFF) (financials: ES, NK, TY, FV, FX)
   Dealer, Asset Manager/Institutional, Leveraged Funds, Other.
   Leveraged Funds is the CTA proxy. Asset Manager is pension funds, insurers and
   mutual funds, not CTAs.

Reading the results:
- Both buckets are much broader than trend following (CTAs + CPOs + hedge funds), while
  the replica targets the 10 managers of the SG Trend Index. COT size is therefore larger
  and the implied AUM is an upper bound.
- Treasuries are dominated by basis trades (cash long + futures short), not directional
  bets, so level comparisons there are misleading. They are flagged in `BASIS_CONTAMINATED`.
- Markets not listed in the US (Z, VG, G, RX, JB, XM, HI, XP) have no CFTC data.

Timing: COT is as of Tuesday close and published on Friday. It is forward-filled onto the
weekly (Friday) grid, so each Friday sees the latest Tuesday snapshot with no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATHS, Paths

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

#: CTA-proxy category columns per report
CATEGORY_COLUMNS = {
    "disagg": ("m_money_positions_long_all", "m_money_positions_short_all"),
    "tff": ("lev_money_positions_long", "lev_money_positions_short"),
}
CATEGORY_LABEL = {
    "disagg": "Managed Money",
    "tff": "Leveraged Funds",
}

#: Treasury futures: Leveraged Funds is contaminated by basis trades, so levels are not comparable
BASIS_CONTAMINATED = {"TY", "FV", "TU", "US"}


@dataclass(frozen=True)
class CFTCMarket:
    """Mapping between our market code and the CFTC market name.

    `market_names` is plural because CFTC renamed contracts around February 2022. With an
    exact match on one name the history is cut at that point (observed: TY/FV/BP/HG only
    from 2022-02, NG only up to 2022-02), so both old and new names are listed and joined.
    """

    code: str
    report: str  # 'disagg' | 'tff'
    market_names: tuple[str, ...]  # new -> old order; looked up by exact match
    note: str = ""

    @property
    def category(self) -> str:
        return CATEGORY_LABEL[self.report]


#: Names must match exactly. A partial match picks up MICRO/MINI/cross contracts (initially
#: MICRO E-mini S&P, `EURO FX/JAPANESE YEN XRATE`, MINI Soybeans and ICE Europe WTI were
#: picked by mistake). When adding a market, the contract with the largest open interest
#: (OI) is the main contract.
CFTC_MARKETS: tuple[CFTCMarket, ...] = (
    # ---- Financial futures: Leveraged Funds ----------------------------------------
    CFTCMarket("ES", "tff", (
        "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
        "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    )),
    CFTCMarket("NQ", "tff", (
        "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE",
        "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE",
    )),
    CFTCMarket("NK", "tff", (
        "NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE",
    )),
    CFTCMarket("TY", "tff", (
        "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
        "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    ), note="Basis trades dominate; levels not comparable"),
    CFTCMarket("FV", "tff", (
        "UST 5Y NOTE - CHICAGO BOARD OF TRADE",
        "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    ), note="Basis trades dominate; levels not comparable"),
    CFTCMarket("TU", "tff", (
        "UST 2Y NOTE - CHICAGO BOARD OF TRADE",
        "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    ), note="Basis trades dominate; levels not comparable"),
    CFTCMarket("US", "tff", (
        "UST BOND - CHICAGO BOARD OF TRADE",
        "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
    ), note="Basis trades dominate; levels not comparable"),
    CFTCMarket("EC", "tff", ("EURO FX - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("BP", "tff", (
        "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
        "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    )),
    CFTCMarket("JY", "tff", ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("AD", "tff", ("AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("CD", "tff", ("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("SF", "tff", ("SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("NV", "tff", ("NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("MP", "tff", ("MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE",)),
    # ---- Commodities: Managed Money ----------------------------------------------
    CFTCMarket("GC", "disagg", ("GOLD - COMMODITY EXCHANGE INC.",)),
    CFTCMarket("SI", "disagg", ("SILVER - COMMODITY EXCHANGE INC.",)),
    CFTCMarket("HG", "disagg", (
        "COPPER- #1 - COMMODITY EXCHANGE INC.",
        "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
    )),
    CFTCMarket("S", "disagg", ("SOYBEANS - CHICAGO BOARD OF TRADE",)),
    CFTCMarket("C", "disagg", ("CORN - CHICAGO BOARD OF TRADE",)),
    CFTCMarket("W", "disagg", (
        "WHEAT-SRW - CHICAGO BOARD OF TRADE",
        "WHEAT - CHICAGO BOARD OF TRADE",
    )),
    # WTI: the main NYMEX contract in 2026 is 'WTI-PHYSICAL' (OI 1.86M). ICE Europe is separate.
    CFTCMarket("CL", "disagg", (
        "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE",
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
    )),
    # Henry Hub: main contract is 'NAT GAS NYME' (OI 1.67M), old name 'NATURAL GAS'
    CFTCMarket("NG", "disagg", (
        "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
    )),
    CFTCMarket("HO", "disagg", (
        "NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE",
        "#2 HEATING OIL, NY HARBOR - NEW YORK MERCANTILE EXCHANGE",
    )),
    CFTCMarket("XB", "disagg", ("GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE",)),
    CFTCMarket("KC", "disagg", ("COFFEE C - ICE FUTURES U.S.",)),
    CFTCMarket("SB", "disagg", ("SUGAR NO. 11 - ICE FUTURES U.S.",)),
    CFTCMarket("CT", "disagg", ("COTTON NO. 2 - ICE FUTURES U.S.",)),
    CFTCMarket("CC", "disagg", ("COCOA - ICE FUTURES U.S.",)),
    CFTCMarket("LC", "disagg", ("LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE",)),
    CFTCMarket("LH", "disagg", ("LEAN HOGS - CHICAGO MERCANTILE EXCHANGE",)),
)

#: Outside CFTC jurisdiction (not US-listed): no data exists
NOT_COVERED = ("Z", "VG", "G", "RX", "JB", "XM", "HI", "XP", "MP", "NV",
               "GX", "CF", "IB", "KM", "QC", "ST", "DU", "OE", "IK", "ER", "IR")



# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #


class CFTCStore:
    """Read-only access to the local COT history `data/external/cftc_positions.csv`.

    The file is wide: first column `date`, then one column per market code holding the
    net position of the CTA proxy bucket (see `CFTC_MARKETS`). This repository does not
    download anything; supply the file yourself (see README).

    Examples
    --------
    >>> net = CFTCStore().load()            # doctest: +SKIP
    >>> net.tail()                          # doctest: +SKIP
    """

    def __init__(self, paths: Paths = PATHS) -> None:
        self.paths = paths
        self.cache = paths.data / "external" / "cftc_positions.csv"

    def read_cache(self) -> pd.DataFrame | None:
        """Return the local history, or None if the file does not exist."""
        if not self.cache.exists():
            return None
        df = pd.read_csv(self.cache, index_col=0, parse_dates=[0])
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df.sort_index()

    def load(self, verbose: bool = True) -> pd.DataFrame:
        cached = self.read_cache()
        if cached is None:
            raise FileNotFoundError(f"CFTC history not found: {self.cache}")
        if verbose:
            print(f"  CFTC history: {cached.index[-1]:%Y-%m-%d}, {len(cached):,} weeks")
        return cached


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass
class CFTCComparison:
    """Replica-estimated contracts vs actual CFTC positions."""

    table: pd.DataFrame  # per-market summary
    ours: pd.DataFrame  # our estimated contracts (weekly grid)
    theirs: pd.DataFrame  # CFTC net contracts (aligned to the weekly grid)
    implied_aum_bn: float  # median of per-market implied AUM
    implied_aum_dispersion: float  # IQR / median (1 or more means identification failed)
    aum_assumed_bn: float
    #: Lead-lag test (lag<0 = we lead)
    lead_lag: pd.DataFrame = None  # type: ignore[assignment]
    #: Position extremeness (crowding) percentile (default 5 years)
    crowding: pd.DataFrame = None  # type: ignore[assignment]
    #: {window years: crowding_table}, for the dashboard selector
    crowding_by_window: dict = None  # type: ignore[assignment]
    #: Rolling sign agreement rate
    rolling_agree: pd.Series = None  # type: ignore[assignment]
    #: Report date of the displayed values (Tuesday close). The snapshot the last Friday of
    #: our weekly grid sees = the forward-filled actual COT date. The as-of date of the
    #: table and crowding numbers.
    asof: pd.Timestamp | None = None
    #: Most recent report date present in the source. If later than `asof`, our price
    #: grid is shorter and a release is not yet reflected.
    cot_last: pd.Timestamp | None = None
    #: Last date of our weekly (Friday) grid
    grid_last: pd.Timestamp | None = None
    #: Per-market report date; catches markets whose mapping broke
    asof_by_market: pd.Series = None  # type: ignore[assignment]
    #: Last modification time of the local file (cftc_positions.csv)
    fetched_at: pd.Timestamp | None = None

    @property
    def released(self) -> pd.Timestamp | None:
        """Release date of the `asof` report; Tuesday data is published that Friday at 15:30 ET."""
        return None if self.asof is None else self.asof + pd.Timedelta(days=3)

    @property
    def stale_weeks(self) -> int:
        """How many weeks the latest source release is ahead of the displayed values (0 = current)."""
        if self.asof is None or self.cot_last is None:
            return 0
        return int(round((self.cot_last - self.asof).days / 7))

    def asof_line(self) -> str:
        """One-line as-of description for the dashboard/log."""
        if self.asof is None:
            return "As-of date unknown"
        s = (f"Report as of {self.asof:%Y-%m-%d} (Tue) close · "
             f"released {self.released:%Y-%m-%d} (Fri)")
        if self.stale_weeks:
            s += f" · latest in source {self.cot_last:%Y-%m-%d} ({self.stale_weeks} releases not reflected)"
        return s

    @property
    def best_lead_lag(self) -> tuple[int, float]:
        """Lag with the highest median correlation, and that value."""
        if self.lead_lag is None or self.lead_lag.empty:
            return 0, float("nan")
        s = self.lead_lag["median"]
        return int(s.idxmax()), float(s.max())

    def summary(self) -> str:
        t = self.table
        usable = t[~t["basis_contaminated"]].dropna(subset=["corr_change"])
        L = ["=" * 74, "CFTC comparison (replica estimate vs reported)", "=" * 74]
        L.append(f"  {self.asof_line()}")
        L.append(f"  Markets covered : {len(t)} (within CFTC jurisdiction)")
        if len(usable):
            L.append(
                f"  Sign agreement  : {usable['sign_agree'].mean():.0%} "
                f"(50% if random)"
            )
            L.append(
                f"  Weekly change corr : median {usable['corr_change'].median():+.3f}  "
                f"mean {usable['corr_change'].mean():+.3f}"
            )
            L.append(
                f"  Level corr      : median {usable['corr_level'].median():+.3f}"
            )
        L.append(
            f"  Implied AUM (median) : ${self.implied_aum_bn:,.0f}BN "
            f"· assumed ${self.aum_assumed_bn:,.0f}BN"
        )
        disp = self.implied_aum_dispersion
        if np.isfinite(disp):
            verdict = "not identified (dispersion across markets too high)" if disp > 0.8 else "usable as reference"
            L.append(f"    Dispersion {disp:.2f} (IQR/median) -> {verdict}")
        if "implied_aum_bn" in self.table.columns:
            s = self.table["implied_aum_bn"].dropna().sort_values()
            if len(s):
                L.append(
                    f"    Range ${s.iloc[0]:,.0f}BN ({s.index[0]}) ~ "
                    f"${s.iloc[-1]:,.0f}BN ({s.index[-1]})"
                )
        L.append("    -> The COT bucket is broader than trend following, so this is an upper bound; in")
        L.append("      low-volatility asset classes (FX, rates) our notional is too large, so dispersion is high (1/sigma effect of naive risk parity)")
        lag, val = self.best_lead_lag
        if np.isfinite(val):
            same = self.lead_lag.loc[0, "median"] if 0 in self.lead_lag.index else np.nan
            L.append(
                f"  Lead-lag        : median corr {val:+.3f} at lag={lag:+d}w "
                f"(contemporaneous {same:+.3f})"
            )
            L.append(
                "    -> " + ("The replica leads COT: Tuesday COT data is only published "
                            "on Friday, a practical 1-week edge"
                            if lag < 0 else "no lead advantage")
            )
        if self.crowding is not None and not self.crowding.empty:
            ext = self.crowding[self.crowding["crowding"].str.startswith("Extreme")]
            if len(ext):
                items = ", ".join(
                    f"{c} {r['crowding']}({r['pct_win']:.0%})"
                    for c, r in ext.iterrows()
                )
                L.append(f"  Crowding alert  : {items}")
        if self.rolling_agree is not None and len(self.rolling_agree):
            L.append(
                f"  Rolling agreement : latest {self.rolling_agree.iloc[-1]:.0%} "
                f"(2-year window, min {self.rolling_agree.min():.0%})"
            )
        bad = t.index[t["basis_contaminated"]].tolist()
        if bad:
            L.append(f"  Excluded for basis contamination: {bad} (Treasuries, not directional bets)")
        return "\n".join(L)


def lead_lag_table(
    ours: pd.DataFrame, theirs: pd.DataFrame, codes: list[str], max_lag: int = 4
) -> pd.DataFrame:
    """Whether our flow estimate leads COT changes.

    `lag < 0` = we lead (our delta predicts the later COT delta).

    Measured (2026-07-31, paper 16-market universe): median correlation +0.273 at
    lag = -1, 3x the contemporaneous +0.096. By market: HG +0.535, S +0.521, GC +0.493,
    CL +0.435.

    Caveat: the timing windows partly overlap (our Friday grid vs COT Tuesday close).
    The practical meaning is still clear: Tuesday COT data is published only that Friday,
    so our estimate is available about 1 week before publication. This confirms the
    paper's "no lag" claim with independent data.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        rec: dict[str, float] = {"lag": lag}
        for c in codes:
            a, b = ours[c].diff(), theirs[c].diff()
            x = a.shift(-lag) if lag < 0 else a
            y = b if lag < 0 else b.shift(lag)
            j = pd.concat([x, y], axis=1).dropna()
            rec[c] = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 100 else np.nan
        rec["median"] = float(np.nanmedian([rec[c] for c in codes]))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("lag")


#: Candidate windows (years) for the crowding percentile.
#: 25 and 30 years are impossible: the Disaggregated/TFF reports start in 2006. Before that
#: only the legacy Non-Commercial bucket exists, which lumps all speculators into a much
#: broader bucket and cannot proxy CTAs.
#: 1 and 3 years ask "is it extreme in the recent regime", 5+ years ask "is it structurally extreme".
CROWDING_WINDOW_YEARS = (1, 3, 5, 10, 15, 20)


def available_windows(theirs: pd.DataFrame) -> dict[int, int]:
    """Keep only the {years: weeks} windows the data length allows."""
    n = int(theirs.notna().any(axis=1).sum())
    out = {y: y * 52 for y in CROWDING_WINDOW_YEARS if y * 52 <= n}
    return out or {max(1, n // 52): n}


def crowding_table(
    theirs: pd.DataFrame, positions: pd.Series, codes: list[str], window: int = 260
) -> pd.DataFrame:
    """Position extremeness (crowding) percentile: information our model cannot give.

    We do not know the total market size, so we cannot say how crowded a market is now.
    COT gives actual contract counts, so a historical percentile can be computed.

    Observed: EC 5-year percentile 0% (5-year max short), JY 3%, HG 94% (extreme long).
    Our model is short EUR/JPY and COT is also at a 5-year max short, a crowding risk.
    Extreme positions are vulnerable to squeezes.

    Changing ``window`` changes the verdict: extreme over 5 years but ordinary over 20
    means the crowding is a product of the recent regime. Hence the dashboard window selector.
    """
    rows = []
    for c in codes:
        s = theirs[c].dropna()
        if len(s) < 100:
            continue
        now = float(s.iloc[-1])
        p_win = float((s.iloc[-window:] < now).mean())
        p_all = float((s < now).mean())
        rows.append({
            "code": c,
            "cftc_net": now,
            "pct_win": p_win,
            "pct_all": p_all,
            "our_position": float(positions.get(c, np.nan)),
            "crowding": ("Extreme long" if p_win > 0.9 else "Extreme short" if p_win < 0.1
                         else "High long" if p_win > 0.75 else "High short" if p_win < 0.25
                         else "Neutral"),
        })
    return pd.DataFrame(rows).set_index("code")


def rolling_agreement(
    ours: pd.DataFrame, theirs: pd.DataFrame, codes: list[str], window: int = 104
) -> pd.Series:
    """Rolling sign agreement rate: a second health check independent of benchmark R-squared.

    The benchmark (SG Trend Index) only gives returns while COT gives positions. The two
    validate the model in different ways, so they are best read together.
    """
    agree = pd.DataFrame(
        {c: (np.sign(ours[c]) == np.sign(theirs[c])).astype(float) for c in codes}
    ).mean(axis=1)
    return agree.rolling(window, min_periods=window // 2).mean().dropna()


class CFTCValidator:
    """Compare our estimated contract counts with CFTC.

    Our contract counts come from `ExposureEstimator.contracts()`, which needs raw
    front-month (C1) prices, contract multipliers and FX. Without them no comparison is possible.
    """

    def __init__(self, result, store: CFTCStore | None = None) -> None:
        self.r = result
        self.store = store or CFTCStore(result.exposure.paths
                                        if hasattr(result.exposure, "paths")
                                        else PATHS)

    def run(
        self, verbose: bool = True, cot: pd.DataFrame | None = None,
    ) -> CFTCComparison | None:
        r = self.r
        if r.front_prices is None:
            if verbose:
                print("  No raw front-month (C1) prices, cannot convert to contracts -> skipping CFTC comparison")
            return None

        try:
            if cot is None:
                cot = self.store.load(verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"  CFTC data load failed: {type(exc).__name__}: {exc}")
            return None

        ours = r.exposure.contracts(r.front_prices, r.fx).dropna(how="all")
        codes = [c for c in ours.columns if c in cot.columns]
        if not codes:
            if verbose:
                print("  No overlapping markets")
            return None

        # COT (Tuesday) -> our weekly grid (Friday). Each Friday sees the latest Tuesday.
        theirs = cot[codes].reindex(ours.index, method="ffill")
        ours = ours[codes]

        # -- As-of date -------------------------------------------------------
        # The displayed numbers are the last row of `theirs`, which is the Tuesday release
        # the last Friday picked up via ffill. It can differ from the latest release in the
        # source (if the price grid is shorter, the latest release is not visible yet), so
        # both are carried separately.
        grid_last = ours.index[-1]
        cot_use = cot[codes]
        seen = cot_use.loc[:grid_last].dropna(how="all")
        asof = seen.index[-1] if len(seen) else None
        full = cot_use.dropna(how="all")
        cot_last = full.index[-1] if len(full) else None
        asof_by_market = pd.Series({
            c: (cot_use[c].loc[:grid_last].dropna().index[-1]
                if cot_use[c].loc[:grid_last].notna().any() else pd.NaT)
            for c in codes
        })

        rows = []
        for c in codes:
            a = ours[c]
            b = theirs[c]
            j = pd.concat([a.rename("ours"), b.rename("theirs")], axis=1).dropna()
            if len(j) < 60:
                continue
            da, db = j["ours"].diff(), j["theirs"].diff()
            chg = pd.concat([da, db], axis=1).dropna()
            m = next(x for x in CFTC_MARKETS if x.code == c)
            rows.append({
                "code": c,
                "category": m.category,
                "basis_contaminated": c in BASIS_CONTAMINATED,
                "weeks": len(j),
                "ours_now": float(a.dropna().iloc[-1]) if a.notna().any() else np.nan,
                "cftc_now": float(b.dropna().iloc[-1]) if b.notna().any() else np.nan,
                "sign_agree": float(
                    (np.sign(j["ours"]) == np.sign(j["theirs"])).mean()
                ),
                "corr_level": float(j["ours"].corr(j["theirs"])),
                "corr_change": float(chg.iloc[:, 0].corr(chg.iloc[:, 1]))
                if len(chg) > 30 else np.nan,
                "scale": float(j["theirs"].abs().mean() / max(j["ours"].abs().mean(), 1e-9)),
                "note": m.note,
            })

        table = pd.DataFrame(rows).set_index("code")
        implied, r2 = self._implied_aum(ours, theirs, table)

        # Extra analyses exclude basis-contaminated markets
        clean = [c for c in codes if c not in BASIS_CONTAMINATED]
        pos = r.ensemble.positions.dropna(how="all").iloc[-1]
        return CFTCComparison(
            table=table.sort_values("corr_change", ascending=False),
            ours=ours, theirs=theirs,
            implied_aum_bn=implied, implied_aum_dispersion=r2,
            aum_assumed_bn=r.exposure.aum / 1e9,
            lead_lag=lead_lag_table(ours, theirs, clean),
            crowding=crowding_table(theirs, pos, clean),
            # Precompute per window; the dashboard swaps them via the selector
            crowding_by_window={
                yrs: crowding_table(theirs, pos, clean, window=w)
                for yrs, w in available_windows(theirs).items()
            },
            rolling_agree=rolling_agreement(ours, theirs, clean),
            asof=asof, cot_last=cot_last, grid_last=grid_last,
            asof_by_market=asof_by_market, fetched_at=self._cache_mtime(),
        )

    def _cache_mtime(self) -> pd.Timestamp | None:
        """Last modification time of the local file (None if absent)."""
        try:
            return pd.Timestamp(self.store.cache.stat().st_mtime, unit="s")
        except OSError:
            return None

    def _implied_aum(self, ours, theirs, table) -> tuple[float, float]:
        """Median of per-market implied AUM, and dispersion.

        Implied AUM of market c = (CFTC contracts / our contracts) x assumed AUM.

        A pooled regression is not used: it was tried and failed (R-squared 0.03) because
        per-market ratios are scattered between 0.0x and 6.5x and a single slope cannot fit.
        The dispersion itself is informative: our model over-allocates notional to
        low-volatility asset classes (FX, rates) because naive risk parity scales by 1/sigma.
        It fits commodities reasonably and deviates strongly in FX.

        Returns
        -------
        (median implied AUM in $BN, IQR / median = dispersion)
        """
        use = [c for c in ours.columns if c in table.index
               and not table.loc[c, "basis_contaminated"]]
        if not use:
            return float("nan"), float("nan")

        per_market = {}
        for c in use:
            j = pd.concat([ours[c].rename("o"), theirs[c].rename("t")], axis=1).dropna()
            if len(j) < 60:
                continue
            denom = j["o"].abs().mean()
            if denom < 1:
                continue
            per_market[c] = j["t"].abs().mean() / denom * self.r.exposure.aum / 1e9

        if not per_market:
            return float("nan"), float("nan")
        s = pd.Series(per_market)
        table["implied_aum_bn"] = s.reindex(table.index)
        med = float(s.median())
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        return med, (iqr / med if med else float("nan"))


__all__ = [
    "CFTC_MARKETS",
    "CFTCMarket",
    "CFTCStore",
    "CFTCValidator",
    "CFTCComparison",
    "BASIS_CONTAMINATED",
    "CATEGORY_LABEL",
    "NOT_COVERED",
]
