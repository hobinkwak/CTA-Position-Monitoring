"""
Universe definition: the 16 futures markets of the replication model.

The market code (Market.code) is the single source of truth across the project. Column
names in data/ and in position DataFrames all use this code.

Contract multiplier and currency are used only to convert notional to contract counts.
They are not needed for signal computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd


class Sector:
    """Sector constants: the four asset classes of the model.

    Note: the source model calls this sector "Rates", but **we label it "Bonds".**
    All members are **bond futures** (TY, G, RX, JB) and position signs follow the
    **futures price**. Calling it "Rates" would make the sign read opposite to the market
    usage of "short rates" (= long bonds), inverting the direction. Short-rate futures
    (SOFR etc.) are not in the universe. The `RATES` alias is kept for backward compatibility.
    """

    EQUITIES = "Equities"
    BONDS = "Bonds"
    RATES = BONDS          # backward-compatible alias; new code should use BONDS
    CURRENCIES = "Currencies"
    COMMODITIES = "Commodities"

    ALL: tuple[str, ...] = (EQUITIES, BONDS, CURRENCIES, COMMODITIES)


@dataclass(frozen=True)
class Market:
    """A single futures market.

    Attributes
    ----------
    code : internal project identifier (ES, Z, VG, ...).
    name : human-readable name.
    sector : one of Sector.
    bbg_generic : Bloomberg front-month generic ticker. One-letter roots include a space ("Z 1 Index").
    bbg_root : root without the generic number ("Z ", "CL").
    currency : contract currency.
    multiplier : contract multiplier (currency amount per 1 price point).
    """

    code: str
    name: str
    sector: str
    bbg_generic: str
    bbg_root: str
    currency: str
    multiplier: float
    #: Quote-unit adjustment. Contract notional = price x price_scale x multiplier
    #:
    #: In some markets the quoted price differs from the contract spec unit. Verified:
    #:   1.0    index / dollar quotes (ES, VG, NK, TY, RX, EC, CL, GC ...)
    #:   0.01   cent quotes or FX x100 (HG 647.45 -> $6.4745, BP 134.75 -> 1.3475,
    #:          AD 70.24 -> 0.7024, S 1177.25 -> $11.7725)
    #:   1e-4   JY 63.15 -> 0.006315 (yen is quoted x10000)
    #:
    #: A wrong value makes the **contract count wrong by 100-10,000x** (the JY bug was
    #: found as a 768x mismatch against CFTC data). Signals and R-squared are unaffected
    #: because only returns are used.
    price_scale: float = 1.0

    def generic(self, n: int = 1) -> str:
        """Nth generic ticker: swaps only the number in bbg_generic."""
        yellow_key = self.bbg_generic.rsplit(" ", 1)[-1]
        return f"{self.bbg_root}{n} {yellow_key}"

    @property
    def yellow_key(self) -> str:
        return self.bbg_generic.rsplit(" ", 1)[-1]


#: Model universe table. Ordered by sector, then in table order.
MARKETS: tuple[Market, ...] = (
    # ---- Equities ----------------------------------------------------------
    Market("ES", "S&P 500 E-mini", Sector.EQUITIES, "ES1 Comdty", "ES", "USD", 50.0),
    Market("Z", "FTSE 100", Sector.EQUITIES, "Z 1 Index", "Z ", "GBP", 10.0),
    Market("VG", "Euro Stoxx 50", Sector.EQUITIES, "VG1 Index", "VG", "EUR", 10.0),
    Market("NK", "Nikkei 225", Sector.EQUITIES, "NK1 Index", "NK", "JPY", 1_000.0),
    # ---- Rates -------------------------------------------------------------
    Market("TY", "US 10 Year", Sector.RATES, "TY1 Comdty", "TY", "USD", 1_000.0),
    Market("G", "Long Gilt", Sector.RATES, "G 1 Comdty", "G ", "GBP", 1_000.0),
    Market("RX", "Euro Bund", Sector.RATES, "RX1 Comdty", "RX", "EUR", 1_000.0),
    Market("JB", "Japan 10 Year", Sector.RATES, "JB1 Comdty", "JB", "JPY", 1_000_000.0),
    # ---- Currencies --------------------------------------------------------
    Market("EC", "Euro", Sector.CURRENCIES, "EC1 Curncy", "EC", "USD", 125_000.0),
    Market("BP", "British Pound", Sector.CURRENCIES, "BP1 Curncy", "BP", "USD", 62_500.0, 0.01),
    Market("JY", "Japanese Yen", Sector.CURRENCIES, "JY1 Curncy", "JY", "USD", 12_500_000.0, 0.0001),
    Market("AD", "Aussie Dollar", Sector.CURRENCIES, "AD1 Curncy", "AD", "USD", 100_000.0, 0.01),
    # ---- Commodities -------------------------------------------------------
    Market("CL", "Crude Oil", Sector.COMMODITIES, "CL1 Comdty", "CL", "USD", 1_000.0),
    Market("GC", "Gold", Sector.COMMODITIES, "GC1 Comdty", "GC", "USD", 100.0),
    Market("HG", "Copper", Sector.COMMODITIES, "HG1 Comdty", "HG", "USD", 25_000.0, 0.01),
    Market("S", "Soybeans", Sector.COMMODITIES, "S 1 Comdty", "S ", "USD", 5_000.0, 0.01),
)


#: Extended market candidates beyond the base universe.
#:
#: The base 16-market universe is much smaller than what a typical diversified trend
#: follower trades. It is a **constraint** of the model, not a recommendation. Actual
#: CTAs trade all of these, so adding them may improve the replication. The only
#: criterion is the **benchmark R-squared**.
#:
#: Caution 1, correlated duplicates: the model is naive risk parity and does not adjust
#:    for correlation. Adding silver (SI), 0.8 correlated with gold (GC), doubles the
#:    precious-metals bet. Heating oil (HO) and gasoline (XB) are about 0.9 with crude
#:    (CL). Little information is added and risk just concentrates. Low-correlation
#:    markets (natural gas, agriculture, base metals, EM currencies) give real diversification.
#:
#: Caution 2, sector balance: the base universe is **deliberately balanced** at 4/4/4/4.
#:    Adding 8 commodities alone doubles commodity sector risk and skews the sector mix.
#:    `ModelParams(sector_risk_parity=True)` fixes the risk budget per sector.
#:
#: Caution 3, verify contract multipliers against exchange specs. They are used **only
#:    for contract-count conversion**, so errors do not contaminate signals or R-squared.
EXTENDED_MARKETS: tuple[Market, ...] = (
    # ---- Equities ----------------------------------------------------------
    Market("NQ", "Nasdaq 100 E-mini", Sector.EQUITIES, "NQ1 Index", "NQ", "USD", 20.0),
    Market("RTY", "Russell 2000 E-mini", Sector.EQUITIES, "RTY1 Index", "RTY", "USD", 50.0),
    Market("GX", "DAX", Sector.EQUITIES, "GX1 Index", "GX", "EUR", 25.0),
    Market("CF", "CAC 40", Sector.EQUITIES, "CF1 Index", "CF", "EUR", 10.0),
    Market("HI", "Hang Seng", Sector.EQUITIES, "HI1 Index", "HI", "HKD", 50.0),
    Market("XP", "ASX SPI 200", Sector.EQUITIES, "XP1 Index", "XP", "AUD", 25.0),
    # ---- Rates -------------------------------------------------------------
    Market("TU", "US 2 Year", Sector.RATES, "TU1 Comdty", "TU", "USD", 2_000.0),
    Market("FV", "US 5 Year", Sector.RATES, "FV1 Comdty", "FV", "USD", 1_000.0),
    Market("US", "US Long Bond", Sector.RATES, "US1 Comdty", "US", "USD", 1_000.0),
    Market("DU", "Euro Schatz 2Y", Sector.RATES, "DU1 Comdty", "DU", "EUR", 1_000.0),
    Market("OE", "Euro Bobl 5Y", Sector.RATES, "OE1 Comdty", "OE", "EUR", 1_000.0),
    Market("IK", "Italy BTP 10Y", Sector.RATES, "IK1 Comdty", "IK", "EUR", 1_000.0),
    Market("XM", "Australia 10 Year", Sector.RATES, "XM1 Comdty", "XM", "AUD", 1_000.0),
    Market("CN", "Canada 10 Year", Sector.RATES, "CN1 Comdty", "CN", "CAD", 1_000.0),
    # ---- Currencies --------------------------------------------------------
    Market("CD", "Canadian Dollar", Sector.CURRENCIES, "CD1 Curncy", "CD", "USD", 100_000.0, 0.01),
    Market("SF", "Swiss Franc", Sector.CURRENCIES, "SF1 Curncy", "SF", "USD", 125_000.0),
    Market("NV", "New Zealand Dollar", Sector.CURRENCIES, "NV1 Curncy", "NV", "USD", 100_000.0, 0.01),
    Market("MP", "Mexican Peso", Sector.CURRENCIES, "MP1 Curncy", "MP", "USD", 500_000.0, 0.01),
    # ---- Commodities -------------------------------------------------------
    Market("SI", "Silver", Sector.COMMODITIES, "SI1 Comdty", "SI", "USD", 5_000.0, 0.01),
    Market("PL", "Platinum", Sector.COMMODITIES, "PL1 Comdty", "PL", "USD", 50.0),
    Market("CO", "Brent Crude", Sector.COMMODITIES, "CO1 Comdty", "CO", "USD", 1_000.0),
    Market("HO", "NY Harbor ULSD", Sector.COMMODITIES, "HO1 Comdty", "HO", "USD", 42_000.0, 0.01),
    Market("XB", "RBOB Gasoline", Sector.COMMODITIES, "XB1 Comdty", "XB", "USD", 42_000.0, 0.01),
    Market("NG", "Natural Gas", Sector.COMMODITIES, "NG1 Comdty", "NG", "USD", 10_000.0),
    Market("LA", "LME Aluminium", Sector.COMMODITIES, "LA1 Comdty", "LA", "USD", 25.0),
    Market("LX", "LME Zinc", Sector.COMMODITIES, "LX1 Comdty", "LX", "USD", 25.0),
    Market("C", "Corn", Sector.COMMODITIES, "C 1 Comdty", "C ", "USD", 5_000.0, 0.01),
    Market("W", "Wheat", Sector.COMMODITIES, "W 1 Comdty", "W ", "USD", 5_000.0, 0.01),
    Market("KC", "Coffee", Sector.COMMODITIES, "KC1 Comdty", "KC", "USD", 37_500.0, 0.01),
    Market("SB", "Sugar No.11", Sector.COMMODITIES, "SB1 Comdty", "SB", "USD", 112_000.0, 0.01),
    Market("CT", "Cotton", Sector.COMMODITIES, "CT1 Comdty", "CT", "USD", 50_000.0, 0.01),
    Market("LC", "Live Cattle", Sector.COMMODITIES, "LC1 Comdty", "LC", "USD", 40_000.0, 0.01),
)

#: **Recommended 8 additions**: +2 per sector (24 markets in total = 6/6/6/6).
#:
#: Only markets meeting all three criteria were chosen:
#:   1. **Low correlation** with the existing 16 (with naive risk parity, duplicates
#:      only concentrate risk)
#:   2. Traded by actual CTAs in **meaningful size** (liquidity)
#:   3. **Sector balance** is kept: no single sector is enlarged
RECOMMENDED_ADDITIONS: tuple[str, ...] = (
    "HI", "XP",    # equities    - Asia ex-Japan, Australia (NQ/RTY/GX excluded: 0.9+ with existing)
    "XM", "FV",    # bonds       - Australia 10Y (different rate cycle), US 5Y (curve dimension)
    "CD", "MP",    # currencies  - CAD (North American commodity currency), MXN (EM carry)
    "NG", "C",     # commodities - natural gas (low correlation with crude), corn (grains complex)
)

#: Extended markets with **high correlation to an existing market**, so little diversification.
#: They can be added, but do not expect an R-squared improvement (risk may just concentrate).
HIGH_CORRELATION_WITH = {
    "SI": "GC",   # silver vs gold        ~0.80
    "PL": "GC",   # platinum vs gold      ~0.65
    "CO": "CL",   # Brent vs WTI          ~0.95
    "HO": "CL",   # heating oil vs WTI    ~0.90
    "XB": "CL",   # gasoline vs WTI       ~0.88
    "NQ": "ES",   # Nasdaq vs S&P         ~0.92
    "RTY": "ES",  # Russell vs S&P        ~0.88
    "GX": "VG",   # DAX vs SX5E           ~0.95
    "CF": "VG",   # CAC vs SX5E           ~0.96
    "FV": "TY",   # 5Y vs 10Y             ~0.93
    "TU": "TY",   # 2Y vs 10Y             ~0.80
    "US": "TY",   # 30Y vs 10Y            ~0.92
    "OE": "RX",   # Bobl vs Bund          ~0.93
    "DU": "RX",   # Schatz vs Bund        ~0.80
    "W": "C",     # wheat vs corn         ~0.70
}

#: Low-correlation candidates that give real diversification (preferred additions).
LOW_CORRELATION_ADDITIONS: tuple[str, ...] = (
    "NG", "LA", "LX", "C", "KC", "SB", "CT", "LC",   # commodities
    "XM", "CN", "IK",                                 # bonds
    "CD", "SF", "NV", "MP",                           # currencies
    "HI", "XP",                                       # equities
)


class Universe:
    """Collection of markets with lookup and aggregation helpers."""

    def __init__(self, markets: Sequence[Market] = MARKETS) -> None:
        codes = [m.code for m in markets]
        if len(set(codes)) != len(codes):
            raise ValueError(f"duplicate market codes: {codes}")
        self._markets: tuple[Market, ...] = tuple(markets)
        self._by_code: dict[str, Market] = {m.code: m for m in markets}

    # -- basic container protocol ------------------------------------------ #

    def __len__(self) -> int:
        return len(self._markets)

    def __iter__(self) -> Iterator[Market]:
        return iter(self._markets)

    def __contains__(self, code: object) -> bool:
        return code in self._by_code

    def __getitem__(self, code: str) -> Market:
        return self.get(code)

    def __repr__(self) -> str:
        return f"<Universe n={len(self)} sectors={len(self.sectors)}>"

    # -- lookup ------------------------------------------------------------ #

    def get(self, code: str) -> Market:
        try:
            return self._by_code[code]
        except KeyError as exc:
            raise KeyError(f"unknown market code {code!r}; known: {self.codes}") from exc

    @property
    def markets(self) -> tuple[Market, ...]:
        return self._markets

    @property
    def codes(self) -> list[str]:
        return [m.code for m in self._markets]

    @property
    def names(self) -> list[str]:
        return [m.name for m in self._markets]

    @property
    def sectors(self) -> list[str]:
        """Sectors actually present in the universe, in Sector.ALL order."""
        present = {m.sector for m in self._markets}
        return [s for s in Sector.ALL if s in present]

    def by_sector(self, sector: str) -> list[Market]:
        return [m for m in self._markets if m.sector == sector]

    def codes_by_sector(self, sector: str) -> list[str]:
        return [m.code for m in self._markets if m.sector == sector]

    @property
    def sector_map(self) -> dict[str, str]:
        """{market_code: sector}, ready for DataFrame groupby."""
        return {m.code: m.sector for m in self._markets}

    @property
    def name_map(self) -> dict[str, str]:
        return {m.code: m.name for m in self._markets}

    def subset(self, codes: Sequence[str]) -> "Universe":
        return Universe([self.get(c) for c in codes])

    # -- tickers ----------------------------------------------------------- #

    def bbg_tickers(self, n: int = 1) -> dict[str, str]:
        """{market_code: 'CL1 Comdty'}. n=2 gives the second generic."""
        return {m.code: m.generic(n) for m in self._markets}

    # -- aggregation ------------------------------------------------------- #

    def aggregate_by_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a per-market-column DataFrame into sector totals.

        Parameters
        ----------
        df : DataFrame whose columns are market codes (e.g. positions, $ exposure)

        Returns
        -------
        DataFrame whose columns are sectors, in Sector.ALL order.
        """
        cols = [c for c in df.columns if c in self._by_code]
        if not cols:
            raise ValueError("no known market codes in df.columns")
        grouped = df[cols].T.groupby(pd.Series(self.sector_map)[cols]).sum().T
        ordered = [s for s in Sector.ALL if s in grouped.columns]
        return grouped[ordered]

    def to_frame(self) -> pd.DataFrame:
        """Universe definition table (for dashboard / documentation)."""
        return pd.DataFrame(
            [
                {
                    "code": m.code,
                    "name": m.name,
                    "sector": m.sector,
                    "bbg_generic": m.bbg_generic,
                    "currency": m.currency,
                    "multiplier": m.multiplier,
                }
                for m in self._markets
            ]
        ).set_index("code")


#: The base 16-market universe (default)
PAPER_UNIVERSE = Universe(MARKETS)

#: Base 16 plus all extended candidates (~49). Only markets with data are actually used.
FULL_UNIVERSE = Universe(MARKETS + EXTENDED_MARKETS)

#: Base 16 plus the recommended 8 = 24 markets (6/6/6/6). **Start here to expand.**
RECOMMENDED_UNIVERSE = Universe(
    MARKETS + tuple(m for m in EXTENDED_MARKETS if m.code in RECOMMENDED_ADDITIONS)
)

#: Base 16 plus all low-correlation candidates, for a wider universe.
DIVERSIFIED_UNIVERSE = Universe(
    MARKETS + tuple(m for m in EXTENDED_MARKETS if m.code in LOW_CORRELATION_ADDITIONS)
)

#: Project default universe = the base 16
UNIVERSE = PAPER_UNIVERSE

#: Universe selection by name (for CLI / comparison scripts)
UNIVERSE_PRESETS: dict[str, Universe] = {
    "paper": PAPER_UNIVERSE,              # 16 (4/4/4/4)  - base universe
    "recommended": RECOMMENDED_UNIVERSE,  # 24 (6/6/6/6)  - recommended expansion
    "diversified": DIVERSIFIED_UNIVERSE,  # 34            - all low-correlation candidates
    "full": FULL_UNIVERSE,                # 49            - everything
}


def get_universe(name: str = "paper") -> Universe:
    """Get a universe by preset name."""
    try:
        return UNIVERSE_PRESETS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown universe preset {name!r}; available: {sorted(UNIVERSE_PRESETS)}"
        ) from exc


__all__ = [
    "Sector",
    "Market",
    "MARKETS",
    "EXTENDED_MARKETS",
    "HIGH_CORRELATION_WITH",
    "LOW_CORRELATION_ADDITIONS",
    "RECOMMENDED_ADDITIONS",
    "Universe",
    "UNIVERSE",
    "PAPER_UNIVERSE",
    "RECOMMENDED_UNIVERSE",
    "DIVERSIFIED_UNIVERSE",
    "FULL_UNIVERSE",
    "UNIVERSE_PRESETS",
    "get_universe",
]
