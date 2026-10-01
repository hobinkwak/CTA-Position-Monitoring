"""
Exposure estimation: % notional, $ notional, contract counts and trade flow.

Exposure is expressed as a percentage per $1 notional allocated to the benchmark,
aggregated by sector (+100% equity exposure approximates a $1 long position per dollar
invested in the benchmark). Assuming a total AUM mimicking the benchmark, the replication
output gives position sizes by market.

Flow sensitivity (`flow_sensitivity`) is an **extension** beyond the replication model
and is labelled as such on the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RUN_CONFIG, RunConfig
from .signals import NormalizedMomentum
from .universe import UNIVERSE, Universe
from .volatility import VolatilityPanel

#: FX column name -> (base, quote), used for USD conversion.
_FX_PAIRS = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
}


@dataclass(frozen=True)
class ExposureSnapshot:
    """Positioning snapshot at a given date."""

    date: pd.Timestamp
    by_market: pd.DataFrame  # index=code, columns=[pct, usd, sector, name, ...]
    by_sector: pd.Series  # % notional by sector
    gross: float
    net: float
    aum: float

    def summary(self) -> str:
        lines = [
            f"as of {self.date:%Y-%m-%d}   assumed AUM ${self.aum / 1e9:,.0f}BN",
            f"  gross={self.gross:,.1%}  net={self.net:,.1%}",
            "  Notional by sector:",
        ]
        for sector, pct in self.by_sector.items():
            lines.append(f"    {sector:<12} {pct:>+8.1%}   ${pct * self.aum / 1e9:>+8.1f}BN")
        return "\n".join(lines)


class ExposureEstimator:
    """Replicated positions -> reportable exposure.

    Parameters
    ----------
    positions : notional weight DataFrame (index=weekly grid, columns=market code)
    universe : source of sector / contract multiplier information
    aum : assumed total trend-following AUM
    """

    def __init__(
        self,
        positions: pd.DataFrame,
        universe: Universe = UNIVERSE,
        aum: float | None = None,
        run_config: RunConfig = RUN_CONFIG,
    ) -> None:
        self.positions = positions
        self.universe = universe
        self.cfg = run_config
        self.aum = float(aum if aum is not None else run_config.aum_usd)

    def __repr__(self) -> str:
        return f"<ExposureEstimator n={len(self.positions)} aum=${self.aum/1e9:,.0f}BN>"

    # ------------------------------------------------------------------ #
    # % notional
    # ------------------------------------------------------------------ #

    def pct_by_market(self) -> pd.DataFrame:
        """% notional exposure per market (per $1 of benchmark)."""
        return self.positions

    def pct_by_sector(self) -> pd.DataFrame:
        """% notional by sector (replication exposure by sector)."""
        return self.universe.aggregate_by_sector(self.positions.fillna(0.0))

    def gross_exposure(self) -> pd.Series:
        return self.positions.abs().sum(axis=1).rename("gross")

    def net_exposure(self) -> pd.Series:
        return self.positions.sum(axis=1).rename("net")

    # ------------------------------------------------------------------ #
    # Risk contribution: can point the opposite way from notional
    # ------------------------------------------------------------------ #

    def risk_contribution(self, vol_annualized: pd.DataFrame) -> pd.DataFrame:
        """Per-market risk contribution = |w| x sigma_ann.

        Why it is needed
        ----------------
        The model uses `w = m / sigma_ann`, so **notional concentrates in low-volatility
        markets.** Observed (2026-07-31): rates carry 34.8% of notional but only 19.7%
        of risk, while equities are 22.1% of notional and 35.8% of risk. Individually,
        crude oil is 1.2% of notional but 7.2% of risk, the same risk as US Treasuries
        (15.8% of notional).

        Reading "rates -106%" as the dominant bet is therefore a **misreading**. Risk
        contribution is closer to the idea that each market gets an equal weight once
        adjusted for volatility.

        Note: correlations are not reflected (same naive risk-parity view as the model),
        so risk within a block (e.g. 4 USD crosses, 4 duration markets) is **summed and
        overstated**. Covariances are needed to see the true diversification effect.
        """
        vol = vol_annualized.reindex(index=self.positions.index, columns=self.positions.columns)
        return (self.positions.abs() * vol).dropna(how="all")

    def risk_share(self, vol_annualized: pd.DataFrame) -> pd.DataFrame:
        """Risk contribution normalised to sum to 1."""
        rc = self.risk_contribution(vol_annualized)
        total = rc.sum(axis=1).replace(0, np.nan)
        return rc.div(total, axis=0)

    def risk_by_sector(self, vol_annualized: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """Risk contribution by sector (default: normalised to shares)."""
        rc = self.risk_share(vol_annualized) if normalize else self.risk_contribution(
            vol_annualized
        )
        return self.universe.aggregate_by_sector(rc.fillna(0.0))

    def notional_vs_risk(self, vol_annualized: pd.DataFrame) -> pd.DataFrame:
        """Latest notional share vs risk share, per market."""
        rc = self.risk_contribution(vol_annualized).dropna(how="all")
        if rc.empty:
            return pd.DataFrame()
        dt = rc.index[-1]
        pos = self.positions.loc[dt]
        vol = vol_annualized.reindex(columns=self.positions.columns).loc[dt]
        risk = rc.loc[dt]

        out = pd.DataFrame(
            {
                "name": pd.Series(self.universe.name_map),
                "sector": pd.Series(self.universe.sector_map),
                "position": pos,
                "vol_ann": vol,
                "notional_share": pos.abs() / pos.abs().sum(),
                "risk_share": risk / risk.sum(),
            }
        ).loc[[c for c in self.universe.codes if c in pos.index]].dropna(subset=["position"])
        out["ratio"] = out["risk_share"] / out["notional_share"].replace(0, np.nan)
        out.attrs["as_of"] = dt
        return out

    # ------------------------------------------------------------------ #
    # $ notional
    # ------------------------------------------------------------------ #

    def usd_by_market(self, aum: float | None = None) -> pd.DataFrame:
        """Dollar notional per market (estimated dollar exposure)."""
        return self.positions * float(aum if aum is not None else self.aum)

    def usd_by_sector(self, aum: float | None = None) -> pd.DataFrame:
        return self.pct_by_sector() * float(aum if aum is not None else self.aum)

    # ------------------------------------------------------------------ #
    # Contract counts
    # ------------------------------------------------------------------ #

    def contracts(
        self,
        front_prices: pd.DataFrame,
        fx: pd.DataFrame | None = None,
        aum: float | None = None,
    ) -> pd.DataFrame:
        """Dollar notional -> number of contracts.

        Pass **raw front-month prices**. Back-adjusted rolled prices are not actual
        traded prices, so the contract counts would be wrong.

            contracts = $ notional / (raw price x contract multiplier x USD rate)

        Parameters
        ----------
        front_prices : raw front-month prices (columns = market code)
        fx : USD conversion rates. Example columns 'EURUSD', 'GBPUSD', 'USDJPY'. If
             None, 1.0 is assumed (accurate only for USD-quoted markets).
        """
        usd = self.usd_by_market(aum)
        px = front_prices.reindex(usd.index, method="ffill").reindex(columns=usd.columns)
        rates = self._usd_rates(usd.index, fx)

        # Omitting price_scale makes contract counts wrong by 100-10,000x.
        # Some markets quote in units different from the contract spec (cents, FX x100/x10000).
        # Found when JY was off by 768x against CFTC data.
        effective = pd.Series(
            {m.code: m.multiplier * m.price_scale for m in self.universe}, dtype=float
        ).reindex(usd.columns)
        currencies = pd.Series({m.code: m.currency for m in self.universe}).reindex(usd.columns)

        contract_value_usd = px.multiply(effective, axis=1)
        for code in usd.columns:
            ccy = currencies[code]
            contract_value_usd[code] = contract_value_usd[code] * rates[ccy]

        return (usd / contract_value_usd.replace(0, np.nan)).round(0)

    def _usd_rates(self, index: pd.DatetimeIndex, fx: pd.DataFrame | None) -> pd.DataFrame:
        """{currency: USD rate} time series. Multiplying by rate[ccy] gives USD."""
        currencies = sorted({m.currency for m in self.universe})
        rates = pd.DataFrame(1.0, index=index, columns=currencies)
        if fx is None:
            return rates

        fx = fx.reindex(index, method="ffill")
        for col in fx.columns:
            pair = _FX_PAIRS.get(col.upper())
            if pair is None:
                continue
            base, quote = pair
            if quote == "USD" and base in rates.columns:
                rates[base] = fx[col]  # e.g. EURUSD -> 1 EUR = x USD
            elif base == "USD" and quote in rates.columns:
                rates[quote] = 1.0 / fx[col].replace(0, np.nan)  # USDJPY -> JPY -> USD
        rates["USD"] = 1.0
        return rates

    # ------------------------------------------------------------------ #
    # Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self, date: pd.Timestamp | None = None) -> ExposureSnapshot:
        """Positioning summary at a given date (default: latest)."""
        valid = self.positions.dropna(how="all")
        if valid.empty:
            raise ValueError("positions has no valid rows.")
        dt = valid.index[-1] if date is None else pd.Timestamp(date)
        if dt not in valid.index:
            dt = valid.index[valid.index <= dt][-1]

        row = valid.loc[dt].fillna(0.0)
        prev = valid.loc[:dt].iloc[-2].fillna(0.0) if len(valid.loc[:dt]) > 1 else row * 0

        by_market = pd.DataFrame(
            {
                "name": pd.Series(self.universe.name_map),
                "sector": pd.Series(self.universe.sector_map),
                "pct": row,
                "usd": row * self.aum,
                "pct_change_wow": row - prev,
                "usd_flow_wow": (row - prev) * self.aum,
            }
        ).loc[[c for c in self.universe.codes if c in row.index]]
        by_market["side"] = np.where(
            by_market["pct"] > 0, "Long", np.where(by_market["pct"] < 0, "Short", "Flat")
        )

        by_sector = by_market.groupby("sector")["pct"].sum()
        by_sector = by_sector.reindex(
            [s for s in self.universe.sectors if s in by_sector.index]
        )

        return ExposureSnapshot(
            date=dt,
            by_market=by_market,
            by_sector=by_sector,
            gross=float(row.abs().sum()),
            net=float(row.sum()),
            aum=self.aum,
        )

    # ------------------------------------------------------------------ #
    # Flow (realised weekly)
    # ------------------------------------------------------------------ #

    def weekly_flow_pct(self) -> pd.DataFrame:
        """Weekly position change (delta w): the size (as a ratio) CTAs would have traded."""
        return self.positions.diff()

    def weekly_flow_usd(self, aum: float | None = None) -> pd.DataFrame:
        """Weekly traded notional ($)."""
        return self.weekly_flow_pct() * float(aum if aum is not None else self.aum)

    def turnover(self) -> pd.Series:
        """Weekly gross turnover, sum of |delta w|."""
        return self.weekly_flow_pct().abs().sum(axis=1).rename("turnover")

    # ------------------------------------------------------------------ #
    # Flow sensitivity (extension beyond the replication model)
    # ------------------------------------------------------------------ #

    def flow_sensitivity(
        self,
        signal_engines: list[NormalizedMomentum],
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        vol_panels: dict[tuple[int, str], VolatilityPanel] | None = None,
        leverage: float = 1.0,
        ensemble_weights: dict[str, float] | None = None,
        shocks: tuple[float, ...] | None = None,
        aum: float | None = None,
    ) -> pd.DataFrame:
        """If next week's price moves X%, how much must CTAs buy or sell.

        Assumes the same % shock independently in each market, recomputes signals and
        positions, and converts the difference from the current position into traded
        notional.

        Returns
        -------
        index = market code, columns = MultiIndex(shock, ['pct','usd'])
        """
        shocks = tuple(shocks if shocks is not None else self.cfg.flow_shocks)
        aum = float(aum if aum is not None else self.aum)
        weights = ensemble_weights or {
            eng.params.label: 1.0 / len(signal_engines) for eng in signal_engines
        }

        current = self.positions.dropna(how="all").iloc[-1].fillna(0.0)

        frames: dict[float, pd.DataFrame] = {}
        for shock in shocks:
            blended = pd.Series(0.0, index=weekly_returns.columns, dtype=float)
            for engine in signal_engines:
                panel = None
                if vol_panels is not None:
                    panel = vol_panels.get(
                        (engine.params.volatility_lookback, engine.params.vol_method)
                    )
                shocked_mom = engine.compute_shocked(
                    weekly_returns, daily_returns, shock, panel
                ).iloc[0]
                sigma_ann = (
                    (panel or engine.vol_estimator.compute(daily_returns))
                    .at(pd.DatetimeIndex([weekly_returns.index[-1]]))
                    .annualized.iloc[0]
                    .reindex(weekly_returns.columns)
                )
                member = (shocked_mom / sigma_ann).replace([np.inf, -np.inf], np.nan)
                blended = blended.add(member.fillna(0.0) * weights[engine.params.label])

            target = blended * leverage
            delta = target.reindex(current.index).fillna(0.0) - current
            frames[shock] = pd.DataFrame({"pct": delta, "usd": delta * aum})

        result = pd.concat(frames, axis=1)
        result.columns.names = ["shock", "metric"]
        return result.loc[[c for c in self.universe.codes if c in result.index]]

    # ------------------------------------------------------------------ #

    def to_csv(self, output_dir, prefix: str = "exposure") -> dict[str, str]:
        """Write the main outputs to CSV."""
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, frame in {
            "pct_by_market": self.pct_by_market(),
            "pct_by_sector": self.pct_by_sector(),
            "usd_by_market": self.usd_by_market(),
            "weekly_flow_usd": self.weekly_flow_usd(),
        }.items():
            path = out / f"{prefix}_{name}.csv"
            frame.to_csv(path)
            written[name] = str(path)
        return written


__all__ = ["ExposureEstimator", "ExposureSnapshot"]
