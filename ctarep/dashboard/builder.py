"""
Single-file HTML dashboard assembly.

No server is needed. Opening the single file `output/dashboard.html` is enough, and
it can be copied and distributed anywhere.

How it works
------------
For every chart a light and a dark Plotly figure JSON are built and stored in the
global `CHARTS`; JS calls `Plotly.react()` with the one matching the current theme.
A tab is rendered and resized the first time it becomes visible (this avoids the
zero-width problem of drawing into a hidden div).

Every chart comes with a **table view**, which serves as relief for low-contrast
colors in light mode and as an accessibility requirement.
"""
from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import (
    BENCHMARK_NAME,
    BEST_SINGLE_PARAMS,
    ENSEMBLE_CAP,
    ENSEMBLE_VOL_LOOKBACK,
    PATHS,
    Paths,
)
from .figures import FigureFactory, _fade
from .theme import THEMES

TEMPLATE_DIR = Path(__file__).parent / "templates"

DASH = "–"  # placeholder for missing values (en dash; no em dash in on-screen text)

#: Sign convention of the Bonds asset class. It is often misread, so it is attached to every card.
#: The paper says "Rates"; we label it "Bonds" (see universe.Sector).
RATES_SIGN_NOTE = (
    "<b>Bonds sign convention</b>: based on the <b>futures price</b> of bond futures "
    "(US 10Y, Gilt, Bund, JGB). "
    "<b>Negative (−) = short bonds = short duration = a bet on higher yields</b>, "
    "which is the opposite direction of the colloquial 'short rates' (long bonds). "
    "The paper's Rates asset class is labeled Bonds here. Short-rate futures (SOFR etc.) are not in the universe."
)


def _as_items(obj):
    """Safely turn a Series/dict/None into a list of (key, value). Avoids the bool(Series) exception."""
    if obj is None:
        return []
    if isinstance(obj, pd.Series):
        return [] if obj.empty else list(obj.items())
    return list(obj.items())


#: 'Independent' scenario mode (no index propagation; each market moves by its own sigma)
INDEP_KEY = "__indep__"
INDEP_LABEL = "Independent (own σ per market)"


@dataclass
class Card:
    """One dashboard card = title + description + chart + (optional) table.

    If `variants` is set, a selector appears at the top of the card and the chart_id /
    table is swapped according to the selection (used for the driver selector in the
    Scenario tab).

    If **two** selectors are needed, pass `variant2_opts`. The keys of `variants` then
    become the combination ``"option1||option2"``, and the options of the first selector
    are given by `variant_opts`.
    (Example: projected crowding = driver x percentile window)
    """

    id: str
    title: str
    subtitle: str = ""
    chart: bool = True
    table_html: str | None = None
    note: str = ""
    wide: bool = True
    #: {option label: chart_id}; with two selectors the key is "option1||option2"
    variants: dict[str, str] | None = None
    #: {option label: table html}; same key convention as variants
    variant_tables: dict[str, str] | None = None
    variant_label: str = "Driver"
    #: Options of selector 1. If None, the variants keys are used as is (single selector).
    variant_opts: list[str] | None = None
    #: Options of selector 2. If given, there are two selectors.
    variant2_opts: list[str] | None = None
    variant2_label: str = "Lookback"
    #: Initial selection. If None, the first option (use when the default is not first in the list).
    variant_default: str | None = None
    variant2_default: str | None = None
    #: Draw the selector as a **button group** instead of a dropdown (for two or three options).
    variant_buttons: bool = False
    variant2_buttons: bool = False
    #: Unit toggle: {"$BN": chart_id, "pp": chart_id}. Gives each card two buttons.
    #: $ amounts are proportional to the AUM assumption, so pp (independent of AUM) can be viewed alongside.
    unit_toggle: dict[str, str] | None = None
    #: Whether to leave the table (details) expanded. If None, the default rule (expanded when there is no chart).
    #: Cards drawn directly by JS have `chart=False` and would be auto-expanded, so turn it off here.
    table_open: bool | None = None
    #: Attach the card's **own** range control (presets + start/end dates).
    #: The global range bar was removed because each chart needs a different window.
    range_control: bool = False

    @property
    def sel1_opts(self) -> list[str]:
        if self.variant_opts:
            return self.variant_opts
        return list((self.variants or self.variant_tables or {}).keys())

    @property
    def default_key(self) -> str:
        """Key used for the initial display."""
        if self.variant2_opts:
            return f"{self.sel1_opts[0]}||{self.variant2_opts[0]}"
        return self.sel1_opts[0] if self.sel1_opts else ""


@dataclass
class Tab:
    id: str
    label: str
    cards: list[Card] = field(default_factory=list)
    intro: str = ""


class DashboardBuilder:
    """PipelineResult -> self-contained HTML.

    Examples
    --------
    >>> builder = DashboardBuilder(result)             # doctest: +SKIP
    >>> path = builder.build(open_browser=True)        # doctest: +SKIP
    """

    def __init__(self, result, paths: Paths = PATHS) -> None:
        self.r = result
        self.paths = paths
        self.factories = {name: FigureFactory(result, theme) for name, theme in THEMES.items()}
        self.charts: dict[str, dict[str, str]] = {}
        #: {chart_id: {axis, decimals}}: $BN charts that the AUM slider rescales
        self.usd_charts: dict[str, dict] = {}
        #: Chart ids whose x axis is a date: targets of the range control
        self.time_charts: list[str] = []
        #: Data for the asset-class expanding chart (drawn by JS)
        self.exposure_grid: dict | None = None
        #: Data for the integrated scenario card (drawn by JS)
        self.scenario_grid: dict | None = None
        #: Data for the trade pressure card (drawn by JS)
        self.pressure_grid: dict | None = None
        #: Data for the position overview (KPIs, regime chips) (drawn by JS)
        self.overview_grid: dict | None = None
        #: Data for the price + flip price card (drawn by JS)
        self.price_grid: dict | None = None
        #: {base chart_id: {asset class: variant chart_id}}: for the sidebar asset-class filter
        self.sector_charts: dict[str, dict[str, str]] = {}
        #: Realized flow + factor decomposition data
        self.flow_grid: dict | None = None
        #: Position age x distance-to-flip scatter data
        self.age_grid: dict | None = None
        #: Risk contribution data: the 'risk contribution' view of the current position card
        self.risk_grid: dict | None = None
        #: 'Scenario assumptions' tables (used in the Data & method tab)
        self.assumption_tables: dict | None = None
        self.assumption_default: str | None = None
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------ #

    def _register(
        self,
        chart_id: str,
        method: str,
        *args,
        usd_axis: str | None = None,
        usd_decimals: int = 1,
        time_axis: bool = False,
        **kwargs,
    ) -> str:
        """Register light/dark figure JSON and return the chart_id.

        Parameters
        ----------
        usd_axis : axis carrying '$BN' values ('y' | 'x' | 'z'). If given, the AUM slider
            rescales that axis by a factor ($ amounts are proportional to AUM, so only a
            multiplication is needed and no model recomputation).
        usd_decimals : number of decimals when labels are regenerated
        """
        variants: dict[str, str] = {}
        for theme_name, factory in self.factories.items():
            fig: go.Figure = getattr(factory, method)(*args, **kwargs)
            variants[theme_name] = pio.to_json(fig, validate=False)
        self.charts[chart_id] = variants
        if time_axis:
            # chart with a date x axis -> the top range control relayouts only this list
            self.time_charts.append(chart_id)
        if usd_axis:
            self.usd_charts[chart_id] = {"axis": usd_axis, "decimals": usd_decimals}
        return chart_id

    def _register_sector(
        self,
        base_id: str,
        method: str,
        make_kwargs,
        *args,
        time_axis: bool = False,
        **kwargs,
    ) -> None:
        """Pre-bake a variant per asset class for the sidebar asset-class filter.

        Numeric arrays in the baked Plotly JSON are base64 binaries, so rows cannot be
        filtered in the browser. Instead one **pre-baked variant** is kept per asset class
        and JS only swaps the chart id (`sector_charts[base_id][sector] = variant_id`).
        Variants are small (4-6 bars), so the file size cost is negligible.

        make_kwargs(sector, codes) -> keyword arguments to pass to that figure method.
        """
        uni = self.r.universe
        mapping: dict[str, str] = {}
        for s in uni.sectors:
            codes = [c for c in uni.codes if uni.get(c).sector == s]
            if not codes:
                continue
            cid = f"{base_id}@{s}"
            self._register(cid, method, *args, time_axis=time_axis,
                           **{**kwargs, **make_kwargs(s, codes)})
            mapping[s] = cid
        self.sector_charts[base_id] = mapping

    # ------------------------------------------------------------------ #
    # Scenario tab (core of the dashboard)
    # ------------------------------------------------------------------ #

    def _scenario_tab(self) -> Tab:
        """"If the index moves this way, how does the CTA change its positioning."

        An extension implementing what the paper deferred to a follow-up note, and the
        reason this dashboard exists.
        """
        r = self.r
        cfg = r.run_config
        drivers = list(r.projections)
        horizon = cfg.scenario_horizon_weeks

        tab = Tab(
            id="scenario",
            label="Scenario",
        )

        # The 'current position' card was moved to the top of the Exposure tab: the
        # snapshot answers 'what do we hold now', which belongs to the exposure narrative.

        if not drivers:
            tab.cards.append(
                Card(
                    id="scenario_missing",
                    title="Scenario not available",
                    subtitle="ScenarioEngine initialization failed (warnings are shown in the Data & method tab)",
                    chart=False,
                )
            )
            return tab

        main = cfg.scenario_driver if cfg.scenario_driver in drivers else drivers[0]

        # -- 1. Required trades / projected positions (headline, integrated card) --
        # Originally two cards: "required trades by asset class ($BN)" and "projected
        # positions (% by market)". They are the same bar chart with three fixed axes
        # (aggregation unit, metric, unit), so the three axes are exposed as toggles
        # and the cards were merged.
        # The chart is drawn by JS: driver x 2 x 2 x 2 combinations are too many to
        # pre-bake, and since $BN = pp x AUM it also reacts to the AUM slider for free.
        self.scenario_grid = self._scenario_grid(drivers, main)
        tab.cards.append(
            Card(
                id="scenario_combined",
                title="Required trades and projected positions by scenario",
                subtitle=(
                    f"Result when the driver moves by each scenario over {horizon} week{'s' if horizon != 1 else ''}<br>"
                    "<b>Required trades</b> = volume the CTA must buy or sell<br>"
                    "<b>Projected position</b> = volume held afterwards<br>"
                ),
                chart=False,
            )
        )

        # -- 2. Certain trades -----------------------------------------------
        # "Average required trade" cannot be used as a metric: scenarios are symmetric
        # at +-sigma, so up and down moves cancel and only the **flat-case trade
        # (window roll-off) remains, 94%.** Probability weighting (normal and empirical
        # quantiles) raises the correlation with drift to 0.998 instead. So a **lower
        # bound** is used instead of an expectation: if the sign does not change across
        # all 5 steps, the endpoint closest to 0 is the 'certain trade'.
        # (All 256 driver x market combinations were confirmed monotonic, so the two ends
        # of the 5 points are the true ends of the range.)
        tab.cards.append(
            Card(
                id="certain_trades",
                title="Certain trades",
                subtitle=(
                    "Minimum trade volume and direction that are certain however the driver moves<br>"
                    "<b>Solid bar</b> = certain part, <b>hatched bar</b> = additional amount that may trade depending on the scenario "
                    "(endpoint is the maximum), <b>vertical tick</b> = position in the flat case, "
                    "<b>gray</b> = direction not certain<br>"
                    f"<b>[n/{len(drivers)}]</b> on the right = number of drivers that give the same direction when swapped"
                ),
                chart=False,
            )
        )

        # -- 2b. Trade pressure (paper extension) -----------------------------
        # Certain trades require the sign to stay unchanged across all 5 steps, so in
        # independent mode most are 'not certain' (the trade direction follows the
        # market's own price direction). Instead of a binary verdict, show the buy
        # probability and expected trade weighted by the market's own historical
        # distribution (KDE).
        self.pressure_grid = self._pressure_grid()
        if self.pressure_grid:
            pg = self.pressure_grid
            tab.cards.append(
                Card(
                    id="trade_pressure",
                    title="Trade pressure",
                    subtitle=(
                        "Buy probability and expected trade when each market moves per its historical weekly return distribution<br>"
                        "<b>Bar</b> = expected trade, <b>dark</b> = buy (blue) or sell (red) "
                        f"probability of {pg['sure_prob']:.0%} or more, <b>light</b> = {pg['lean_prob']:.0%} or more, "
                        "<b>gray</b> = neutral<br>"
                        "<b>Vertical tick</b> = trade in the flat case, <b>[ ]</b> on the left = direction and probability"
                    ),
                    chart=False,
                )
            )

        # -- 2c. Position age x distance to flip ------------------------------
        self.age_grid = self._age_grid()
        if self.age_grid:
            tab.cards.append(
                Card(
                    id="agewatch",
                    title="Position age and flip proximity",
                    subtitle=(
                        "X axis = weeks the current sign has held, Y axis = remaining distance to flip<br>"
                        "Color = long (blue) / short (red), size = position size<br>"
                        "<b>Bottom right</b> = flip imminent after a long hold (largest reversal risk), "
                        "<b>bottom left</b> = flip imminent right after a new entry<br>"
                        "Y axis in σ = comparable across markets, in % = market volatility differences not reflected"
                    ),
                    chart=False,
                )
            )

        # -- 2d. Price and flip price -----------------------------------------
        # Overlay the 'flip price' of the flip table on the price chart.
        self.price_grid = self._price_grid()
        if self.price_grid:
            # Placed in the Main tab (right below the position overview); picked up in _build_tabs.
            self._price_card = (
                Card(
                    id="price_flip",
                    title="Price and flip price",
                    subtitle="",
                    chart=False,
                )
            )

        # -- 3. Market detail matrix (removed) --------------------------------
        # Absorbed into the 'asset heatmap' view of the integrated card above.
        # FigureFactory.scenario_market_matrix is kept for pulling a single figure
        # from the console.

        # -- 5. Trigger levels ------------------------------------------------
        # This used to be a per-driver table, but the flip point is determined only by
        # the market's own momentum and is independent of the driver (see the
        # _table_flip docstring). The driver selector only blanked out low-beta markets,
        # so it was removed and a **shock horizon (weeks)** selector was added instead.
        flip_tables = {
            f"{h}w": self._table_flip(h)
            for h in cfg.trigger_horizon_choices
            if h in (r.triggers_flip or {})
        }
        if flip_tables:
            default_h = f"{cfg.trigger_horizon_weeks}w"
            if default_h not in flip_tables:
                default_h = next(iter(flip_tables))
            # With a single horizon, no selector is created (one button is noise).
            multi = len(flip_tables) > 1
            tab.cards.append(
                Card(
                    id="scenario_triggers",
                    title=f"Position flip conditions ({default_h})",
                    subtitle=(
                        "Table of how far each market's <b>own</b> price must move over the shock horizon "
                        "for the position sign to flip<br>"
                        "Markets that already flip in the flat case are marked with ⚠️"
                    ),
                    chart=False,
                    wide=True,
                    variant_label="Shock horizon",
                    variant_opts=list(flip_tables) if multi else None,
                    variant_default=default_h if multi else None,
                    variant_buttons=multi,
                    variant_tables=flip_tables if multi else None,
                    table_html=flip_tables[default_h],
                )
            )

        # -- 7. Assumptions ---------------------------------------------------
        # Changing the driver changes all betas and shocks, so these are kept as
        # variants like the other cards.
        # 'Scenario assumptions' is for methodology checks, so it was moved to the
        # Data & method tab. Here they are only built and stored.
        assum_tables = {
            self._driver_label(d): self._table_scenario_assumptions(d) for d in drivers
        }
        self.assumption_tables = assum_tables
        self.assumption_default = self._driver_label(main)
        return tab

    @staticmethod
    def _driver_label(code: str) -> str:
        from ..universe import UNIVERSE

        try:
            return f"{code} · {UNIVERSE.get(code).name}"
        except KeyError:
            return code

    # ------------------------------------------------------------------ #

    def _build_tabs(self) -> list[Tab]:
        r = self.r
        cfg = r.run_config
        aum_bn = cfg.aum_usd / 1e9

        scenario_tab = self._scenario_tab()

        # ---------------------------------------------------------------- #
        exposure_tab = Tab(
            id="exposure",
            label="Exposure",
            intro=(
                "Exposure by market, shown as a notional ratio per $1 of benchmark. "
                "(e.g. +100% = $1 long per $1 invested in the benchmark)"
            ),
        )
        # -- Current position (snapshot) --------------------------------------
        # Drawn directly by JS (instead of FigureFactory.current_positioning), because
        # hiding/restoring markets by clicking the y-axis names requires filtering the
        # data arrays in the browser, and plotly.py bakes arrays into JSON as base64
        # binaries that are hard to modify.
        # Values reuse 'current' of the scenario grid (SG), so there is no duplicate calculation.
        # Risk contribution is a toggle **in the same card as notional**. As a separate
        # card nobody would click it, and since the model has w = m~/sigma, showing
        # notional alone invites misreading.
        self.risk_grid = self._risk_grid()
        main_tab = Tab(id="main", label="Main", intro="")
        # -- Position overview (KPIs, regime chips, asset-class filter) ---------
        self.overview_grid = self._overview_grid()
        if self.overview_grid:
            ov = self.overview_grid
            main_tab.cards.append(
                Card(
                    id="position_overview",
                    chart=False,
                    title="Position overview",
                    subtitle="",
                )
            )
        if getattr(self, "_price_card", None) is not None:
            main_tab.cards.append(self._price_card)
        exposure_tab.cards.append(
            Card(
                id="current_positioning",
                chart=False,
                table_open=False,      # the chart is the main element, so the table is collapsed
                title="Current position (notional and risk basis)",
                subtitle=(
                    f"As of {r.snapshot.date:%Y-%m-%d} · blue = long, red = short<br>"
                ),
                table_html=self._table_positions(),
            )
        )
        self.exposure_grid = self._exposure_grid()
        exposure_tab.cards.append(
            Card(
                id="exposure_expand",
                title="Replica exposure by market over time",
                chart=False,
                range_control=True,
            )
        )
        # (Position heatmap: merged into the card above as a view toggle.
        # FigureFactory.position_heatmap is kept for console use.)

        # -- Exposure change by period -----------------------------------------
        exposure_tab.cards.append(
            Card(
                id="expo_change",
                title="Exposure change by period",
                subtitle=(
                    "Difference between the current position and past points in time, multiple periods compared<br>"
                ),
                chart=False,
            )
        )
        self._register("momentum_signals", "momentum_signals")
        self._register_sector("momentum_signals", "momentum_signals",
                              lambda s, cs: {"codes": cs})
        exposure_tab.cards.append(
            Card(
                id="momentum_signals",
                title="Normalized momentum by trend speed (current)",
                subtitle=(
                    "The <b>signal</b> that builds the position<br>"
                    "m = (mean weekly return / weekly volatility) × √L : t-statistic of the trend. "
                    f"Horizontal line = cap ±{r.ensemble.models[0].params.momentum_cap:g}. "
                ),
                table_html=self._table_momentum(),
            )
        )
        self._register("market_volatility", "market_volatility", time_axis=True)
        self._register_sector("market_volatility", "market_volatility",
                              lambda s, cs: {"codes": cs})
        exposure_tab.cards.append(
            Card(
                id="market_volatility",
                title="Realized volatility by market",
            )
        )

        # ---------------------------------------------------------------- #
        flow = Tab(
            id="flow",
            label="Flow",
        )
        # -- Realized flow (factor decomposition integrated) -------------------
        # Previously 4 windows x 2 units = 8 charts were pre-baked; adding a 'factor
        # decomposition' view would make 6 windows x 2 units x 2 views = 24. The needed
        # numbers are only market x window x 2 components = 192, so **only values are
        # shipped and the browser draws** ($BN = pp x AUM, so it follows the AUM slider for free).
        self.flow_grid = self._flow_grid()
        flow.cards.append(
            Card(
                id="realized_flow",
                title="Cumulative net trades by period",
                subtitle=(
                    "Blue = net buy, red = net sell<br>"
                    "<b>Factor decomposition</b>: splits trades into a <b>trend change</b> part and a <b>volatility change</b> part<br>"
                ),
                chart=False,
            )
        )

        # -- 2b. Cumulative flow (single market) -------------------------------
        # Cumulative flow = cumulative weekly position changes = (current - starting position).
        # The browser builds it from EG.market_series (% notional by market), which is already shipped.
        flow.cards.append(
            Card(
                id="cumflow",
                title="Cumulative net trades of a single market",
                subtitle=(
                    "Blue above the fan = net buy, red below = net sell"
                ),
                chart=False,
            )
        )

        # Put the single-market cumulative net trades card on top (stable sort keeps the rest of the order)
        flow.cards.sort(key=lambda c: c.id != "cumflow")

        # (Flow sensitivity card removed: same question as the 'asset heatmap' of the
        # scenario card; sigma-based steps compare better across markets than fixed %.
        # FigureFactory.flow_sensitivity / _table_flow_sensitivity are kept.)

        # ---------------------------------------------------------------- #
        # Model monitoring: replication quality + staleness watch + parameter sensitivity in one tab
        # ---------------------------------------------------------------- #
        sweep_tab = Tab(
            id="monitoring",
            label="Model monitoring",
            intro=(
                "Page for checking replication quality, signs of model staleness and parameter sensitivity."
            ),
        )

        # -- 1. Health summary (traffic lights) --------------------------------
        sweep_tab.cards.append(
            Card(
                id="model_health",
                title="Model health summary",
                subtitle="Traffic-light verdict (normal, caution, warning) per replication quality metric",
                chart=False,
                table_html=self._table_model_health(),
            )
        )

        # -- 2. Tracking ability ------------------------------------------------
        self._register("cumulative_comparison", "cumulative_comparison", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="cumulative_comparison",
                title="Cumulative return: replica vs benchmark",
                subtitle=(
                    "Both indexed to 100 at the start (log scale)"
                ),
                table_html=self._table_performance(),
            )
        )
        self._register("cumulative_gap", "cumulative_gap", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="cumulative_gap",
                title="Cumulative gap: benchmark vs replica",
                subtitle="Replica cumulative multiple ÷ benchmark cumulative multiple − 1 (0 = perfect tracking)",
                wide=False,
            )
        )
        self._register("rolling_r2", "rolling_r2", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="rolling_r2",
                title=f"Rolling R² ({cfg.rolling_r2_window_weeks}w)",
                subtitle=(
                    "A <b>sustained drop below 50%</b> is a watch item"
                ),
                wide=False,
            )
        )
        self._register("rolling_tracking_error", "rolling_tracking_error", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="rolling_tracking_error",
                title="Rolling tracking error (104w, annualized)",
                subtitle=(
                    "Volatility of the weekly (replica − benchmark) difference. "
                    "A <b>sustained level above 1.5x</b> the full-sample mean is a watch item"
                ),
                wide=False,
            )
        )

        # -- 3. Independent validation (CFTC) ----------------------------------
        if r.cftc is not None and r.cftc.rolling_agree is not None:
            clean_agree = r.cftc.table[~r.cftc.table["basis_contaminated"]]
            self._register("cftc_rolling_agreement", "cftc_rolling_agreement", time_axis=True)
            sweep_tab.cards.append(
                Card(
                    id="cftc_rolling_agreement",
                    title="CFTC sign agreement rate",
                    subtitle=(
                        f"Share of weeks where the model's estimated direction matches the actual COT direction. Overall mean "
                        f"<b>{clean_agree['sign_agree'].mean():.0%}</b> (random 50%)<br>"
                        "<b>Staying below 60%</b> is a watch item"
                    ),
                    wide=False,
                )
            )

        # -- 4. Model internals -------------------------------------------------
        self._register("turnover", "turnover", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="turnover",
                title="Weekly turnover",
                subtitle=(
                    "Σ|Δw| = sum of absolute weekly notional changes. "
                    "For checking the effect of transaction costs on replica performance"
                ),
                wide=False,
            )
        )
        self._register("member_dispersion", "member_dispersion", time_axis=True)
        sweep_tab.cards.append(
            Card(
                id="member_dispersion",
                title="Position disagreement across trend speeds",
                subtitle=(
                    "How differently the 16/32/52w member models position the same market<br>"
                    "A <b>spike</b> often marks a phase where the trend is breaking, "
                    "and replication error tends to rise at the same time."
                ),
                wide=False,
            )
        )

        # -- 5. Parameter sensitivity -------------------------------------------
        sweep_tab.cards.append(
            Card(
                id="sweep_intro",
                title="Parameter sensitivity",
                subtitle=(
                    "Momentum lookback 5 × cap 5 × volatility lookback 5 = "
                    "<b>125 combinations</b>, objective is <b>replication accuracy (R²)</b>"
                ),
                chart=False,
                table_html=self._table_stats(),
            )
        )
        if r.sweep is not None:
            self._register("sweep_sorted", "sweep_sorted")
            sweep_tab.cards.append(
                Card(
                    id="sweep_sorted",
                    title="All 125 combinations (by R²)",
                    subtitle="Performance jumps where the momentum lookback changes from 4w to 8w and from 8w to 16w",
                    table_html=self._table_sweep_top(),
                )
            )
            # Sensitivity slices are cut at the **values the actual model uses** (not the paper's 90 days).
            hold_mom = BEST_SINGLE_PARAMS.momentum_lookback
            hold_cap = ENSEMBLE_CAP
            hold_vol = ENSEMBLE_VOL_LOOKBACK
            self._register(
                "sweep_mom", "sweep_sensitivity", "momentum_lookback",
                {"cap": hold_cap, "vol_lookback": hold_vol}, "momentum lookback (weeks)",
            )
            sweep_tab.cards.append(
                Card(
                    id="sweep_mom",
                    title="Sensitivity: momentum lookback",
                    subtitle=f"Cap {hold_cap:g}, volatility {hold_vol}d fixed · most sensitive parameter",
                    wide=False,
                )
            )
            self._register(
                "sweep_cap", "sweep_sensitivity", "cap",
                {"momentum_lookback": hold_mom, "vol_lookback": hold_vol}, "normalized momentum cap",
            )
            sweep_tab.cards.append(
                Card(
                    id="sweep_cap",
                    title="Sensitivity: momentum cap",
                    subtitle=f"Momentum {hold_mom}w, volatility {hold_vol}d fixed · cap 0.0 = unlimited",
                    wide=False,
                )
            )
            self._register(
                "sweep_vol", "sweep_sensitivity", "vol_lookback",
                {"momentum_lookback": hold_mom, "cap": hold_cap}, "volatility lookback (days)",
            )
            sweep_tab.cards.append(
                Card(
                    id="sweep_vol",
                    title="Sensitivity: volatility lookback",
                    subtitle=f"Momentum {hold_mom}w, cap {hold_cap:g} fixed",
                    wide=False,
                )
            )
        # If the sweep is off, **no card is created**: an empty card is noise.
        # (With `--no-sweep` this tab keeps only the monitoring metrics.)

        # ---------------------------------------------------------------- #
        cftc_tab = self._cftc_tab()

        # ---------------------------------------------------------------- #
        data_tab = Tab(
            id="data",
            label="Data & method",
            intro="Input data quality checks and model settings",
        )
        # Scenario assumptions are for methodology checks, so they live here (PMs only look at the Scenario tab)
        if self.assumption_tables:
            data_tab.cards.append(
                Card(
                    id="scenario_assumptions",
                    title="Scenario assumptions",
                    subtitle=(
                        "Values used for shock size and cross-market propagation "
                        "(<b>β is relative to the selected driver</b>; changing the driver recomputes "
                        f"the β of all {len(r.universe)} markets against that asset)"
                    ),
                    chart=False,
                    variant_tables=self.assumption_tables,
                    variant_default=self.assumption_default,
                    variant_opts=list(self.assumption_tables),
                    table_html=self.assumption_tables[self.assumption_default],
                )
            )
        data_tab.cards.append(
            Card(
                id="universe",
                title="Universe",
                chart=False,
                table_html=self._table_universe(),
            )
        )
        data_tab.cards.append(
            Card(
                id="validation",
                title="Data validation",
                chart=False,
                table_html=self._table_validation(),
            )
        )
        data_tab.cards.append(
            Card(
                id="method",
                title="Model settings",
                chart=False,
                table_html=self._table_config(),
            )
        )

        # Reading order: what we hold now (Exposure) -> how it moved (Flow)
        # -> what happens if something occurs (Scenario) -> model state (Model monitoring)
        # -> methodology (Data & method) -> comparison with actuals (CFTC)
        tabs = [
            main_tab,
            exposure_tab,
            flow,
            scenario_tab,
        ]
        tabs.append(sweep_tab)
        tabs.append(data_tab)
        # CFTC is a supplementary cross-check, so it goes last
        if cftc_tab is not None:
            tabs.append(cftc_tab)
        return tabs

    # ------------------------------------------------------------------ #
    # CFTC independent validation tab
    # ------------------------------------------------------------------ #

    def _cftc_tab(self) -> Tab | None:
        c = self.r.cftc
        if c is None:
            return None

        clean = c.table[~c.table["basis_contaminated"]]

        tab = Tab(
            id="cftc",
            label="CFTC positioning",
            intro=(
                "<b>Positions per the CFTC COT report</b> (Tuesday close, published Friday)<br>"
                + self._cftc_asof_note()
            ),
        )

        wins = sorted((c.crowding_by_window or {5: c.crowding}).keys())
        crowd_variants, crowd_tables = {}, {}
        for y in wins:
            lab = f"{y}y"
            crowd_variants[lab] = self._register(f"cftc_crowding__{y}",
                                                 "cftc_crowding", years=y)
            self._register_sector(f"cftc_crowding__{y}", "cftc_crowding",
                                  lambda s, cs: {"codes": cs}, years=y)
            crowd_tables[lab] = self._table_cftc_crowding(y)
        # Default is 5y: after adding 1y and 3y as candidates, using wins[0] as is would
        # push the default to 1y. 5y is the existing baseline, so keep it.
        first = "5y" if "5y" in crowd_variants else f"{wins[0]}y"
        tab.cards.append(
            Card(
                id=crowd_variants[first],
                title="Position crowding",
                subtitle=(
                    "<b>0% = largest short in the percentile window, 100% = largest long</b> "
                    "(percentile of CFTC Managed Money net position within its own history)"
                ),
                variants=crowd_variants,
                variant_tables=crowd_tables,
                variant_label="Percentile window",
                variant_default=first,
                table_html=crowd_tables[first],
                note=self._cftc_asof_chip(),
            )
        )

        tab.cards.append(
            Card(
                id="cftc_markets",
                title="CFTC COT positions vs CTA model positions",
                subtitle="Contracts, sign agreement and correlation by market",
                chart=False,
                table_html=self._table_cftc_markets(),
                note=self._cftc_asof_chip(),
            )
        )
        return tab

    def _exposure_grid(self) -> dict:
        """Data for the asset-class expanding chart: asset-class totals and per-market series together.

        JS picks asset-class lines or market lines depending on which asset classes are 'expanded'.

        Color assignment: **fixed per entity** (it does not change whatever is toggled on or off).
        The theme has 12 categorical slots (defined in theme.py), split into two layers:
          - slots 1-4 = asset-class total lines only
          - slots 5-12 = individual markets only, 4 per asset class starting from the first slot
        The two sets do not overlap, so a total line and a market line **can never share a
        color**. The 4 markets within an asset class take slots 5, 6, 7, 8 one each and so
        differ from each other; across asset classes the **line dash** differs (Equities solid /
        Bonds dashed / Currencies dotted / Commodities long-dashed). Color + dash makes all 16
        markets unique. Some of the markets' colors are low-saturation and cannot be told apart
        by color alone, so dash and direct labels are used together.
        """
        r = self.r
        uni = r.universe
        by_mkt = r.exposure.pct_by_market().dropna(how="all")
        by_sec = r.exposure.pct_by_sector().reindex(by_mkt.index)
        sectors = [s for s in uni.sectors if s in by_sec.columns]
        codes = [c for c in uni.codes if c in by_mkt.columns]

        # Weekly data, so shipping it as is costs little (16 x ~1500). Truncated to 5 decimals.
        def ser(df, col):
            return [None if pd.isna(v) else round(float(v), 5) for v in df[col]]

        light = self.factories["light"]
        dark = self.factories["dark"]
        sec_colors = {
            "light": light.t.sector_colors(sectors),
            "dark": dark.t.sector_colors(sectors),
        }
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in by_mkt.index],
            "sectors": sectors,
            "members": {s: [c for c in codes if uni.get(c).sector == s]
                        for s in sectors},
            "names": {c: uni.get(c).name for c in codes},
            "sector_series": {s: ser(by_sec, s) for s in sectors},
            "market_series": {c: ser(by_mkt, c) for c in codes},
            "sector_colors": sec_colors,
            "slot_colors": {"light": list(light.t.series), "dark": list(dark.t.series)},
            **self._market_style(uni, sectors, codes, len(light.t.series)),
        }

    @staticmethod
    def _market_style(uni, sectors: list[str], codes: list[str],
                      n_slots: int) -> dict:
        """Fixed (color slot, dash) per market. Uses only slots 5-8 (index 4-7); 1-4 belong to asset-class total lines."""
        detail_lo = min(4, n_slots // 2)          # start of the market-only slots
        n_detail = n_slots - detail_lo
        slot: dict[str, int] = {}
        dash: dict[str, int] = {}
        for si, sec in enumerate(sectors):
            mem = [c for c in codes if uni.get(c).sector == sec]
            for mi, code in enumerate(mem):
                slot[code] = detail_lo + mi % n_detail
                # if an asset class has more markets than slots, shift the dash by one more step
                dash[code] = si + len(sectors) * (mi // n_detail)
        return {"market_slot": slot, "market_dash": dash}

    # -- Monitoring ------------------------------------------------------ #

    #: (label, ok threshold, warning threshold); direction is set by `higher_is_better`
    _HEALTH_OK, _HEALTH_WARN, _HEALTH_BAD = "🟢 Normal", "🟡 Caution", "🔴 Warning"

    @staticmethod
    def _health_flag(value: float, ok: float, bad: float,
                     higher_is_better: bool = True) -> str:
        if not np.isfinite(value):
            return DASH
        if higher_is_better:
            return (DashboardBuilder._HEALTH_OK if value >= ok else
                    DashboardBuilder._HEALTH_BAD if value < bad else
                    DashboardBuilder._HEALTH_WARN)
        return (DashboardBuilder._HEALTH_OK if value <= ok else
                DashboardBuilder._HEALTH_BAD if value > bad else
                DashboardBuilder._HEALTH_WARN)

    def _table_model_health(self) -> str:
        """Model health traffic lights.

        The thresholds are not absolute standards but practical levels meaning 'worth a
        look below this'. One red alone may be noise; the real signal is when **two or
        more deteriorate at the same time**.
        """
        r = self.r
        rows: list[dict] = []

        # 1. Rolling R²
        rr = r.rolling_r2.dropna()
        v = float(rr.iloc[-1]) if len(rr) else float("nan")
        rows.append({
            "Metric": "Rolling R² (104w)",
            "What it shows": "Return explanatory power",
            "Current": f"{v:.1%}" if np.isfinite(v) else DASH,
            "Threshold": "Normal ≥60% · Warning <50%",
            "Status": self._health_flag(v, 0.60, 0.50),
        })

        # 2. CFTC sign agreement rate (independent of returns)
        agree = float("nan")
        if r.cftc is not None and r.cftc.rolling_agree is not None and len(r.cftc.rolling_agree):
            agree = float(r.cftc.rolling_agree.dropna().iloc[-1])
        rows.append({
            "Metric": "CFTC sign agreement (104w)",
            "What it shows": "Position direction (independent of R²)",
            "Current": f"{agree:.0%}" if np.isfinite(agree) else DASH,
            "Threshold": "Normal ≥65% · Warning <60%",
            "Status": self._health_flag(agree, 0.65, 0.60),
        })

        # 3. Cumulative path drift
        j = pd.concat([r.ensemble.returns.rename("a"),
                       r.benchmark_returns.rename("b")], axis=1).dropna()
        gap = float((1 + j["a"]).prod() / (1 + j["b"]).prod() - 1) if len(j) else float("nan")
        rows.append({
            "Metric": "Cumulative path drift",
            "What it shows": "Return path (what R² misses)",
            "Current": f"{gap:+.1%}" if np.isfinite(gap) else DASH,
            "Threshold": "Normal ≤±10% · Warning >±25%",
            "Status": self._health_flag(abs(gap), 0.10, 0.25, higher_is_better=False),
        })

        # 4. Rolling tracking error / full sample
        ratio = float("nan")
        if len(j) > 120:
            d = j["a"] - j["b"]
            cur = float(d.rolling(104).std(ddof=1).iloc[-1] * np.sqrt(52))
            full = float(d.std(ddof=1) * np.sqrt(52))
            ratio = cur / full if full > 0 else float("nan")
        rows.append({
            "Metric": "Tracking error / full sample",
            "What it shows": "Change in tracking precision",
            "Current": f"{ratio:.2f}x" if np.isfinite(ratio) else DASH,
            "Threshold": "Normal ≤1.25x · Warning >1.50x",
            "Status": self._health_flag(ratio, 1.25, 1.50, higher_is_better=False),
        })

        # 5. Data freshness
        last = r.weekly_returns.dropna(how="all").index[-1]
        stale = (pd.Timestamp.now().normalize() - last).days
        rows.append({
            "Metric": "Data freshness",
            "What it shows": f"Last observation {last:%Y-%m-%d}",
            "Current": f"{stale} days ago",
            "Threshold": "Normal ≤10 days · Warning >21 days",
            "Status": self._health_flag(stale, 10, 21, higher_is_better=False),
        })

        # 6. Run warnings
        n_warn = len(r.warnings)
        rows.append({
            "Metric": "Pipeline warnings",
            "What it shows": "Data quality gate and roll diagnostics",
            "Current": f"{n_warn}",
            "Threshold": "Normal 0 · Warning ≥3",
            "Status": self._health_flag(n_warn, 0, 2, higher_is_better=False),
        })

        out = pd.DataFrame(rows).set_index("Metric")
        html = self._df_to_html(out, "Metric")
        if r.warnings:
            items = "".join(f"<li>{w}</li>" for w in r.warnings)
            html += (
                "<details class='muted' style='margin-top:8px'>"
                f"<summary>Pipeline warnings: {n_warn}</summary>"
                f"<ul style='margin:6px 0 0 18px'>{items}</ul></details>"
            )
        return html

    # -- CFTC tables ----------------------------------------------------- #

    def _cftc_asof_note(self) -> str:
        """As-of text for the CFTC tab.

        Displayed contracts and percentiles are the **Tuesday close**; precisely, they are
        the release that the last row of our weekly grid (Friday) picked up via ffill. This
        can differ from the latest source release (when the price grid is shorter), so both
        are stated.
        """
        c = self.r.cftc
        if c is None or c.asof is None:
            return "<span class='muted'>The report as-of date is not available.</span>"
        parts = [
            f"COT figures in this tab are as of the <b>{c.asof:%Y-%m-%d} (Tue) close</b> and "
            f"were published on {c.released:%Y-%m-%d} (Fri)"
        ]
        if c.stale_weeks:
            parts.append(
                f"⚠ The source has releases up to {c.cot_last:%Y-%m-%d}, but the price grid ends at "
                f"{c.grid_last:%Y-%m-%d}, so {c.stale_weeks} release(s) are not reflected"
            )
        if c.fetched_at is not None:
            parts.append(f"Fetched {c.fetched_at:%Y-%m-%d %H:%M}")
        parts.append(
            "New releases are fetched incrementally on each run, "
            "skipped if the previous fetch was within 12 hours"
        )
        return "<span class='muted'>" + " · ".join(parts) + "</span>"

    def _cftc_asof_chip(self) -> str:
        """Short as-of text for the chip at the top of a card. Details are in the tab intro."""
        c = self.r.cftc
        if c is None or c.asof is None:
            return ""
        s = f"COT as of {c.asof:%Y-%m-%d} (Tue) close"
        if c.stale_weeks:
            s += f" · latest source release {c.cot_last:%Y-%m-%d} not reflected"
        return s

    def _table_cftc_crowding(self, years: int = 5) -> str:
        c = self.r.cftc
        if c is None or c.crowding is None or c.crowding.empty:
            return "<p class='muted'>No data.</p>"
        src = (c.crowding_by_window or {}).get(years, c.crowding)
        d = src.sort_values("pct_win")
        out = pd.DataFrame({
            "Market": [self.r.universe.get(i).name for i in d.index],
            "COT net position (contracts)": d["cftc_net"].map("{:+,.0f}".format),
            f"{years}y percentile": d["pct_win"].map("{:.0%}".format),
            "Full-sample percentile": d["pct_all"].map("{:.0%}".format),
            "Crowding": d["crowding"],
            "Our position": d["our_position"].map(
                lambda v: DASH if pd.isna(v) else f"{v:+.1%}"),
        })
        return self._df_to_html(out, "Code")

    def _table_cftc_markets(self) -> str:
        c = self.r.cftc
        if c is None:
            return "<p class='muted'>No data.</p>"
        d = c.table
        # As-of date per market: all markets should share the same date. If only one lags,
        # the contract-name mapping has broken (e.g. CFTC renamed it) and it stands out.
        asof_m = c.asof_by_market if c.asof_by_market is not None else pd.Series(dtype="O")
        out = pd.DataFrame({
            "Market": [self.r.universe.get(i).name if i in self.r.universe else i
                       for i in d.index],
            "Category": d["category"],
            "Weeks": d["weeks"],
            "COT as of": [
                DASH if pd.isna(asof_m.get(i, pd.NaT)) else f"{asof_m[i]:%Y-%m-%d}"
                for i in d.index
            ],
            "Our contracts": d["ours_now"].map("{:+,.0f}".format),
            "CFTC contracts": d["cftc_now"].map("{:+,.0f}".format),
            "Sign agreement": d["sign_agree"].map("{:.0%}".format),
            "Level corr": d["corr_level"].map("{:+.3f}".format),
            "Change corr": d["corr_change"].map(
                lambda v: DASH if pd.isna(v) else f"{v:+.3f}"),
            "Basis contaminated": d["basis_contaminated"].map(lambda v: "⚠ Yes" if v else "No"),
        })
        return self._df_to_html(out, "Code")

    # ------------------------------------------------------------------ #
    # Tables (accessibility relief)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _df_to_html(df: pd.DataFrame, index_name: str = "") -> str:
        df = df.copy()
        if index_name:
            df.index.name = index_name
        return df.to_html(
            classes="data-table", border=0, na_rep=DASH, escape=False, justify="left"
        )

    def _price_asof_note(self) -> str:
        """As-of text for prices.

        Prices are pulled by calendar date and ffilled, so **the last row of the file is
        not necessarily the latest close.** Observed case: the last row was 08-04, but 14
        of 16 markets were copies of the 08-03 close. So the last date on which each
        market's value actually changed is used, and markets are grouped when dates differ.
        """
        asof = getattr(self.r, "price_asof", None)
        if asof is None or len(asof) == 0:
            return ""
        groups: dict[str, list[str]] = {}
        for code, d in asof.items():
            groups.setdefault(f"{d:%Y-%m-%d}", []).append(code)
        parts = []
        for d in sorted(groups, reverse=True):
            mkts = groups[d]
            parts.append(f"<b>{d}</b> ({len(mkts)} markets"
                         + (f": {', '.join(mkts)}" if len(mkts) <= 4 else "")
                         + ")")
        body = " · ".join(parts)
        extra = ""
        return f"<p class='muted'>Latest price as of: {body}{extra}</p>"

    @staticmethod
    def _fmt_frame(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
        """Format a numeric frame as strings (NaN becomes a dash)."""
        return df.apply(
            lambda col: col.map(lambda v: DASH if pd.isna(v) else fmt.format(v))
        )

    # -- Scenario tables ------------------------------------------------- #

    def _scenario_proj(self, driver: str | None):
        if driver is None:
            return self.r.projection_independent
        return self.r.projections.get(driver)

    #: Realized flow cumulative windows (weeks). 1-2w catch recent flips, 52w is one full cycle.
    FLOW_WINDOWS = (1, 2, 4, 13, 26, 52)
    #: Percentile rolling windows (weeks). Same candidates as the cumulative flow card.
    PCT_WINDOWS = (26, 52, 104, 156, 260)

    def _flow_grid(self) -> dict | None:
        """Cumulative net trades + momentum/volatility factor decomposition.

        Since the position is `w = M / σ` (M = blended momentum numerator, σ = annualized
        volatility), the two endpoint values of a period split it into two components
        **exactly**::

            Δw = (M₁ − M₀)/σ₁  +  M₀·(1/σ₁ − 1/σ₀)
                 ─────────────     ────────────────
                 momentum part      volatility part

        This is an identity, not an approximation (checked: max error 1e-16). The cross term
        is assigned to the momentum part, because evaluating at the new σ reads as 'trend
        change as of now'. The total matches the existing
        `weekly_flow_usd().iloc[-N:].sum()` exactly (max error 6e-14 $BN).
        """
        pos = self.r.exposure.pct_by_market()
        if pos is None or pos.empty:
            return None
        vol = self.r.ensemble.models[0].signals.vol_annualized
        codes = [c for c in self.r.universe.codes if c in pos.columns]
        pos, vol = pos[codes], vol.reindex(pos.index)[codes]

        out: dict[str, dict] = {}
        w1, s1 = pos.iloc[-1], vol.iloc[-1]
        for n in self.FLOW_WINDOWS:
            if len(pos) <= n:
                continue
            w0, s0 = pos.iloc[-1 - n], vol.iloc[-1 - n]
            m1, m0 = w1 * s1, w0 * s0
            mom = (m1 - m0) / s1
            vv = m0 * (1.0 / s1 - 1.0 / s0)
            out[str(n)] = {
                c: [None if pd.isna(mom[c]) else round(float(mom[c]) * 100, 4),
                    None if pd.isna(vv[c]) else round(float(vv[c]) * 100, 4)]
                for c in codes
            }
        if not out:
            return None

        # Percentile: "is this trade unusual for this market".
        # $BN and pp come out **structurally large for low-volatility markets** because
        # w = k·m~/σ (observed corr(|$BN|, 1/σ) = +0.66; pp is $BN/AUM, so the same).
        # Ranking against the market's own history removes that bias (corr = -0.005).
        pctile: dict[str, dict] = {}
        for n in self.FLOW_WINDOWS:
            key = str(n)
            if key not in out:
                continue
            hist = pos.diff(n)
            cur = (w1 - pos.iloc[-1 - n])
            per: dict[str, dict] = {}
            for label, wl in [("exp", None)] + [(str(v), v) for v in self.PCT_WINDOWS]:
                vals = {}
                for c in codes:
                    hs = hist[c].dropna()
                    if wl is not None:
                        hs = hs.iloc[-wl:]
                    if hs.empty or pd.isna(cur[c]):
                        vals[c] = None
                    else:
                        vals[c] = round(float((hs <= cur[c]).mean() * 100), 1)
                per[label] = vals
            pctile[key] = per

        return {
            "markets": codes,
            "names": {c: self.r.universe.get(c).name for c in codes},
            "windows": [str(n) for n in self.FLOW_WINDOWS if str(n) in out],
            "decomp": out,      # {window: {market: [momentum contribution pp, volatility contribution pp]}}
            "pctile": pctile,   # {window: {'exp'|window: {market: percentile}}}
            "pct_windows": [str(v) for v in self.PCT_WINDOWS],
        }

    def _risk_grid(self) -> dict | None:
        """Risk contribution grid for the **risk contribution view** of the 'current position' card.

        Why it ranks equal with notional
        --------------------------------
        The model has `w = m~ / σ_ann`, so **notional piles up structurally in
        low-volatility markets.** Showing notional alone makes "Bonds -106%" read like the
        dominant bet, which it is not on a risk basis (see the measurements in the
        `exposure.risk_contribution` docstring: bonds are 34.8% of notional but 19.7% of
        risk; equities are 22.1% of notional but 35.8% of risk). The same bar must be
        viewable on both yardsticks to avoid misreading.

        Uses the **sign-preserving** risk contribution share::

            rc_i = sign(w_i) · |w_i|·σ_i / Σ_j |w_j|·σ_j

        Dropping the sign breaks the diverging color (blue long / red short) and makes the
        question "what % of risk is the duration **short**" unanswerable. The denominator is
        a sum of absolute values, so Σ|rc| = 1.

        ⚠️ Correlations are not reflected (same view as the paper's naive risk parity).
        Risk within the same block (4 duration, 4 USD cross) is summed and **overstated**.
        """
        vol = self._vol_for_risk()
        if vol is None:
            return None
        cmp = self.r.exposure.notional_vs_risk(vol)
        if cmp.empty:
            return None

        uni = self.r.universe
        codes = [c for c in uni.codes if c in cmp.index]
        cmp = cmp.loc[codes]
        sign = np.sign(cmp["position"]).replace(0.0, 1.0)
        signed = sign * cmp["risk_share"]

        def num(s, c, nd=5):
            v = s.get(c)
            return None if pd.isna(v) else round(float(v), nd)

        # Asset-class contrast: lets "largest by notional but not by risk" be read in one line.
        sec = cmp.groupby("sector")[["notional_share", "risk_share"]].sum()
        sec_signed = signed.groupby(cmp["sector"]).sum()
        sectors = [s for s in uni.sectors if s in sec.index]

        asof = cmp.attrs.get("as_of")
        return {
            "markets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "sector_of": {c: uni.get(c).sector for c in codes},
            # signed risk contribution share (Σ|·| = 1)
            "risk_signed": {c: num(signed, c) for c in codes},
            "notional": {c: num(cmp["position"], c) for c in codes},
            "notional_share": {c: num(cmp["notional_share"], c) for c in codes},
            "risk_share": {c: num(cmp["risk_share"], c) for c in codes},
            "vol_ann": {c: num(cmp["vol_ann"], c) for c in codes},
            "sectors": sectors,
            "sector_notional": {s: num(sec["notional_share"], s) for s in sectors},
            "sector_risk": {s: num(sec["risk_share"], s) for s in sectors},
            "sector_risk_signed": {s: num(sec_signed, s) for s in sectors},
            "asof": f"{asof:%Y-%m-%d}" if asof is not None else None,
        }

    def _age_grid(self) -> dict | None:
        """Position age x distance to flip: watch-list scatter.

        Age = how many weeks the current sign has held (sign run length).
        An old position close to flipping carries the largest reversal risk.
        The y axis toggles between σ and %. Since % is not comparable across markets
        (for CL, -33% is 2.15σ; for TY, +2.2% is 2.02σ), σ is the default.
        """
        flips = getattr(self.r, "triggers_flip", None) or {}
        if not flips:
            return None
        pos = self.r.exposure.pct_by_market()
        if pos is None or pos.empty:
            return None
        uni = self.r.universe
        codes = [c for c in uni.codes if c in pos.columns]

        age = {}
        for c in codes:
            sign = np.sign(pos[c].dropna())
            if sign.empty:
                continue
            last, n = sign.iloc[-1], 0
            for v in sign.to_numpy()[::-1]:
                if v == last:
                    n += 1
                else:
                    break
            age[c] = int(n)

        # ⚠️ Prices are on the **raw contract (C1)** basis. Momentum and volatility are
        #    computed on adjusted rolled futures as in the paper, but **adjusted prices must
        #    not be used when showing price levels on screen**: adjusted prices are
        #    retroactively revised history, not actual quotes.
        #    ScenarioEngine.triggers(prices=front_prices) is passed C1, and since ratio
        #    back-adjustment does not touch the front contract, the adjusted price and C1
        #    are identical at the latest point (checked: all 16 markets match).
        def num(t, c, col, mult=1.0, nd=3):
            if c not in t.index or pd.isna(t.at[c, col]):
                return None
            return round(float(t.at[c, col]) * mult, nd)

        per_h = {}
        for h, trig in sorted(flips.items()):
            t = trig.table
            per_h[str(h)] = {
                c: [num(t, c, "own_sigma"), num(t, c, "own_move_pct", 100.0),
                    num(t, c, "flip_price", 1.0, 4), num(t, c, "last_price", 1.0, 4)]
                for c in codes
            }
        return {
            "markets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "sector": {c: uni.get(c).sector for c in codes},
            "age": age,
            "pos": {c: round(float(pos[c].iloc[-1]) * 100, 2) for c in codes},
            "horizons": list(per_h),
            "trig": per_h,          # {horizon: {market: [σ, %]}}
            # Watch zone boundary: held 40 weeks or more and within 1.5σ
            "watch": [40, 1.5],
            # Price as-of date (per market). Can differ from the last file row because of ffill.
            "asof": {c: f"{d:%Y-%m-%d}"
                     for c, d in _as_items(getattr(self.r, "price_asof", None))
                     if c in codes},
        }

    def _scenario_grid(self, drivers: list[str], main: str) -> dict | None:
        """Data for the integrated scenario card: JS draws the bar chart itself.

        There are three axes (aggregation unit, metric, unit), so pre-baking combinations
        would give dozens of charts (driver x 2 x 2 x 2). Instead only the **ratio (pp)
        values** are shipped and combined in the browser. Since $BN = pp x AUM($BN), it
        also follows the AUM slider without recomputation.
        """
        uni = self.r.universe
        drivers = [d for d in drivers if self._scenario_proj(d) is not None]
        if not drivers:
            return None

        def frame(df: pd.DataFrame, keys: list[str], order: list[str]) -> dict:
            """{scenario_key: [values...]}: ordered by `order`, truncated to 5 decimals."""
            d = df.reindex(order)
            return {k: [None if pd.isna(v) else round(float(v), 5) for v in d[k]]
                    for k in keys}

        first = self._scenario_proj(drivers[0])
        codes = [c for c in uni.codes if c in first.positions.index]
        sectors = [s for s in uni.sectors
                   if any(uni.get(c).sector == s for c in codes)]

        per_driver: dict[str, dict] = {}
        for d in drivers:
            proj = self._scenario_proj(d)
            keys = proj.keys
            sec_pos = proj.by_sector("positions").reindex(sectors)
            sec_trd = proj.by_sector("trades_pct").reindex(sectors)
            cur = proj.current.reindex(codes)
            cur_sec = uni.aggregate_by_sector(
                cur.fillna(0.0).to_frame().T).T.iloc[:, 0].reindex(sectors)
            per_driver[d] = {
                "asset": {"positions": frame(proj.positions, keys, codes),
                          "trades": frame(proj.trades_pct, keys, codes)},
                "sector": {"positions": frame(sec_pos, keys, sectors),
                           "trades": frame(sec_trd, keys, sectors)},
                "current": {"asset": [None if pd.isna(v) else round(float(v), 5)
                                      for v in cur],
                            "sector": [None if pd.isna(v) else round(float(v), 5)
                                       for v in cur_sec]},
                # By what % the driver is assumed to move in each scenario
                "moves": {k: round(float(proj.driver_move()[k]), 5) for k in keys},
                # % move each market actually assumes (after propagation). Differs by market in independent mode.
                "shocks": frame(proj.shocks_pct, keys, codes),
            }

        # 'Independent' is not put in the driver dropdown but shipped separately:
        # driver_move() is empty (the shock differs by market), which would break the
        # 'assumption' text of the existing card.
        indep = self.r.projection_independent
        if indep is not None:
            keys = indep.keys
            sec_pos = indep.by_sector("positions").reindex(sectors)
            sec_trd = indep.by_sector("trades_pct").reindex(sectors)
            cur = indep.current.reindex(codes)
            cur_sec = uni.aggregate_by_sector(
                cur.fillna(0.0).to_frame().T).T.iloc[:, 0].reindex(sectors)
            per_driver[INDEP_KEY] = {
                "asset": {"positions": frame(indep.positions, keys, codes),
                          "trades": frame(indep.trades_pct, keys, codes)},
                "sector": {"positions": frame(sec_pos, keys, sectors),
                           "trades": frame(sec_trd, keys, sectors)},
                "current": {"asset": [None if pd.isna(v) else round(float(v), 5)
                                      for v in cur],
                            "sector": [None if pd.isna(v) else round(float(v), 5)
                                       for v in cur_sec]},
                "moves": {},
                "shocks": frame(indep.shocks_pct, keys, codes),
            }

        light, dark = self.factories["light"], self.factories["dark"]
        sig = first.sigmas()
        return {
            "drivers": [{"code": d, "label": self._driver_label(d)} for d in drivers],
            "default_driver": main if main in drivers else drivers[0],
            "scenarios": [{"key": s.key, "label": s.label_ko,
                           "sigma": round(first.sigma_of(s.key), 3)}
                          for s in first.scenarios],
            "scen_colors": {"light": light.t.scenario_colors(sig),
                            "dark": dark.t.scenario_colors(sig)},
            "sectors": sectors,
            "assets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "sector_of": {c: uni.get(c).sector for c in codes},
            "data": per_driver,
            # σ range covered by the scenarios: the basis for saying "certain within this range"
            "sigma_span": [min(sig), max(sig)],
            # Number of drivers agreeing on direction {code: [sell, buy, uncertain]}: unconditional robustness
            "agreement": self._direction_agreement(drivers, codes),
            # Independent-mode key/label: default of the cumulative flow card
            "indep_key": INDEP_KEY if INDEP_KEY in per_driver else None,
            "indep_label": INDEP_LABEL,
            # WTD: the part realized since the last close of the weekly grid.
            # The model only uses complete weeks, so this stretch is **actual movement not yet
            # reflected**. Next to the scenario assumption it shows e.g. "assumed -3.1% but
            # already -0.4% so far".
            # ⚠️ Do not wrap a Series in `or {}`: bool(Series) raises an exception.
            "wtd": {c: round(float(v), 6)
                    for c, v in _as_items(getattr(self.r, "wtd", None))
                    if c in codes and pd.notna(v)},
            "wtd_meta": {
                "days": int(getattr(self.r, "wtd_days", 0) or 0),
                "from": (f"{self.r.wtd_from:%Y-%m-%d}"
                         if getattr(self.r, "wtd_from", None) is not None else None),
                "to": (f"{self.r.wtd_to:%Y-%m-%d}"
                       if getattr(self.r, "wtd_to", None) is not None else None),
            },
            "bar_colors": self._bar_colors(),
        }

    def _bar_colors(self) -> dict:
        """Long/short diverging bar colors (light, dark). Shared by the scenario and trade pressure cards."""
        return {
            name: {
                "pos": f.t.pos_pole, "neg": f.t.neg_pole, "mid": f.t.baseline,
                "posFade": _fade(f.t.pos_pole, f.t.surface, 0.55),
                "negFade": _fade(f.t.neg_pole, f.t.surface, 0.55),
                "midFade": _fade(f.t.baseline, f.t.surface, 0.45),
                # diverging scale for heatmaps (uses the same pole colors as the bars)
                "scale": f.t.diverging_scale(),
            }
            for name, f in (("light", self.factories["light"]),
                            ("dark", self.factories["dark"]))
        }

    #: Tabs where the sidebar asset-class filter applies. The Model monitoring and Data tabs
    #: hold values for one portfolio of all 16 markets combined (R², tracking error,
    #: turnover, etc.), which cannot be split by asset class.
    FILTER_TABS = ("main", "exposure", "scenario", "flow", "cftc")

    def _sector_meta(self) -> dict:
        """For the sidebar: asset-class list and market -> asset-class mapping."""
        uni = self.r.universe
        sectors = [s for s in uni.sectors if any(uni.get(c).sector == s for c in uni.codes)]
        return {
            "sectors": sectors,
            "of": {c: uni.get(c).sector for c in uni.codes},
            "counts": {s: sum(1 for c in uni.codes if uni.get(c).sector == s) for s in sectors},
            "total": len(uni.codes),
        }

    def _overview_grid(self) -> dict | None:
        """Data for the position overview: current notional per market and change versus N weeks ago.

        JS classifies the regime (net long / net short / neutral, long build / short build,
        ...) using thresholds. $ is notional x AUM, so it follows the slider.
        """
        pos = self.r.exposure.pct_by_market()
        cfg = self.r.run_config
        weeks = cfg.regime_flow_weeks
        if pos is None or pos.empty or len(pos.dropna(how="all")) <= weeks:
            return None
        pos = pos.dropna(how="all")
        uni = self.r.universe
        codes = [c for c in uni.codes if c in pos.columns]
        cur, prev = pos.iloc[-1], pos.iloc[-1 - weeks]
        # Percentile versus own history (signed notional, 0 = largest short, 1 = largest long)
        win = pos[pos.index > pos.index[-1] - pd.DateOffset(years=cfg.overview_pctile_years)]
        pct = pd.Series({c: (win[c].dropna() <= cur[c]).mean() for c in codes
                         if pd.notna(cur.get(c)) and win[c].notna().any()})
        # Flow threshold: quantile of |N-week change in weekly notional| over the last N years per market
        thr = win.diff(weeks).abs().quantile(cfg.regime_flow_quantile)

        def col(s: pd.Series) -> list:
            return [None if pd.isna(s.get(c)) else round(float(s[c]), 5) for c in codes]

        return {
            "assets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "sector_of": {c: uni.get(c).sector for c in codes},
            "sectors": [s for s in uni.sectors
                        if any(uni.get(c).sector == s for c in codes)],
            "notional": col(cur),
            "change": col(cur - prev),
            "pctile": col(pct),
            "pctile_years": cfg.overview_pctile_years,
            "asof": f"{pos.index[-1]:%Y-%m-%d}",
            "prev": f"{pos.index[-1 - weeks]:%Y-%m-%d}",
            "flow_weeks": weeks,
            "flow_threshold": col(thr),
            "flow_quantile": cfg.regime_flow_quantile,
            "bar_colors": self._bar_colors(),
        }

    def _price_grid(self) -> dict | None:
        """Price history + weekly notional + flip price (per market). JS draws the chart.

        Prices are the **raw contract (C1)**. The flip price is the last C1 close times
        (1 + required move), so the same yardstick is needed for them to overlay on the
        chart. Adjusted rolled futures cannot be used because their history is
        retroactively revised. The ffill tail after each market's latest close date
        (price_asof) is cut.
        """
        r = self.r
        cfg = r.run_config
        fp = r.front_prices
        flips = r.triggers_flip or {}
        pos = r.exposure.pct_by_market()
        if fp is None or not flips or pos is None or pos.empty:
            return None
        H = cfg.trigger_horizon_weeks if cfg.trigger_horizon_weeks in flips else next(iter(flips))
        tbl = flips[H].table
        if tbl is None or tbl.empty:
            return None

        uni = r.universe
        codes = [c for c in uni.codes
                 if c in tbl.index and c in fp.columns and c in pos.columns]
        if not codes:
            return None

        asof_map = {c: pd.Timestamp(v) for c, v in _as_items(getattr(r, "price_asof", None))
                    if pd.notna(v)}
        end = max(asof_map.values()) if asof_map else pd.Timestamp(fp.index[-1])
        start = end - pd.DateOffset(years=cfg.price_chart_years)
        dates = pd.bdate_range(start, end)

        def num(v, nd=6):
            return None if pd.isna(v) else float(f"{float(v):.{nd}g}")

        # Flip table per horizon. The 4w one is the existing triggers_flip; others are computed on the fly only when equal to the scenario engine horizon
        flip_tabs: dict[int, pd.DataFrame] = {}
        for h in cfg.price_flip_horizons:
            t = flips[h].table if h in flips else None
            eng = getattr(r, "scenario_engine", None)
            if t is None and eng is not None and eng.horizon == h:
                try:
                    t = eng.triggers(mode="independent", driver=None, prices=fp,
                                     sigma_range=12.0, n_points=241).table
                except Exception:
                    t = None
            if t is not None and not t.empty:
                flip_tabs[int(h)] = t

        px, meta = {}, {}
        age_map = dict((self.age_grid or {}).get("age") or {})
        wtd = {c: v for c, v in _as_items(getattr(r, "wtd", None)) if pd.notna(v)}
        for c in codes:
            s = fp[c].dropna()
            if c in asof_map:
                s = s[s.index <= asof_map[c]]
            px[c] = [num(v) for v in s.reindex(dates)]
            # Weeks where the position sign changed (zeros are skipped). The price is the C1 close just before that date.
            sg = np.sign(pos[c].dropna())
            sg = sg[sg != 0]
            chg = sg[(sg != sg.shift()) & sg.shift().notna() & (sg.index >= start)]
            fs = fp[c].dropna()
            flips_c = []
            for d, sv in chg.items():
                pv_ = fs.asof(d)
                if pd.notna(pv_):
                    flips_c.append([f"{d:%Y-%m-%d}", num(pv_), "long" if sv > 0 else "short"])
            row = tbl.loc[c]
            meta[c] = {
                "flips": flips_c,
                "last_price": num(row.get("last_price")),
                "flip_h": {
                    str(h): {
                        "price": num(t.loc[c].get("flip_price")),
                        "move": num(t.loc[c].get("own_move_pct"), 5),
                        "sigma": num(t.loc[c].get("own_sigma"), 4),
                        "flat_pos": num(t.loc[c].get("flat_pos"), 5),
                    }
                    for h, t in flip_tabs.items() if c in t.index
                },
                "current_pos": num(row.get("current_pos"), 5),
                "asof": f"{asof_map[c]:%Y-%m-%d}" if c in asof_map else None,
                "age": age_map.get(c),
                "wtd": num(wtd.get(c), 5),
            }

        pw = pos.loc[pos.index >= start, codes]
        default = (tbl.loc[codes, "own_sigma"].abs().sort_values().index[0]
                   if tbl.loc[codes, "own_sigma"].notna().any() else codes[0])
        return {
            "horizon": int(H),
            "horizons": sorted(flip_tabs),
            # Horizon colors: categorical slots 1 and 3, so they do not clash with the long/short blue and red. Theme independent.
            "horizon_colors": dict(zip(map(str, sorted(flip_tabs)),
                                       (self.factories["light"].t.series[0],
                                        self.factories["light"].t.series[2]))),
            "assets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "default": default,
            "dates": [f"{d:%Y-%m-%d}" for d in dates],
            "price": px,
            "pos_dates": [f"{d:%Y-%m-%d}" for d in pw.index],
            "pos": {c: [num(v, 5) for v in pw[c]] for c in codes},
            "meta": meta,
            "bar_colors": self._bar_colors(),
        }

    #: Trade pressure tier labels (pressure.TIERS order: sell -> buy)
    PRESSURE_TIER_LABELS = {
        "sell_sure": "Strong sell pressure", "sell_lean": "Sell pressure", "mixed": "Neutral",
        "buy_lean": "Buy pressure", "buy_sure": "Strong buy pressure",
    }

    def _pressure_grid(self) -> dict | None:
        """Data for the trade pressure card (JS draws it). $ is pp x AUM, so it follows the slider."""
        tp = getattr(self.r, "trade_pressure", None)
        if tp is None or tp.table.empty:
            return None
        uni = self.r.universe
        t = tp.table
        codes = [c for c in uni.codes if c in t.index]

        def col(name: str, nd: int = 5) -> list:
            return [None if pd.isna(v) else round(float(v), nd) for v in t.loc[codes, name]]

        def txt(name: str) -> list:
            return [None if pd.isna(v) else str(v) for v in t.loc[codes, name]]

        return {
            "assets": codes,
            "names": {c: uni.get(c).name for c in codes},
            "p_buy": col("p_buy", 4),
            "p_buy_norm": col("p_buy_norm", 4),
            "e_trade": col("e_trade_pct"),
            "sd": col("sd_pct"),
            "flat": col("flat_pct"),
            "flip_z": col("flip_z", 3),
            "flip_pct": col("flip_pct", 5),
            "flip_above": txt("flip_above"),
            "tier": txt("tier"),
            "method": txt("method"),
            "tier_labels": self.PRESSURE_TIER_LABELS,
            "sure_prob": tp.sure_prob,
            "lean_prob": tp.lean_prob,
            "bar_colors": self._bar_colors(),
        }

    def _direction_agreement(self, drivers: list[str], codes: list[str]) -> dict:
        """Per market, 'how many drivers give a certain trade in the same direction'.

        Fixing one driver, some markets keep the same sign across all 5 steps (= certain
        trade). This counts that verdict over each of the 16 drivers. It gives no size, but
        is a direction robustness metric that can be stated **unconditionally**.
        Example: ES is unanimous sell at 16/16, JB only 6/16.
        """
        out: dict[str, list[int]] = {}
        for c in codes:
            sell = buy = mixed = 0
            for d in drivers:
                proj = self._scenario_proj(d)
                if proj is None or c not in proj.trades_pct.index:
                    continue
                row = proj.trades_pct.loc[c]
                if (row < 0).all():
                    sell += 1
                elif (row > 0).all():
                    buy += 1
                else:
                    mixed += 1
            out[c] = [sell, buy, mixed]
        return out

    def _table_scenario_sector(self, driver: str | None) -> str:
        proj = self._scenario_proj(driver)
        if proj is None:
            return "<p class='muted'>Not available</p>"
        trades = proj.by_sector("trades_usd") / 1e9
        positions = proj.by_sector("positions")

        out = pd.DataFrame(index=trades.index)
        for s in proj.scenarios:
            out[f"{s.label_ko} trade $BN"] = trades[s.key].map("{:+,.1f}".format)
            out[f"{s.label_ko} position"] = positions[s.key].map("{:+.0%}".format)
        return self._df_to_html(out, "Sector")

    def _table_scenario_market(self, driver: str | None) -> str:
        proj = self._scenario_proj(driver)
        if proj is None:
            return "<p class='muted'>Not available</p>"
        order = [c for c in self.r.universe.codes if c in proj.positions.index]
        usd = proj.trades_usd.loc[order] / 1e9
        pos = proj.positions.loc[order]

        out = pd.DataFrame(
            {
                "Market": pd.Series(self.r.universe.name_map).reindex(order),
                "Current position": proj.current.reindex(order).map("{:+.1%}".format),
            }
        )
        for s in proj.scenarios:
            out[f"{s.label_ko} $BN"] = usd[s.key].map("{:+,.1f}".format)
            out[f"{s.label_ko} pos"] = pos[s.key].map("{:+.0%}".format)
        return self._df_to_html(out, "Code")

    def _table_flip(self, horizon: int) -> str:
        """Flip trigger: table of "how far must it move for long <-> short to flip".

        Uses **independent mode** (no driver). Reason:
        the index propagation mode has `shock_i = β(i vs driver) × driver_move`, so
        markets with a small β never reach a flip inside the search range (±12σ) and
        become NaN (e.g. β(TY vs ES)=0.02 makes TY vanish from the table).
        But **the point where a market's sign flips depends only on that market's own
        momentum, so it is independent of the driver**, as verified empirically (switching
        the driver between ES/TY, markets with the same value differ by at most 0.0005pp).
        So the driver only changed "which rows survive" and was noise; computing in
        independent mode gives values for all 16 markets.
        """
        trig = (self.r.triggers_flip or {}).get(horizon)
        if trig is None or trig.table.empty:
            return "<p class='muted'>Not available.</p>"
        df = trig.table.copy()

        asof_map = {c: v for c, v in _as_items(getattr(self.r, "price_asof", None))}
        age_map = dict((self.age_grid or {}).get("age") or {})
        wtd_map = {c: float(v) for c, v in _as_items(getattr(self.r, "wtd", None))}
        wtd_days = int(getattr(self.r, "wtd_days", 0) or 0)
        wtd_hdr = f" ({self.r.wtd_from:%m-%d}→{self.r.wtd_to:%m-%d})" if wtd_days else ""
        if not wtd_days:
            wtd_map = {}

        def per_week(v):
            if pd.isna(v):
                return DASH
            return f"{(1.0 + v) ** (1.0 / horizon) - 1.0:+.2%}"

        out = pd.DataFrame(
            {
                "Market": df["name"],
                "Sector": df["sector"],
                "Current position": df["current_pos"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1%}"
                ),
                # The scatter's table view was removed and its unique information (weeks the
                # sign has held) was absorbed here. The point is that "short for 85 weeks,
                # +2.0% to go" reads in one line.
                "Position age": [
                    DASH if c not in age_map else f"{age_map[c]}w"
                    for c in df.index
                ],
                # ⚠️ Direction is based on the **position after the flat case**, not the
                #    current position. Because the momentum window slides, some markets
                #    flip sign even if the price does not move.
                f"Position after {horizon}w flat": df["flat_pos"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1%}"
                ),
                "Flip direction": [
                    DASH if pd.isna(mv)
                    else ("long → short" if f > 0 else ("short → long" if f < 0 else "neutral → " +
                          ("long" if mv > 0 else "short")))
                    + ("  ⚠️ already flips in the flat case"
                       if (pd.notna(c) and pd.notna(f) and c * f < 0) else "")
                    for c, f, mv in zip(df["current_pos"], df["flat_pos"],
                                        df["own_move_pct"])
                ],
                "Required move": df["own_move_pct"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.2%}"
                ),
                # The move that has already happened: placed right next to the required
                # move, "how much is left" reads as a comparison of two numbers.
                f"Realized WTD{wtd_hdr}": [
                    DASH if c not in wtd_map or pd.isna(wtd_map[c])
                    else f"{wtd_map[c]:+.2%}"
                    for c in df.index
                ],
                "Current price": df["last_price"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:,.2f}"
                ),
                # Without stating **as of when** the price is, it misleads (because of ffill,
                # the last file row != the market's latest close date)
                "As of": [
                    DASH if c not in asof_map else f"{asof_map[c]:%m-%d}"
                    for c in df.index
                ],
                "Flip price": df["flip_price"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:,.2f}"
                ),
                # Derived and technical metrics go to the far right
                "Per week": df["own_move_pct"].map(per_week),
                f"{horizon}w σ": df["own_sigma"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1f}σ"
                ),
            }
        )
        # Closest to flipping first (by σ; % is not comparable because volatility differs by market)
        out = out.loc[df["own_sigma"].abs().sort_values(na_position="last").index]

        # The explanation is already in the card subtitle, and 'already flips in the flat
        # case' is attached to the row with ⚠️, so below the table only the WTD reference
        # window is left in one line.
        note = (
            "<p class='muted'>Realized WTD = "
            + (f"{self.r.wtd_from:%Y-%m-%d} → {self.r.wtd_to:%Y-%m-%d} "
               f"({wtd_days} business days)" if wtd_days else "no new data")
            + "</p>"
        )
        return self._df_to_html(out, "Code") + note + self._price_asof_note()

    def _table_triggers(self, driver: str) -> str:
        trig = self.r.triggers.get(driver)
        if trig is None or trig.table.empty:
            return "<p class='muted'>Not available</p>"
        df = trig.table.copy()
        h = trig.horizon_weeks

        out = pd.DataFrame(
            {
                "Market": df["name"],
                "Sector": df["sector"],
                "Current position": df["current_pos"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1%}"
                ),
                "→": [
                    "No flip (robust)" if pd.isna(mv)
                    else ("Flip to short" if p > 0 else "Flip to long")
                    for p, mv in zip(df["current_pos"], df["own_move_pct"])
                ],
                f"Own price move ({h}w)": df["own_move_pct"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.2%}"
                ),
                "σ multiple": df["own_sigma"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1f}σ"
                ),
                "Trigger price": df["flip_price"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:,.2f}"
                ),
                "Current price": df["last_price"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:,.2f}"
                ),
                f"β vs {driver}": df["beta_vs_driver"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.2f}"
                ),
                f"{driver} move (reference)": df["driver_move_pct"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:+.1%}"
                ),
            }
        )
        # Closest to flipping first
        out = out.loc[df["own_move_pct"].abs().sort_values(na_position="last").index]

        note = (
            f"<p class='muted'><b>How to read</b>: the primary metric is <b>'Own price move'</b>: "
            f"if that market <b>itself</b> moves that much over {h} week{'s' if h != 1 else ''}, the position sign flips. "
            "It is always valid regardless of the driver.<br>"
            f"<b>'{driver} move' is for reference only</b> and is shown <b>only for markets with a sufficiently large β</b>. "
            "A small β divides by a small number and the value explodes. For example, for a market with β=−0.04 to "
            f"move +0.2%, {driver} would have to fall −66%, but this does "
            f"<b>not mean that {driver} causes that flip</b>.<br>"
            f"Why the horizon is <b>{h} weeks</b>: if too short (1 week), momentum is a 16-52 week <b>average</b>, so "
            "a flip needs an unrealistic shock (−5σ or more, recurrence of hundreds of thousands of years); "
            "if too long (13 weeks or more), most of the momentum window is covered by the shock and "
            "<b>the flip point automatically becomes 0%</b>, which is meaningless. "
            f"{h} weeks is the balance point that covers only 25% of the minimum lookback (16 weeks).<br>"
            "<b>'No flip (robust)'</b> means that no move within that horizon "
            "flips the position: all three trend speeds (16/32/52w) point the same way, "
            "so the position is very robust.</p>"
        )
        if self.r.front_prices is None:
            note += (
                "<p class='muted'>Trigger <b>prices</b> need raw contract (C1) prices to be accurate: "
                "add <code>data/raw/futures_c1.csv</code>. "
                "The move in % is valid without prices.</p>"
            )
        return note + self._df_to_html(out, "Code")

    def _table_scenario_assumptions(self, driver: str) -> str:
        r = self.r
        proj = r.projections.get(driver)
        engine = r.scenario_engine
        if proj is None or engine is None:
            return "<p class='muted'>Not available.</p>"

        moves = proj.driver_move()
        scen = pd.DataFrame(
            {
                "Target frequency": [
                    DASH if s.target_prob is None else f"Top/bottom {s.target_prob:.0%}"
                    for s in proj.scenarios
                ],
                "σ multiple (empirical)": [f"{proj.sigma_of(s.key):+.2f}σ" for s in proj.scenarios],
                "Empirical frequency": [
                    DASH if not np.isfinite(proj.empirical_freq.get(s.key, np.nan))
                    else f"{proj.empirical_freq[s.key]:.1%}"
                    for s in proj.scenarios
                ],
                "Times per year": [
                    DASH if not np.isfinite(proj.empirical_freq.get(s.key, np.nan))
                    else f"{proj.empirical_freq[s.key] * 52:.1f}"
                    for s in proj.scenarios
                ],
                f"{driver} assumed move": [f"{moves[s.key]:+.2%}" for s in proj.scenarios],
            },
            index=[s.display for s in proj.scenarios],
        )
        # Normality diagnostics are filled from actual data on every build (no hardcoding)
        try:
            ns = engine.normality_summary()
            lb = engine.ensemble.models[0].params.volatility_lookback
            normality = (
                f"Normality is rejected for {ns['n_rejected']:.0f} of {ns['n_markets']:.0f} markets' weekly returns, "
                f"with a mean excess kurtosis of {ns['excess_kurtosis_raw']:.1f}. "
                f"After standardizing by the {lb}-day σ, the mean excess kurtosis is {ns['excess_kurtosis_z']:.1f}, "
                f"and the 5-95% quantiles differ from the normal distribution by at most {ns['max_prob_gap'] * 100:.1f}pp in probability."
            )
        except Exception:
            normality = ""
        html = (
            "<h3 class='subhead'>Definition of the 5 scenarios</h3>"
            "<p class='muted'>Scenarios are defined by <b>target frequency</b>, and σ multiples are "
            "converted from the <b>empirical quantiles of the standardized return z = r/σ</b> "
            "(no normality assumption). " + normality + "</p>"
            + self._df_to_html(scen, "Scenario")
        )

        try:
            betas = engine.betas(driver)
            shocks = proj.shocks_pct
            prop = pd.DataFrame(
                {
                    "Market": pd.Series(r.universe.name_map),
                    f"β vs {driver}": betas.map("{:+.2f}".format),
                    "Move when Strong up": shocks["strong_up"].map("{:+.2%}".format),
                    "Move when Strong down": shocks["strong_down"].map("{:+.2%}".format),
                }
            ).loc[[c for c in r.universe.codes if c in betas.index]]
            html += (
                "<h3 class='subhead'>Cross-market propagation (beta)</h3>"
                f"<p class='muted'>Univariate betas on the past {r.run_config.scenario_beta_window_weeks} weeks of "
                "weekly returns. The replica model does not use correlations "
                "(naive risk parity); betas are used <b>only for scenario propagation</b>.</p>"
                + self._df_to_html(prop, "Code")
            )
        except Exception as exc:
            html += f"<p class='muted'>Could not compute betas: {exc}</p>"

        return html

    # -- Other tables ---------------------------------------------------- #

    def _vol_for_risk(self):
        """Annualized volatility panel for the risk contribution calculation."""
        model = self.r.ensemble.models[0]
        return None if model.signals is None else model.signals.vol_annualized

    def _table_notional_vs_risk(self) -> str:
        vol = self._vol_for_risk()
        if vol is None:
            return "<p class='muted'>No volatility panel.</p>"
        cmp = self.r.exposure.notional_vs_risk(vol)
        if cmp.empty:
            return "<p class='muted'>Not available.</p>"
        out = pd.DataFrame(
            {
                "Market": cmp["name"],
                "Sector": cmp["sector"],
                "Position": cmp["position"].map("{:+.1%}".format),
                "σ_ann": cmp["vol_ann"].map("{:.1%}".format),
                "Notional share": cmp["notional_share"].map("{:.1%}".format),
                "Risk share": cmp["risk_share"].map("{:.1%}".format),
                "Risk/notional": cmp["ratio"].map(
                    lambda v: DASH if pd.isna(v) else f"{v:.2f}x"
                ),
            }
        ).sort_values("Risk share", ascending=False)

        sec = cmp.groupby("sector")[["notional_share", "risk_share"]].sum()
        rows = "".join(
            f"<tr><th>{s}</th><td>{r['notional_share']:.1%}</td>"
            f"<td>{r['risk_share']:.1%}</td>"
            f"<td>{r['risk_share']/r['notional_share']:.2f}x</td></tr>"
            for s, r in sec.iterrows()
        )
        summary = (
            "<h3 class='subhead'>Asset class summary</h3>"
            "<table class='data-table'><thead><tr><th>Sector</th>"
            "<th>Notional share</th><th>Risk share</th><th>Ratio</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "<h3 class='subhead'>By market</h3>"
        )
        return summary + self._df_to_html(out, "Code")

    def _table_positions(self) -> str:
        df = self.r.snapshot.by_market.copy()
        out = pd.DataFrame(
            {
                "Market": df["name"],
                "Sector": df["sector"],
                "Side": df["side"],
                "Notional %": df["pct"].map("{:+.1%}".format),
                "$BN": (df["usd"] / 1e9).map("{:+,.1f}".format),
                "Δ week %": df["pct_change_wow"].map("{:+.1%}".format),
                "Δ week $BN": (df["usd_flow_wow"] / 1e9).map("{:+,.1f}".format),
            }
        )
        return self._df_to_html(out, "Code")

    def _table_momentum(self) -> str:
        rows = {}
        for model in self.r.ensemble.models:
            if model.signals is None:
                continue
            mom = model.signals.momentum.dropna(how="all")
            if mom.empty:
                continue
            rows[f"{model.params.momentum_lookback}w"] = mom.iloc[-1]
        if not rows:
            return "<p class='muted'>No signals.</p>"
        raw = pd.DataFrame(rows)
        out = self._fmt_frame(raw, "{:+.2f}")
        out.insert(0, "Market", pd.Series(self.r.universe.name_map))
        return self._df_to_html(out, "Code")

    def _table_stats(self) -> str:
        df = self.r.stats_table.copy()
        out = pd.DataFrame(
            {
                "R²": df["R2"].map("{:.1%}".format),
                "Correlation": df["Correlation"].map("{:.3f}".format),
                "Beta": df["Beta"].map("{:.3f}".format),
                "Alpha (ann)": df["Alpha (ann)"].map("{:+.2%}".format),
                "Tracking error (ann)": df["Tracking Error (ann)"].map("{:.2%}".format),
                "Weeks": df["Weeks"].map("{:,.0f}".format),
            }
        )
        return self._df_to_html(out, "Model")

    def _table_performance(self) -> str:
        return self._df_to_html(self.r.performance.round(3), "Series")

    def _table_recent(self, df: pd.DataFrame, fmt: str, n: int = 12) -> str:
        if df is None or df.empty:
            return "<p class='muted'>No data.</p>"
        tail = df.dropna(how="all").iloc[-n:]
        formatted = self._fmt_frame(tail, fmt)
        formatted.index = [d.strftime("%Y-%m-%d") for d in tail.index]
        return self._df_to_html(formatted[::-1], "Week ending")

    def _table_sweep_top(self, n: int = 20) -> str:
        sweep = self.r.sweep
        if sweep is None or sweep.empty:
            return "<p class='muted'>The sweep was not run.</p>"
        top = sweep.head(n).copy()
        out = pd.DataFrame(
            {
                "Params (mom/cap/vol)": top["label"].values,
                "Momentum (w)": top["momentum_lookback"].values,
                "Cap": [f"{v:g}" for v in top["cap"]],
                "Vol lookback (d)": top["vol_lookback"].values,
                "R²": [f"{v:.1%}" for v in top["r2"]],
                "Sharpe (unlevered)": [f"{v:.2f}" for v in top["sharpe"]],
            },
            index=np.arange(1, len(top) + 1),
        )
        return self._df_to_html(out, "Rank")

    def _table_flow_sensitivity(self) -> str:
        flow = self.r.flow_sensitivity
        if flow is None or flow.empty:
            return "<p class='muted'>Not available</p>"
        usd = flow.xs("usd", axis=1, level="metric") / 1e9
        out = self._fmt_frame(usd, "{:+,.1f}")
        out.columns = [f"{c:+.0%}" for c in usd.columns]
        out.insert(0, "Market", pd.Series(self.r.universe.name_map))
        return self._df_to_html(out, "Code")

    def _table_universe(self) -> str:
        uni = self.r.universe.to_frame().copy()
        uni["multiplier"] = uni["multiplier"].map("{:,.0f}".format)
        uni.columns = ["Market", "Sector", "Ticker (C1)", "Currency", "Multiplier"]
        return self._df_to_html(uni, "Code")

    def _table_validation(self) -> str:
        rep = self.r.validation
        frame = rep.to_frame()
        if frame.empty:
            return "<p class='muted'>No validation results.</p>"
        frame.columns = [
            "First valid", "Non-positive px", "Max gap (d)", "Max stale (d)", "Extreme moves",
        ]
        head = "<p class='muted'>" + rep.summary().replace("\n", "<br>") + "</p>"
        return head + self._df_to_html(frame, "Code")

    def _table_config(self) -> str:
        r, cfg = self.r, self.r.run_config
        ens = r.ensemble
        fit = ens.fit_result
        rows = {
            "Benchmark": BENCHMARK_NAME,
            "Member models (momentum/cap/vol)": ", ".join(ens.labels),
            "Member model weights": ", ".join(f"{k}={v:.0%}" for k, v in ens.weights.items()),
            "Leverage scaling": fit.method if fit else DASH,
            "Ensemble rescale factor": f"{ens.rescale_factor:.3f}",
            "Target volatility (benchmark, annualized)": f"{fit.target_vol:.2%}" if fit else DASH,
            "AUM assumption": f"${cfg.aum_usd / 1e9:,.0f}BN",
            "Regression lookback": f"{cfg.regression_window_weeks}w",
            "Rolling R² lookback": f"{cfg.rolling_r2_window_weeks}w",
            "Sample period": (
                f"{r.weekly_returns.index[0]:%Y-%m-%d} ~ "
                f"{r.weekly_returns.index[-1]:%Y-%m-%d}"
            ),
            "Weekly observations": f"{int(ens.returns.notna().sum()):,}",
            "Number of markets": f"{len(r.universe)}",
            "Run time": f"{r.elapsed_sec:.1f}s",
        }
        html = self._df_to_html(pd.DataFrame({"Value": pd.Series(rows)}), "Item")
        if r.warnings:
            items = "".join(f"<li>{w}</li>" for w in r.warnings)
            html += f"<div class='warnbox'><b>Warnings</b><ul>{items}</ul></div>"
        return html

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(
        self,
        output_path: Path | str | None = None,
        open_browser: bool = False,
        default_theme: str = "light",
        standalone: bool = False,
    ) -> Path:
        """Build the HTML, save it and return the path.

        Parameters
        ----------
        standalone : if True, embed plotly.js (~4.8MB) in the file.
            It opens without internet/CDN access, so use it when **distributing the file
            to others**. If False, plotly.js is loaded from a CDN, so the file is light
            but needs a network.
        """
        self.paths.ensure()
        path = Path(output_path or self.paths.dashboard_html)

        tabs = self._build_tabs()

        # Build the variant map with the same card key the template creates (`<card.id>-<index>`)
        variants: dict[str, dict] = {}
        for tab in tabs:
            for i, card in enumerate(tab.cards):
                if not (card.variants or card.variant_tables):
                    continue
                variants[f"{card.id}-{i}"] = {
                    "chartIds": card.variants or {},
                    "tables": card.variant_tables or {},
                }

        plotlyjs = ""
        if standalone:
            from plotly.offline import get_plotlyjs

            plotlyjs = get_plotlyjs()

        template = self.env.get_template("dashboard.html.j2")
        html = template.render(
            plotlyjs=plotlyjs,
            title="CTA Positioning Replication",
            subtitle=(
                "Kestner (2020) · <i>Replicating CTA Positioning: An Improved Method</i> "
                "· SSRN 3674828"
            ),
            as_of=self.r.snapshot.date.strftime("%Y-%m-%d"),
            generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            tabs=tabs,
            charts_json=json.dumps(
                {
                    cid: {t: json.loads(v) for t, v in variants.items()}
                    for cid, variants in self.charts.items()
                },
                separators=(",", ":"),
            ),
            variants_json=json.dumps(variants, separators=(",", ":")),
            usd_charts_json=json.dumps(self.usd_charts, separators=(",", ":")),
            time_charts_json=json.dumps(self.time_charts, separators=(",", ":")),
            exposure_grid_json=json.dumps(self.exposure_grid,
                                          separators=(",", ":"),
                                          ensure_ascii=False)
            if self.exposure_grid else "null",
            date_min=str(self.r.weekly_returns.index[0].date()),
            date_max=str(self.r.weekly_returns.index[-1].date()),
            scenario_grid_json=json.dumps(self.scenario_grid, separators=(",", ":"),
                                          ensure_ascii=False) if self.scenario_grid else "null",
            sector_meta_json=json.dumps(self._sector_meta(), separators=(",", ":"),
                                        ensure_ascii=False),
            sector_charts_json=json.dumps(self.sector_charts, separators=(",", ":"),
                                          ensure_ascii=False),
            filter_tabs=list(self.FILTER_TABS),
            price_grid_json=json.dumps(self.price_grid, separators=(",", ":"),
                                       ensure_ascii=False) if self.price_grid else "null",
            overview_grid_json=json.dumps(self.overview_grid, separators=(",", ":"),
                                          ensure_ascii=False) if self.overview_grid else "null",
            pressure_grid_json=json.dumps(self.pressure_grid, separators=(",", ":"),
                                          ensure_ascii=False) if self.pressure_grid else "null",
            flow_grid_json=json.dumps(self.flow_grid, separators=(",", ":"),
                                      ensure_ascii=False) if self.flow_grid else "null",
            age_grid_json=json.dumps(self.age_grid, separators=(",", ":"),
                                     ensure_ascii=False) if self.age_grid else "null",
            risk_grid_json=json.dumps(self.risk_grid, separators=(",", ":"),
                                      ensure_ascii=False) if self.risk_grid else "null",
            aum_base=self.r.run_config.aum_usd / 1e9,
            default_theme=default_theme,
            benchmark=BENCHMARK_NAME,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

        if open_browser:
            webbrowser.open(path.resolve().as_uri())
        return path


__all__ = ["DashboardBuilder", "Card", "Tab"]
