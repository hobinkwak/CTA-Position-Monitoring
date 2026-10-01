"""
Plotly figure factory.

Kept separate from rendering (HTML assembly), so this file can be reused as is
if the dashboard moves to Dash or Streamlit.

Design rules:
  * No dual axes. Two measures on different scales go in separate charts.
  * Categorical colors are fixed per entity, never reassigned by rank.
  * Thin marks (2px lines), recessive solid hairline grid/axes (no dashes).
  * Selective direct labels (line endpoints) only. No numbers on every point.
  * Legend always shown when there are two or more series.
  * In light mode some series colors have contrast below 3:1, so direct labels
    and a table view act as relief.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import BENCHMARK_NAME
from ..universe import Universe
from .theme import Theme

#: Common chart margin
_MARGIN = dict(l=56, r=96, t=16, b=44)
_LINE_WIDTH = 2
_HAIRLINE = 1


class FigureFactory:
    """PipelineResult -> Plotly figures.

    Parameters
    ----------
    result : PipelineResult
    theme : Theme (light / dark)

    Notes
    -----
    Each method returns a `go.Figure`. `builder.py` turns it into an HTML fragment.
    """

    def __init__(self, result, theme: Theme) -> None:
        self.r = result
        self.t = theme
        self.universe: Universe = result.universe

    # ------------------------------------------------------------------ #
    # Common layout
    # ------------------------------------------------------------------ #

    def _layout(self, **overrides) -> dict:
        t = self.t
        base = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=t.font, size=12, color=t.text_secondary),
            margin=_MARGIN,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=t.surface,
                bordercolor=t.border,
                font=dict(family=t.font, size=12, color=t.text_primary),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=t.text_secondary, size=11),
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                linecolor=t.baseline,
                linewidth=_HAIRLINE,
                tickcolor=t.baseline,
                tickfont=dict(color=t.muted, size=11),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=t.gridline,
                gridwidth=_HAIRLINE,
                zeroline=True,
                zerolinecolor=t.baseline,
                zerolinewidth=_HAIRLINE,
                showline=False,
                tickfont=dict(color=t.muted, size=11),
            ),
            height=380,
            # Drag zoom is kept (x direction only). It works alongside the top
            # period control; pressing "All" there also resets the drag zoom.
            dragmode="zoom",
        )
        base.update(overrides)
        return base

    def _endpoint_label(
        self, fig: go.Figure, x, y, text: str, color: str, fmt: str = "{:.0%}"
    ) -> None:
        """Direct label at the line end. Also serves as relief for low-contrast light-mode colors."""
        if y is None or (isinstance(y, float) and not np.isfinite(y)):
            return
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{text}</b> {fmt.format(y)}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(family=self.t.font, size=11, color=color),
            bgcolor="rgba(0,0,0,0)",
        )

    @staticmethod
    def _empty(message: str, theme: Theme) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            showarrow=False,
            font=dict(family=theme.font, size=13, color=theme.muted),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=220,
            margin=_MARGIN,
        )
        return fig

    # ================================================================== #
    # 1. Replication quality
    # ================================================================== #

    def cumulative_comparison(self) -> go.Figure:
        """Replica vs benchmark **log cumulative return** (both indexed to 100, log y-axis).

        Why a log axis: equal vertical distance means **equal compound return**.
        On a linear axis over 26 years only the later period is visible and early
        tracking is lost. Ticks stay in index values (100/200/400) for readability.
        """
        t = self.t
        rep = self.r.ensemble.returns
        bm = self.r.benchmark_returns
        joined = pd.concat([rep.rename("Replica"), bm.rename(BENCHMARK_NAME)], axis=1).dropna()
        if joined.empty:
            return self._empty("No overlapping period between replica and benchmark.", t)

        indexed = 100 * (1 + joined).cumprod()
        fig = go.Figure()
        colors = {"Replica": t.series[0], BENCHMARK_NAME: t.series[1]}
        for col in indexed.columns:
            fig.add_trace(
                go.Scatter(
                    x=indexed.index,
                    y=indexed[col],
                    name=col,
                    mode="lines",
                    line=dict(color=colors[col], width=_LINE_WIDTH),
                    hovertemplate="%{y:.1f}<extra>" + col + "</extra>",
                )
            )
            self._endpoint_label(
                fig, indexed.index[-1], float(indexed[col].iloc[-1]), col, colors[col], "{:.0f}"
            )

        layout = self._layout()
        layout["yaxis"] = _ax(
            layout["yaxis"], type="log",
            title="Index (start = 100, log scale)",
        )
        fig.update_layout(**layout)
        return fig

    def return_scatter(self) -> go.Figure:
        """Weekly return scatter, replica vs benchmark."""
        t = self.t
        joined = pd.concat(
            [self.r.ensemble.returns.rename("rep"), self.r.benchmark_returns.rename("bm")], axis=1
        ).dropna()
        if len(joined) < 3:
            return self._empty("Not enough observations.", t)

        r2 = float(joined["rep"].corr(joined["bm"]) ** 2)
        slope, intercept = np.polyfit(joined["rep"], joined["bm"], 1)
        xs = np.linspace(joined["rep"].min(), joined["rep"].max(), 50)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=joined["rep"],
                y=joined["bm"],
                mode="markers",
                name="Weekly return",
                marker=dict(
                    color=t.series[0], size=6, opacity=0.55,
                    line=dict(width=2, color=t.surface),  # 2px surface ring
                ),
                hovertemplate="Replica %{x:.2%}<br>Benchmark %{y:.2%}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=slope * xs + intercept,
                mode="lines",
                name="Regression line",
                line=dict(color=t.text_secondary, width=_LINE_WIDTH),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"<b>R² = {r2:.1%}</b>   β = {slope:.2f}   n = {len(joined)} weeks",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(family=t.font, size=12, color=t.text_primary),
        )
        layout = self._layout(hovermode="closest", height=380)
        layout["xaxis"] = _ax(
            layout["xaxis"], title="Replica weekly return", tickformat=".0%",
            showgrid=True, gridcolor=t.gridline, zeroline=True, zerolinecolor=t.baseline,
        )
        layout["yaxis"] = _ax(layout["yaxis"], title=f"{BENCHMARK_NAME} weekly return", tickformat=".0%")
        fig.update_layout(**layout)
        return fig

    def rolling_r2(self) -> go.Figure:
        """Rolling R², replication quality over time."""
        t = self.t
        s = self.r.rolling_r2.dropna()
        if s.empty:
            return self._empty("Not enough observations to compute rolling R².", t)

        fig = go.Figure(
            go.Scatter(
                x=s.index, y=s, mode="lines", name="Rolling R²",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:.1%}<extra>Rolling R²</extra>",
            )
        )
        full = self.r.ensemble_r2
        fig.add_hline(
            y=full, line=dict(color=t.text_secondary, width=_HAIRLINE),
            annotation_text=f"Full sample {full:.1%}",
            annotation_position="top left",
            annotation_font=dict(family=t.font, size=11, color=t.text_secondary),
        )
        self._endpoint_label(fig, s.index[-1], float(s.iloc[-1]), "R²", t.series[0])
        layout = self._layout()
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%", range=[0, 1])
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 1b. Model monitoring: staleness checks
    # ================================================================== #

    def cumulative_gap(self) -> go.Figure:
        """Replica / benchmark cumulative ratio - 1. Tracks **path drift**.

        R² is scale invariant, so it misses a leverage mismatch. This chart fills
        that gap: the further from 0, the more the return path diverges from the
        benchmark.
        """
        t = self.t
        rep, bm = self.r.ensemble.returns, self.r.benchmark_returns
        j = pd.concat([rep.rename("r"), bm.rename("b")], axis=1).dropna()
        if len(j) < 52:
            return self._empty("Overlapping period is too short.", t)
        gap = (1 + j["r"]).cumprod() / (1 + j["b"]).cumprod() - 1.0

        fig = go.Figure(
            go.Scatter(
                x=gap.index, y=gap, mode="lines", name="Cumulative gap",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:+.1%}<extra>Cumulative gap</extra>",
            )
        )
        fig.add_hline(y=0.0, line=dict(color=t.baseline, width=_HAIRLINE))
        for v, lab in ((0.25, "+25%"), (-0.25, "−25%")):
            fig.add_hline(y=v, line=dict(color=t.baseline, width=_HAIRLINE),
                          annotation_text=lab, annotation_position="right",
                          annotation_font=dict(family=t.font, size=10, color=t.muted))
        self._endpoint_label(fig, gap.index[-1], float(gap.iloc[-1]), "Gap",
                             t.series[0], "{:+.1%}")
        layout = self._layout(height=340)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat="+.0%",
                              title="Cumulative ratio - 1 (0 = perfect tracking)")
        fig.update_layout(**layout)
        return fig

    def rolling_tracking_error(self, window: int = 104) -> go.Figure:
        """Rolling annualized tracking error: volatility of weekly (replica - benchmark) differences."""
        t = self.t
        rep, bm = self.r.ensemble.returns, self.r.benchmark_returns
        j = pd.concat([rep.rename("r"), bm.rename("b")], axis=1).dropna()
        if len(j) < window + 10:
            return self._empty("Not enough observations.", t)
        te = (j["r"] - j["b"]).rolling(window).std(ddof=1) * np.sqrt(52)
        te = te.dropna()

        fig = go.Figure(
            go.Scatter(
                x=te.index, y=te, mode="lines", name="Tracking error",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:.1%}<extra>Tracking error</extra>",
            )
        )
        full = float((j["r"] - j["b"]).std(ddof=1) * np.sqrt(52))
        fig.add_hline(
            y=full, line=dict(color=t.text_secondary, width=_HAIRLINE),
            annotation_text=f"Full sample {full:.1%}", annotation_position="top left",
            annotation_font=dict(family=t.font, size=11, color=t.text_secondary),
        )
        self._endpoint_label(fig, te.index[-1], float(te.iloc[-1]), "TE", t.series[0])
        layout = self._layout(height=340)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%",
                              title=f"{window}w rolling annualized tracking error")
        fig.update_layout(**layout)
        return fig

    def member_dispersion(self, window: int = 26) -> go.Figure:
        """Disagreement across trend speeds, an early sign of regime change.

        Larger when the 16/32/52-week members take different positions in the
        same market. A spike often marks a trend turning point, when **replication
        error also grows.**
        """
        t = self.t
        try:
            disp = self.r.ensemble.member_dispersion()
        except Exception:
            return self._empty("Member dispersion cannot be computed.", t)
        if disp is None or disp.empty:
            return self._empty("Member dispersion cannot be computed.", t)
        s = disp.mean(axis=1).rolling(window, min_periods=window // 2).mean().dropna()
        if s.empty:
            return self._empty("Not enough observations.", t)

        fig = go.Figure(
            go.Scatter(
                x=s.index, y=s, mode="lines", name="Member dispersion",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:.3f}<extra>Member dispersion</extra>",
            )
        )
        med = float(s.median())
        fig.add_hline(
            y=med, line=dict(color=t.text_secondary, width=_HAIRLINE),
            annotation_text=f"Median {med:.3f}", annotation_position="top left",
            annotation_font=dict(family=t.font, size=11, color=t.text_secondary),
        )
        self._endpoint_label(fig, s.index[-1], float(s.iloc[-1]), "Disp.",
                             t.series[0], "{:.3f}")
        layout = self._layout(height=340)
        layout["yaxis"] = _ax(layout["yaxis"],
                              title=f"{window}w average, cross-member position std dev by market")
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 2. Exposure
    # ================================================================== #

    def sector_exposure(self, usd: bool = False) -> go.Figure:
        """Notional exposure time series by sector."""
        t = self.t
        data = self.r.exposure.usd_by_sector() / 1e9 if usd else self.r.exposure.pct_by_sector()
        data = data.dropna(how="all")
        if data.empty:
            return self._empty("No exposure data.", t)

        colors = t.sector_colors(list(data.columns))
        fig = go.Figure()
        fmt = "{:+,.0f}" if usd else "{:+.0%}"
        hover = "%{y:+,.0f}B" if usd else "%{y:+.0%}"
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[col], name=col, mode="lines",
                    line=dict(color=colors[col], width=_LINE_WIDTH),
                    hovertemplate=hover + "<extra>" + col + "</extra>",
                )
            )
            self._endpoint_label(
                fig, data.index[-1], float(data[col].iloc[-1]), col, colors[col], fmt
            )

        layout = self._layout(height=420)
        layout["yaxis"] = _ax(
            layout["yaxis"],
            tickformat=",.0f" if usd else ".0%",
            title="Notional, $BN" if usd else "Exposure (% of AUM)",
        )
        fig.update_layout(**layout)
        return fig

    def current_positioning(self) -> go.Figure:
        """Current position by market, diverging by sign (long blue / short red)."""
        t = self.t
        snap = self.r.snapshot
        df = snap.by_market.sort_values("pct")
        if df.empty:
            return self._empty("No position data.", t)

        colors = [t.sign_color(v) for v in df["pct"]]
        labels = [f"{code} · {row['name']}" for code, row in df.iterrows()]
        fig = go.Figure(
            go.Bar(
                y=labels, x=df["pct"], orientation="h",
                marker=dict(color=colors, line=dict(width=2, color=t.surface)),
                text=[f"{v:+.0%}" for v in df["pct"]],
                textposition="outside",
                textfont=dict(family=t.font, size=11, color=t.text_secondary),
                hovertemplate="%{y}<br>%{x:+.1%} · $%{customdata:+,.1f}B<extra></extra>",
                customdata=df["usd"] / 1e9,
                showlegend=False,
            )
        )
        layout = self._layout(hovermode="closest", height=520)
        layout["margin"] = dict(l=190, r=72, t=16, b=44)
        layout["xaxis"] = _ax(layout["xaxis"], tickformat="+.0%", showgrid=True,
                               gridcolor=t.gridline, zeroline=True, zerolinecolor=t.baseline)
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        return fig

    def position_heatmap(self, freq: str = "ME") -> go.Figure:
        """Market x time position heatmap (diverging scale).

        Weekly data is too large, so it is resampled to month end.
        """
        t = self.t
        pos = self.r.exposure.pct_by_market().dropna(how="all")
        if pos.empty:
            return self._empty("No position data.", t)

        monthly = pos.resample(freq).last()
        order = [c for c in self.universe.codes if c in monthly.columns]
        monthly = monthly[order]

        ylabels = [f"{c} · {self.universe.get(c).name}" for c in order]
        limit = float(np.nanpercentile(np.abs(monthly.values), 98)) or 1.0
        colorscale = [
            [0.0, t.neg_pole], [0.25, _fade(t.neg_pole, t.mid, 0.5)],
            [0.5, t.mid],
            [0.75, _fade(t.pos_pole, t.mid, 0.5)], [1.0, t.pos_pole],
        ]
        fig = go.Figure(
            go.Heatmap(
                z=monthly.T.values,
                x=monthly.index,
                y=ylabels,
                colorscale=colorscale,
                zmid=0, zmin=-limit, zmax=limit,
                hovertemplate="%{y}<br>%{x|%Y-%m}<br>%{z:+.1%}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Notional", font=dict(family=t.font, size=11, color=t.muted)),
                    tickformat="+.0%",
                    tickfont=dict(family=t.font, size=10, color=t.muted),
                    outlinewidth=0, thickness=10, len=0.85,
                ),
            )
        )
        # Fixed height proportional to row count. Height must **not change** when the period is narrowed.
        layout = self._layout(hovermode="closest", height=max(520, 30 * len(order) + 90))
        layout["margin"] = dict(l=190, r=16, t=16, b=44)
        # `tickmode="array"` **forces every market label** to be drawn.
        # Otherwise Plotly skips category labels when it judges space is short
        # (narrowing the period adds x labels and y labels were actually cut off).
        layout["yaxis"] = _ax(
            layout["yaxis"], showgrid=False, zeroline=False, autorange="reversed",
            type="category", tickmode="array", tickvals=ylabels, ticktext=ylabels,
            automargin=False,
        )
        # Pin x to a date axis so a selected period is not mistaken for categories
        layout["xaxis"] = _ax(layout["xaxis"], type="date")
        fig.update_layout(**layout)
        return fig

    def _vol_panel(self) -> pd.DataFrame | None:
        model = self.r.ensemble.models[0]
        if model.signals is None:
            return None
        return model.signals.vol_annualized

    def notional_vs_risk(self, codes: list[str] | None = None) -> go.Figure:
        """Notional share vs risk share: the same markets on two yardsticks.

        Shows the point of volatility-adjusted equal weighting directly. Notional
        piles into low-volatility markets (rates) but their risk is not as large.

        codes: asset class filter. The share denominator stays the **whole
        portfolio** (rescaling to 100% after filtering would lose the
        "share of total" information).
        """
        t = self.t
        vol = self._vol_panel()
        if vol is None:
            return self._empty("No volatility data.", t)
        cmp = self.r.exposure.notional_vs_risk(vol)
        if cmp.empty:
            return self._empty("Risk contribution cannot be computed.", t)
        if codes is not None:
            cmp = cmp.loc[[c for c in cmp.index if c in codes]]
            if cmp.empty:
                return self._empty("No data for this asset class.", t)

        cmp = cmp.sort_values("risk_share", ascending=True)
        labels = [f"{c} · {cmp.loc[c, 'name']}" for c in cmp.index]

        fig = go.Figure()
        for name, col, color in (
            ("Notional share", "notional_share", t.baseline),
            ("Risk share", "risk_share", t.series[0]),
        ):
            fig.add_trace(
                go.Bar(
                    y=labels, x=cmp[col], orientation="h", name=name,
                    marker=dict(color=color, line=dict(width=2, color=t.surface)),
                    hovertemplate="%{y}<br>" + name + " %{x:.1%}<extra></extra>",
                )
            )
        layout = self._layout(hovermode="closest", height=560, barmode="group", bargap=0.25)
        layout["margin"] = dict(l=190, r=40, t=16, b=44)
        layout["xaxis"] = _ax(layout["xaxis"], tickformat=".0%", showgrid=True,
                              gridcolor=t.gridline, title="Share")
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        return fig

    def risk_by_sector(self) -> go.Figure:
        """Risk contribution share by sector over time (sums to 100%)."""
        t = self.t
        vol = self._vol_panel()
        if vol is None:
            return self._empty("No volatility data.", t)
        data = self.r.exposure.risk_by_sector(vol).dropna(how="all")
        if data.empty:
            return self._empty("No risk contribution data.", t)

        colors = t.sector_colors(list(data.columns))
        fig = go.Figure()
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[col], name=col, mode="lines",
                    stackgroup="one", line=dict(color=colors[col], width=0),
                    fillcolor=colors[col],
                    hovertemplate="%{y:.1%}<extra>" + col + "</extra>",
                )
            )
        layout = self._layout(height=420)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%", range=[0, 1],
                              title="Risk contribution share")
        fig.update_layout(**layout)
        return fig

    def risk_by_market(self, sector: str) -> go.Figure:
        """Risk contribution share by market within one sector over time (denominator is the whole portfolio).

        `risk_by_sector` sums the four sectors to 100%. When an asset class is
        picked in the sidebar, that sector is broken down by market. The sum
        equals the sector's share of the total.
        """
        t = self.t
        vol = self._vol_panel()
        if vol is None:
            return self._empty("No volatility data.", t)
        share = self.r.exposure.risk_share(vol).dropna(how="all")
        codes = [c for c in self.universe.codes
                 if c in share.columns and self.universe.get(c).sector == sector]
        if share.empty or not codes:
            return self._empty("No risk contribution data.", t)

        fig = go.Figure()
        for i, c in enumerate(codes):
            fig.add_trace(
                go.Scatter(
                    x=share.index, y=share[c].fillna(0.0), mode="lines",
                    name=f"{c} · {self.universe.get(c).name}", stackgroup="one",
                    line=dict(color=t.series[i % len(t.series)], width=0),
                    fillcolor=t.series[i % len(t.series)],
                    hovertemplate="%{y:.1%}<extra>" + c + "</extra>",
                )
            )
        layout = self._layout(height=420)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%",
                              title=f"{sector} risk contribution (share of total)")
        fig.update_layout(**layout)
        return fig

    def gross_net(self) -> go.Figure:
        """Gross / net notional, leverage level over time."""
        t = self.t
        gross = self.r.exposure.gross_exposure().dropna()
        net = self.r.exposure.net_exposure().dropna()
        if gross.empty:
            return self._empty("No exposure data.", t)

        fig = go.Figure()
        # (legend name, series, color, endpoint label); endpoint label is short to fit the right margin
        for name, s, color, short in (
            ("Gross exposure", gross, t.series[0], "Gross"),
            ("Net exposure", net, t.series[1], "Net"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=s.index, y=s, name=name, mode="lines",
                    line=dict(color=color, width=_LINE_WIDTH),
                    hovertemplate="%{y:+.0%}<extra>" + name + "</extra>",
                )
            )
            self._endpoint_label(fig, s.index[-1], float(s.iloc[-1]), short, color)

        layout = self._layout()
        layout["yaxis"] = _ax(layout["yaxis"], tickformat="+.0%", title="Exposure (% of AUM)")
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 3. Scenario positioning (core of the dashboard)
    # ================================================================== #

    def _projection(self, driver: str | None):
        if driver is None:
            return self.r.projection_independent
        return self.r.projections.get(driver)

    def scenario_sector_trades(self, driver: str | None = None) -> go.Figure:
        """Required trades by sector x scenario ($BN), the key chart.

        "If the index moves this much, how much must CTAs buy or sell in each sector."
        """
        t = self.t
        proj = self._projection(driver)
        if proj is None:
            return self._empty("Scenarios cannot be computed.", t)

        trades = proj.by_sector("trades_usd") / 1e9  # index=sector, columns=scenario
        colors = t.scenario_colors(proj.sigmas())

        fig = go.Figure()
        for scenario, color in zip(proj.scenarios, colors):
            vals = trades[scenario.key]
            fig.add_trace(
                go.Bar(
                    x=list(trades.index),
                    y=vals.values,
                    name=scenario.label_ko,
                    marker=dict(color=color, line=dict(width=2, color=t.surface)),
                    text=[f"{v:+,.0f}" for v in vals.values],
                    textposition="outside",
                    textfont=dict(family=t.font, size=10, color=t.text_secondary),
                    hovertemplate=(
                        "%{x}<br>" + scenario.label_ko + "<br>$%{y:+,.1f}B<extra></extra>"
                    ),
                )
            )

        layout = self._layout(hovermode="closest", height=440, barmode="group", bargap=0.28)
        layout["xaxis"] = _ax(layout["xaxis"], type="category")
        layout["yaxis"] = _ax(
            layout["yaxis"], title="Required trade, $BN  (+ buy / − sell)", tickformat=",.0f"
        )
        fig.update_layout(**layout)
        return fig

    def scenario_market_matrix(self, driver: str | None = None) -> go.Figure:
        """Required trades heatmap by market x scenario ($BN), detail view."""
        t = self.t
        proj = self._projection(driver)
        if proj is None:
            return self._empty("Scenarios cannot be computed.", t)

        usd = proj.trades_usd / 1e9
        order = [c for c in self.universe.codes if c in usd.index]
        usd = usd.loc[order]
        limit = float(np.nanmax(np.abs(usd.values))) or 1.0

        fig = go.Figure(
            go.Heatmap(
                z=usd.values,
                x=[s.label_ko for s in proj.scenarios],
                y=[f"{c} · {self.universe.get(c).name}" for c in order],
                colorscale=t.diverging_scale(),
                zmid=0, zmin=-limit, zmax=limit,
                text=[[f"{v:+,.1f}" for v in row] for row in usd.values],
                texttemplate="%{text}",
                textfont=dict(family=t.font, size=10),
                hovertemplate="%{y}<br>%{x}<br>Required trade $%{z:+,.1f}B<extra></extra>",
                colorbar=dict(
                    title=dict(text="$BN", font=dict(family=t.font, size=11, color=t.muted)),
                    tickfont=dict(family=t.font, size=10, color=t.muted),
                    outlinewidth=0, thickness=10, len=0.85,
                ),
            )
        )
        layout = self._layout(hovermode="closest", height=560)
        layout["margin"] = dict(l=190, r=16, t=16, b=52)
        layout["xaxis"] = _ax(layout["xaxis"], side="top", type="category")
        layout["yaxis"] = _ax(
            layout["yaxis"], showgrid=False, zeroline=False, autorange="reversed"
        )
        fig.update_layout(**layout)
        return fig

    def scenario_positions(self, driver: str | None = None) -> go.Figure:
        """Projected position by scenario (vs current), bars by market."""
        t = self.t
        proj = self._projection(driver)
        if proj is None:
            return self._empty("Scenarios cannot be computed.", t)

        order = [c for c in self.universe.codes if c in proj.positions.index]
        colors = t.scenario_colors(proj.sigmas())

        fig = go.Figure()
        for scenario, color in zip(proj.scenarios, colors):
            vals = proj.positions.loc[order, scenario.key]
            fig.add_trace(
                go.Bar(
                    x=order, y=vals.values, name=scenario.label_ko,
                    marker=dict(color=color, line=dict(width=2, color=t.surface)),
                    hovertemplate=(
                        "%{x}<br>" + scenario.label_ko + "<br>%{y:+.1%}<extra></extra>"
                    ),
                )
            )
        # Current position overlaid as a thin marker (acts as the reference line)
        fig.add_trace(
            go.Scatter(
                x=order, y=proj.current.reindex(order).values,
                name="Current", mode="markers",
                marker=dict(
                    symbol="line-ew", size=16,
                    line=dict(color=t.text_primary, width=2),
                ),
                hovertemplate="%{x}<br>Current %{y:+.1%}<extra></extra>",
            )
        )

        layout = self._layout(hovermode="x unified", height=420, barmode="group", bargap=0.3)
        layout["xaxis"] = _ax(layout["xaxis"], type="category")
        layout["yaxis"] = _ax(
            layout["yaxis"], tickformat="+.0%", title="Projected notional (per $1 of benchmark)"
        )
        fig.update_layout(**layout)
        return fig

    def cftc_crowding(self, years: int | None = None,
                      codes: list[str] | None = None) -> go.Figure:
        """Position extremity percentile, information our model cannot provide.

        With `years`, plots the percentile over that lookback window (5/10/15/20 years).
        With `codes`, only those markets (asset class filter).
        """
        t = self.t
        c = self.r.cftc
        if c is None or c.crowding is None or c.crowding.empty:
            return self._empty("No CFTC data.", t)

        src = c.crowding
        if years is not None and getattr(c, "crowding_by_window", None):
            src = c.crowding_by_window.get(years, src)
        yrs = years or 5
        d = src.sort_values("pct_win")
        if codes is not None:
            d = d.loc[[i for i in d.index if i in codes]]
            if d.empty:
                return self._empty("No CFTC data for this asset class.", t)
        labels = [f"{i} · {self.universe.get(i).name}" for i in d.index]
        # Extremes (>90% or <10%) highlighted with diverging colors, the rest recessive
        colors = [
            t.pos_pole if p > 0.9 else t.neg_pole if p < 0.1 else t.baseline
            for p in d["pct_win"]
        ]
        fig = go.Figure(
            go.Bar(
                y=labels, x=d["pct_win"], orientation="h",
                marker=dict(color=colors, line=dict(width=2, color=t.surface)),
                text=[f"{p:.0%}" for p in d["pct_win"]], textposition="outside",
                textfont=dict(family=t.font, size=11, color=t.text_secondary),
                customdata=np.column_stack([d["cftc_net"], d["our_position"]]),
                hovertemplate=("%{y}<br>" + f"{yrs}y" + " percentile %{x:.0%}<br>"
                               "COT net position %{customdata[0]:+,.0f} contracts<br>"
                               "Model position %{customdata[1]:+.1%}<extra></extra>"),
                showlegend=False,
            )
        )
        for v in (0.1, 0.9):
            fig.add_vline(x=v, line=dict(color=t.baseline, width=_HAIRLINE))
        layout = self._layout(hovermode="closest", height=460)
        layout["margin"] = dict(l=190, r=64, t=16, b=44)
        layout["xaxis"] = _ax(layout["xaxis"], tickformat=".0%", range=[0, 1.08],
                              title=f"{yrs}y percentile of COT net position (0% = max short, 100% = max long)",
                              showgrid=True, gridcolor=t.gridline)
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        return fig

    def cftc_rolling_agreement(self) -> go.Figure:
        """Rolling sign agreement rate, a second health check independent of the benchmark."""
        t = self.t
        c = self.r.cftc
        if c is None or c.rolling_agree is None or not len(c.rolling_agree):
            return self._empty("No CFTC data.", t)

        s = c.rolling_agree
        fig = go.Figure(
            go.Scatter(
                x=s.index, y=s, mode="lines", name="Sign agreement rate",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:.0%}<extra>Sign agreement rate</extra>",
            )
        )
        fig.add_hline(
            y=0.5, line=dict(color=t.baseline, width=_HAIRLINE),
            annotation_text="Random 50%", annotation_position="bottom left",
            annotation_font=dict(family=t.font, size=10, color=t.muted),
        )
        self._endpoint_label(fig, s.index[-1], float(s.iloc[-1]), "Agree", t.series[0])
        layout = self._layout(height=340)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%", range=[0.3, 1.0],
                              title="2y rolling sign agreement rate")
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 4. Flow
    # ================================================================== #

    def recent_flow(self, weeks: int = 26, usd: bool = True) -> go.Figure:
        """Recent weekly trades, cumulative change by market.

        usd=True gives $BN, False gives **share in pp**. pp is independent of the AUM assumption.
        """
        t = self.t
        src = (self.r.exposure.weekly_flow_usd() / 1e9 if usd
               else self.r.exposure.pct_by_market().diff() * 100)
        flow = src.dropna(how="all")
        if flow.empty:
            return self._empty("No trade data.", t)

        recent = flow.iloc[-weeks:].sum().sort_values()
        colors = [t.sign_color(v) for v in recent]
        labels = [f"{c} · {self.universe.get(c).name}" for c in recent.index]
        fmt = "{:+,.0f}" if usd else "{:+,.1f}"
        hov = ("%{y}<br>$%{x:+,.1f}B<extra></extra>" if usd
               else "%{y}<br>%{x:+,.1f}pp<extra></extra>")
        fig = go.Figure(
            go.Bar(
                y=labels, x=recent.values, orientation="h",
                marker=dict(color=colors, line=dict(width=2, color=t.surface)),
                text=[fmt.format(v) for v in recent.values],
                textposition="outside",
                textfont=dict(family=t.font, size=11, color=t.text_secondary),
                hovertemplate=hov,
                showlegend=False,
            )
        )
        layout = self._layout(hovermode="closest", height=520)
        layout["margin"] = dict(l=190, r=72, t=16, b=44)
        layout["xaxis"] = _ax(layout["xaxis"], title=f"Cumulative net trade, last {weeks}w, " + ("$BN" if usd else "pp"),
                               showgrid=True, gridcolor=t.gridline,
                               zeroline=True, zerolinecolor=t.baseline)
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        return fig

    def flow_sensitivity(self, usd: bool = True) -> go.Figure:
        """Required trades by price shock scenario (extension beyond the original model).

        usd=True gives $BN, False gives **share in pp** (independent of the AUM assumption).
        """
        t = self.t
        flow = self.r.flow_sensitivity
        if flow is None or flow.empty:
            return self._empty("Flow sensitivity cannot be computed.", t)

        usd_flag = bool(usd)
        if usd_flag:
            usd = flow.xs("usd", axis=1, level="metric") / 1e9
        else:
            lv = list(flow.columns.get_level_values("metric").unique())
            key = "pct" if "pct" in lv else lv[0]
            usd = flow.xs(key, axis=1, level="metric") * 100
        shocks = list(usd.columns)
        order = [c for c in self.universe.codes if c in usd.index]
        usd = usd.loc[order]

        limit = float(np.nanmax(np.abs(usd.values))) or 1.0
        colorscale = [
            [0.0, t.neg_pole], [0.25, _fade(t.neg_pole, t.mid, 0.5)],
            [0.5, t.mid],
            [0.75, _fade(t.pos_pole, t.mid, 0.5)], [1.0, t.pos_pole],
        ]
        fig = go.Figure(
            go.Heatmap(
                z=usd.values,
                x=[f"{s:+.0%}" for s in shocks],
                y=[f"{c} · {self.universe.get(c).name}" for c in order],
                colorscale=colorscale, zmid=0, zmin=-limit, zmax=limit,
                text=[[f"{v:+,.1f}" for v in row] for row in usd.values],
                texttemplate="%{text}",
                textfont=dict(family=t.font, size=10),
                hovertemplate=("%{y}<br>Shock %{x}<br>Required trade $%{z:+,.1f}B<extra></extra>"
                               if usd_flag else
                               "%{y}<br>Shock %{x}<br>Required trade %{z:+,.1f}pp<extra></extra>"),
                colorbar=dict(
                    title=dict(text="$BN" if usd_flag else "pp",
                               font=dict(family=t.font, size=11, color=t.muted)),
                    tickfont=dict(family=t.font, size=10, color=t.muted),
                    outlinewidth=0, thickness=10, len=0.85,
                ),
            )
        )
        layout = self._layout(hovermode="closest", height=560)
        layout["margin"] = dict(l=190, r=16, t=16, b=52)
        layout["xaxis"] = _ax(layout["xaxis"], title="Next-week price shock", side="top")
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False, autorange="reversed")
        fig.update_layout(**layout)
        return fig

    def turnover(self) -> go.Figure:
        """Weekly turnover Σ|Δw|."""
        t = self.t
        s = self.r.exposure.turnover().dropna()
        if s.empty:
            return self._empty("No turnover data.", t)
        fig = go.Figure(
            go.Scatter(
                x=s.index, y=s, mode="lines", name="Turnover",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                hovertemplate="%{y:.0%}<extra>Turnover</extra>",
            )
        )
        self._endpoint_label(fig, s.index[-1], float(s.iloc[-1]), "Turnover", t.series[0])
        layout = self._layout(height=320)
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%", title="Weekly Σ|Δw|")
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 4. Parameter sweep
    # ================================================================== #

    def sweep_sorted(self) -> go.Figure:
        """R² of all 125 parameter combinations sorted ascending."""
        t = self.t
        sweep = self.r.sweep
        if sweep is None or sweep.empty:
            return self._empty("Parameter sweep was not run.", t)

        s = sweep.sort_values("r2").reset_index(drop=True)
        fig = go.Figure(
            go.Scatter(
                x=np.arange(1, len(s) + 1), y=s["r2"],
                mode="lines", name="R²",
                line=dict(color=t.series[0], width=_LINE_WIDTH),
                customdata=s[["label", "momentum_lookback"]].values,
                hovertemplate="%{customdata[0]}<br>R² %{y:.1%}<extra></extra>",
                showlegend=False,
            )
        )
        best = s.iloc[-1]
        fig.add_annotation(
            x=len(s), y=best["r2"],
            text=f"<b>Best</b> {best['label']} · {best['r2']:.1%}",
            showarrow=False, xanchor="right", yshift=14,
            font=dict(family=t.font, size=11, color=t.text_primary),
        )
        layout = self._layout(hovermode="closest", height=360)
        layout["xaxis"] = _ax(layout["xaxis"], title="Parameter combination (R² ascending)")
        layout["yaxis"] = _ax(layout["yaxis"], tickformat=".0%", title="R² vs benchmark")
        fig.update_layout(**layout)
        return fig

    def sweep_sensitivity(self, vary: str, hold: dict, title: str) -> go.Figure:
        """Sensitivity to a single varied parameter."""
        t = self.t
        sweep = self.r.sweep
        if sweep is None or sweep.empty:
            return self._empty("Parameter sweep was not run.", t)

        from ..evaluation import ReplicationEvaluator

        sub = ReplicationEvaluator.sweep_sensitivity(sweep, vary, hold)
        if sub.empty:
            return self._empty("No matching combinations.", t)

        fig = go.Figure(
            go.Bar(
                x=[str(v) for v in sub[vary]], y=sub["r2"],
                marker=dict(color=t.series[0], line=dict(width=2, color=t.surface)),
                text=[f"{v:.1%}" for v in sub["r2"]],
                textposition="outside",
                textfont=dict(family=t.font, size=11, color=t.text_secondary),
                hovertemplate="%{x}: R² %{y:.1%}<extra></extra>",
                showlegend=False,
            )
        )
        layout = self._layout(hovermode="closest", height=300)
        layout["xaxis"] = _ax(layout["xaxis"], title=title, type="category")
        layout["yaxis"] = _ax(
            layout["yaxis"], tickformat=".0%",
            range=[0, max(0.4, float(sub["r2"].max()) * 1.3)],
        )
        fig.update_layout(**layout)
        return fig

    # ================================================================== #
    # 5. Diagnostics
    # ================================================================== #

    def market_volatility(self, codes: list[str] | None = None) -> go.Figure:
        """Current annualized realized volatility by market, the denominator of risk-parity weights."""
        t = self.t
        model = self.r.ensemble.models[0]
        if model.signals is None:
            return self._empty("No signal data.", t)

        vol = model.signals.vol_annualized.dropna(how="all")
        if vol.empty:
            return self._empty("No volatility data.", t)
        latest = vol.iloc[-1].dropna().sort_values()
        if codes is not None:
            latest = latest[[c for c in latest.index if c in codes]]
            if latest.empty:
                return self._empty("No data for this asset class.", t)

        sectors = pd.Series(self.universe.sector_map)
        sector_colors = t.sector_colors(self.universe.sectors)
        colors = [sector_colors[sectors[c]] for c in latest.index]
        labels = [f"{c} · {self.universe.get(c).name}" for c in latest.index]

        fig = go.Figure(
            go.Bar(
                y=labels, x=latest.values, orientation="h",
                marker=dict(color=colors, line=dict(width=2, color=t.surface)),
                text=[f"{v:.0%}" for v in latest.values],
                textposition="outside",
                textfont=dict(family=t.font, size=11, color=t.text_secondary),
                hovertemplate="%{y}<br>Annualized %{x:.1%}<extra></extra>",
                showlegend=False,
            )
        )
        layout = self._layout(hovermode="closest", height=520)
        layout["margin"] = dict(l=190, r=72, t=16, b=44)
        layout["xaxis"] = _ax(layout["xaxis"], tickformat=".0%",
                               title=f"Realized volatility (annualized, {model.params.volatility_lookback} business days)",
                               showgrid=True, gridcolor=t.gridline)
        layout["yaxis"] = _ax(layout["yaxis"], showgrid=False, zeroline=False)
        fig.update_layout(**layout)
        return fig

    def momentum_signals(self, codes: list[str] | None = None) -> go.Figure:
        """Current normalized momentum (t-stat) by member: agreement or disagreement across trend speeds."""
        t = self.t
        fig = go.Figure()
        order = [c for c in self.universe.codes if codes is None or c in codes]
        any_data = False

        for i, model in enumerate(self.r.ensemble.models):
            if model.signals is None:
                continue
            mom = model.signals.momentum.dropna(how="all")
            if mom.empty:
                continue
            latest = mom.iloc[-1].reindex(order)
            any_data = True
            fig.add_trace(
                go.Bar(
                    x=order, y=latest.values, name=f"{model.params.momentum_lookback}w",
                    marker=dict(color=t.series[i % len(t.series)],
                                line=dict(width=2, color=t.surface)),
                    hovertemplate="%{x}: %{y:+.2f}<extra>"
                                  + f"{model.params.momentum_lookback}w</extra>",
                )
            )
        if not any_data:
            return self._empty("No momentum signals.", t)

        cap = self.r.ensemble.models[0].params.effective_cap
        if cap is not None:
            for y in (cap, -cap):
                fig.add_hline(y=y, line=dict(color=t.baseline, width=_HAIRLINE))

        layout = self._layout(hovermode="x unified", height=380, barmode="group")
        layout["xaxis"] = _ax(layout["xaxis"], type="category")
        layout["yaxis"] = _ax(layout["yaxis"], title="Normalized momentum (capped t-stat)")
        fig.update_layout(**layout)
        return fig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ax(base: dict, **overrides) -> dict:
    """Merge axis dicts. Later values win (dict(**a, b=1) fails on duplicate keys)."""
    return {**base, **overrides}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _fade(color: str, toward: str, ratio: float) -> str:
    """Linear interpolation between two colors, for diverging scale midpoints."""
    a, b = _hex_to_rgb(color), _hex_to_rgb(toward)
    mixed = tuple(int(round(a[i] * (1 - ratio) + b[i] * ratio)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


__all__ = ["FigureFactory"]
