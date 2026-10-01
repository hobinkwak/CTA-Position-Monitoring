"""
End-to-end orchestration.

    load data -> (roll if needed) -> validate -> weekly grid -> signals/positions
    -> ensemble -> evaluation (R-squared, 125-combination sweep) -> exposure -> result bundle

`main.py` only calls this class and renders the dashboard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import Benchmark
from .cftc import CFTCComparison
from .config import (
    BENCHMARK_NAME,
    PARAM_GRID,
    PATHS,
    RUN_CONFIG,
    DataConfig,
    DATA_CONFIG,
    ModelParams,
    Paths,
    RunConfig,
    ensemble_params,
)
from .dataloader import DataLoader, ValidationReport
from .ensemble import EnsembleReplicator
from .expiry import ExpiryCalendar, RollFromGenericTicker
from .evaluation import ReplicationEvaluator
from .exposure import ExposureEstimator, ExposureSnapshot
from .portfolio import ReplicationModel
from .pressure import TradePressure, TradePressureEstimator
from .rolling import FuturesRoller, RollDiagnostics
from .scenario import (
    DEFAULT_SCENARIOS,
    ScenarioEngine,
    ScenarioProjection,
    TriggerLevels,
)
from .timegrid import SessionReturns, WeeklyGrid
from .universe import UNIVERSE, Universe
from .volatility import RealizedVolatility, VolatilityPanel


@dataclass
class _CFTCInput:
    """Adapter holding only the fields `CFTCValidator` needs.

    Needed because validation runs before PipelineResult is complete.
    """

    exposure: "ExposureEstimator"
    front_prices: pd.DataFrame | None
    fx: pd.DataFrame | None
    ensemble: EnsembleReplicator


@dataclass
class PipelineResult:
    """Bundle of pipeline outputs, consumed by the dashboard."""

    # Data
    prices: pd.DataFrame
    daily_returns: pd.DataFrame
    weekly_returns: pd.DataFrame
    front_prices: pd.DataFrame | None
    fx: pd.DataFrame | None
    grid: WeeklyGrid
    validation: ValidationReport
    roll_diagnostics: RollDiagnostics | None

    # Model
    benchmark: Benchmark
    benchmark_returns: pd.Series
    ensemble: EnsembleReplicator
    best_single: ReplicationModel

    # Evaluation
    evaluator: ReplicationEvaluator
    stats_table: pd.DataFrame
    performance: pd.DataFrame
    rolling_r2: pd.Series
    sweep: pd.DataFrame | None

    # Exposure
    exposure: ExposureEstimator
    snapshot: ExposureSnapshot
    flow_sensitivity: pd.DataFrame | None

    # CFTC independent validation (independent of the AUM assumption)
    cftc: "CFTCComparison | None"

    # Scenario positioning (core of the dashboard)
    scenario_engine: "ScenarioEngine | None"
    projections: dict[str, "ScenarioProjection"]  # {driver_code: projection}
    projection_independent: "ScenarioProjection | None"
    response_curves: dict[str, pd.DataFrame]  # {driver_code: curve}
    triggers: dict[str, "TriggerLevels"]  # {driver_code: triggers}
    #: {horizon (weeks): TriggerLevels} for independent mode. With no driver, all 16
    #: markets get values (index propagation mode gives NaN for low-beta markets).
    triggers_flip: dict[int, "TriggerLevels"]

    # Meta
    universe: Universe
    run_config: RunConfig
    #: Return so far in the incomplete week (WTD): last weekly-grid close -> latest daily
    #: close. The weekly model uses only complete weeks (drop_partial_last=True), so this
    #: stretch is realized movement not yet in the model. Next to a scenario assumption it
    #: reads as "assumed -3.1%, already -0.4% realized".
    wtd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    wtd_from: pd.Timestamp | None = None
    wtd_to: pd.Timestamp | None = None
    wtd_days: int = 0
    #: Per-market date the price last actually changed. Source data is forward-filled on
    #: calendar dates, so the last file row is not necessarily that market's latest close.
    #: Always show this date together with a price on screen.
    price_asof: pd.Series = field(default_factory=lambda: pd.Series(dtype='datetime64[ns]'))
    #: Trade pressure (extension beyond the paper): buy probability and expected trade weighted by own price distribution
    trade_pressure: "TradePressure | None" = None
    elapsed_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #

    @property
    def ensemble_r2(self) -> float:
        return float(self.stats_table.loc["ensemble", "R2"])

    @property
    def headline(self) -> dict[str, object]:
        """Dashboard top KPIs."""
        snap = self.snapshot
        return {
            "as_of": snap.date,
            "ensemble_r2": self.ensemble_r2,
            "gross": snap.gross,
            "net": snap.net,
            "aum": snap.aum,
            "n_markets": len(self.universe),
            "n_weeks": int(self.ensemble.returns.notna().sum()),
            "start": self.weekly_returns.index[0],
            "end": self.weekly_returns.index[-1],
        }

    def to_csv(self, output_dir: Path | str | None = None) -> dict[str, str]:
        """Export the main outputs."""
        out = Path(output_dir or PATHS.output)
        out.mkdir(parents=True, exist_ok=True)
        written = self.exposure.to_csv(out)
        self.ensemble.all_returns().to_csv(out / "replication_returns.csv")
        self.stats_table.to_csv(out / "replication_stats.csv")
        if self.sweep is not None:
            self.sweep.to_csv(out / "parameter_sweep.csv", index=False)
        written.update(
            {
                "returns": str(out / "replication_returns.csv"),
                "stats": str(out / "replication_stats.csv"),
            }
        )

        # Scenario outputs (core of the dashboard), one file set per driver
        for driver, proj in self.projections.items():
            proj.trades_usd.to_csv(out / f"scenario_trades_usd_{driver}.csv")
            proj.positions.to_csv(out / f"scenario_positions_{driver}.csv")
            written[f"scenario_{driver}"] = str(out / f"scenario_trades_usd_{driver}.csv")
        for driver, trig in self.triggers.items():
            trig.table.to_csv(out / f"scenario_triggers_{driver}.csv")
        if self.projection_independent is not None:
            self.projection_independent.trades_usd.to_csv(
                out / "scenario_trades_usd_independent.csv"
            )
        return written


class CTAReplicationPipeline:
    """Paper replication pipeline.

    Examples
    --------
    >>> pipe = CTAReplicationPipeline()          # doctest: +SKIP
    >>> result = pipe.run()                      # doctest: +SKIP
    >>> result.ensemble_r2                       # doctest: +SKIP
    """

    def __init__(
        self,
        paths: Paths = PATHS,
        universe: Universe = UNIVERSE,
        run_config: RunConfig = RUN_CONFIG,
        data_config: DataConfig = DATA_CONFIG,
        verbose: bool = True,
    ) -> None:
        self.paths = paths
        self.universe = universe
        self.cfg = run_config
        self.data_cfg = data_config
        self.verbose = verbose

        self.loader = DataLoader(paths, universe, data_config)
        self.roller = FuturesRoller(universe, data_config)
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self._log(f"  ! {msg}")

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #

    #: A rolled-futures cache this many business days shorter than the source (C1) is considered stale
    ROLLED_STALE_TOLERANCE = 60

    @staticmethod
    def _trim_stale_tail(prices: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
        """Trim stale rows at the very end (rows identical to the previous day in every market).

        Source files are forward-filled, so dates not yet updated arrive filled with the
        previous close (observed: the 2026-08-03 row was a copy of 07-31, 16/16 identical).
        Left alone, this creates fake trading days with zero return, which understates
        volatility and inflates notional. WTD would also show a false "1 day elapsed, 0% move".

        The probability that all 16 markets match to the last decimal is practically zero,
        so the test is safe. Interior rows are untouched: weekends and holidays are handled
        in other steps.
        """
        if prices is None or len(prices) < 2:
            return prices
        out = prices
        for _ in range(max_rows):
            if len(out) < 2:
                break
            a, b = out.iloc[-1], out.iloc[-2]
            same = ((a.isna() & b.isna()) | (a == b)).all()
            if not same:
                break
            out = out.iloc[:-1]
        return out

    @staticmethod
    def _asof_by_market(prices: pd.DataFrame, max_rows: int = 8) -> pd.Series:
        """Per-market date the price last actually changed.

        Source data is forward-filled on calendar dates, so the last file row is not
        necessarily that market's latest close. If pulled before the close, markets not
        yet settled arrive with the previous close copied in.

        Observed (2026-08-05 KST): the last file row was 2026-08-04, but 14 of 16 markets
        were copies of the 08-03 close and only the Asian contracts (NK, JB) were 08-04.
        Writing "current price as of 08-04" on screen would be wrong for most markets.

        Rule: from the tail, a stretch exactly equal to the previous value is stale. A day
        where the close truly did not change may be caught (rare), but that only pushes the
        date back by one day, which is safer than showing a fake latest date.
        """
        if prices is None or prices.empty:
            return pd.Series(dtype="datetime64[ns]")
        out = {}
        for c in prices.columns:
            s = prices[c].dropna()
            if s.empty:
                continue
            idx = len(s) - 1
            for _ in range(max_rows):
                if idx == 0 or s.iloc[idx] != s.iloc[idx - 1]:
                    break
                idx -= 1
            out[c] = s.index[idx]
        return pd.Series(out, dtype="datetime64[ns]")

    @classmethod
    def _mask_stale_tail(cls, prices: pd.DataFrame) -> pd.DataFrame:
        """Blank each market's stale tail with NaN (so no fake zero enters returns).

        Leaving forward-filled values makes that day's return 0 and understates volatility:
        one zero in a 60-day window lowers sigma by about 0.8%. Pulling data intraday every
        day triggers this each time. Price levels remain valid (previous close), so the
        original is used for display and trigger computation and only the panel used for
        return calculation is blanked.
        """
        if prices is None or prices.empty:
            return prices
        asof = cls._asof_by_market(prices)
        out = prices.copy()
        for c, d in asof.items():
            if c in out.columns:
                out.loc[out.index > d, c] = np.nan
        return out

    @staticmethod
    def _week_to_date(prices: pd.DataFrame, grid) -> tuple:
        """Return so far in the incomplete week.

        The weekly grid holds only complete weeks, so daily closes after its last date are
        realized movement not yet in the model. It is measured on ratio-adjusted rolled
        prices, so it is in the same unit as the scenario shock (%).
        """
        empty = (pd.Series(dtype=float), None, None, 0)
        if prices is None or prices.empty or len(grid.dates) == 0:
            return empty
        base_date = grid.dates[-1]
        tail = prices.loc[prices.index > base_date]
        if tail.empty:
            return empty
        # Measure from each market's last valid value: when a panel with the ffill tail
        # blanked comes in, the last valid date differs per market (e.g. only Asian
        # contracts one day later).
        last = tail.apply(lambda col: col.dropna().iloc[-1]
                          if col.notna().any() else np.nan)
        wtd = (last / prices.loc[base_date] - 1.0)
        wtd = wtd.replace([np.inf, -np.inf], np.nan)
        return wtd, base_date, tail.index[-1], int(len(tail))

    def _rolled_cache_stale(self, rolled: pd.DataFrame) -> str:
        """Decide whether the `data/processed/futures_rolled` cache must be discarded.

        Earlier only missing markets were checked; after the source history of VG was
        extended from 2015 to 1998, the cache (starting 2015) was still reused and the
        fix was silently ignored. Coverage and latest date must be checked too.

        Returns
        -------
        Reason string for invalidation; an empty string means the cache is valid.
        """
        expected = list(self.universe.codes)
        missing = [c for c in expected if c not in rolled.columns]
        if missing:
            return (f"{len(missing)} markets missing {missing[:6]}"
                    f"{' ...' if len(missing) > 6 else ''}")

        c1 = self.loader.load_front_prices(required=False)
        if c1 is None:
            return ""            # No source to compare with: trust the cache

        # 1) Is the latest date behind the source?
        # The old threshold was 5 business days, which silently drops the latest day.
        # WTD needs daily data as current as the source, so not even 1 day is tolerated.
        # Re-rolling takes seconds, so rebuilding every day is fine.
        c1 = self._trim_stale_tail(c1)      # avoid rebuilding every time because of stale rows
        if len(c1) and len(rolled) and c1.index[-1] > rolled.index[-1]:
            gap = int(np.busday_count(rolled.index[-1].date(), c1.index[-1].date()))
            return f"source is {gap} business days newer"

        # 2) Is any market coverage much shorter than the source? (VG case)
        short = []
        for c in expected:
            if c not in c1.columns:
                continue
            a = c1[c].dropna()
            b = rolled[c].dropna() if c in rolled.columns else rolled.iloc[:0]
            if a.empty:
                continue
            if b.empty:
                short.append(f"{c}(empty)")
                continue
            lag = int(np.busday_count(a.index[0].date(), b.index[0].date()))
            if lag > self.ROLLED_STALE_TOLERANCE:
                short.append(f"{c}({b.index[0]:%Y-%m} vs source {a.index[0]:%Y-%m})")
        if short:
            return f"coverage shortened {', '.join(short[:4])}"
        return ""

    def load_prices(self) -> tuple[pd.DataFrame, RollDiagnostics | None]:
        """Obtain continuous futures prices.

        Priority 1: use `futures_rolled.*` as is (already roll-adjusted)
        Priority 2: roll in-house from `futures_c1/c2 + roll_dates`
        """
        rolled = self.loader.load_rolled_prices(required=False)
        if rolled is not None:
            stale = self._rolled_cache_stale(rolled)
            if stale:
                self._log(f"  Rolled futures cache invalid ({stale}) -> rebuilding from C1/C2")
            else:
                self._log(f"  Rolled futures loaded: {rolled.shape[1]} markets, {len(rolled)} days")
                diag = self.roller.detect_roll_jumps(rolled)
                return self._log_trim(rolled), diag

        self._log("  No futures_rolled -> rolling in-house from C1/C2")
        c1 = self.loader.load_front_prices(required=True)
        c2 = self.loader.load_second_prices(required=True)

        # Roll date priority:
        #   1. generic ticker history (exact, if present)
        #   2. roll_dates.csv (user-supplied)
        #   3. rule-based expiry calendar (ctarep/expiry.py)
        roll_dates = self._resolve_roll_dates(c1)

        # Markets without C2 cannot be rolled -> use C1 as is, with a warning
        usable = [
            code for code in c1.columns
            if code in c2.columns and c2[code].notna().sum() > 100
        ]
        unusable = [c for c in c1.columns if c not in usable]
        if unusable:
            self._warn(
                f"Cannot roll without C2 -> using raw C1 (risk of contaminated returns): {unusable}"
            )

        rolled, roll_yields = self.roller.build(
            c1[usable], c2[usable], roll_dates, convention="last_old_day"
        )
        for code in unusable:
            rolled[code] = c1[code]
        rolled = rolled[[c for c in self.universe.codes if c in rolled.columns]]

        # Ratio adjustment requires positive prices. A print such as WTI on 2020-04-20
        # (-37.63) would blow up % returns. Real CTAs had already rolled by then (expiring
        # front month), so excluding that observation is the realistic choice.
        nonpos = (rolled <= 0)
        if nonpos.any().any():
            hits = {c: int(n) for c, n in nonpos.sum().items() if n}
            rolled = rolled.mask(nonpos)
            self._warn(
                f"Excluded prices <= 0 (ratio adjustment impossible): {hits}. "
                "These are expiring front-month prints, by which time real CTAs had already rolled."
            )

        self.loader.save_processed(rolled, "futures_rolled")
        self._log(
            f"  Rolling done: {len(usable)} markets (avg "
            f"{np.mean([len(roll_dates.get(c, [])) for c in usable]):.0f} rolls) "
            "-> cached in data/processed/"
        )

        diag = self.roller.detect_roll_jumps(rolled)
        if diag.suspicious_markets:
            self._warn(f"Markets with extreme moves remaining after adjustment: {diag.suspicious_markets}")
        return self._log_trim(rolled), diag

    def _log_trim(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Trim the stale tail and log it."""
        out = self._trim_stale_tail(prices)
        n = len(prices) - len(out)
        if n:
            self._log(f"  Dropped {n} stale rows (source not updated, previous close copied)"
                      f" -> last date {out.index[-1]:%Y-%m-%d}")
        return out

    def _quality_gate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Exclude unusable markets.

        A stale series understates realized volatility, and since the position is
        `m~/sigma_ann` the denominator shrinks and notional explodes. One market can
        contaminate the whole model, so such markets are filtered out automatically.
        """
        cfg = self.data_cfg

        # Drop markets absent from the data from the universe. Otherwise the sector map and
        # reports would show markets that do not exist.
        absent = [c for c in self.universe.codes if c not in prices.columns]
        if absent:
            self._warn(f"Excluded from the universe (not in data): {absent}")
            self.universe = self.universe.subset(
                [c for c in self.universe.codes if c in prices.columns]
            )

        drop: dict[str, str] = {}
        for code in prices.columns:
            s = prices[code].dropna()
            if len(s) < cfg.min_observations:
                drop[code] = f"{len(s)} observations (< {cfg.min_observations})"
                continue
            stale = float((prices[code].loc[s.index[0]:].diff() == 0).mean())
            if stale > cfg.max_stale_ratio:
                drop[code] = (
                    f"same-as-previous-day ratio {stale:.0%} (> {cfg.max_stale_ratio:.0%}), "
                    "possible ticker error"
                )
        if drop:
            for code, why in drop.items():
                self._warn(f"Market excluded [{code}]: {why}")
            self._warn(
                f"To keep the {len(drop)} excluded markets, fix their source series "
                "(stale or too short)."
            )
            prices = prices.drop(columns=list(drop))
            # Narrow the universe to surviving markets so reports stay consistent
            self.universe = self.universe.subset(
                [c for c in self.universe.codes if c in prices.columns]
            )
        return prices

    def _resolve_roll_dates(self, c1: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
        """Get roll dates from the best available source."""
        # 1) Generic ticker history: most accurate if present
        try:
            tickers = self._load_generic_tickers()
        except Exception as exc:
            self._warn(f"Failed to read futures_gen_ticker: {exc}")
            tickers = None
        if tickers is not None and not tickers.empty:
            self._log("  Roll dates: from generic ticker history (exact)")
            return RollFromGenericTicker().roll_dates(tickers)

        # 2) User-supplied roll_dates.csv
        explicit = self.loader.load_roll_dates(required=False)
        if explicit:
            self._log("  Roll dates: using data/raw/roll_dates.csv")
            return explicit

        # 3) Rule-based expiry calendar
        self._log(
            f"  Roll dates: rule-based expiry calendar "
            f"({self.data_cfg.roll_days_before_expiry} business days before expiry)"
        )
        return ExpiryCalendar().roll_dates(
            c1.index,
            days_before=self.data_cfg.roll_days_before_expiry,
            codes=list(c1.columns),
        )

    def _load_generic_tickers(self) -> pd.DataFrame | None:
        """`futures_gen_ticker.csv` (if present): the actual contract each generic points to."""
        path = self.loader.resolve("futures_gen_ticker", required=False)
        if path is None:
            return None
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, index_col=0, parse_dates=[0])
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, index_col=0, parse_dates=[0])
        else:
            df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.columns = [str(c).strip() for c in df.columns]
        return df[[c for c in self.universe.codes if c in df.columns]]

    # ------------------------------------------------------------------ #
    # 2. Run
    # ------------------------------------------------------------------ #

    def run(
        self,
        ensemble_param_list: list[ModelParams] | None = None,
        run_sweep: bool | None = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        self.warnings = []
        self.paths.ensure()

        # -- Data ---------------------------------------------------------
        self._log("[1/7] Loading data")
        prices, roll_diag = self.load_prices()
        front = self.loader.load_front_prices(required=False)
        fx = self.loader.load_fx(required=False)
        benchmark_level = self.loader.load_benchmark(required=True)

        if self.cfg.start or self.cfg.end:
            prices = prices.loc[self.cfg.start : self.cfg.end]
            benchmark_level = benchmark_level.loc[self.cfg.start : self.cfg.end]

        # -- Quality gate: auto-exclude mis-mapped or stale markets ---------
        prices = self._quality_gate(prices)

        validation = self.loader.validate(prices)
        if self.verbose:
            print(validation.summary())
        if validation.nonpositive:
            self._warn(
                f"Markets with prices <= 0: {list(validation.nonpositive)} "
                "-> check that the series is ratio-adjusted continuous futures."
            )

        # -- Returns / weekly grid ----------------------------------------
        self._log("[2/7] Weekly grid and returns")
        # Blank the ffill tail before computing returns.
        #    The exchange session filter only removes holidays; the issue here is not a
        #    holiday but data pulled intraday with unsettled closes copied in.
        #    Observed (2026-08-05 KST): 14/16 markets had an exactly-zero return on the
        #    08-04 row. Left in, fake zeros enter the 60-day window and understate sigma.
        #    Price levels are valid, so the original `prices` is kept for display and triggers.
        price_asof = self._asof_by_market(prices)
        prices_ret = self._mask_stale_tail(prices)
        n_masked = int(prices_ret.isna().sum().sum() - prices.isna().sum().sum())
        if n_masked:
            self._log(f"  Masked {n_masked} ffill-tail observations "
                      f"(per-market latest close {price_asof.min():%m-%d} ~ {price_asof.max():%m-%d})")
        if self.data_cfg.use_exchange_sessions:
            # Remove zero returns on exchange holidays (avoids understating volatility).
            # Returns that span a holiday are preserved.
            sess = SessionReturns()
            daily_returns = sess.returns(prices_ret)
            diag = sess.diagnostics(prices_ret)
            self._log(
                f"  Exchange session filter: {int(diag['dropped'].sum()):,} rows dropped · "
                f"zero returns {diag['zero_pct_raw'].mean():.1f}% -> "
                f"{diag['zero_pct_clean'].mean():.1f}% · "
                f"sigma avg {diag['sigma_lift_pct'].mean():+.2f}% "
                f"(max {diag['sigma_lift_pct'].max():+.1f}% {diag['sigma_lift_pct'].idxmax()})"
            )
        else:
            daily_returns = prices_ret.pct_change()
        # drop_partial_last=True: treating an unfinished week as a complete week would
        #    count 3 days of return as a weekly return and distort both the momentum
        #    numerator and volatility. The part of the incomplete week is not discarded;
        #    it is shown separately as `wtd`.
        grid = WeeklyGrid.from_index(prices.index, drop_partial_last=True)
        weekly_returns = grid.weekly_returns(prices)

        # WTD: movement realized after the last weekly-grid close (on rolled futures, so the
        # same unit as the scenario shock). If daily ends on the same day as weekly, 0 days
        # = empty. Measured on the panel with the ffill tail blanked, so only each market's
        # last actual close is reflected.
        wtd, wtd_from, wtd_to, wtd_days = self._week_to_date(prices_ret, grid)
        if wtd_days and len(price_asof):
            wtd_to = price_asof.max()
            wtd_days = int(np.busday_count(wtd_from.date(), wtd_to.date()))

        benchmark = Benchmark(benchmark_level, name=BENCHMARK_NAME)
        benchmark_returns = benchmark.weekly_returns(grid)

        common = weekly_returns.dropna(how="all").index.intersection(
            benchmark_returns.dropna().index
        )
        self._log(
            f"  {len(grid)} weekly observations / {len(common)} common with benchmark "
            f"({common.min():%Y-%m-%d} ~ {common.max():%Y-%m-%d})"
            if len(common)
            else "  [ERROR] No overlap with the benchmark"
        )
        if len(common) < 52:
            raise ValueError(
                f"Replica/benchmark common period is only {len(common)} weeks. "
                "Check that the data periods overlap."
            )

        # -- Ensemble model -----------------------------------------------
        self._log("[3/7] Fitting the ensemble replication model")
        ensemble = EnsembleReplicator(
            ensemble_param_list or ensemble_params(),
            sector_map=self.universe.sector_map,
        )
        ensemble.fit(
            weekly_returns,
            daily_returns,
            benchmark_returns,
            method=self.cfg.leverage_scaling,
            expanding_min_weeks=self.cfg.expanding_min_weeks,
            rescale=self.cfg.rescale_ensemble,
        )
        self._log(f"  {ensemble.fit_result.summary()}")

        best_single = max(
            ensemble.models,
            key=lambda m: _safe_r2(m.returns, benchmark_returns),
        )

        # -- Evaluation ---------------------------------------------------
        self._log("[4/7] Evaluating replication quality")
        evaluator = ReplicationEvaluator(
            weekly_returns, daily_returns, benchmark_returns,
            sector_map=self.universe.sector_map,
        )

        stats_rows = {
            label: evaluator.stats(ensemble.member_returns[label]).to_series()
            for label in ensemble.labels
        }
        stats_rows["ensemble"] = evaluator.stats(ensemble.returns).to_series()
        stats_table = pd.DataFrame(stats_rows).T

        all_rtn = ensemble.all_returns().join(benchmark_returns.rename(BENCHMARK_NAME))
        performance = evaluator.performance_table(all_rtn)
        rolling_r2 = evaluator.rolling_r2(
            ensemble.returns, window=self.cfg.rolling_r2_window_weeks
        )

        self._log(
            f"  Ensemble R-squared = {stats_table.loc['ensemble', 'R2']:.1%}  "
            f"(members: "
            + ", ".join(f"{lab} {stats_table.loc[lab, 'R2']:.1%}" for lab in ensemble.labels)
            + ")"
        )

        # -- Parameter sweep ------------------------------------------------
        do_sweep = self.cfg.run_parameter_sweep if run_sweep is None else run_sweep
        sweep = None
        if do_sweep:
            self._log(f"[6/7] Parameter sweep ({len(PARAM_GRID.combinations())} combinations)")
            sweep = evaluator.parameter_sweep(PARAM_GRID, progress=self.verbose)
            top = sweep.iloc[0]
            self._log(f"  Best R-squared: {top['label']} -> {top['r2']:.1%}")
        else:
            self._log("[6/7] Parameter sweep skipped")

        # -- Exposure ------------------------------------------------------
        self._log("[7/8] Computing exposure")
        exposure = ExposureEstimator(
            ensemble.positions, self.universe, aum=self.cfg.aum_usd, run_config=self.cfg
        )
        snapshot = exposure.snapshot()
        if self.verbose:
            print(snapshot.summary())

        flow = self._flow_sensitivity(ensemble, exposure, weekly_returns, daily_returns)

        # -- CFTC independent validation -------------------------------------
        # COT is read from the local file (data/external/cftc_positions.csv).
        cftc_cmp = None
        cot_net = None
        if self.cfg.run_cftc_validation:
            try:
                from .cftc import CFTCStore, CFTCValidator

                cot_net = CFTCStore().load(verbose=self.verbose)
                # PipelineResult does not exist yet, so a shim with minimal info is used
                shim = _CFTCInput(
                    exposure=exposure, front_prices=front, fx=fx, ensemble=ensemble
                )
                cftc_cmp = CFTCValidator(shim).run(verbose=self.verbose, cot=cot_net)
                if cftc_cmp is not None and self.verbose:
                    lag, val = cftc_cmp.best_lead_lag
                    print(
                        f"  CFTC comparison: sign agreement "
                        f"{cftc_cmp.table[~cftc_cmp.table['basis_contaminated']]['sign_agree'].mean():.0%}"
                        f" · lead lag={lag:+d} corr {val:+.3f}"
                    )
            except Exception as exc:
                self._warn(f"CFTC validation failed: {type(exc).__name__}: {exc}")

        # -- Scenario positioning (core of the dashboard) ---------------------
        self._log("[8/8] Scenario positioning")
        (
            scenario_engine,
            projections,
            projection_independent,
            response_curves,
            triggers,
            triggers_flip,
        ) = self._scenarios(ensemble, weekly_returns, daily_returns, front)

        # -- Trade pressure (independent-mode reaction curve x historical KDE) --
        trade_pressure = None
        if scenario_engine is not None:
            try:
                trade_pressure = TradePressureEstimator(scenario_engine, self.cfg).run()
                if self.verbose:
                    print("  Trade pressure: " + " · ".join(
                        f"{k} {v}" for k, v in trade_pressure.tier_counts().items()))
            except Exception as exc:
                self._warn(f"Trade pressure computation failed: {type(exc).__name__}: {exc}")

        elapsed = time.perf_counter() - t0
        self._log(f"\nDone ({elapsed:.1f}s)")

        return PipelineResult(
            prices=prices,
            daily_returns=daily_returns,
            weekly_returns=weekly_returns,
            front_prices=front,
            fx=fx,
            grid=grid,
            validation=validation,
            roll_diagnostics=roll_diag,
            benchmark=benchmark,
            benchmark_returns=benchmark_returns,
            ensemble=ensemble,
            best_single=best_single,
            evaluator=evaluator,
            stats_table=stats_table,
            performance=performance,
            rolling_r2=rolling_r2,
            sweep=sweep,
            exposure=exposure,
            snapshot=snapshot,
            flow_sensitivity=flow,
            cftc=cftc_cmp,
            scenario_engine=scenario_engine,
            projections=projections,
            projection_independent=projection_independent,
            response_curves=response_curves,
            triggers=triggers,
            triggers_flip=triggers_flip,
            trade_pressure=trade_pressure,
            universe=self.universe,
            run_config=self.cfg,
            wtd=wtd,
            wtd_from=wtd_from,
            wtd_to=wtd_to,
            wtd_days=wtd_days,
            price_asof=price_asof,
            elapsed_sec=elapsed,
            warnings=list(self.warnings),
        )

    # ------------------------------------------------------------------ #

    def _scenarios(
        self,
        ensemble: EnsembleReplicator,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
        front_prices: pd.DataFrame | None,
    ):
        """Five-step scenario projection, the core output of the dashboard.

        Projections, reaction curves and triggers are precomputed for each driver
        candidate so the dashboard can switch between them via a dropdown.
        """
        cfg = self.cfg
        empty: tuple = (None, {}, None, {}, {}, {})
        try:
            engine = ScenarioEngine(
                ensemble=ensemble,
                weekly_returns=weekly_returns,
                daily_returns=daily_returns,
                universe=self.universe,
                aum=cfg.aum_usd,
                horizon_weeks=cfg.scenario_horizon_weeks,
                scenarios=DEFAULT_SCENARIOS,
                beta_window_weeks=cfg.scenario_beta_window_weeks,
                update_vol=cfg.scenario_update_vol,
            )
        except Exception as exc:
            self._warn(f"ScenarioEngine initialization failed: {exc}")
            return empty

        # None = the whole universe as driver candidates
        if cfg.scenario_driver_choices is None:
            drivers = list(engine.codes)
        else:
            drivers = [d for d in cfg.scenario_driver_choices if d in engine.codes]
        if cfg.scenario_driver in engine.codes and cfg.scenario_driver not in drivers:
            drivers.insert(0, cfg.scenario_driver)
        if not drivers:
            self._warn("No market available as a scenario driver.")
            return empty
        self._log(f"  {len(drivers)} drivers: {drivers}")

        projections: dict[str, ScenarioProjection] = {}
        curves: dict[str, pd.DataFrame] = {}
        triggers: dict[str, TriggerLevels] = {}
        triggers_flip: dict[int, TriggerLevels] = {}
        price_source = front_prices if front_prices is not None else None

        # Triggers use a separate, longer horizon (see the config comments).
        # With a 1-week horizon a flip needs an unrealistic shock and the table is mostly blank.
        trigger_engine = engine
        if cfg.trigger_horizon_weeks != cfg.scenario_horizon_weeks:
            try:
                trigger_engine = ScenarioEngine(
                    ensemble=ensemble,
                    weekly_returns=weekly_returns,
                    daily_returns=daily_returns,
                    universe=self.universe,
                    aum=cfg.aum_usd,
                    horizon_weeks=cfg.trigger_horizon_weeks,
                    scenarios=DEFAULT_SCENARIOS,
                    beta_window_weeks=cfg.scenario_beta_window_weeks,
                    update_vol=cfg.scenario_update_vol,
                )
            except Exception as exc:
                self._warn(f"Trigger-only engine creation failed, using default horizon: {exc}")

        for driver in drivers:
            try:
                projections[driver] = engine.project(mode="index", driver=driver)
                curves[driver] = engine.response_curve(mode="index", driver=driver)
                triggers[driver] = trigger_engine.triggers(
                    mode="index", driver=driver, prices=price_source
                )
            except Exception as exc:
                self._warn(f"Scenario computation failed (driver={driver}): {exc}")

        independent = None
        try:
            independent = engine.project(mode="independent")
        except Exception as exc:
            self._warn(f"Independent-mode scenario failed: {exc}")

        main = cfg.scenario_driver if cfg.scenario_driver in projections else (
            drivers[0] if drivers else None
        )
        if self.verbose and main:
            print(projections[main].summary())
            print(triggers[main].summary())

        # Independent-mode triggers, once per horizon. With no driver, no market becomes NaN
        # because of a small beta ("how far must this market move on its own to flip").
        # If the appended days exceed half the volatility lookback, the sigma estimate is
        # dominated by synthetic values (beyond the lookback sigma = 0 and all positions
        # collapse to 0). Block it instead of failing silently.
        vol_lb = min(m.params.volatility_lookback for m in ensemble.models)
        for H in cfg.trigger_horizon_choices:
            if H * 5 > vol_lb * 0.5:
                self._warn(
                    f"Trigger horizon {H}w excluded: the {H * 5} appended days exceed half the "
                    f"{vol_lb}-day volatility lookback, so the volatility estimate would be dominated by synthetic values."
                )
                continue
            try:
                eng_h = engine if H == cfg.scenario_horizon_weeks else ScenarioEngine(
                    ensemble=ensemble,
                    weekly_returns=weekly_returns,
                    daily_returns=daily_returns,
                    universe=self.universe,
                    aum=cfg.aum_usd,
                    horizon_weeks=H,
                    scenarios=DEFAULT_SCENARIOS,
                    beta_window_weeks=cfg.scenario_beta_window_weeks,
                    update_vol=cfg.scenario_update_vol,
                )
                triggers_flip[H] = eng_h.triggers(
                    mode="independent", driver=None, prices=price_source,
                    sigma_range=12.0, n_points=241,
                )
            except Exception as exc:
                self._warn(f"Flip trigger computation failed (H={H}w): {exc}")

        return engine, projections, independent, curves, triggers, triggers_flip

    # ------------------------------------------------------------------ #

    def _flow_sensitivity(
        self,
        ensemble: EnsembleReplicator,
        exposure: ExposureEstimator,
        weekly_returns: pd.DataFrame,
        daily_returns: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Required trades per price-shock scenario (extension beyond the paper)."""
        try:
            vol_panels: dict[tuple[int, str], VolatilityPanel] = {}
            for model in ensemble.models:
                key = (model.params.volatility_lookback, model.params.vol_method)
                if key not in vol_panels:
                    vol_panels[key] = RealizedVolatility.from_params(model.params).compute(
                        daily_returns
                    )
            return exposure.flow_sensitivity(
                signal_engines=[m.signal_engine for m in ensemble.models],
                weekly_returns=weekly_returns,
                daily_returns=daily_returns,
                vol_panels=vol_panels,
                leverage=ensemble.rescale_factor
                * _mean_member_leverage(ensemble),
                ensemble_weights=ensemble.weights,
                shocks=tuple(self.cfg.flow_shocks),
            )
        except Exception as exc:  # not a feature worth blocking the whole dashboard
            self._warn(f"flow_sensitivity computation failed: {exc}")
            return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _safe_r2(rep: pd.Series | None, bm: pd.Series) -> float:
    if rep is None:
        return -np.inf
    joined = pd.concat([rep.rename("a"), bm.rename("b")], axis=1).dropna()
    if len(joined) < 3:
        return -np.inf
    return float(joined["a"].corr(joined["b"]) ** 2)


def _mean_member_leverage(ensemble: EnsembleReplicator) -> float:
    """Average member leverage k by ensemble weights (to scale the flow scenarios)."""
    total = 0.0
    for model in ensemble.models:
        k = model.fit_result.leverage if model.fit_result else 1.0
        k = float(pd.Series(k).iloc[-1]) if not np.isscalar(k) else float(k)
        total += k * ensemble.weights[model.label]
    return total


__all__ = ["CTAReplicationPipeline", "PipelineResult"]
