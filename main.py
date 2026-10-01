"""Entry point: run the CTA replication model and build the dashboard.

    python main.py                      # build output/dashboard.html and open it
    python main.py --no-open --no-sweep # faster, no browser
    python main.py --check              # report the state of data/ and exit

All computation lives in `ctarep/`; this file only wires things together.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ctarep import PATHS, RUN_CONFIG, CTAReplicationPipeline, DataLoader  # noqa: E402
from ctarep.dashboard import DashboardBuilder  # noqa: E402
from ctarep.universe import get_universe  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="CTA Position Monitoring: replicate CTA positioning and build the dashboard",
    )
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    p.add_argument("--no-sweep", action="store_true",
                   help="skip the 125-combination parameter sweep (about 3x faster)")
    p.add_argument("--aum", type=float, default=None, metavar="BN",
                   help="trend-following AUM assumption in $BN (default 300)")
    p.add_argument("--start", type=str, default=None, help="analysis start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="analysis end date (YYYY-MM-DD)")
    p.add_argument("--scaling", choices=("in_sample", "expanding"), default=None,
                   help="leverage scaling (default in_sample, as in the paper)")
    p.add_argument("--horizon", type=int, default=None, metavar="W",
                   help="weeks over which the price shock is realised (default 1)")
    p.add_argument("--driver", type=str, default=None, metavar="CODE",
                   help="default driver index code (default ES)")
    p.add_argument("--drivers", type=str, default=None, metavar="A,B,C",
                   help="comma-separated driver choices shown in the dropdown")
    p.add_argument("--beta-window", type=int, default=None, metavar="W",
                   help="weeks used to estimate cross-market betas (default 156)")
    p.add_argument("--no-vol-update", action="store_true",
                   help="ignore the effect of the shock on realised volatility")
    p.add_argument("--theme", choices=("light", "dark"), default="light",
                   help="default dashboard theme")
    p.add_argument("--output", type=str, default=None, help="output HTML path")
    p.add_argument("--csv", action="store_true", help="also export result CSV files")
    p.add_argument("--standalone", action="store_true",
                   help="embed plotly.js in the HTML (about +5 MB, opens offline)")
    p.add_argument("--check", action="store_true", help="report data status and exit")
    p.add_argument("--quiet", action="store_true", help="minimal logging")
    p.add_argument("--universe", default="paper",
                   choices=("paper", "recommended", "diversified", "full"),
                   help="market universe (default paper = the 16 markets of the paper)")
    args, unknown = p.parse_known_args(argv)
    if unknown and argv is None:
        print(f"[ignored arguments] {unknown}")
    return args


def check_data() -> int:
    """Show which input files are present under data/."""
    loader = DataLoader()
    print("=" * 72)
    print("data/ status")
    print("=" * 72)
    print(loader.available().to_string())
    print()

    prices = loader.load_rolled_prices(required=False)
    if prices is None:
        print("No back-adjusted continuous futures (futures_rolled) found.")
        c1 = loader.load_front_prices(required=False)
        if c1 is not None:
            print("  futures_c1 exists. Add futures_c2 to let the code roll it.")
        print("\nSee the Data section of README.md for the expected files.")
        return 1

    report = loader.validate(prices)
    print(report.summary())
    print()
    print(report.to_frame().to_string())
    return 0


def run(
    *,
    universe: str = "paper",
    sweep: bool = True,
    aum_bn: float | None = None,
    start: str | None = None,
    end: str | None = None,
    scaling: str | None = None,
    horizon: int | None = None,
    driver: str | None = None,
    drivers: str | list[str] | None = None,
    beta_window: int | None = None,
    vol_update: bool = True,
    dashboard: bool = True,
    standalone: bool = False,
    theme: str = "light",
    output: str | Path | None = None,
    open_browser: bool = True,
    csv: bool = False,
    verbose: bool = True,
):
    """Run the pipeline and return a `PipelineResult`.

    Examples
    --------
    >>> from main import run                      # doctest: +SKIP
    >>> r = run(sweep=False)                      # doctest: +SKIP
    >>> r.snapshot.by_market                      # current positions
    >>> r.projections['ES'].trades_usd / 1e9      # scenario trades in $BN
    >>> r.ensemble_r2                             # replication R-squared

    Returns
    -------
    PipelineResult (with `.dashboard_path` set when `dashboard=True`).
    """
    cfg = RUN_CONFIG
    overrides: dict = {}
    if not sweep:
        overrides["run_parameter_sweep"] = False
    if aum_bn is not None:
        overrides["aum_usd"] = aum_bn * 1e9
    if start:
        overrides["start"] = start
    if end:
        overrides["end"] = end
    if scaling:
        overrides["leverage_scaling"] = scaling
    if horizon is not None:
        overrides["scenario_horizon_weeks"] = horizon
    if driver:
        overrides["scenario_driver"] = str(driver).strip().upper()
    if drivers:
        seq = drivers.split(",") if isinstance(drivers, str) else drivers
        overrides["scenario_driver_choices"] = tuple(
            str(d).strip().upper() for d in seq if str(d).strip()
        )
    if beta_window is not None:
        overrides["scenario_beta_window_weeks"] = beta_window
    if not vol_update:
        overrides["scenario_update_vol"] = False
    if overrides:
        cfg = replace(cfg, **overrides)

    print("=" * 72)
    print("Replicating CTA Positioning (Kestner, 2020)")
    print("=" * 72)

    result = CTAReplicationPipeline(
        run_config=cfg, universe=get_universe(universe), verbose=verbose
    ).run()

    print("\n" + "-" * 72)
    print("Replication quality")
    print("-" * 72)
    print(result.stats_table.round(4).to_string())
    print("\nPerformance")
    print(result.performance.to_string())

    if csv:
        written = result.to_csv()
        print(f"\n{len(written)} CSV files -> {PATHS.output}")

    result.dashboard_path = None
    if dashboard:
        print("\n" + "-" * 72)
        print("Building dashboard")
        print("-" * 72)
        builder = DashboardBuilder(result)
        path = builder.build(
            output_path=output,
            open_browser=open_browser,
            default_theme=theme,
            standalone=standalone,
        )
        result.dashboard_path = path
        mode = "standalone (plotly.js embedded)" if standalone else "plotly.js from CDN"
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB, "
              f"{len(builder.charts)} charts, {mode})")
        if not open_browser:
            print(f"  Open in a browser: {path.resolve().as_uri()}")

    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        return check_data()
    try:
        run(
            universe=args.universe,
            sweep=not args.no_sweep,
            aum_bn=args.aum,
            start=args.start,
            end=args.end,
            scaling=args.scaling,
            horizon=args.horizon,
            driver=args.driver,
            drivers=args.drivers,
            beta_window=args.beta_window,
            vol_update=not args.no_vol_update,
            standalone=args.standalone,
            theme=args.theme,
            output=args.output,
            open_browser=not args.no_open,
            csv=args.csv,
            verbose=not args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"\n[missing data] {exc}\n")
        print("Run `python main.py --check` to see which input files were found.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
