"""
Model formula checks.

    python -m pytest tests/ -v
    python tests/test_model.py          # also runs without pytest

Checks that the formulas are implemented exactly, that there is no lookahead, and
that the rolled-futures construction removes roll gaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctarep.config import DAYS_PER_WEEK, TRADING_DAYS_PER_YEAR, ModelParams  # noqa: E402
from ctarep.ensemble import EnsembleReplicator  # noqa: E402
from ctarep.evaluation import r_squared  # noqa: E402
from ctarep.portfolio import ReplicationModel  # noqa: E402
from ctarep.rolling import FuturesRoller  # noqa: E402
from ctarep.signals import NormalizedMomentum  # noqa: E402
from ctarep.timegrid import WeeklyGrid  # noqa: E402
from ctarep.universe import UNIVERSE  # noqa: E402
from ctarep.volatility import RealizedVolatility  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _panel(n_days: int = 900, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, WeeklyGrid]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    codes = UNIVERSE.codes[:4]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n_days, len(codes))), axis=0)),
        index=dates,
        columns=codes,
    )
    daily = prices.pct_change()
    grid = WeeklyGrid.from_index(dates)
    weekly = grid.weekly_returns(prices)
    return daily, weekly, grid


# --------------------------------------------------------------------------- #
# 1. Normalized momentum formula
# --------------------------------------------------------------------------- #


def test_normalized_momentum_matches_closed_form():
    """Compare m = (r̄ / σ_w) × √L with a hand calculation."""
    daily, weekly, _ = _panel()
    params = ModelParams(momentum_lookback=32, momentum_cap=0.0, volatility_lookback=90)
    panel = NormalizedMomentum(params).compute(weekly, daily)

    sigma_d = RealizedVolatility.from_params(params).daily(daily)
    dt = panel.momentum.dropna(how="all").index[-1]
    code = weekly.columns[0]

    r_bar = weekly[code].loc[:dt].iloc[-32:].mean()
    sigma_w = sigma_d[code].asof(dt) * np.sqrt(DAYS_PER_WEEK)
    expected = (r_bar / sigma_w) * np.sqrt(32)

    assert np.isclose(panel.momentum.loc[dt, code], expected, rtol=1e-10), (
        f"got {panel.momentum.loc[dt, code]}, expected {expected}"
    )
    print(f"  m = {expected:+.4f}  (closed form matches)")


def test_momentum_equals_tstat_and_window_sharpe():
    """m is (1) the t-statistic of the mean and (2) L-week cumulative return / L-week volatility; both agree."""
    daily, weekly, _ = _panel()
    L = 16
    params = ModelParams(momentum_lookback=L, momentum_cap=0.0, volatility_lookback=60)
    panel = NormalizedMomentum(params).compute(weekly, daily)

    dt = panel.momentum.dropna(how="all").index[-1]
    code = weekly.columns[1]
    window = weekly[code].loc[:dt].iloc[-L:]
    sigma_w = panel.vol_weekly.loc[dt, code]

    tstat = window.mean() / (sigma_w / np.sqrt(L))
    window_sharpe = (window.sum()) / (sigma_w * np.sqrt(L))

    assert np.isclose(panel.momentum.loc[dt, code], tstat, rtol=1e-10)
    assert np.isclose(panel.momentum.loc[dt, code], window_sharpe, rtol=1e-10)
    print(f"  t-stat = window-sharpe = {tstat:+.4f}")


def test_cap_clips_and_zero_means_no_cap():
    """The cap actually clips, and a cap of 0.0 means unlimited."""
    daily, weekly, _ = _panel()
    uncapped = NormalizedMomentum(
        ModelParams(momentum_lookback=8, momentum_cap=0.0, volatility_lookback=30)
    ).compute(weekly, daily)
    capped = NormalizedMomentum(
        ModelParams(momentum_lookback=8, momentum_cap=0.5, volatility_lookback=30)
    ).compute(weekly, daily)

    assert capped.momentum.abs().max().max() <= 0.5 + 1e-12
    assert uncapped.momentum.abs().max().max() > 0.5, "the sample needs signals above the cap for the check to be meaningful"
    # cap 0.0 -> same as raw
    pd.testing.assert_frame_equal(uncapped.momentum, uncapped.raw_momentum)
    print(f"  uncapped max |m| = {uncapped.momentum.abs().max().max():.2f}, capped <= 0.5 ✓")


def test_volatility_scaling():
    """σ_w = σ_d√5, σ_ann = σ_d√252."""
    daily, _, _ = _panel()
    rv = RealizedVolatility(lookback=60)
    panel = rv.compute(daily)
    assert np.allclose(
        panel.weekly.dropna().values, panel.daily.dropna().values * np.sqrt(DAYS_PER_WEEK)
    )
    assert np.allclose(
        panel.annualized.dropna().values,
        panel.daily.dropna().values * np.sqrt(TRADING_DAYS_PER_YEAR),
    )
    print("  σ_w = σ_d√5, σ_ann = σ_d√252 ✓")


# --------------------------------------------------------------------------- #
# 2. Portfolio construction
# --------------------------------------------------------------------------- #


def test_no_lookahead():
    """R_τ uses only w_{τ−1}: changing the last week's return leaves earlier returns unchanged."""
    daily, weekly, _ = _panel()
    model = ReplicationModel(ModelParams(momentum_lookback=8, volatility_lookback=30))
    pos = model.build_positions(weekly, daily)

    base = ReplicationModel.portfolio_returns(pos, weekly)

    tampered = weekly.copy()
    tampered.iloc[-1] = tampered.iloc[-1] + 0.5  # large shock in the last week
    shocked = ReplicationModel.portfolio_returns(pos, tampered)

    pd.testing.assert_series_equal(base.iloc[:-1], shocked.iloc[:-1])
    assert not np.isclose(base.iloc[-1], shocked.iloc[-1])
    print("  last-week shock does not affect past returns ✓")


def test_leverage_matches_benchmark_vol():
    """After fitting k, replica annualized volatility == benchmark annualized volatility."""
    daily, weekly, _ = _panel()
    rng = np.random.default_rng(3)
    bm = pd.Series(rng.normal(0.001, 0.02, len(weekly)), index=weekly.index, name="bm")

    model = ReplicationModel(ModelParams(momentum_lookback=16, volatility_lookback=60))
    model.fit(weekly, daily, bm, method="in_sample")

    common = model.returns.dropna().index.intersection(bm.dropna().index)
    rep_vol = model.returns.loc[common].std(ddof=1) * np.sqrt(52)
    bm_vol = bm.loc[common].std(ddof=1) * np.sqrt(52)
    assert np.isclose(rep_vol, bm_vol, rtol=1e-8), f"{rep_vol:.4f} vs {bm_vol:.4f}"
    print(f"  replica vol {rep_vol:.2%} == benchmark vol {bm_vol:.2%} ✓")


def test_r2_is_scale_invariant():
    """R² does not depend on leverage, which justifies using unlevered positions in the sweep."""
    daily, weekly, _ = _panel()
    rng = np.random.default_rng(11)
    bm = pd.Series(rng.normal(0.001, 0.02, len(weekly)), index=weekly.index, name="bm")

    model = ReplicationModel(ModelParams(momentum_lookback=16, volatility_lookback=60))
    raw = model.build_positions(weekly, daily)
    unlevered = ReplicationModel.portfolio_returns(raw, weekly)

    assert np.isclose(r_squared(unlevered, bm), r_squared(unlevered * 7.3, bm), rtol=1e-12)
    print(f"  R² = {r_squared(unlevered, bm):.4f} (same at 7.3x leverage) ✓")


def test_ensemble_position_average_equals_return_average():
    """Average of positions == average of returns (returns are linear in w)."""
    daily, weekly, _ = _panel()
    rng = np.random.default_rng(5)
    bm = pd.Series(rng.normal(0.001, 0.02, len(weekly)), index=weekly.index, name="bm")

    ens = EnsembleReplicator(
        [ModelParams(momentum_lookback=lb, volatility_lookback=60) for lb in (8, 16, 32)]
    )
    ens.fit(weekly, daily, bm, rescale=False)

    # Lookbacks differ, so only some members are live early on. The ensemble there
    # treats immature members as flat (1/3 weight), so the equivalence is checked
    # on the span where all members are active.
    mature = ens.member_returns.dropna().index
    from_positions = ens.returns.reindex(mature)
    from_returns = ens.member_returns.loc[mature].mean(axis=1)

    assert len(mature) > 100, f"only {len(mature)} mature weeks"
    assert np.allclose(from_positions.values, from_returns.values, atol=1e-12), (
        f"max diff = {np.abs(from_positions - from_returns).max():.3e}"
    )
    print(f"  position average == return average (checked on {len(mature)} weeks) ✓")


def test_naive_risk_parity_equalizes_risk():
    """m̃/σ_ann weights equalize every market's risk contribution (after adjusting for signal size)."""
    daily, weekly, _ = _panel()
    params = ModelParams(momentum_lookback=16, momentum_cap=1.0, volatility_lookback=60)
    model = ReplicationModel(params)
    pos = model.build_positions(weekly, daily)

    dt = pos.dropna(how="all").index[-1]
    vol_ann = model.signals.vol_annualized.loc[dt]
    mom = model.signals.momentum.loc[dt]

    # position x volatility = risk contribution -> must be exactly proportional to normalized momentum
    risk = pos.loc[dt] * vol_ann
    assert np.allclose(risk.dropna().values, mom.reindex(risk.dropna().index).values, rtol=1e-10)
    print("  w × σ_ann = m̃ (naive risk parity) ✓")


# --------------------------------------------------------------------------- #
# 3. Rolled futures
# --------------------------------------------------------------------------- #


def test_rolling_removes_roll_gaps():
    """Whether rolled futures built from C1/C2 remove the artificial price jump at each roll.

    The most important data-processing step in this project.
    """
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)

    # 'true' price (no roll gaps)
    true_px = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))

    # 5% contango: C2 is always 5% more expensive than C1
    contango = 1.05
    roll_positions = [150, 300, 450]

    # observed C1: at each roll it switches to a contract 5% higher
    step = np.zeros(n)
    for j, pos in enumerate(roll_positions, start=1):
        step[pos + 1 :] = j
    c1 = pd.DataFrame({"CL": true_px * contango**step}, index=dates)
    c2 = pd.DataFrame({"CL": c1["CL"] * contango}, index=dates)

    roll_dates = {"CL": dates[roll_positions]}
    roller = FuturesRoller(UNIVERSE.subset(["CL"]))
    rolled, _ = roller.build(c1, c2, roll_dates, method="relative")

    raw_rtn = c1["CL"].pct_change()
    adj_rtn = rolled["CL"].pct_change()
    true_rtn = pd.Series(true_px, index=dates).pct_change()

    # the raw return on the day after a roll must carry exactly the contango multiple
    # (holds deterministically, regardless of the random-walk returns)
    gap_days = [dates[p + 1] for p in roll_positions]
    inflation = (1 + raw_rtn.loc[gap_days]) / (1 + true_rtn.loc[gap_days])
    assert np.allclose(inflation.values, contango), f"test setup error: {inflation.values}"

    # after adjustment the gap is gone and returns match the 'true' returns exactly
    assert np.allclose(adj_rtn.dropna().values, true_rtn.dropna().values, atol=1e-12), (
        f"roll gap remains: max diff = "
        f"{np.abs(adj_rtn.dropna() - true_rtn.dropna()).max():.2e}"
    )
    print(
        f"  roll gap {contango - 1:.0%} (raw C1) -> "
        f"{np.abs(adj_rtn.dropna() - true_rtn.dropna()).max():.0e} (adjusted) ✓"
    )


def test_roll_gap_corrupts_momentum_if_unadjusted():
    """Numerically confirm that raw C1 corrupts the momentum signal.

    The justification for why rolled futures are needed.
    """
    rng = np.random.default_rng(1)
    n = 1200
    dates = pd.bdate_range("2019-01-01", periods=n)
    true_px = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))

    contango = 1.04
    roll_positions = list(range(60, n, 63))
    step = np.zeros(n)
    for j, pos in enumerate(roll_positions, start=1):
        step[pos + 1 :] = j

    rolled = pd.DataFrame({"CL": true_px}, index=dates)
    unadjusted = pd.DataFrame({"CL": true_px * contango**step}, index=dates)

    params = ModelParams(momentum_lookback=32, momentum_cap=0.0, volatility_lookback=90)
    engine = NormalizedMomentum(params)

    results = {}
    for name, px in (("rolled", rolled), ("unadjusted", unadjusted)):
        grid = WeeklyGrid.from_index(px.index)
        panel = engine.compute(grid.weekly_returns(px), px.pct_change())
        results[name] = panel.momentum["CL"].dropna()

    diff = (results["unadjusted"] - results["rolled"]).abs().mean()
    assert diff > 0.5, f"contamination not detected (mean |diff| = {diff:.3f})"
    print(
        f"  mean momentum distortion with raw C1 = {diff:.2f} (signal units), "
        f"severe against a cap of 1.0 ✓"
    )


# --------------------------------------------------------------------------- #
# 4. Weekly grid
# --------------------------------------------------------------------------- #


def test_weekly_grid_uses_last_trading_day():
    """If a holiday removes Friday, the week uses its last actual trading day."""
    dates = pd.bdate_range("2024-01-01", "2024-02-29")
    good_friday = pd.Timestamp("2024-01-19")
    dates = dates[dates != good_friday]  # remove one Friday

    grid = WeeklyGrid.from_index(dates)
    week = [d for d in grid.dates if pd.Timestamp("2024-01-15") <= d <= pd.Timestamp("2024-01-21")]
    assert len(week) == 1
    assert week[0] == pd.Timestamp("2024-01-18"), week[0]  # pulled back to Thursday
    print("  Friday closed -> automatically moves to Thursday ✓")


# --------------------------------------------------------------------------- #

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"\n{fn.__name__}")
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
