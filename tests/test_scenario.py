"""
Scenario engine checks.

    python tests/test_scenario.py

Key checks:
  * the projected position matches the replica model formula exactly (against an independent recomputation)
  * feeding in the realized returns reproduces the actual next-week position
  * positions are monotonic in the shock (flat in the capped range)
  * the position is actually 0 at the trigger point
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctarep.config import DAYS_PER_WEEK, TRADING_DAYS_PER_YEAR, ModelParams  # noqa: E402
from ctarep.ensemble import EnsembleReplicator  # noqa: E402
from ctarep.scenario import DEFAULT_SCENARIOS, ScenarioEngine  # noqa: E402
from ctarep.timegrid import WeeklyGrid  # noqa: E402
from ctarep.universe import UNIVERSE  # noqa: E402

CODES = UNIVERSE.codes[:6]


def _setup(n_days: int = 1600, seed: int = 21):
    """Return a fitted ensemble and data for a small universe."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    common = rng.standard_normal(n_days)
    cols = {}
    for i, code in enumerate(CODES):
        idio = rng.standard_normal(n_days)
        shocks = 0.7 * idio + 0.3 * common
        trend = np.zeros(n_days)
        for t in range(1, n_days):
            trend[t] = 0.04 * trend[t - 1] + shocks[t]
        cols[code] = 100 * np.exp(np.cumsum(0.011 * trend))
    prices = pd.DataFrame(cols, index=dates)

    daily = prices.pct_change()
    grid = WeeklyGrid.from_index(dates)
    weekly = grid.weekly_returns(prices)
    bm = pd.Series(rng.normal(0.001, 0.02, len(weekly)), index=weekly.index, name="bm")

    ens = EnsembleReplicator(
        [
            ModelParams(momentum_lookback=lb, momentum_cap=1.0, volatility_lookback=90)
            for lb in (16, 32, 52)
        ]
    )
    ens.fit(weekly, daily, bm)
    return prices, daily, weekly, ens


def _engine(daily, weekly, ens, horizon=1, **kw):
    return ScenarioEngine(
        ensemble=ens,
        weekly_returns=weekly,
        daily_returns=daily,
        universe=UNIVERSE.subset(CODES),
        aum=300e9,
        horizon_weeks=horizon,
        **kw,
    )


# --------------------------------------------------------------------------- #


def test_projection_matches_independent_recomputation():
    """Recompute the engine's projected position in pure numpy and compare exactly."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens, horizon=1)

    shock = pd.Series(0.03, index=CODES)  # all markets +3%
    got = eng.positions_after(shock)

    # ---- independent recomputation ----
    r_week = shock  # horizon=1, so used as is
    r_day = (1 + r_week) ** (1 / DAYS_PER_WEEK) - 1

    ext_daily = pd.concat(
        [
            daily[CODES],
            pd.DataFrame(
                np.tile(r_day.values, (DAYS_PER_WEEK, 1)),
                index=pd.bdate_range(daily.index[-1] + pd.Timedelta(days=1), periods=DAYS_PER_WEEK),
                columns=CODES,
            ),
        ]
    )
    ext_weekly = pd.concat(
        [
            weekly[CODES],
            pd.DataFrame([r_week.values], index=[weekly.index[-1] + pd.Timedelta(days=7)],
                         columns=CODES),
        ]
    )

    expected = pd.Series(0.0, index=CODES)
    for model in ens.models:
        p = model.params
        sigma_d = ext_daily.rolling(
            p.volatility_lookback, min_periods=p.effective_vol_min_periods
        ).std(ddof=1).iloc[-1]
        m = (ext_weekly.iloc[-p.momentum_lookback :].mean() / (sigma_d * np.sqrt(DAYS_PER_WEEK))
             ) * np.sqrt(p.momentum_lookback)
        m = m.clip(-p.effective_cap, p.effective_cap)
        k = float(model.fit_result.leverage)
        expected += (m / (sigma_d * np.sqrt(TRADING_DAYS_PER_YEAR))) * k * ens.weights[model.label]
    expected *= ens.rescale_factor

    assert np.allclose(got.values, expected.values, rtol=1e-10), (
        f"max diff = {np.abs(got - expected).max():.3e}"
    )
    print(f"  matches independent recomputation exactly (max diff {np.abs(got - expected).max():.1e}) ✓")


def test_realized_returns_reproduce_actual_next_position():
    """Feeding the realized weekly returns as the shock yields the actual next-week position.

    A practical check that the engine reproduces 'future price -> future position'.
    (Volatility is approximated by spreading the weekly return evenly over 5 days,
    so the match is not exact.)
    """
    _, daily, weekly, ens = _setup()

    # drop the last week and feed that week's realized return in as the 'shock'
    cut_weekly = weekly.iloc[:-1]
    last_week_end = weekly.index[-1]
    cut_daily = daily.loc[: cut_weekly.index[-1]]
    actual_shock = weekly.iloc[-1][CODES]

    eng = ScenarioEngine(
        ensemble=ens,
        weekly_returns=cut_weekly,
        daily_returns=cut_daily,
        universe=UNIVERSE.subset(CODES),
        horizon_weeks=1,
    )
    projected = eng.positions_after(actual_shock)
    actual = ens.positions.loc[last_week_end, CODES]

    corr = float(np.corrcoef(projected.values, actual.values)[0, 1])
    max_err = float((projected - actual).abs().max())
    scale = float(actual.abs().mean())

    assert corr > 0.98, f"corr = {corr:.4f}"
    assert max_err < 0.25 * scale, f"max_err {max_err:.4f} vs mean |pos| {scale:.4f}"
    print(
        f"  realized-return projection vs actual position: corr {corr:.4f}, "
        f"max err {max_err:.4f} (mean |pos| {scale:.3f}) ✓"
    )


def test_position_monotonic_in_shock():
    """With volatility held fixed, the position never decreases in the shock (pure signal effect).

    A market already at the cap is flat, which also satisfies monotonicity.
    """
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens, update_vol=False)

    curve = eng.response_curve(mode="independent", driver=None, sigma_range=6.0, n_points=61)
    for code in CODES:
        diffs = curve[code].diff().dropna()
        assert (diffs >= -1e-12).all(), (
            f"{code}: position decreases over part of the shock range (min Δ = {diffs.min():.2e})"
        )
    span = (curve[CODES].iloc[-1] - curve[CODES].iloc[0]).abs()
    responsive = int((span > 1e-6).sum())
    assert responsive >= len(CODES) // 2, (
        f"only {responsive}/{len(CODES)} respond; most are stuck at the cap"
    )
    print(
        f"  monotonic over the full range with fixed volatility (no decrease) ✓ "
        f"({responsive}/{len(CODES)} markets respond, mean response {span.mean():.2f})"
    )


def test_cap_creates_plateau():
    """Once the momentum cap binds (volatility fixed), the position is exactly flat."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens, update_vol=False)
    # wide enough that every market is certain to hit the cap at both ends
    curve = eng.response_curve(mode="independent", driver=None, sigma_range=60.0, n_points=25)

    left = curve[CODES].iloc[:4].diff().abs().max().max()
    right = curve[CODES].iloc[-4:].diff().abs().max().max()
    middle = curve[CODES].iloc[11:14].diff().abs().mean().mean()

    assert left < 1e-12, f"left capped range is not flat: max Δ = {left:.2e}"
    assert right < 1e-12, f"right capped range is not flat: max Δ = {right:.2e}"
    assert middle > 1e-6, f"middle does not respond: Δ = {middle:.2e}"
    print(f"  cap plateau confirmed: Δ = 0 at both ends, middle Δ {middle:.4f} ✓")


def test_vol_targeting_deleverages_past_the_cap():
    """Under a large shock beyond the cap, notional shrinks: volatility-targeting deleveraging.

    Momentum is already capped and cannot grow, while realized volatility keeps rising,
    so position = cap / σ_ann shrinks. This is observed in real CTAs; it means not
    "CTAs keep selling without limit as prices fall" but "past some point, volatility
    makes them cut notional instead". It is visible in the dashboard response curve.
    """
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens, update_vol=True)
    curve = eng.response_curve(mode="independent", driver=None, sigma_range=8.0, n_points=17)

    peak_abs = curve[CODES].abs().max()
    edge_abs = curve[CODES].abs().iloc[[0, -1]].max()  # larger of the two ends
    reduced = int((edge_abs < peak_abs * 0.95).sum())

    assert reduced >= 1, (
        "no market shows reduced notional under an extreme shock "
        f"(min edge/peak = {(edge_abs / peak_abs).min():.3f})"
    )
    worst = (edge_abs / peak_abs).min()
    print(
        f"  notional reduced under extreme shock in {reduced}/{len(CODES)} markets "
        f"(up to {1 - worst:.0%} lower), volatility-targeting deleveraging ✓"
    )


def test_trigger_point_gives_zero_position():
    """Applying the shock at a point found by the trigger search gives a position near 0."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)
    driver = CODES[0]
    # flips are rare at a 1-week horizon (CTA positions are sticky), so scan widely
    trig = eng.triggers(mode="index", driver=driver, sigma_range=25.0, n_points=201)

    checked = 0
    for code, row in trig.table.iterrows():
        if not np.isfinite(row["flip_sigma"]):
            continue
        shock = eng.shocks_for(row["flip_sigma"], mode="index", driver=driver)
        pos = eng.positions_after(shock)[code]
        scale = float(eng.current.abs().mean())
        assert abs(pos) < 0.05 * scale, f"{code}: position at flip point {pos:+.4f} (scale {scale:.3f})"
        checked += 1

    assert checked > 0, "no flip point was found"
    print(f"  position ≈ 0 at the flip point for {checked} markets ✓")


def test_index_mode_propagates_via_beta():
    """In index mode each market's shock = beta × driver shock."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)
    driver = CODES[0]

    shock = eng.shocks_for(1.0, mode="index", driver=driver)
    betas = eng.betas(driver)
    driver_move = shock[driver]

    assert np.isclose(betas[driver], 1.0)
    assert np.allclose(shock.values, (betas * driver_move).reindex(shock.index).values)
    print(f"  β propagation confirmed (driver {driver} {driver_move:+.2%}, β range "
          f"{betas.min():+.2f}~{betas.max():+.2f}) ✓")


def test_independent_mode_uses_own_sigma():
    """In independent mode each market moves by a multiple of its own weekly σ."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)

    sigma = 2.0
    shock = eng.shocks_for(sigma, mode="independent")
    ratio = (shock / eng._weekly_vol.reindex(shock.index)).dropna()
    assert np.allclose(ratio.values, sigma), f"{ratio.values}"
    print(f"  each market shock = {sigma}σ ✓ (range {shock.min():+.2%}~{shock.max():+.2%})")


def test_scenario_ordering_and_trades():
    """The 5 scenarios run from down to up, and trade = projected - current."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)
    proj = eng.project(mode="index", driver=CODES[0])

    # σ multiples are converted via empirical z quantiles, so look at sigma_map, not Scenario.sigma
    sigmas = proj.sigmas()
    assert sigmas == sorted(sigmas), f"scenario order is not down -> up: {sigmas}"
    assert len(proj.scenarios) == 5
    # check the target frequencies are reproduced.
    # With 2,000+ samples the empirical quantiles are used and match exactly by
    # definition; with fewer it falls back to normal quantiles, so a few pp of
    # deviation is possible.
    n_z = len(eng._z)
    tol = 0.01 if n_z >= 2000 else 0.04
    for s in proj.scenarios:
        if s.target_prob is None:
            continue
        got = proj.empirical_freq[s.key]
        assert abs(got - s.target_prob) < tol, (
            f"{s.key}: target {s.target_prob:.1%} vs empirical {got:.1%} "
            f"(n_z={n_z}, tol={tol:.0%})"
        )
    print(f"  target frequencies reproduced (n_z={n_z}, "
          f"{'empirical quantiles' if n_z >= 2000 else 'normal fallback'})")

    for s in proj.scenarios:
        expected = proj.positions[s.key] - proj.current.reindex(proj.positions.index).fillna(0)
        assert np.allclose(proj.trades_pct[s.key].values, expected.values)

    assert np.allclose(proj.trades_usd.values, proj.trades_pct.values * proj.aum)
    # the driver move in the flat scenario is 0
    assert np.isclose(proj.driver_move()["flat"], 0.0)
    print(f"  5-step ordering / trade consistency ✓ (drivers move "
          f"{proj.driver_move()['strong_down']:+.2%}~{proj.driver_move()['strong_up']:+.2%})")


def test_horizon_scales_shock():
    """For a horizon of H weeks the shock grows by √H (random-walk assumption)."""
    _, daily, weekly, ens = _setup()
    one = _engine(daily, weekly, ens, horizon=1).shocks_for(2.0, mode="independent")
    four = _engine(daily, weekly, ens, horizon=4).shocks_for(2.0, mode="independent")
    assert np.allclose(four.values, one.values * 2.0, rtol=1e-10)  # √4 = 2
    print(f"  4-week shock = 1-week shock × 2 ✓ ({one.iloc[0]:+.2%} -> {four.iloc[0]:+.2%})")


def test_flat_scenario_shows_trend_decay():
    """Even when flat, the position changes, because the oldest week drops out of the lookback."""
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)
    flat = eng.positions_after(pd.Series(0.0, index=CODES))
    drift = (flat - eng.current.reindex(flat.index)).abs()
    assert (drift > 1e-9).any(), "flat scenario is identical to the current position: no trend decay"
    print(f"  positions move by {drift.mean():.4f} on average even when flat (trend decay) ✓")


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
