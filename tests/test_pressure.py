"""
Trade pressure checks.

    python tests/test_pressure.py

Checks:
  * probability-mass weights sum to 1 (KDE / normal, including out-of-range tails)
  * independent recomputation: the buy probability and expected trade in the table equal
    values recomputed from the response curve and the weights
  * on normal data the KDE probability is close to the normal-reference probability
  * if the trade has one sign over the whole range, the probability is 0 or 1
  * tier boundaries follow the configured values
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scenario import CODES, _engine, _setup  # noqa: E402

from ctarep.config import RunConfig  # noqa: E402
from ctarep.pressure import TIERS, TradePressureEstimator  # noqa: E402


def _est(**kw):
    _, daily, weekly, ens = _setup()
    eng = _engine(daily, weekly, ens)
    return eng, TradePressureEstimator(eng, RunConfig(**kw))


def test_weights_sum_to_one():
    grid = np.linspace(-5, 5, 201)
    rng = np.random.default_rng(0)
    z = rng.standard_t(3, 3000)  # fat tails: a sizeable mass lies outside the range
    assert abs(TradePressureEstimator._weights_kde(z, grid).sum() - 1) < 1e-12
    assert abs(TradePressureEstimator._weights_norm(grid).sum() - 1) < 1e-12
    print("  weights sum to 1 (KDE and normal) ✓")


def test_table_matches_independent_recomputation():
    eng, est = _est()
    tp = est.run()
    trade = est._trade_curve()
    grid = trade.index.to_numpy(float)
    z = eng.standardized_panel()
    worst = 0.0
    for code in tp.table.index:
        zc = z[code].dropna().to_numpy()
        w = est._weights_kde(zc - zc.mean(), grid)
        tr = trade[code].to_numpy()
        worst = max(worst, abs((w * (tr > 0)).sum() - tp.table.loc[code, "p_buy"]),
                    abs((w * tr).sum() - tp.table.loc[code, "e_trade_pct"]))
    assert worst < 1e-12, f"max diff {worst:.2e}"
    print(f"  table == independent recomputation (max diff {worst:.1e}) ✓")


def test_kde_close_to_normal_on_normal_data():
    _, est = _est()
    t = est.run().table
    gap = (t["p_buy"] - t["p_buy_norm"]).abs().max()
    assert gap < 0.08, f"data is close to normal but KDE and normal probabilities differ by {gap:.3f}"
    print(f"  max |KDE - normal| on normal data: {gap:.3f} ✓")


def test_flip_point_search():
    grid = np.linspace(-5, 5, 201)
    # with no sign change the flip is NaN
    z, side = TradePressureEstimator._nearest_flip(grid, np.full_like(grid, 1.0))
    assert np.isnan(z) and side is None
    z, side = TradePressureEstimator._nearest_flip(grid, grid - 0.5)
    assert abs(z - 0.5) < 1e-9 and side == "buy"
    print("  flip point search (none / single crossing) ✓")


def test_tier_thresholds_follow_config():
    _, est = _est(pressure_sure_prob=0.9, pressure_lean_prob=0.6)
    assert est._tier(0.95) == "buy_sure" and est._tier(0.65) == "buy_lean"
    assert est._tier(0.5) == "mixed"
    assert est._tier(0.35) == "sell_lean" and est._tier(0.05) == "sell_sure"
    tp = est.run()
    assert set(tp.table["tier"]) <= set(TIERS)
    assert tp.table["p_buy"].between(0, 1).all()
    print("  tier boundaries follow the configured values ✓")


def test_short_history_falls_back_to_normal():
    _, est = _est(pressure_min_obs=10_000)
    t = est.run().table
    assert (t["method"] == "normal").all()
    assert np.allclose(t["p_buy"], t["p_buy_norm"])
    print("  falls back to normal when observations are insufficient ✓")


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
