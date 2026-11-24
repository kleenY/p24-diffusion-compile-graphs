"""S4-P P4b probe step 1: A-tier simulator (p24/sim/simulator_tier_async_S4P.py + p24/policy/tier_async_S4P.py) against hand-computed toy examples. CPU only.
Run: cd code && python -B -m pytest tests/test_tier_async_S4P.py -q"""
import numpy as np
import pandas as pd
import pytest

from p24.policy import CostView, Shape
from p24.policy.tier_async_S4P import ATier
from p24.sim.costs import TIERS, CostTable
from p24.sim.simulator_tier_async_S4P import TierOptions, run_pieces, simulate_tier_async

A = Shape(512, 512, 1, 0)
S1 = Shape(512, 512, 1, 1)   # shares te (B=1) and vae keys with A; own UNet key (batch 2)


def row(arm, shape, steady_ms, resid=None, fc=None, pool=None, cap_s=None):
    r = dict(arm=arm, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, steady_ms=steady_ms, source="measured", fc_source="measured",
             capture_pool_mb=pool if pool is not None else np.nan, capture_s=cap_s if cap_s is not None else np.nan)
    for t in TIERS:
        r[f"resid_s__{t}"] = (resid or {}).get(t, np.nan)
        for c in ("te", "unet", "vae"):
            r[f"fc_{c}_s__{t}"] = ((fc or {}).get(t) or {}).get(c, np.nan)
    return r


def toy(c1g1_pool=50.0, c1g1_resid=0.1):
    cold = dict(te=0.5, unet=1.0, vae=0.5)            # C1: + resid 0.5 -> compile wall 2.5 s at the cold tier (g=1)
    inp = dict(te=0.25, unet=0.5, vae=0.25)           # + resid 0.25 -> 1.25 s at the in-process tier
    rows = []
    for s in (A, S1):
        rows.append(row("E0", s, 1000.0, resid={"cold_diskcache": 0.2, "inproc_diskcache": 0.1}))
        rows.append(row("G1", s, 800.0, resid={"cold_diskcache": 0.3, "inproc_diskcache": 0.3}, pool=100.0, cap_s=0.1))
        rows.append(row("C1", s, 500.0, resid={"cold_diskcache": 0.5, "inproc_diskcache": 0.25}, fc={"cold_diskcache": cold, "inproc_diskcache": inp}))
        rows.append(row("C1+G1", s, 400.0, resid={"cold_diskcache": c1g1_resid, "inproc_diskcache": c1g1_resid},
                        fc={"cold_diskcache": cold, "inproc_diskcache": inp}, pool=c1g1_pool, cap_s=0.1))
    return CostTable(pd.DataFrame(rows))


def reqs(shapes):
    return [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg) for i, s in enumerate(shapes)]


def run(shapes, trigger="kth_request", k=1, f_g1=0.0, f_eager=0.0, g=1.0, budget=None, prior=None, events=False):
    ct = toy()
    cv = CostView(ct, "warm")
    pol = ATier(cv, dict(), prior if prior is not None else {A: 1.0}, trigger, k)
    opt = TierOptions("warm", f_g1, f_eager, g, mem_budget_mb=budget, overhead_aware=False, keep_events=events)
    return simulate_tier_async(reqs(shapes), pol, ct, opt)


def test_run_pieces():
    # 0.3 s at rate 1/2 then 0.8 s at rate 0.8 inside [0, 2.5): 0.6 + 1.0 = 1.6
    t, interf = run_pieces(0.0, [(0.3, 1.0), (0.8, 0.25)], [(0.0, 2.5)])
    assert t == pytest.approx(1.6) and interf == pytest.approx(1.6 - 1.1)


def test_single_shape_f0_g1_k1():
    # compile wall 2.5 s from t=0 (cold tier). r1: G1 capture 0.8 + 0.3 = 1.1; r2, r3: G1 replay 0.8 -> 2.7; r4 starts 2.7 >= 2.5:
    # switch re-capture 0.1 + C1+G1 0.4; r5..r10: 6 x 0.4
    r = run([A] * 10)
    assert r["service_s"] == pytest.approx(1.1 + 0.8 + 0.8 + (0.1 + 0.4) + 6 * 0.4)
    assert r["g1"] == 3 and r["c1g1"] == 7 and r["capture"] == 1 and r["recapture"] == 1 and r["eager"] == 0
    assert r["bg_compile_wall_s"] == pytest.approx(2.5) and r["n_queued"] == 1 and r["n_compiled_by_end"] == 1


def test_single_shape_with_interference_k1():
    # f_eager=1 (rate 1/2), f_g1=0.25 (rate 0.8); window [0, 2.5):
    # r1: 0.3 -> 0.6, 0.8 work at rate 0.8 -> 1.0 s: 1.6.  r2: 0.8 work from 1.6: 0.9 s left in window = 0.72 work, 0.08 at rate 1 -> 2.58.
    # r3 starts 2.58 >= 2.5: switch 0.1 (rate 1) + 0.4, then 7 x 0.4
    r = run([A] * 10, f_g1=0.25, f_eager=1.0)
    assert r["service_s"] == pytest.approx(2.58 + 0.5 + 7 * 0.4)
    assert r["g1"] == 2 and r["c1g1"] == 8
    assert r["interf_eager_s"] == pytest.approx(0.3) and r["interf_g1_s"] == pytest.approx((1.0 - 0.8) + (2.58 - 1.6 - 0.8))


def test_g1_not_recorded_falls_back_to_eager_with_interference():
    # graph budget 10 MB < pool 100 MB: capture denied -> eager (f_eager=1: rate 1/2 inside [0, 2.5)).  r1: (1.0 + 0.2) -> 2.4;
    # r2 from 2.4: 0.1 s in window = 0.05 work, 0.95 at rate 1 -> 3.45; r3 starts 3.45: switch (graph-less: C1, no re-capture), 8 x 0.5
    r = run([A] * 10, f_g1=0.25, f_eager=1.0, budget=10.0)
    assert r["capture_denied"] == 1 and r["eager"] == 2 and r["c1"] == 8 and r["g1"] == 0 and r["recapture"] == 0
    assert r["service_s"] == pytest.approx(3.45 + 8 * 0.5)


def test_k2_trigger_starts_at_second_request():
    # r1 capture 1.1; r2 starts 1.1: compile [1.1, 3.6]; r2..r5 G1 (start 1.1, 1.9, 2.7, 3.5 < 3.6) -> 4.3; r6 switch 0.1 + 0.4 ; r7..r10 4 x 0.4
    r = run([A] * 10, k=2)
    assert r["g1"] == 5 and r["service_s"] == pytest.approx(1.1 + 4 * 0.8 + 0.5 + 4 * 0.4)
    assert r["compile_log"][0][0] == 1 and r["compile_log"][0][4] == pytest.approx(1.1)


def test_ski_rental_trigger():
    # U(C1+G1) at the in-process tier = resid 0.1 + fc 0.25+0.5+0.25 = 1.1; saving per G1 replay 0.8-0.4 = 0.4 (capture request does not count):
    # after r2, r3, r4 the sum is 1.2 >= 1.1 -> queued at the start of r5 (t = 1.1 + 3 x 0.8 = 3.5), wall 2.5 (cold, first shape) -> [3.5, 6.0]
    # r5 3.5->4.3, r6 ->5.1, r7 ->5.9, r8 (5.9 < 6.0) ->6.7, r9 switch 0.1 + 0.4 -> 7.2, r10 -> 7.6
    r = run([A] * 10, trigger="ski_rental")
    assert r["compile_log"][0][0] == 4 and r["compile_log"][0][4] == pytest.approx(3.5)
    assert r["service_s"] == pytest.approx(7.6) and r["g1"] == 8 and r["c1g1"] == 2


def test_switch_pool_does_not_fit_releases_graph():
    ct = toy(c1g1_pool=150.0)
    cv = CostView(ct, "warm")
    pol = ATier(cv, dict(), {A: 1.0}, "kth_request", 1)
    r = simulate_tier_async(reqs([A] * 10), pol, ct, TierOptions("warm", 0.0, 0.0, 1.0, mem_budget_mb=120.0, overhead_aware=False))
    # r1 capture 1.1 (G1 pool 100 <= 120), r2, r3 G1 -> 2.7; r4: C1+G1 pool 150 > 120 even with its own G1 freed -> graph released, C1 0.5 x 7
    assert r["released_at_switch"] == 1 and r["recapture"] == 0 and r["c1"] == 7
    assert r["service_s"] == pytest.approx(2.7 + 7 * 0.5)


def test_two_shapes_fifo_and_key_credit():
    # A first (cold tier, wall 2.5, [0, 2.5]); S1 (k=1) queued at its first request: inproc tier, te + vae keys credited by A -> resid 0.25 + unet 0.5 = 0.75,
    # FIFO start = max(t, 2.5) = 2.5, end 3.25
    r = run([A, S1, S1, S1, S1, S1, S1], prior={A: 0.5, S1: 0.5})
    log = {x[1]: x for x in r["compile_log"]}
    assert log[S1.tag()][2] == "inproc_diskcache" and log[S1.tag()][3] == pytest.approx(0.75)
    assert log[S1.tag()][4] == pytest.approx(2.5) and log[S1.tag()][5] == pytest.approx(3.25)
    assert r["n_queued"] == 2


def test_never_blocks_first_request():
    # the first request only pays the G1 capture (0.3) + G1 steady (0.8) x interference; the compile is never on the request path
    r = run([A], f_g1=0.3, f_eager=1.0, g=1.5)
    assert r["service_s"] == pytest.approx(0.3 * 2.0 + 0.8 * 1.3)


def test_bounded_by_no_compile_baseline():
    # with f=0 the compile is free: service <= pure G1 serving of the same stream (every request G1 0.8 + capture 0.3)
    r = run([A] * 50)
    assert r["service_s"] < 0.3 + 50 * 0.8
