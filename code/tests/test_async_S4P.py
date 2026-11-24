"""S4-P P4 probe step 1: A-async simulator (p24/sim/simulator_async_S4P.py) against hand-computed toy examples. CPU only.
Run: cd code && python -B -m pytest tests/test_async_S4P.py -q"""
import math

import numpy as np
import pandas as pd
import pytest

from p24.policy import Shape
from p24.policy.async_S4P import AAsync
from p24.sim.costs import TIERS, CostTable
from p24.sim.simulator_async_S4P import AsyncOptions, run_work, simulate_async

A = Shape(512, 512, 1, 0)
S1 = Shape(512, 512, 1, 1)   # shares te (B=1) and vae keys with A; own UNet key (batch 2)


def row(arm, shape, steady_ms, resid=None, fc=None):
    r = dict(arm=arm, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, steady_ms=steady_ms, source="measured", fc_source="measured")
    for t in TIERS:
        r[f"resid_s__{t}"] = (resid or {}).get(t, np.nan)
        for c in ("te", "unet", "vae"):
            r[f"fc_{c}_s__{t}"] = ((fc or {}).get(t) or {}).get(c, np.nan)
    return r


def toy():
    cold = dict(te=0.5, unet=1.0, vae=0.5)            # + resid 0.5 -> compile one-time 2.5 s at the cold tier
    inp = dict(te=0.25, unet=0.5, vae=0.25)           # + resid 0.25 -> 1.25 s at the in-process tier
    rows = []
    for s in (A, S1):
        rows.append(row("E0", s, 1000.0, resid={"cold_diskcache": 0.2, "inproc_diskcache": 0.1}))
        rows.append(row("C1", s, 500.0, resid={"cold_diskcache": 0.5, "inproc_diskcache": 0.25},
                        fc={"cold_diskcache": cold, "inproc_diskcache": inp}))
    return CostTable(pd.DataFrame(rows))


def reqs(shapes):
    return [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg) for i, s in enumerate(shapes)]


def test_run_work_piecewise():
    assert run_work(0, 1, 0.0, [(0, 5)]) == 1
    assert run_work(0, 1, 0.5, []) == 1
    assert run_work(0, 1, 0.5, [(0, 2.5)]) == pytest.approx(1.5)
    assert run_work(1.8, 1.0, 0.5, [(0, 2.5)]) == pytest.approx(2.5 + (1.0 - 0.7 / 1.5))       # straddles the end of the interval
    # gap before the interval: 1 s at rate 1, then 1 s of work at rate 1/2 takes 2 s
    assert run_work(0, 2, 1.0, [(1, 5)]) == pytest.approx(3.0)


def test_single_shape_f0_g1():
    # compile wall 2.5 s from t=0; eager first request 1.0 + 0.2, then 1.0 each; at t=3.2 the shape is ready -> C1 0.5 each
    r = simulate_async(reqs([A] * 10), AAsync(), toy(), AsyncOptions("warm", 0.0, 1.0))
    assert r["n_eager"] == 3 and r["n_c1"] == 7
    assert r["service_s"] == pytest.approx(1.2 + 1.0 + 1.0 + 7 * 0.5)
    assert r["bg_compile_wall_s"] == pytest.approx(2.5)


def test_single_shape_f05():
    r = simulate_async(reqs([A] * 10), AAsync(), toy(), AsyncOptions("warm", 0.5, 1.0))
    # req1 work 1.2 inside [0,2.5): ends 1.8; req2 starts 1.8, 0.7 s of interval left -> 0.4667 work, rest 0.5333 at rate 1 -> 3.0333; then C1
    assert r["n_eager"] == 2 and r["n_c1"] == 8
    assert r["service_s"] == pytest.approx(1.8 + (2.5 - 1.8) + (1.0 - 0.7 / 1.5) + 8 * 0.5)


def test_g_scales_compile_wall():
    r = simulate_async(reqs([A] * 10), AAsync(), toy(), AsyncOptions("warm", 0.0, 2.0))      # wall 5 s: eager until t>=5: 1.2,2.2,3.2,4.2,5.2
    assert r["n_eager"] == 5 and r["service_s"] == pytest.approx(1.2 + 4 * 1.0 + 5 * 0.5)


def test_two_shapes_queue_and_key_credit():
    # A first (cold, 2.5 s, [0,2.5]); S1 first seen at request 2 (t=1.2): inproc tier, te+vae keys credited by A, unet 0.5 + resid 0.25 = 0.75;
    # FIFO: starts at 2.5, ends 3.25.
    r = simulate_async(reqs([A, S1, S1, S1, S1, S1, S1]), AAsync(), toy(), AsyncOptions("warm", 0.0, 1.0))
    log = {x[1]: x for x in r["compile_log"]}
    assert log[S1.tag()][2] == "inproc_diskcache"
    assert log[S1.tag()][3] == pytest.approx(0.75) and log[S1.tag()][4] == pytest.approx(2.5) and log[S1.tag()][5] == pytest.approx(3.25)
    ev = r["events"]
    # r1 A eager_first 1.2 (->1.2); r2 S1 eager_first 1.0+0.1 (->2.3); r3 S1 eager 1.0 (->3.3 >= 3.25); r4.. C1 0.5
    assert ev[1]["event"] == "eager_first" and ev[1]["t_end_s"] == pytest.approx(2.3)
    assert ev[2]["event"] == "steady_eager" and ev[2]["t_end_s"] == pytest.approx(3.3)
    assert all(e["event"] == "c1" for e in ev[3:]) and r["service_s"] == pytest.approx(3.3 + 4 * 0.5)


def test_f_scope_all_slows_c1():
    # same stream, f=1 scope 'all': C1 requests that start/overlap a compile interval are slower
    r0 = simulate_async(reqs([A, S1, S1, S1, S1, S1, S1]), AAsync(), toy(), AsyncOptions("warm", 1.0, 1.0, "eager"))
    r1 = simulate_async(reqs([A, S1, S1, S1, S1, S1, S1]), AAsync(), toy(), AsyncOptions("warm", 1.0, 1.0, "all"))
    assert r1["service_s"] >= r0["service_s"]


def test_a_async_never_blocks_first_request():
    # the policy never charges the compile to the request path: the first request is eager and costs steady + eager one-time only
    r = simulate_async(reqs([A]), AAsync(), toy(), AsyncOptions("warm", 0.3, 1.5))
    assert r["service_s"] == pytest.approx((1.0 + 0.2) * 1.3)


def test_switch_fn_blocking_load_with_key_credit():
    # subprocess alternative: the main process loads each finished shape from the disk cache (blocking); keys already loaded are credited by switch_fn
    def sw(shape, loaded):
        return 0.5 if shape == A else (0.2 if ("unet", 512, 512, 2) not in loaded else 0.0)
    r = simulate_async(reqs([A] * 4 + [S1] * 4), AAsync(), toy(), AsyncOptions("warm", 0.0, 1.0, switch_fn=sw))
    r0 = simulate_async(reqs([A] * 4 + [S1] * 4), AAsync(), toy(), AsyncOptions("warm", 0.0, 1.0))
    assert r["n_c1"] == r0["n_c1"] and r["service_s"] == pytest.approx(r0["service_s"] + 0.5 + 0.2)
