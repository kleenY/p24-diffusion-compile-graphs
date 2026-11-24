"""S2 step 2.5 / 2.6 code: request-level simulator, break-even formulas, cost-table helpers, serve_trace replay loop.
CPU only (no torch import): hand-computed examples must be reproduced exactly."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from p24.policy import Shape, component_keys, make_policy
from p24.sim import breakeven as be
from p24.sim.costs import TIERS, CostTable, MissingCost
from p24.sim.simulator import SimOptions, simulate
from scripts import build_cost_table as bct
from scripts import serve_trace as st_

A = Shape(512, 512, 1, 0)
B_ = Shape(768, 768, 1, 0)
S1 = Shape(512, 512, 1, 1)   # B1 with CFG: UNet batch 2
S2 = Shape(512, 512, 2, 0)   # B2 without CFG: UNet batch 2 (shares the UNet graph with S1)


def row(arm, shape, steady_ms, resid=None, fc=None, pool=np.nan, weights=2000.0, peak=2500.0, fc_source="measured"):
    r = dict(arm=arm, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, steady_ms=steady_ms, capture_pool_mb=pool,
             weights_alloc_mb=weights, steady_peak_reserved_mb=peak, transient_excess_s=np.nan, source="measured",
             fc_source=fc_source)
    for t in TIERS:
        r[f"resid_s__{t}"] = (resid or {}).get(t, np.nan)
        for c in ("te", "unet", "vae"):
            r[f"fc_{c}_s__{t}"] = ((fc or {}).get(t) or {}).get(c, np.nan)
    return r


def table(rows):
    return CostTable(pd.DataFrame(rows))


def reqs(shapes):
    return [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg, noise_seed=i, prompt="p") for i, s in enumerate(shapes)]


# ---------------------------------------------------------------- hand-computed examples
def test_example1_b0_same_shape():
    """3 requests, one shape, eager: first call 0.5 s extra + 3 x 0.1 s = 0.8 s."""
    ct = table([row("E0", A, 100.0, resid={"cold_diskcache": 0.5})])
    r = simulate(reqs([A, A, A]), make_policy("B0"), ct, SimOptions(load_s=7.0))
    assert [e["event"] for e in r.events] == ["eager_first", "steady", "steady"]
    assert r.summary["service_s"] == pytest.approx(0.8, abs=1e-12)
    assert r.summary["total_incl_load_s"] == pytest.approx(7.8, abs=1e-12)
    assert [e["dur_s"] for e in r.events] == pytest.approx([0.6, 0.1, 0.1], abs=1e-12)


def test_example2_b4_two_shapes_alternating():
    """A,B,A,B with capture on first encounter. A is the process's first shape (cold tier 1.0 s), B a later shape
    (in-process tier 2.0 s): (1.0+0.05) + (2.0+0.2) + 0.05 + 0.2 = 3.5 s; graph memory 100 + 300 MB."""
    ct = table([row("G1", A, 50.0, resid={"cold_diskcache": 1.0, "inproc_diskcache": 9.0}, pool=100.0),
                row("G1", B_, 200.0, resid={"cold_diskcache": 8.0, "inproc_diskcache": 2.0}, pool=300.0)])
    r = simulate(reqs([A, B_, A, B_]), make_policy("B4"), ct)
    assert [e["event"] for e in r.events] == ["capture", "capture", "steady", "steady"]
    assert [e["tier"] for e in r.events] == ["cold_diskcache", "inproc_diskcache", "", ""]
    assert r.summary["service_s"] == pytest.approx(3.5, abs=1e-12)
    assert r.summary["peak_graph_mb"] == 400.0
    assert r.summary["n_events_with_fallback"] == 0


def test_example3_b1_shared_unet_pays_only_te_and_vae():
    """B1: S1 (B1, CFG on) then S2 (B2, CFG off): the compiled UNet (batch 2, same resolution) is reused, so the second
    shape pays te + vae first calls and its residual only.
    req1 0.5+1+10+2 + 0.12 = 13.62 ; req2 0.25+1.5+2.5 + 0.15 = 4.40 ; req3 0.15 ; req4 0.12  -> 18.29 s"""
    assert component_keys(S1)["unet"] == component_keys(S2)["unet"]
    assert component_keys(S1)["te"] != component_keys(S2)["te"] and component_keys(S1)["vae"] != component_keys(S2)["vae"]
    ct = table([
        row("C1", S1, 120.0, resid={"cold_diskcache": 0.5}, fc={"cold_diskcache": dict(te=1.0, unet=10.0, vae=2.0)}),
        row("C1", S2, 150.0, resid={"inproc_diskcache": 0.25},
            fc={"inproc_diskcache": dict(te=1.5, unet=11.0, vae=2.5)})])
    r = simulate(reqs([S1, S2, S2, S1]), make_policy("B1"), ct)
    assert [e["event"] for e in r.events] == ["compile", "compile", "steady", "steady"]
    assert [e["dur_s"] for e in r.events] == pytest.approx([13.62, 4.40, 0.15, 0.12], abs=1e-12)
    assert r.summary["service_s"] == pytest.approx(18.29, abs=1e-12)


def test_b1_cfg_does_not_share_unet_when_batches_differ():
    """B2 CFG-on (UNet batch 4) after B1 CFG-on (batch 2): nothing shared, everything is paid."""
    s3 = Shape(512, 512, 2, 1)
    ct = table([row("C1", S1, 100.0, resid={"cold_diskcache": 0.0}, fc={"cold_diskcache": dict(te=1, unet=10, vae=2)}),
                row("C1", s3, 100.0, resid={"inproc_diskcache": 0.0}, fc={"inproc_diskcache": dict(te=1, unet=10, vae=2)})])
    r = simulate(reqs([S1, s3]), make_policy("B1"), ct)
    assert r.summary["service_s"] == pytest.approx(13.1 + 13.1, abs=1e-12)


def test_cold_empty_mode_uses_empty_tiers_and_optional_transient():
    ct = table([dict(row("C1", A, 100.0, resid={"cold_empty": 1.0, "cold_diskcache": 0.1},
                         fc={"cold_empty": dict(te=5, unet=40, vae=15), "cold_diskcache": dict(te=1, unet=2, vae=1)}),
                     transient_excess_s=3.0)])
    r0 = simulate(reqs([A, A]), make_policy("B1"), ct, SimOptions(cache_mode="empty"))
    assert r0.summary["service_s"] == pytest.approx(1 + 60 + 0.1 + 0.1, abs=1e-12)
    r1 = simulate(reqs([A, A]), make_policy("B1"), ct, SimOptions(cache_mode="empty", use_transient=True))
    assert r1.summary["service_s"] == pytest.approx(64.2, abs=1e-12)
    rw = simulate(reqs([A, A]), make_policy("B1"), ct, SimOptions(cache_mode="warm", use_transient=True))
    assert rw.summary["service_s"] == pytest.approx(0.1 + 4 + 0.1 + 0.1, abs=1e-12)  # transient only in empty tiers


def test_missing_tier_falls_back_and_is_flagged():
    ct = table([row("G1", A, 50.0, resid={"cold_diskcache": 1.0}, pool=10.0),
                row("G1", B_, 100.0, resid={"cold_diskcache": 2.0}, pool=10.0)])  # no inproc tier at all
    r = simulate(reqs([A, B_]), make_policy("B4"), ct)
    assert r.summary["service_s"] == pytest.approx(1.05 + 2.1, abs=1e-12)
    assert r.summary["n_events_with_fallback"] == 1
    assert "inproc_diskcache->cold_diskcache" in r.events[1]["fallbacks"]


def test_missing_cold_tier_raises_instead_of_crossing_cache_states():
    """No cold_empty cost: the simulator must not substitute the (10x cheaper) disk-cache value."""
    ct = table([row("C1", A, 100.0, resid={"cold_diskcache": 0.1},
                    fc={"cold_diskcache": dict(te=1, unet=2, vae=1)})])
    with pytest.raises(MissingCost):
        simulate(reqs([A]), make_policy("B1"), ct, SimOptions(cache_mode="empty"))
    # in-process tier of the same cache state may fall back to its cold tier (flagged)
    ct2 = table([row("C1", A, 100.0, resid={"cold_empty": 1.0}, fc={"cold_empty": dict(te=1, unet=2, vae=1)}),
                 row("C1", B_, 100.0, resid={"cold_empty": 2.0}, fc={"cold_empty": dict(te=3, unet=4, vae=5)})])
    r = simulate(reqs([A, B_]), make_policy("B1"), ct2, SimOptions(cache_mode="empty"))
    # te key = B only (S-5): B_ (768x768 B1) reuses the B1 text encoder compiled for A -> te (3) not charged again
    assert r.summary["service_s"] == pytest.approx((1 + 4 + 0.1) + (2 + 9 + 0.1), abs=1e-12)
    assert r.summary["n_events_with_fallback"] == 1


def test_missing_shape_raises():
    ct = table([row("E0", A, 100.0, resid={"cold_diskcache": 0.5})])
    with pytest.raises(MissingCost):
        simulate(reqs([B_]), make_policy("B0"), ct)


def test_fc_missing_cells_put_everything_in_resid():
    ct = table([row("C1", A, 100.0, resid={"cold_diskcache": 9.0}, fc_source="missing")])
    r = simulate(reqs([A, A]), make_policy("B1"), ct)
    assert r.summary["service_s"] == pytest.approx(9.2, abs=1e-12)
    assert "fc:none" in r.events[0]["fallbacks"]


def test_capture_budget_denial_falls_back_to_eager():
    """Budget 150 MB, pools 100 + 300: A is captured, B is denied and served eagerly (E0 cost incl. its first call)."""
    ct = table([row("G1", A, 50.0, resid={"cold_diskcache": 1.0}, pool=100.0),
                row("G1", B_, 200.0, resid={"inproc_diskcache": 2.0}, pool=300.0),
                row("E0", B_, 400.0, resid={"inproc_diskcache": 0.5})])
    r = simulate(reqs([A, B_, B_]), make_policy("B4"), ct, SimOptions(mem_budget_mb=150.0))
    assert [e["event"] for e in r.events] == ["capture", "capture_denied", "capture_denied"]
    assert r.summary["service_s"] == pytest.approx(1.05 + (0.5 + 0.4) + 0.4, abs=1e-12)
    assert r.summary["peak_graph_mb"] == 100.0


# ---------------------------------------------------------------- break-even formulas
def test_breakeven_formulas_and_boundaries():
    assert be.n_capture(0.10, 0.08, 1.0) == pytest.approx(50.0)
    assert be.n_compile(0.10, 0.05, 5.0, 1.0) == pytest.approx(120.0)
    assert be.n_upgrade(0.08, 0.05, 5.0) == pytest.approx(5.0 / 0.03)
    # denominators <= 0 -> inf
    assert math.isinf(float(be.n_capture(0.08, 0.08, 1.0)))
    assert math.isinf(float(be.n_capture(0.08, 0.09, 1.0)))
    assert math.isinf(float(be.n_compile(0.05, 0.05, 5.0, 1.0)))
    assert math.isinf(float(be.n_upgrade(0.05, 0.08, 5.0)))
    # negative one-time cost is clipped to 0 requests (capture is free), not negative
    assert float(be.n_capture(0.10, 0.08, -0.5)) == 0.0
    # vectorised, NaN stays NaN
    v = be.n_capture(np.array([0.1, 0.1, np.nan]), np.array([0.08, 0.1, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert v[0] == pytest.approx(50.0) and math.isinf(v[1]) and np.isnan(v[2])


def test_breakeven_bootstrap_constant_input_and_inf_ci():
    n = 5
    g = dict(e_s=np.full(n, 0.10), x_e=np.full(n, 0.5), g_s=np.full(n, 0.08), x_g=np.full(n, 1.5), o_s=np.full(n, 0.05),
             x_o=np.full(n, 6.5), x_c=np.full(n, 5.5))
    ci = be.bootstrap(g, n_boot=500, seed=0)
    assert ci["n_capture"] == pytest.approx((50.0, 50.0))      # cg = 1.0, e-g = 0.02
    assert ci["n_compile"] == pytest.approx((6.0 / 0.05, 6.0 / 0.05))
    # graph never faster -> inf (also in the CI), no nan arithmetic on inf
    g2 = dict(g, g_s=np.full(n, 0.11))
    ci2 = be.bootstrap(g2, n_boot=500, seed=0)
    assert math.isinf(ci2["n_capture"][0]) and math.isinf(ci2["n_capture"][1])
    # deterministic given the seed, and needs >= 2 processes per group
    g3 = dict(g, e_s=np.array([0.09, 0.10, 0.11, 0.10, 0.12]))
    assert be.bootstrap(g3, 500, 0) == be.bootstrap(g3, 500, 0)
    assert be.bootstrap(dict(g, e_s=np.array([0.1])), 500, 0) is None


def test_breakeven_table_from_cost_table_point_values():
    s = A
    ct = table([row("E0", s, 100.0, resid={"cold_diskcache": 0.5}),
                row("G1", s, 80.0, resid={"cold_diskcache": 1.5}, pool=10),
                row("C1", s, 95.0, resid={"cold_diskcache": 0.5}, fc={"cold_diskcache": dict(te=1, unet=3, vae=1)}),
                row("C1+G1", s, 50.0, resid={"cold_diskcache": 2.0}, fc={"cold_diskcache": dict(te=1, unet=3, vae=1)},
                    pool=10)])
    t = be.breakeven_table(ct)
    r = t[t.tier == "cold_diskcache"].iloc[0]
    # Cg = 1.5-0.5 = 1.0 ; Cc = (0.5+5) - 0.5 = 5.0 ; direct = (2+5) - 0.5 = 6.5
    assert r.cg_s == pytest.approx(1.0) and r.cc_s == pytest.approx(5.0)
    assert r.n_capture == pytest.approx(1.0 / 0.02)
    assert r.n_compile == pytest.approx(6.0 / 0.05)
    assert r.n_upgrade == pytest.approx(5.0 / 0.03)
    assert r.n_compile_direct == pytest.approx(6.5 / 0.05)
    assert np.isnan(r.n_capture_lo)   # no process-level data


# ---------------------------------------------------------------- B3 interpolation / cost-table helpers
def _cells():
    def c(arm, B, steady, src="measured"):
        d = dict(arm=arm, H=512, W=512, B=B, cfg=0, n_proc=5, steady_ms=steady, steady_ci_lo=steady, steady_ci_hi=steady,
                 capture_pool_mb=10.0 * B, block=f"b{B}", source=src, fc_source="measured", equiv_pass=np.nan,
                 **{f"resid_s__{t}": np.nan for t in TIERS})
        d["resid_s__cold_diskcache"] = float(B)
        return d
    return [c("C1", 2, 100.0), c("C1", 4, 200.0), c("E0", 2, 90.0), c("E0", 3, 160.0), c("E0", 4, 200.0)]


def test_b3_interpolation_and_check():
    ct, chk = bct.interpolate_b3(pd.DataFrame(_cells()))
    c1b3 = ct[(ct.arm == "C1") & (ct.B == 3)].iloc[0]
    assert c1b3.steady_ms == pytest.approx(150.0)
    assert c1b3.source == "interpolated_B3"
    assert c1b3.capture_pool_mb == pytest.approx(30.0)
    assert c1b3.resid_s__cold_diskcache == pytest.approx(3.0)
    assert np.isnan(c1b3.resid_s__cold_empty)                 # NaN in either neighbour stays NaN
    e0b3 = ct[(ct.arm == "E0") & (ct.B == 3)]
    assert len(e0b3) == 1 and e0b3.iloc[0].source == "measured" and e0b3.iloc[0].steady_ms == 160.0   # measured B3 kept
    assert len(chk) == 1 and chk.iloc[0].steady_ms_interp == pytest.approx(145.0)
    assert chk.iloc[0].rel_err == pytest.approx((145.0 - 160.0) / 160.0)


def _raw(arm, shape, tier, pid, seq_idx, first_req_s, fc, steady=100.0, n=6):
    d = dict(arm=arm, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, proc_id=pid, cold_tier=tier, status="ok",
             rep=np.arange(n), wall_ms=steady, gpu_ms=steady, first_req_s=first_req_s, contended=0, load1_start=1.0,
             load1_end=1.0, row_kind="steady", seq_idx=seq_idx, block="t", tier_violation=0, pcore_pinned=1,
             recompile_limit_hit=0, capture_pool_mb=np.nan, static_io_mb=np.nan, steady_peak_reserved_mb=3000.0,
             weights_alloc_mb=2000.0, resident_mb=2500.0, capture_s=np.nan)
    for k, v in zip(("first_call_te_s", "first_call_unet_s", "first_call_vae_s"), fc):
        d[k] = v
    return pd.DataFrame(d)


def test_per_process_intrinsic_components_in_a_sequence():
    """seq: S1 (new) then S2 (UNet shared with S1): S2's UNet first call must not enter the intrinsic cost table."""
    raw = pd.concat([_raw("C1", S1, "cold_diskcache", 0, 0, 14.1, (1.0, 10.0, 2.0)),
                     _raw("C1", S2, "inproc_diskcache", 0, 1, 4.5, (1.5, 0.05, 2.5), steady=150.0)], ignore_index=True)
    pp = bct.per_process(raw, {})
    p1 = pp[(pp.B == 1)].iloc[0]
    p2 = pp[(pp.B == 2)].iloc[0]
    assert p1.tier == "cold_diskcache" and p2.tier == "inproc_diskcache"
    assert p1.first_call_unet_s == 10.0 and np.isnan(p2.first_call_unet_s)
    assert p1.resid_s == pytest.approx(14.1 - 0.1 - 13.0)
    assert p2.resid_s == pytest.approx(max(4.5 - 0.15 - (1.5 + 2.5), 0.0))   # unet NaN counts 0
    assert np.isfinite(p1.x_s) and np.isnan(p2.x_s)                          # shared -> no intrinsic x
    cells = bct.cells_from_procs(pp)
    c2 = cells[cells.B == 2].iloc[0]
    assert np.isnan(c2["fc_unet_s__inproc_diskcache"]) and c2["fc_te_s__inproc_diskcache"] == 1.5


def test_load_dir_maps_inproc_first_by_meta_cache_mode(tmp_path):
    d = _raw("C1", S2, "inproc_first", 0, 1, 4.5, (1.5, 0.05, 2.5))
    d.to_csv(tmp_path / "C1_seq1_x_p0.csv", index=False)
    (tmp_path / "C1_seq_p0.meta.json").write_text(json.dumps({"cache_mode": "empty"}))
    out = bct.load_dir(tmp_path, "blk", seq=True)
    assert set(out.cold_tier) == {"inproc_empty"} and set(out.block) == {"blk"}


def test_cache_independent_arms_fill_both_cold_tiers():
    raw = pd.concat([_raw("E0", A, "na", p, -1, 0.6, (np.nan,) * 3) for p in range(3)], ignore_index=True)
    cells = bct.cells_from_procs(bct.per_process(raw, {}))
    r = cells.iloc[0]
    assert r["resid_s__cold_empty"] == pytest.approx(0.5) == pytest.approx(r["resid_s__cold_diskcache"])
    assert np.isnan(r["resid_s__inproc_diskcache"])


# ---------------------------------------------------------------- serve_trace replay loop (mock runner, CPU)
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MockRunner:
    """prepare / run advance a fake clock by tabulated amounts; records the calls (key = plain Shape here)."""

    def __init__(self, clock, prep_s, run_s):
        self.clock, self.prep_s, self.run_s = clock, prep_s, run_s
        self.calls = []

    def prepare(self, key, inp):
        self.calls.append(("prepare", key))
        self.clock.t += self.prep_s
        return {"capture_pool_mb": 100.0, "capture_s": 0.1}

    def run(self, inp, key, marks=None):
        self.calls.append(("run", key))
        self.clock.t += self.run_s


def _serve(policy_name, shapes, prep_s=1.0, run_s=0.1, input_s=0.01):
    clk = FakeClock()
    runner = MockRunner(clk, prep_s, run_s)

    def make_inputs(req, shape):
        clk.t += input_s
        return object()

    rows, status, err = st_.serve(reqs(shapes), runner, make_policy(policy_name), make_inputs, clk, lambda: None,
                                  lambda: (4096.0, 2048.0))
    return rows, status, runner, clk


def test_serve_trace_event_sequences_and_timing():
    rows, status, runner, clk = _serve("B4", [A, B_, A, B_, A])
    assert status == "ok"
    assert [r["event"] for r in rows] == ["capture", "capture", "steady", "steady", "steady"]
    assert [c for c in runner.calls if c[0] == "prepare"] == [("prepare", A), ("prepare", B_)]
    assert rows[0]["input_ms"] == pytest.approx(10.0) and rows[0]["prepare_ms"] == pytest.approx(1000.0)
    assert rows[0]["run_ms"] == pytest.approx(100.0) and rows[0]["dur_ms"] == pytest.approx(1110.0)
    assert rows[2]["prepare_ms"] == 0.0 and rows[2]["dur_ms"] == pytest.approx(110.0)
    assert rows[1]["graph_mem_mb"] == 200.0
    m = st_.summarize_times(rows, load_s=5.0)
    assert m["service_s"] == pytest.approx(2 * 1.11 + 3 * 0.11)
    assert m["service_s_excl_input"] == pytest.approx(2 * 1.1 + 3 * 0.1)
    assert m["total_incl_load_s"] == pytest.approx(5.0 + m["service_s"])
    assert m["event_counts"] == {"capture": 2, "steady": 3}


def test_serve_trace_b0_never_prepares_and_b1_prepares_each_new_shape():
    rows0, _, r0, _ = _serve("B0", [A, A, B_])
    assert [r["event"] for r in rows0] == ["eager_first", "steady", "eager_first"]
    assert not [c for c in r0.calls if c[0] == "prepare"]
    rows1, _, r1, _ = _serve("B1", [S1, S2, S2, S1])
    assert [r["event"] for r in rows1] == ["compile", "compile", "steady", "steady"]
    assert [c[1] for c in r1.calls if c[0] == "prepare"] == [S1, S2]


@pytest.mark.parametrize("pol", ["B0", "B4", "B1"])
def test_serve_events_equal_simulator_events(pol):
    shapes = [A, S1, S2, A, S2, B_, S1, B_]
    rows, _, _, _ = _serve(pol, shapes)
    arm_cost = {"B0": "E0", "B4": "G1", "B1": "C1"}[pol]
    rs = []
    for s in set(shapes):
        rs.append(row(arm_cost, s, 100.0, resid={t: 0.5 for t in TIERS}, pool=50.0,
                      fc={t: dict(te=1, unet=2, vae=1) for t in TIERS}))
    sim = simulate(reqs(shapes), make_policy(pol), table(rs))
    assert [r["event"] for r in rows] == [e["event"] for e in sim.events]


def test_serve_failure_is_recorded_and_stops():
    clk = FakeClock()

    class Boom(MockRunner):
        def prepare(self, key, inp):
            raise RuntimeError("capture failed")
    runner = Boom(clk, 1, 0.1)
    rows, status, err = st_.serve(reqs([A, B_]), runner, make_policy("B4"), lambda r, s: object(), clk, lambda: None,
                                  lambda: (0.0, 0.0))
    assert status == "fail" and "capture failed" in err and len(rows) == 1 and rows[0]["status"] == "fail"


def test_serve_trace_and_sim_import_without_torch():
    """Planning code (policy, simulator, break-even, cost-table builder, serve loop) must not need torch / a GPU."""
    import pathlib
    import subprocess
    import sys
    code = ("import sys, scripts.serve_trace, scripts.build_cost_table, p24.sim.breakeven, p24.sim.simulator; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=pathlib.Path(__file__).resolve().parents[1],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-500:]


def test_cache_independent_inproc_any_label_fills_inproc_tiers_not_cold():
    raw = pd.concat([_raw("E0", A, "na", p, -1, 0.6, (np.nan,) * 3) for p in range(2)] +
                    [_raw("E0", B_, "inproc_env", 0, 1, 0.2, (np.nan,) * 3, steady=100.0)], ignore_index=True)
    cells = bct.cells_from_procs(bct.per_process(raw, {}))
    a = cells[cells.H == 512].iloc[0]
    b = cells[cells.H == 768].iloc[0]
    assert np.isnan(b["resid_s__cold_diskcache"]) and np.isnan(b["resid_s__cold_empty"])    # not a cold-tier value
    assert b["resid_s__inproc_empty"] == pytest.approx(0.1) == pytest.approx(b["resid_s__inproc_diskcache"])
    assert np.isnan(a["resid_s__inproc_empty"])


def test_extrapolate_ratio_to_c1():
    """Steady: median steady ratio. One-time (fix 2026-11-16 01:35): per tier, median ratio of MEASURED first-request
    times arm / C1 -- NOT the residual ratio (residuals can differ wildly when the arm's cost sits in other components)."""
    def c(arm, B, steady, resid, first):
        d = dict(arm=arm, H=512, W=512, B=B, cfg=0, n_proc=5, steady_ms=steady, steady_ci_lo=steady, steady_ci_hi=steady,
                 block="b", source="measured", fc_source="measured", equiv_pass=np.nan,
                 **{f"resid_s__{t}": np.nan for t in TIERS}, **{f"first_req_s__{t}": np.nan for t in TIERS})
        d["resid_s__cold_diskcache"], d["first_req_s__cold_diskcache"] = resid, first
        return d
    # C2 measured at B1: first request 1.1x C1, but its residual is 10x C1's (the old rule would scale by 10)
    ct = pd.DataFrame([c("C1", 1, 100.0, 0.1, 10.0), c("C1", 4, 200.0, 0.2, 20.0), c("C2", 1, 110.0, 1.0, 11.0)])
    out = bct.extrapolate(ct, ["C2"])
    new = out[(out.arm == "C2") & (out.B == 4)].iloc[0]
    assert new.source == "extrapolated_ratio_C1" and new.steady_ms == pytest.approx(220.0)
    assert new.first_req_s__cold_diskcache == pytest.approx(22.0)
    assert new.resid_s__cold_diskcache == pytest.approx(0.22)
    assert np.isnan(new.first_req_s__cold_empty)                 # no shared measurement in that tier -> no value
    assert len(out[(out.arm == "C2") & (out.B == 1)]) == 1       # the measured row is untouched


def test_build_cost_table_main_end_to_end_and_no_overwrite(tmp_path):
    raw = tmp_path / "blk"
    raw.mkdir()
    for arm, steady in (("E0", 100.0), ("G1", 80.0)):
        for p in range(3):
            d = _raw(arm, A, "cold_diskcache", p, -1, 0.6 if arm == "E0" else 1.1, (np.nan,) * 3, steady=steady)
            d.to_csv(raw / f"{arm}_512x512_b1_nocfg_p{p}.csv", index=False)
    out = tmp_path / "out"
    bct.main(["--raw", str(raw), "--out-dir", str(out)])
    ct = pd.read_csv(out / "cost_table.csv")
    assert set(ct.arm) == {"E0", "G1"} and (out / "cost_table_procs.csv").exists() and (out / "build_meta.json").exists()
    assert ct[ct.arm == "G1"].iloc[0]["resid_s__cold_empty"] == pytest.approx(1.1 - 0.08)
    with pytest.raises(SystemExit):
        bct.main(["--raw", str(raw), "--out-dir", str(out)])      # never overwrites


def test_serve_trace_budget_denies_with_predicted_pool_and_serves_eagerly():
    """S2 2.6 B4 with a graph budget: admission uses the predicted pool (same as the simulator); denied -> eager_run."""
    clk = FakeClock()
    runner = MockRunner(clk, 1.0, 0.1)
    eager_calls = []

    def make_inputs(req, shape):
        return object()

    pred = {A: 150.0, B_: 150.0, S1: 40.0}
    rows, status, _ = st_.serve(reqs([A, B_, S1, B_, A]), runner, make_policy("B4"), make_inputs, clk, lambda: None,
                                lambda: (0.0, 0.0), mem_budget_mb=200.0, graph_mb_fn=lambda s: pred[s],
                                eager_run=lambda inp, key: eager_calls.append(key))
    assert status == "ok"
    # A admitted (150 <= 200; measured pool 100 registered), B_ denied (100 + 150 > 200), S1 admitted (100 + 40)
    assert [r["event"] for r in rows] == ["capture", "capture_denied", "capture", "capture_denied", "steady"]
    assert eager_calls == [B_, B_]
    assert [c[1] for c in runner.calls if c[0] == "prepare"] == [A, S1]


def test_component_cost_looked_up_by_component_key_before_cold_fallback():
    """fc measured on the shape that introduced a component key is reused for another shape with the same key."""
    S1u = Shape(512, 512, 1, 1)  # UNet batch 2 at 512^2, shared with S2 = 512x512 B2 no-CFG
    ct = table([row("C1", A, 100.0, resid={"cold_diskcache": 0.1, "inproc_diskcache": 0.1},
                    fc={"cold_diskcache": dict(te=1, unet=5, vae=1)}),
                row("C1", S1u, 100.0, resid={"inproc_diskcache": 0.1},
                    fc={"inproc_diskcache": dict(te=0.5, unet=2, vae=0.3), "cold_diskcache": dict(te=1, unet=5, vae=1)}),
                row("C1", S2, 100.0, resid={"inproc_diskcache": 0.1},
                    fc={"inproc_diskcache": dict(te=0.6, unet=float("nan"), vae=0.4),
                        "cold_diskcache": dict(te=1, unet=5, vae=1)})])
    r = simulate(reqs([A, S2]), make_policy("B1"), ct)
    # S2 as 2nd shape: te (B2) new 0.6, unet key shared with S1u -> 2 (not the cold 5), vae new 0.4
    assert r.events[1]["fallbacks"] == "fc_unet:shared_key"
    assert r.summary["service_s"] == pytest.approx((0.1 + 7 + 0.1) + (0.1 + 0.6 + 2 + 0.4 + 0.1), abs=1e-9)
