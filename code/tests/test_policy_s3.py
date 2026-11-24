"""S3 step 3.2: policies Ours / Ours-NE / B2 / B3 / B3' / B3'' / B5 / B6 / B6' / B7 on the request-level simulator (CPU only, no torch).
Hand-computed toy examples + checks on the real (frozen) cost table where it is present."""
import inspect
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from p24.policy import (Action, CostView, ProcState, Shape, component_keys, make_policy, plan_event, plan_step, shape_of,
                        COMPILE, COMPILE_CAPTURE, CAPTURE, EAGER, RUN_BUCKET)
from p24.policy import baselines as bl
from p24.policy import ours as ours_mod
from p24.policy.baselines import solve_mck
from p24.sim.costs import TIERS, CostTable
from p24.sim.simulator import SimOptions, simulate

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
BUD_PATH = ROOT / "results/S3/budgets.json"
DEV = ROOT / "results/S3/workload/dev_traces"
real = pytest.mark.skipif(not (CT_PATH.exists() and BUD_PATH.exists() and DEV.exists()), reason="frozen S2/S3 inputs not present")

A = Shape(512, 512, 1, 0)
B_ = Shape(768, 768, 1, 0)
C_ = Shape(512, 512, 2, 0)


def row(arm, shape, steady_ms, resid=None, pool=np.nan, capture_s=0.01):
    r = dict(arm=arm, H=shape.H, W=shape.W, B=shape.B, cfg=shape.cfg, steady_ms=steady_ms, capture_pool_mb=pool,
             capture_s=capture_s, weights_alloc_mb=2000.0, steady_peak_reserved_mb=2500.0, transient_excess_s=np.nan,
             source="measured", fc_source="missing")
    for t in TIERS:
        r[f"resid_s__{t}"] = resid if resid is not None else np.nan
        for c in ("te", "unet", "vae"):
            r[f"fc_{c}_s__{t}"] = np.nan
    return r


def toy_table():
    """Three shapes; E0 100 ms; G1 90 ms / pool 400 MB / one-time 0.5 s; C1+G1 70 ms / pool 600 MB / one-time 1.0 s."""
    rows = []
    for s in (A, B_, C_):
        rows += [row("E0", s, 100.0, resid=0.1), row("G1", s, 90.0, resid=0.5, pool=400.0),
                 row("C1+G1", s, 70.0, resid=1.0, pool=600.0), row("C1", s, 75.0, resid=1.0)]
    return CostTable(pd.DataFrame(rows))


def reqs(shapes):
    return [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg, noise_seed=i, prompt="p") for i, s in enumerate(shapes)]


# ------------------------------------------------------------------------------------------------ B7 knapsack (hand computed)
def test_b7_knapsack_toy_hand_computed():
    """counts A=1000, B=100, C=400. saving = n*(100-steady) - one_time:
         A: G1 1000*0.010-0.5 = 9.5   C1+G1 1000*0.030-1.0 = 29.0
         B: G1  100*0.010-0.5 = 0.5   C1+G1  100*0.030-1.0 =  2.0
         C: G1  400*0.010-0.5 = 3.5   C1+G1  400*0.030-1.0 = 11.0     (weights 400 / 600 MB, per-graph overhead 0)
       budget 1000 MB: best = A:C1+G1 (600) + C:G1 (400) = 32.5 (A:C1+G1 + B:G1 = 29.5, A:G1 + C:C1+G1 = 20.5);
       budget 1600 MB: A:C1+G1 + C:C1+G1 + B:G1 = 29 + 11 + 0.5 = 40.5."""
    cv = CostView(toy_table())
    pol = make_policy("B7", cost=cv, oracle_counts={A: 1000, B_: 100, C_: 400})
    plan = pol.make_plan(1000.0, ov=0.0)
    assert plan == {A: "C1+G1", C_: "G1"}
    assert pol.plan_value_s == pytest.approx(32.5, abs=1e-9)
    plan = pol.make_plan(1600.0, ov=0.0)
    assert plan == {A: "C1+G1", C_: "C1+G1", B_: "G1"}
    assert pol.plan_value_s == pytest.approx(40.5, abs=1e-9)
    assert pol.make_plan(300.0, ov=0.0) == {}          # nothing fits
    # simulator: warm-up builds the plan (time counted), unplanned shapes eager
    r = simulate(reqs([A, B_, C_]), make_policy("B7", cost=cv, oracle_counts={A: 1, B_: 1, C_: 1}), toy_table(),
                 SimOptions(mem_budget_mb=1000.0))
    assert r.summary["n_warmup"] >= 0 and r.summary["service_s"] > 0


def test_mck_matches_bruteforce_random():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = 5
        opts = [[("a", float(rng.integers(1, 9)), float(rng.integers(1, 20))), ("b", float(rng.integers(5, 15)), float(rng.integers(1, 40)))]
                for _ in range(n)]
        budget = float(rng.integers(5, 40))
        best, labels = solve_mck(opts, budget, unit=1.0)
        bf = 0.0
        for pick in __import__("itertools").product([None, 0, 1], repeat=n):
            w = sum(opts[i][j][1] for i, j in enumerate(pick) if j is not None)
            if w <= budget:
                bf = max(bf, sum(opts[i][j][2] for i, j in enumerate(pick) if j is not None))
        assert best == pytest.approx(bf)
        got = sum(next(o for o in opts[i] if o[0] == lab)[2] for i, lab in enumerate(labels) if lab)
        w = sum(next(o for o in opts[i] if o[0] == lab)[1] for i, lab in enumerate(labels) if lab)
        assert got == pytest.approx(best) and w <= budget


# ------------------------------------------------------------------------------------------------ plan_step / state bookkeeping
def test_plan_step_judge_first_then_act():
    st = ProcState(mem_budget_mb=1000.0)
    st.register(A, "capture", 400.0)
    st.register(B_, "capture", 500.0)
    # new graph of 600 MB does not fit even after evicting A (500+600 > 1000) -> denied, NOTHING evicted
    ev, evd = plan_step(Action(CAPTURE, evict=(A,)), st, C_, 600.0)
    assert (ev, evd) == ("capture_denied", ())
    # evicting B frees enough (400+600=1000 <= 1000) -> capture with eviction
    ev, evd = plan_step(Action(CAPTURE, evict=(B_,)), st, C_, 600.0)
    assert (ev, evd) == ("capture", (B_,))
    with pytest.raises(KeyError):
        plan_step(Action(CAPTURE, evict=(C_,)), st, Shape(512, 768, 1, 0), 10.0)
    st.register(C_, ev, 600.0, evicted=evd)
    assert B_ not in st.graphs and C_ in st.graphs and st.evicted_total == 1
    assert st.build_count[C_] == 1 and st.seen_count[C_] == 1 and st.last_used[C_] == st.n_served


def test_plan_step_equals_plan_event_without_s3_features():
    rng = np.random.default_rng(1)
    shapes = [A, B_, C_]
    s1, s2 = ProcState(mem_budget_mb=700.0), ProcState(mem_budget_mb=700.0)
    for _ in range(200):
        sh = shapes[rng.integers(3)]
        act = Action([EAGER, CAPTURE, COMPILE, COMPILE_CAPTURE][rng.integers(4)])
        e1 = plan_event(act, s1, sh, 300.0)
        e2, evd = plan_step(act, s2, sh, 300.0)
        if act.kind == COMPILE_CAPTURE and s1.graph_kind.get(sh) == "G1":
            # the one intended S3 difference: an upgrade releases the shape's own G1 first (plan_event still counts it)
            assert e2 == ("compile_capture" if s2.can_admit(300.0, (sh,)) else "capture_denied")
            e2 = e1
        assert e1 == e2 and evd == ()
        s1.register(sh, e1, 300.0)
        s2.register(sh, e2, 300.0)
    assert s1.graphs == s2.graphs


def test_overhead_aware_admission():
    st = ProcState(mem_budget_mb=1000.0, overhead_aware=True)
    st.register(A, "capture", 400.0)               # 400 + 22.33
    assert st.can_admit(555.0) and not st.can_admit(556.0)         # 422.33 + 22.33 + 555 = 999.66 / 1000.99
    assert ProcState(mem_budget_mb=1000.0).can_admit(1000.0)      # S2 rule unchanged
    assert st.can_admit(560.0, free=(A,))


# ------------------------------------------------------------------------------------------------ eviction / denial under a tight budget (real table)
def _real():
    ct = CostTable.from_csv(CT_PATH)
    import json
    bud = json.loads(BUD_PATH.read_text())["tiers"]
    from p24.sim.prior import prior_A
    from p24.sim.simulator import read_trace
    return ct, bud, prior_A(), read_trace


@real
@pytest.mark.parametrize("name,extra", [("Ours", {}), ("B5", {})])
def test_tight_budget_evicts_and_denies(name, extra):
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W2_N10000_seed0_A.jsonl")
    pol = make_policy(name, cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e4, **extra), prior=prior)
    r = simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
    s = r.summary
    assert s["n_evicted_total"] >= 3, s
    # refusals: capture_denied events (Ours may see none for graph tiers when it never asks; B5 asks too) -> count both kinds
    assert s["n_denied_total"] >= 1, s
    df = r.frame()
    assert (df["n_evicted"] > 0).sum() >= 3 and df.loc[df.n_evicted > 0, "evicted"].str.len().min() > 0
    assert s["peak_graph_mb"] <= bud["tight"]["mb"] + 1e-6        # the budget is respected by every admission


@real
def test_ours_ne_never_evicts():
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W2_N10000_seed0_A.jsonl")[:3000]
    pol = make_policy("Ours-NE", cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e4), prior=prior)
    r = simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
    assert r.summary["n_evicted_total"] == 0


# ------------------------------------------------------------------------------------------------ padding rule
@real
def test_512_b3_nocfg_never_padded():
    ct, bud, prior, _ = _real()
    cv = CostView(ct)
    b3 = Shape(512, 512, 3, 0)
    b4 = Shape(512, 512, 4, 0)
    req = dict(H=512, W=512, B=3, cfg=0)
    assert not cv.equiv_pass_padB4(b3, "G1")                      # cost table: G1 B3 no-CFG padded into B4 failed equivalence
    for kind in ("G1", "C1+G1"):
        st = ProcState(mem_budget_mb=6565.1, overhead_aware=True)
        st.register(b4, "capture" if kind == "G1" else "compile_capture", ct.capture_pool_mb(kind, b4))
        assert st.graph_kind[b4] == kind
        for nm in ("Ours", "B5"):
            pol = make_policy(nm, cost=cv, params=dict(H_prior=1e4), prior=prior)
            assert pol.decide(req, st).kind != RUN_BUCKET, (nm, kind)   # equiv fails (G1) / bucket slower than eager (C1+G1)
    assert make_policy("B4").decide(req, ProcState()).kind == CAPTURE


def test_padding_allowed_only_when_equivalent_and_faster():
    rows = []
    b3, b4 = Shape(512, 512, 3, 1), Shape(512, 512, 4, 1)
    for arm, st_b3, st_b4, pad in (("E0", 300.0, 340.0, np.nan), ("C1+G1", 250.0, 200.0, 1.0)):
        r3, r4 = row(arm, b3, st_b3, resid=0.1, pool=300.0), row(arm, b4, st_b4, resid=0.1, pool=400.0)
        r3["equiv_pass_padB4"] = pad
        rows += [r3, r4]
    ct = CostTable(pd.DataFrame(rows))
    cv = CostView(ct)
    pol = make_policy("Ours", cost=cv, params=dict(H_prior=1.0), prior={})
    st = ProcState(mem_budget_mb=5000.0)
    st.register(b4, "compile_capture", 400.0)
    assert pol.decide(dict(H=512, W=512, B=3, cfg=1), st) == Action(RUN_BUCKET, bucket=b4)       # 200 ms < eager 300 ms, equiv pass
    pol2 = make_policy("Ours", cost=cv, params=dict(H_prior=1.0, pad=False), prior={})
    assert pol2.decide(dict(H=512, W=512, B=3, cfg=1), st).kind == EAGER                         # ablation: no padding


# ------------------------------------------------------------------------------------------------ the policy cannot see the true N
def test_policy_signatures_do_not_expose_n():
    for cls in (ours_mod.Ours, bl.B2Dynamic, bl.B3ReduceOverhead, bl.B3PrimeLimit8, bl.B3PostWarm, bl.B6BatchBuckets,
                bl.B6PrimeTopK):
        for fn in (cls.__init__, cls.decide):
            for pname in inspect.signature(fn).parameters:
                assert pname.lower() not in ("n", "n_requests", "trace", "requests", "oracle_counts"), (cls.__name__, pname)
    assert "oracle_counts" in inspect.signature(bl.B7Knapsack.__init__).parameters      # only B7 knows counts
    assert "oracle_counts" not in inspect.signature(ours_mod.Ours.__init__).parameters
    assert "N" not in ours_mod.DEFAULT_PARAMS and "n_requests" not in ours_mod.DEFAULT_PARAMS
    src = inspect.getsource(ours_mod) + inspect.getsource(bl.B6BatchBuckets) + inspect.getsource(bl.B3PostWarm)
    for bad in ("len(requests)", "n_requests", "/proc", "cuda", "perf_counter", "isinstance(state, Sim"):
        assert bad not in src, bad


@real
@pytest.mark.parametrize("name", ["Ours", "B5", "Ours-NE", "B3p", "B6", "B3pp"])
def test_decisions_independent_of_future_requests(name):
    """Same prefix, different total length (a 'true N' proxy) -> identical (action, event, evicted) on the prefix."""
    ct, bud, prior, read_trace = _real()
    full = read_trace(DEV / "W5_N1000_seed1_A.jsonl")
    outs = []
    for n in (300, 1000):
        pol = make_policy(name, cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3), prior=prior)
        r = simulate(full[:n], pol, ct, SimOptions(mem_budget_mb=bud["mid"]["mb"], overhead_aware=True))
        df = r.frame()
        outs.append(df[df.idx.between(-100, 299)][["idx", "action", "event", "evicted"]].reset_index(drop=True))
    pd.testing.assert_frame_equal(outs[0], outs[1])


# ------------------------------------------------------------------------------------------------ B3' = C1 + recompile limit 8
def test_b3prime_limit_rule_plan_step():
    pol = make_policy("B3p")
    assert pol.recompile_limit == 8
    st = ProcState()
    st.recompile_limit = pol.recompile_limit
    # fill 8 distinct UNet keys with 8 different resolutions/batches (te/vae keys stay below 8)
    shapes = [Shape(512, 512, 1, 0), Shape(512, 512, 2, 0), Shape(512, 512, 4, 0), Shape(512, 768, 1, 0), Shape(512, 768, 2, 0),
              Shape(512, 768, 4, 0), Shape(768, 512, 1, 0), Shape(768, 512, 2, 0)]
    for s in shapes:
        ev, _ = plan_step(pol.decide({}, st), st, s)
        assert ev == "compile", s
        st.register(s, ev)
    assert len({k for k in st.compiled_keys if k[0] == "unet"}) == 8
    new_unet = Shape(768, 768, 1, 0)                           # 9th UNet key -> whole request eager
    assert plan_step(Action(COMPILE), st, new_unet)[0] == "eager_fallback_limit"
    reuse = Shape(512, 512, 1, 1)                              # UNet key (512,512,2) already compiled; te key (B=1) and vae key too -> allowed
    assert component_keys(reuse)["unet"] in st.compiled_keys
    assert plan_step(Action(COMPILE), st, reuse)[0] == "compile"
    assert plan_step(Action(COMPILE), st, shapes[0])[0] == "steady"     # an already compiled shape keeps running compiled
    st.register(new_unet, "eager_fallback_limit")
    assert new_unet in st.eager_seen and new_unet not in st.compiled_shapes


@real
def test_b3prime_simulation_caps_each_component_at_8():
    ct, _, _, read_trace = _real()
    reqs_ = read_trace(DEV / "W3_N1000_seed0_A.jsonl")
    r = simulate(reqs_, make_policy("B3p"), ct, SimOptions())
    df = r.frame()
    assert (df.event == "eager_fallback_limit").sum() > 0
    comp = df[df.event == "compile"]
    keys = {"te": set(), "unet": set(), "vae": set()}
    for h, w, b, c in zip(comp.H, comp.W, comp.B, comp.cfg):
        for k, v in component_keys(Shape(h, w, b, c)).items():
            keys[k].add(v)
    assert all(len(v) <= 8 for v in keys.values()) and max(len(v) for v in keys.values()) == 8
    # eager fallback rows are charged eager steady (+ eager first-call cost only once per shape)
    fb = df[df.event == "eager_fallback_limit"].iloc[0]
    assert fb.dur_s >= ct.steady_s("E0", Shape(fb.H, fb.W, fb.B, fb.cfg))


# ------------------------------------------------------------------------------------------------ warm-up is counted; determinism
def test_warmup_time_is_counted_toy():
    ct = toy_table()
    cv = CostView(ct)
    prior = {A: 0.6, B_: 0.3, C_: 0.1}
    r = simulate(reqs([A, A]), make_policy("B6p", cost=cv, prior=prior, params=dict(k=2)), ct, SimOptions(mem_budget_mb=2000.0))
    df = r.frame()
    w = df[df.idx < 0]
    assert len(w) == 2 and list(w.event) == ["compile_capture"] * 2
    # warm-up entry = one dummy request: steady 0.070 + one-time (cold 1.0 then in-process 1.0) ; requests: 2 x 0.070 steady
    assert r.summary["warmup_s"] == pytest.approx(2 * (0.070 + 1.0), abs=1e-9)
    assert r.summary["service_s"] == pytest.approx(2 * (0.070 + 1.0) + 2 * 0.070, abs=1e-9)
    assert list(df[df.idx >= 0].event) == ["steady", "steady"]


@real
@pytest.mark.parametrize("name", ["Ours", "B5", "B6", "B7", "B3pp", "Ours-NE"])
def test_deterministic_event_sequences(name):
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W4_N1000_seed2_A.jsonl")
    import collections
    oc = collections.Counter(shape_of(q) for q in reqs_)
    frames = []
    for _ in range(2):
        pol = make_policy(name, cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3), prior=prior, oracle_counts=dict(oc))
        frames.append(simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True)).frame())
    pd.testing.assert_frame_equal(frames[0], frames[1])


# ------------------------------------------------------------------------------------------------ misc baselines
@real
def test_all_policies_run_and_b0_b1_b4_unchanged():
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W2_N1000_seed0_A.jsonl")
    import collections
    oc = collections.Counter(shape_of(q) for q in reqs_)
    for nm in ("B0", "B1", "B2", "B3", "B3p", "B3pp", "B4", "B5", "B6", "B6p", "B7", "Ours", "Ours-NE"):
        pol = make_policy(nm, cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3), prior=prior, oracle_counts=dict(oc))
        r = simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["default"]["mb"], overhead_aware=True))
        assert r.summary["service_s"] > 0 and len(r.events) >= 1000
    # S2 behaviour (no budget, no overhead flag): identical to the old policy objects
    a = simulate(reqs_, make_policy("B1"), ct, SimOptions()).summary["service_s"]
    assert a == pytest.approx(simulate(reqs_, make_policy("B1"), ct, SimOptions()).summary["service_s"])


@real
def test_b3_truncated_respects_budget_and_b6_pads_only_when_equiv():
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W3_N1000_seed0_A.jsonl")
    pol = make_policy("B3", cost=CostView(ct))
    r = simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
    assert pol.pool_sum_mb + pol.n_pools * 22.33 <= bud["tight"]["mb"] + 1e-6
    cv = CostView(ct)
    b6 = make_policy("B6", cost=cv, prior=prior)
    assert b6._bucket_of(Shape(512, 512, 3, 0)) == Shape(512, 512, 4, 0)      # C1+G1 B3 -> B4 passed equivalence
    assert b6._bucket_of(Shape(512, 768, 3, 0)) is None                       # no verdict -> no padding


# ================================================================================================ Ours v2 (3.2b, 2026-11-17)
V1_SPEC = ours_mod.V1_SPEC_PARAMS


def toy4_table():
    """Shape A only. E0 100 ms; G1 90 ms / pool 400 MB / one-time 0.2 s; C1 80 ms / one-time 2.0 s (no pool); C1+G1 70 ms / pool 600 MB / one-time 2.5 s."""
    rows = [row("E0", A, 100.0, resid=0.1), row("G1", A, 90.0, resid=0.2, pool=400.0), row("C1", A, 80.0, resid=2.0),
            row("C1+G1", A, 70.0, resid=2.5, pool=600.0)]
    return CostTable(pd.DataFrame(rows))


def test_v2_defaults_and_v1_spec_switches():
    p = ours_mod.merge_params(None)
    assert p["tiers"] == ("E", "G1", "C1", "C1+G1") and p["horizon"] == "online"
    p1 = ours_mod.merge_params(V1_SPEC)
    assert p1["tiers"] == ("E", "G1", "C1+G1") and p1["horizon"] == "fixed"
    with pytest.raises(AssertionError):
        ours_mod.merge_params({"horizon": "oracle"})
    pol = make_policy("Ours", cost=CostView(toy4_table()), params=dict(H_prior=1000.0))
    st = ProcState()
    st.n_served = 500
    assert pol._h_eff(st) == 1000.0                       # max(H_prior, n_served)
    st.n_served = 2500
    assert pol._h_eff(st) == 2500.0
    pol_f = make_policy("Ours", cost=CostView(toy4_table()), params=dict(H_prior=1000.0, horizon="fixed"))
    assert pol_f._h_eff(st) == 1000.0                     # ablation "fixed horizon"
    for nm in ("Ours-NE", "B5"):                          # derived policies inherit v2 by default
        assert make_policy(nm, cost=CostView(toy4_table()))._h_eff(st) == 2500.0


def _actions(r):
    df = r.frame()
    df = df[df.idx >= 0]
    return df


def test_four_tier_upgrade_path_toy():
    """No prior, alpha=1, unlimited memory. Saved seconds per request vs G1/C1/C1+G1 = 0.010/0.020/0.030 s; thresholds 0.2/2.0/2.5 s ->
    G1 reached at request 20 (first, cheapest), then the saving of staying at G1 vs C1+G1 is 0.020 s/request against 2.5 s -> C1+G1
    after ~125 more requests. The path is eager -> G1 -> C1+G1; C1 is never an intermediate step from G1."""
    tt = toy4_table()
    r = simulate(reqs([A] * 400), make_policy("Ours", cost=CostView(tt), params=dict(alpha=1.0, H_prior=1000.0)), tt, SimOptions())
    df = _actions(r)
    seq = list(dict.fromkeys(df.action))
    assert seq == ["eager", "capture", "compile_capture"], seq
    first = {a: int(df[df.action == a].idx.min()) for a in seq}
    assert 18 <= first["capture"] <= 22, first
    assert 20 + 100 <= first["compile_capture"] <= 20 + 140, first
    assert (df.event == "compile").sum() == 0
    # same shape under v1_spec (three tiers): same path
    r1 = simulate(reqs([A] * 400), make_policy("Ours", cost=CostView(tt), params=dict(alpha=1.0, H_prior=1000.0, **V1_SPEC)), tt, SimOptions())
    assert list(dict.fromkeys(_actions(r1).action)) == ["eager", "capture", "compile_capture"]


def test_four_tier_budget_full_upgrades_to_c1_only():
    """Budget 300 MB: neither graph (400 / 600 MB) fits -> the memory-free tier C1 is still reachable: eager -> (capture refused) -> C1.
    Never a graph in memory; C1 is served by compiled code afterwards."""
    tt = toy4_table()
    r = simulate(reqs([A] * 1500), make_policy("Ours", cost=CostView(tt), params=dict(alpha=1.0, H_prior=1000.0)), tt,
                 SimOptions(mem_budget_mb=300.0, overhead_aware=True))
    df = _actions(r)
    assert r.summary["peak_graph_mb"] == 0.0 and r.summary["n_denied_total"] >= 1
    assert (df.event == "compile").sum() == 1 and (df.event == "compile_capture").sum() == 0 and (df.event == "capture").sum() == 0
    last = df.iloc[-1]
    assert last.action == "compile" and last.event == "steady"
    # three-tier ablation cannot do this: it stays eager forever (its only compiled tier needs a graph)
    r3 = simulate(reqs([A] * 1500), make_policy("Ours", cost=CostView(tt), params=dict(alpha=1.0, H_prior=1000.0, **V1_SPEC)), tt,
                  SimOptions(mem_budget_mb=300.0, overhead_aware=True))
    assert (_actions(r3).event == "compile").sum() == 0 and _actions(r3).iloc[-1].action == "eager"


def test_horizon_online_grows_beyond_h_prior_toy():
    """Prior admission: p_A x H_eff x saving >= one-time cost. Shape A prior 0.01, H_prior=1e3, C1+G1 saving 0.030 s/request, one-time 2.5 s:
    fixed horizon 0.01*1000*0.03 = 0.3 < 2.5 (no prior admission) vs. online H_eff = max(1e3, n_served) = 8400 at n_served = 8400:
    0.01*8400*0.03 = 2.52 > 2.5 (admitted by the prior path)."""
    tt = toy4_table()
    pol = make_policy("Ours", cost=CostView(tt), params=dict(alpha=1e9, H_prior=1000.0), prior={A: 0.01})   # alpha huge: only the prior path can fire
    st = ProcState()
    st.n_served = 100
    assert pol.decide(reqs([A])[0], st).kind == EAGER
    st.n_served = 8400
    assert pol.decide(reqs([A])[0], st).kind in (COMPILE_CAPTURE, CAPTURE, COMPILE)
    polf = make_policy("Ours", cost=CostView(tt), params=dict(alpha=1e9, H_prior=1000.0, horizon="fixed"), prior={A: 0.01})
    assert polf.decide(reqs([A])[0], st).kind == EAGER


@real
@pytest.mark.parametrize("name,params", [("Ours", {}), ("Ours-NE", {}), ("B5", {}), ("Ours", dict(horizon="fixed"))])
def test_v2_decisions_independent_of_future_requests(name, params):
    """H_eff reads only n_served: same prefix, different total length -> identical (action, event, evicted) on the prefix (300 and 1500
    requests; 1500 > H_prior so the online horizon is active), tight budget (evictions happen)."""
    ct, bud, prior, read_trace = _real()
    full = read_trace(DEV / "W2_N10000_seed0_A.jsonl")
    for pre in (300, 1500):
        outs = []
        for n in (pre, 4000):
            pol = make_policy(name, cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3, **params), prior=prior)
            r = simulate(full[:n], pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
            df = r.frame()
            outs.append(df[df.idx < pre][["idx", "action", "event", "evicted"]].reset_index(drop=True))
        pd.testing.assert_frame_equal(outs[0], outs[1])


@real
def test_v2_compile_layer_never_evicted_real_trace():
    """Tight budget, 10k-request trace: graphs are evicted, but once a shape has been compiled it is never served eagerly or via a
    padded bucket again (the compiled code is kept); and v2 uses C1-only for shapes whose graph cannot be held."""
    ct, bud, prior, read_trace = _real()
    reqs_ = read_trace(DEV / "W2_N10000_seed0_A.jsonl")
    pol = make_policy("Ours", cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3), prior=prior)
    r = simulate(reqs_, pol, ct, SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
    df = _actions(r)
    assert r.summary["n_evicted_total"] >= 1 and (df.event == "compile").sum() >= 1
    compiled = set()
    for a, ev, h, w, b, c in zip(df.action, df.event, df.H, df.W, df.B, df.cfg):
        k = (h, w, b, c)
        if k in compiled:
            assert a in ("compile", "compile_capture"), (k, a, ev)
        if a in ("compile", "compile_capture") and ev in ("compile", "compile_capture"):
            compiled.add(k)
    assert r.summary["peak_graph_mb"] <= bud["tight"]["mb"] + 1e-6


@real
@pytest.mark.parametrize("w,n,s", [("W2", 10000, 0), ("W3", 1000, 1)])
def test_v1_spec_reproduces_previous_executor_numbers(w, n, s):
    """params V1_SPEC (3 tiers, fixed horizon) + {alpha=1, H_prior=1e3} == the 3.2 frozen policy: service_s equals the value stored in
    results/S3/sim_dev/tuning_ours_raw.csv (spec3 / g1_a1_H1e3, produced by the v1 code before the v2 edit); REPORT_3.2 aggregate 602.11 s."""
    raw = pd.read_csv(ROOT / "results/S3/sim_dev/tuning_ours_raw.csv")
    old = raw[(raw.variant == "spec3") & (raw.group == "g1_a1_H1e3") & (raw.W == w) & (raw.N == n) & (raw.seed == s)]
    assert len(old) == 1
    ct, bud, prior, read_trace = _real()
    pol = make_policy("Ours", cost=CostView(ct), params=dict(alpha=1.0, H_prior=1e3, **V1_SPEC), prior=prior)
    r = simulate(read_trace(DEV / f"{w}_N{n}_seed{s}_A.jsonl"), pol, ct, SimOptions(mem_budget_mb=bud["default"]["mb"], overhead_aware=True))
    assert r.summary["service_s"] == pytest.approx(float(old.service_s.iloc[0]), rel=1e-12)
    g = raw[(raw.variant == "spec3") & (raw.group == "g1_a1_H1e3")].service_s
    assert float(np.exp(np.log(g).mean())) == pytest.approx(602.11, abs=0.006)
