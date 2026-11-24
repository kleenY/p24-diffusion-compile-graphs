"""S4-P P1: Ours v3 (cost-aware admission, p24/policy/ours_v3.py) on the request-level simulator. CPU only, no torch.
Hand-computed toy cost tables (hot disk cache -> compile at first sight; empty cache -> watch), key-sharing credit, alpha_c1, v2 equivalence
with the switches off, no access to the true N / future requests; real-table checks run only where the frozen S2/S3 inputs exist."""
import inspect
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from p24.policy import Action, CostView, ProcState, Shape, make_policy, COMPILE, EAGER
from p24.policy import ours_v3 as v3mod
from p24.policy.factory_v3 import make_policy as make_policy_v3
from p24.policy.ours_v3 import OursV3
from p24.sim.costs import TIERS, CostTable
from p24.sim.simulator import SimOptions, simulate

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
BUD_PATH = ROOT / "results/S3/budgets.json"
DEV = ROOT / "results/S3/workload/dev_traces"
PARAMS_S3 = ROOT / "results/S3/policy_params.json"
real = pytest.mark.skipif(not (CT_PATH.exists() and BUD_PATH.exists() and DEV.exists() and PARAMS_S3.exists()), reason="frozen S2/S3 inputs not present")

A = Shape(512, 512, 1, 0)      # te key (te,1)  unet key (unet,512,512,1)  vae key (vae,512,512,1)
D = Shape(512, 512, 1, 1)      # te key (te,1)  unet key (unet,512,512,2)  vae key (vae,512,512,1)  -> shares te + vae with A
F = Shape(768, 768, 1, 0)      # shares only the te key with A


def mk_row(arm, s, steady_ms, resid_disk, resid_empty=None, pool=np.nan, fc=None):
    """One cost-table row. fc = (te, unet, vae) first-call seconds, same at every tier (None -> fc_source 'missing': everything in resid)."""
    r = dict(arm=arm, H=s.H, W=s.W, B=s.B, cfg=s.cfg, steady_ms=steady_ms, capture_pool_mb=pool, capture_s=0.01, weights_alloc_mb=2000.0,
             steady_peak_reserved_mb=2500.0, transient_excess_s=np.nan, source="measured", fc_source="missing" if fc is None else "measured")
    for t in TIERS:
        r[f"resid_s__{t}"] = resid_empty if (t.endswith("empty") and resid_empty is not None) else resid_disk
        for c, v in zip(("te", "unet", "vae"), fc or (np.nan, np.nan, np.nan)):
            r[f"fc_{c}_s__{t}"] = v
    return r


def toy_table(shapes=(A,), resid_disk=0.4, resid_empty=4.0, fc=None, all_arms=False):
    """E0 100 ms (one-time 0.1 s); C1 80 ms -> saving 20 ms/request.  C1 one-time: disk-cache tiers resid_disk, empty tiers resid_empty
    (+ component first calls fc, if given).  all_arms also adds G1 / C1+G1 rows (needed by the four-tier ladder)."""
    rows = []
    for s in shapes:
        rows += [mk_row("E0", s, 100.0, 0.1), mk_row("C1", s, 80.0, resid_disk, resid_empty, fc=fc)]
        if all_arms:
            rows += [mk_row("G1", s, 90.0, 0.2, pool=400.0), mk_row("C1+G1", s, 70.0, resid_disk + 0.5, None if resid_empty is None else resid_empty + 0.5,
                                                               pool=600.0, fc=fc)]
    return CostTable(pd.DataFrame(rows))


def reqs(shapes):
    return [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg, noise_seed=i, prompt="p") for i, s in enumerate(shapes)]


TWO = ("E", "C1")      # two-tier ladder (E0 and C1 rows only)


def df_of(r):
    d = r.frame()
    return d[d.idx >= 0].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ payback gate (hand computed)
def test_hot_cache_first_sight_compile_empty_cache_watches():
    """E0 100 ms, C1 80 ms -> saving 0.020 s/request.  Disk-cache one-time 0.4 s -> n_star = 0.4/0.02 = 20 <= K_pay=50: compile at the FIRST request.
    Empty cache one-time 4.0 s -> n_star = 200 > 50: no first-sight compile; v2 watches (ski-rental alpha=1: saved 0.02 s x requests >= 4.0 s ->
    the compile starts after ~200 eager requests)."""
    tt = toy_table()
    pol = make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=50.0, tiers=TWO))
    st = ProcState()
    assert pol.payback_requests(A, st) == pytest.approx(20.0)
    assert pol.decide(reqs([A])[0], st) == Action(COMPILE)
    d = df_of(simulate(reqs([A] * 300), make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=50.0, tiers=TWO)), tt,
                       SimOptions(cache_mode="warm")))
    assert d.iloc[0].action == "compile" and d.iloc[0].event == "compile" and (d.iloc[1:].event == "steady").all()
    # empty cache
    pe = make_policy_v3("Ours-v3", cost=CostView(tt, "empty"), params=dict(K_pay=50.0, tiers=TWO))
    assert pe.payback_requests(A, ProcState()) == pytest.approx(200.0)
    assert pe.decide(reqs([A])[0], ProcState()).kind == EAGER
    de = df_of(simulate(reqs([A] * 400), make_policy_v3("Ours-v3", cost=CostView(tt, "empty"), params=dict(K_pay=50.0, tiers=TWO)), tt,
                        SimOptions(cache_mode="empty")))
    first_compile = int(de[de.event == "compile"].idx.min())
    assert 199 <= first_compile <= 201, first_compile
    assert (de.iloc[:first_compile].action == "eager").all()


def test_threshold_is_inclusive_and_k_zero_is_off():
    tt = toy_table()                                    # n_star(warm) = 20
    cv = CostView(tt, "warm")
    st = ProcState()
    assert make_policy_v3("Ours-v3", cost=cv, params=dict(K_pay=20.0 + 1e-9, tiers=TWO)).decide(reqs([A])[0], st).kind == COMPILE
    assert make_policy_v3("Ours-v3", cost=cv, params=dict(K_pay=19.0, tiers=TWO)).decide(reqs([A])[0], st).kind == EAGER
    assert make_policy_v3("Ours-v3", cost=cv, params=dict(K_pay=0.0, tiers=TWO)).decide(reqs([A])[0], st).kind == EAGER


def test_key_sharing_credit_enters_the_payback_count():
    """Component first calls te/unet/vae 1.0 s each, shape resid 0.1 s, saving 0.02 s -> cold A: U = 3.1 s, n_star = 155.
    After A is compiled by this process, D (A with CFG on) shares the te and vae keys and only pays its new unet key: U = 0.1 + 1.0 = 1.1 s,
    n_star = 55; F shares only the te key: U = 2.1 s, n_star = 105.  K_pay = 60: A and F are not admitted at first sight, D is (once A is compiled)."""
    tt = toy_table((A, D, F), resid_disk=0.1, resid_empty=0.1, fc=(1.0, 1.0, 1.0))
    pol = make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=60.0, tiers=TWO))
    st = ProcState()
    assert pol.payback_requests(A, st) == pytest.approx(155.0)
    assert pol.payback_requests(D, st) == pytest.approx(155.0)
    st.register(A, "compile")                              # A compiled by this process -> its component keys are credited
    assert pol.payback_requests(D, st) == pytest.approx(55.0)
    assert pol.payback_requests(F, st) == pytest.approx(105.0)
    assert pol.decide(reqs([D])[0], st) == Action(COMPILE)
    assert pol.decide(reqs([F])[0], st).kind == EAGER
    # key_decomp ablation: no credit -> D is not cheap
    pol_n = make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=60.0, tiers=TWO, key_decomp=False))
    assert pol_n.payback_requests(D, st) == pytest.approx(155.0)


def test_gate_sequence_in_simulator_toy():
    """A is requested 160 times (ski-rental alpha=1, K_pay=60: A watches until its saved seconds 0.02 x count reach U=3.1 s -> compile at about the
    156th request), then D appears for the first time: n_star(D | A compiled) = 55 <= 60 -> compiled at its first request.  With K_pay=0 (v2)
    the new shape D starts eager."""
    tt = toy_table((A, D), resid_disk=0.1, resid_empty=0.1, fc=(1.0, 1.0, 1.0))
    trace = reqs([A] * 160 + [D] * 5)
    on = df_of(simulate(trace, make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=60.0, tiers=TWO)), tt, SimOptions()))
    off = df_of(simulate(trace, make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(K_pay=0.0, tiers=TWO)), tt, SimOptions()))
    assert 150 <= int(on[(on.action == "compile") & (on.H == 512) & (on.cfg == 0)].idx.min()) <= 158
    assert on.iloc[160].cfg == 1 and on.iloc[160].action == "compile" and on.iloc[160].event == "compile"
    assert off.iloc[160].action == "eager" and off.iloc[160].event == "eager_first"


def test_alpha_c1_hand_computed():
    """Empty cache, U = 4.0 s, saving 0.02 s/request: ski-rental threshold alpha x U / 0.02 = 200 x alpha requests.  alpha_c1 = 0.5 -> compile
    after ~100 eager requests; alpha_c1 = None / 1.0 -> after ~200."""
    tt = toy_table()
    res = {}
    for a_c1 in (None, 1.0, 0.5, 0.25):
        d = df_of(simulate(reqs([A] * 400), make_policy_v3("Ours-v3", cost=CostView(tt, "empty"), params=dict(alpha_c1=a_c1, tiers=TWO)), tt,
                           SimOptions(cache_mode="empty")))
        res[a_c1] = int(d[d.event == "compile"].idx.min())
    assert 199 <= res[None] <= 201 and res[None] == res[1.0]
    assert 99 <= res[0.5] <= 101 and 49 <= res[0.25] <= 51, res


# ------------------------------------------------------------------------------------------------ affordability regime (second pass, hand computed)
def _first(d, event="compile"):
    x = d[d.event == event]
    return int(x.idx.min()) if len(x) else None


def test_affordability_regime_posterior_admission_hand_computed():
    """E0 100 ms (W grows 0.1 s per eager request), C1 80 ms (saving 0.02 s), C1 one-time 1.0 s (disk) / 10.0 s (empty), prior(A)=0.01, H_prior=1000.
    v2: prior admission n_hat x saving = 0.01 x 1000 x 0.02 = 0.2 s < U -> none; ski-rental compiles after U/0.02 = 50 (disk) / 500 (empty) requests.
    v3 afford_rho = 0.5: the regime starts when U <= 0.5 x 0.1 x t -> t = 20 (disk) / t = 200 (empty); then the POSTERIOR frequency of a one-shape stream
    p_hat = (t + 0.5)/(t + 50) gives net = p_hat x 1000 x 0.02 - U > 0 (disk t >= 3; empty at t = 200: 0.802 x 20 = 16.0 > 10) -> compile at t = 20 / 200."""
    prior = {A: 0.01}
    for cm, resid_e, v2_at, v3_at in (("warm", None, 50, 20), ("empty", 10.0, 500, 200)):
        tt = toy_table(resid_disk=1.0, resid_empty=resid_e)
        d2 = df_of(simulate(reqs([A] * 800), make_policy_v3("Ours-v3", cost=CostView(tt, cm), params=dict(tiers=TWO), prior=prior), tt, SimOptions(cache_mode=cm)))
        d3 = df_of(simulate(reqs([A] * 800), make_policy_v3("Ours-v3", cost=CostView(tt, cm), params=dict(tiers=TWO, afford_rho=0.5), prior=prior), tt,
                            SimOptions(cache_mode=cm)))
        assert v2_at - 2 <= _first(d2) <= v2_at + 2, (cm, _first(d2))
        assert v3_at <= _first(d3) <= v3_at + 1, (cm, _first(d3))


def test_affordability_regime_short_stream_never_pays_a_large_premium():
    """One-time cost 6 s, 100 eager requests of one shape = 10 s of work: 6 <= 0.5 x W needs W >= 12 s (t >= 120) -> no compile within 100 requests
    (and no ski-rental either: 0.02 x 100 = 2 s < 6 s).  A longer stream compiles from t = 120."""
    tt = toy_table(resid_disk=6.0)
    prior = {A: 0.01}
    p = lambda: make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(tiers=TWO, afford_rho=0.5), prior=prior)
    assert (df_of(simulate(reqs([A] * 100), p(), tt, SimOptions())).event != "compile").all()
    c = _first(df_of(simulate(reqs([A] * 400), p(), tt, SimOptions())))
    assert 120 <= c <= 121, c


def test_affordability_regime_alpha_c1_hand_computed():
    """No prior (online path only), empty cache U = 10 s: v2 compiles after 500 requests.  afford_rho = 0.5 -> regime from t = 200; alpha_c1 = 0.5
    -> threshold 0.5 x 10 = 5 s = 250 requests (> 200, so the regime is already on): compile at ~250.  With afford_rho tiny the regime never starts -> v2."""
    tt = toy_table(resid_disk=1.0, resid_empty=10.0)
    mk = lambda **kw: make_policy_v3("Ours-v3", cost=CostView(tt, "empty"), params=dict(tiers=TWO, **kw))
    d = lambda pol: df_of(simulate(reqs([A] * 800), pol, tt, SimOptions(cache_mode="empty")))
    assert 498 <= _first(d(mk())) <= 502
    assert 248 <= _first(d(mk(afford_rho=0.5, alpha_c1=0.5))) <= 252
    assert 498 <= _first(d(mk(afford_rho=1e-9, alpha_c1=0.5))) <= 502
    pol = mk(afford_rho=0.5)
    simulate(reqs([A] * 50), pol, tt, SimOptions(cache_mode="empty"))
    assert pol._work == pytest.approx(5.0)               # 50 eager requests x 0.1 s


def test_affordability_work_uses_the_serving_path_steady_time():
    tt = toy_table(resid_disk=0.1)                       # C1 compiled at the first request (n_star = 5 <= K_pay)
    pol = make_policy_v3("Ours-v3", cost=CostView(tt, "warm"), params=dict(tiers=TWO, afford_rho=0.5, K_pay=10.0))
    simulate(reqs([A] * 10), pol, tt, SimOptions())
    assert pol._work == pytest.approx(10 * 0.080)        # every request served by C1 (80 ms)


# ------------------------------------------------------------------------------------------------ equivalence with v2 when the switches are off
@pytest.mark.parametrize("params", [dict(), dict(K_pay=0.0, alpha_c1=None, afford_rho=None), dict(alpha=0.5, H_prior=2000.0),
                                     dict(afford_rho=1e-12, alpha_c1=0.1)])   # regime never reached == v2
def test_switches_off_equal_v2_toy(params):
    shapes = [A, D, F, A, A, D, F, F, A, D] * 40
    tt = toy_table((A, D, F), resid_disk=0.4, resid_empty=4.0, fc=(0.2, 0.2, 0.2), all_arms=True)
    for cm in ("warm", "empty"):
        r2 = simulate(reqs(shapes), make_policy("Ours", cost=CostView(tt, cm), params={k: v for k, v in params.items() if k not in v3mod.V3_DEFAULTS}), tt,
                      SimOptions(cache_mode=cm))
        r3 = simulate(reqs(shapes), OursV3(CostView(tt, cm), params), tt, SimOptions(cache_mode=cm))
        cols = ["idx", "action", "event", "dur_s", "evicted"]
        pd.testing.assert_frame_equal(r2.frame()[cols], r3.frame()[cols])
        assert r2.summary["service_s"] == r3.summary["service_s"]


def test_factory_v3_superset_and_unknown_param_rejected():
    tt = toy_table(all_arms=True)
    cv = CostView(tt)
    assert isinstance(make_policy_v3("Ours-v3", cost=cv), OursV3)
    assert make_policy_v3("B1").name == "B1" and make_policy_v3("B0").name == "B0"
    assert type(make_policy_v3("Ours", cost=cv)).__name__ == "Ours"
    with pytest.raises(ValueError):
        make_policy_v3("Ours-v3")
    with pytest.raises(KeyError):
        OursV3(cv, dict(not_a_param=1))
    with pytest.raises(AssertionError):
        OursV3(cv, dict(K_pay=-1.0))
    with pytest.raises(AssertionError):
        OursV3(cv, dict(afford_rho=0.0))


# ------------------------------------------------------------------------------------------------ the policy cannot see the true N
def test_v3_signatures_and_source_do_not_expose_n():
    for fn in (OursV3.__init__, OursV3.decide, OursV3.payback_requests, OursV3._targets, OursV3.affordable, OursV3.on_event):
        for pname in inspect.signature(fn).parameters:
            assert pname.lower() not in ("n", "n_requests", "trace", "requests", "oracle_counts"), pname
    assert "N" not in v3mod.V3_DEFAULTS and "n_requests" not in v3mod.V3_DEFAULTS
    src = inspect.getsource(v3mod)
    for bad in ("len(requests)", "n_requests", "/proc", "cuda", "perf_counter", "isinstance(state, Sim", "oracle_counts", "read_trace"):
        assert bad not in src, bad


def test_toy_decisions_independent_of_future_requests():
    """Same prefix, different total length (a 'true N' proxy) -> identical (action, event) on the prefix, gate and alpha_c1 active."""
    tt = toy_table((A, D, F), resid_disk=0.1, resid_empty=2.0, fc=(0.5, 0.5, 0.5), all_arms=True)
    full = [A, D, A, F, A, A, D, F] * 60
    for cm in ("warm", "empty"):
        outs = []
        for n in (120, 480):
            r = simulate(reqs(full[:n]), OursV3(CostView(tt, cm), dict(K_pay=40.0, alpha_c1=0.5)), tt, SimOptions(cache_mode=cm))
            outs.append(df_of(r).iloc[:120][["idx", "action", "event"]].reset_index(drop=True))
        pd.testing.assert_frame_equal(outs[0], outs[1])


# ------------------------------------------------------------------------------------------------ real (frozen) inputs
def _real():
    from p24.sim.prior import prior_A
    from p24.sim.simulator import read_trace
    ct = CostTable.from_csv(CT_PATH)
    bud = json.loads(BUD_PATH.read_text())["tiers"]
    pv2 = json.loads(PARAMS_S3.read_text())["ours"]["params"]
    pv2 = dict(pv2, tiers=tuple(pv2["tiers"]))
    return ct, bud, prior_A(), read_trace, pv2


@real
@pytest.mark.parametrize("w,n,s", [("W2", 1000, 0), ("W5", 10000, 1), ("W1", 10000, 2)])
def test_v3_switches_off_reproduces_v2_on_real_dev_trace(w, n, s):
    ct, bud, prior, read_trace, pv2 = _real()
    full = read_trace(DEV / f"{w}_N{n}_seed{s}_A.jsonl")
    for cm in ("warm", "empty"):
        opt = SimOptions(cache_mode=cm, mem_budget_mb=bud["default"]["mb"], overhead_aware=True)
        a = simulate(full, make_policy("Ours", cost=CostView(ct, cm), params=pv2, prior=prior), ct, opt)
        b = simulate(full, OursV3(CostView(ct, cm), dict(pv2), prior), ct, opt)
        pd.testing.assert_frame_equal(a.frame()[["idx", "action", "event", "dur_s"]], b.frame()[["idx", "action", "event", "dur_s"]])
        assert a.summary["service_s"] == b.summary["service_s"]


@real
@pytest.mark.parametrize("k,a1,rho", [(30.0, None, None), (100.0, 0.5, None), (0.0, 0.5, 0.5), (0.0, None, 0.2)])
def test_v3_decisions_independent_of_future_requests_real(k, a1, rho):
    ct, bud, prior, read_trace, pv2 = _real()
    full = read_trace(DEV / "W2_N10000_seed0_A.jsonl")
    for pre in (300, 1500):
        outs = []
        for n in (pre, 4000):
            r = simulate(full[:n], OursV3(CostView(ct), dict(pv2, K_pay=k, alpha_c1=a1, afford_rho=rho), prior), ct,
                         SimOptions(mem_budget_mb=bud["tight"]["mb"], overhead_aware=True))
            outs.append(r.frame().pipe(lambda d: d[d.idx < pre])[["idx", "action", "event", "evicted"]].reset_index(drop=True))
        pd.testing.assert_frame_equal(outs[0], outs[1])


@real
def test_v3_gate_is_cache_state_aware_on_real_table():
    """Same shape, same process state: disk-cache payback count is ~10x smaller than the empty-cache one (cost table, cache mode of the CostView)."""
    ct, _, prior, _, pv2 = _real()
    s = Shape(512, 512, 1, 0)
    nd = OursV3(CostView(ct, "warm"), dict(pv2, K_pay=1.0), prior).payback_requests(s, ProcState())
    ne = OursV3(CostView(ct, "empty"), dict(pv2, K_pay=1.0), prior).payback_requests(s, ProcState())
    assert 50 < nd < 400 and ne > 5 * nd, (nd, ne)
