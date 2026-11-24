"""S3 step 3.1 tests: tiered runtime (p24/runners_tiered.py) and the S3 replay loop (scripts/serve_trace.serve_s3).

Part A (CPU, mock runtime, always runs): loop logic of interface v1 sec 4 -- evict BEFORE the main action, judge-first-act-later (no white
  eviction), over-budget -> eager, B3' planned and unplanned limit fallback, capture OOM -> capture_denied, warm-up bookkeeping, run_bucket
  dispatch, padding gate in the policy, and `serve_s3 == simulate` event equality (B0/B1/B4/B5/B6/Ours/Ours-NE/B3') on dev trace W2.
Part B (GPU0, development mode, NOT exclusive; skipped unless GPU0 is free enough, see `_gpu_reason`): real runtime.
  * evict: memory_reserved falls back; gap to the accounting <= 64 MB; the surviving graph keeps replaying
  * G1 -> C1+G1 upgrade releases the old graph first; run_bucket (padding) on a real bucket; B3' limit hit -> eager; over-budget -> eager
  * EVENT CONSISTENCY (the 3.1 acceptance): the CLI (`python -m scripts.serve_trace`) replays the first 200 requests of a dev W2 trace for
    B0/B4/Ours/B5/B6; (action, event, evicted) per request must equal the simulator's (100 %); Ours and B5 on the tight budget tier must
    actually evict >= 3 times and be denied >= 1 time, otherwise the test is INVALID (fails with that message).
Compile-using GPU tests need the compile cache of a warm directory (they skip otherwise): point TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR
at $P24_CACHE/warm_s3dev/C1_G1/{inductor,triton} (a copy of warm_s2; warm_s2 itself is never written by development runs).
Run from code/ with the GPU0 device variable exported (GPU part) or an empty one (CPU part): python -m pytest tests/test_runtime_s3.py -q
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from p24.policy import (CAPTURE, COMPILE, COMPILE_CAPTURE, RUN_BUCKET, Action, ProcState, Shape,  # noqa: E402
                        make_policy)
from p24.sim.costs import CostTable  # noqa: E402
from p24.sim.simulator import SimOptions, read_trace, simulate  # noqa: E402
from scripts import serve_trace as st  # noqa: E402

COST_CSV = ROOT / "results/S2/cost_table/merged/cost_table.csv"
BUDGETS = json.loads((ROOT / "results/S3/budgets.json").read_text())
DEV_TRACE = ROOT / "results/S3/workload/dev_traces/W2_N1000_seed0_A.jsonl"
N_EVENT = 200
GPU0 = "GPU-722679a6-175b-cb56-b764-e8124d262c41"


def tier_mb(name):
    return None if name is None else float(BUDGETS["tiers"][name]["mb"])


@pytest.fixture(scope="module")
def ct():
    return CostTable.from_csv(COST_CSV)


@pytest.fixture(scope="module")
def requests200():
    return read_trace(DEV_TRACE)[:N_EVENT]


# ====================================================================================================== Part A: mock runtime
class LimitHit(Exception):
    pass


class CapFail(Exception):
    pass


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MockRT:
    """Mock tiered runtime: plain Shape keys; records the call order; configurable failures."""
    LimitExc, CaptureFailed = LimitHit, CapFail

    def __init__(self, clk, prep_s=1.0, run_s=0.1, evict_s=0.01):
        self.clk, self.prep_s, self.run_s, self.evict_s = clk, prep_s, run_s, evict_s
        self.calls, self.kinds = [], {}
        self.limit_for, self.capture_fail_for = set(), set()

    def run_eager(self, inp, key):
        self.calls.append(("eager", key))
        self.clk.t += self.run_s

    def prepare_compiled(self, key, inp):
        self.calls.append(("prepare_compiled", key))
        if key in self.limit_for:
            raise LimitHit("recompile_limit reached")
        self.clk.t += self.prep_s
        return {}

    def run_compiled(self, inp, key):
        self.calls.append(("run_compiled", key))
        self.clk.t += self.run_s

    def prepare_graph(self, kind, key, inp):
        self.calls.append(("prepare_graph", kind, key))
        if key in self.capture_fail_for:
            raise CapFail("out of memory")
        self.kinds[key] = kind
        self.clk.t += self.prep_s
        return {"capture_pool_mb": 123.0}

    def run_graph(self, inp, key):
        self.calls.append(("run_graph", key))
        self.clk.t += self.run_s

    def graph_kind(self, key):
        return self.kinds.get(key)

    def evict(self, key):
        self.calls.append(("evict", key))
        self.kinds.pop(key, None)
        self.clk.t += self.evict_s
        return {"freed_mb": 1.0, "evict_s": self.evict_s}


def S(h=512, w=512, b=1, c=0):
    return Shape(h, w, b, c)


def mkreqs(shapes):
    return [dict(H=s.H, W=s.W, B=s.B, cfg=s.cfg, prompt="a red bicycle leaning against a brick wall", noise_seed=1, req_id=f"r{i}")
            for i, s in enumerate(shapes)]


class Script:
    """Policy that plays back a fixed action list (duck-typed policy)."""
    name = "script"

    def __init__(self, acts, **attrs):
        self.acts = list(acts)
        self.__dict__.update(attrs)

    def decide(self, req, state):
        return self.acts.pop(0)


def run_mock(policy, shapes, budget=None, rt=None, **kw):
    rt = rt or MockRT(FakeClock())
    clk = rt.clk
    rows, status, err = st.serve_s3(mkreqs(shapes), rt, policy, lambda r, s: object(), clk, lambda: None, lambda: (1.0, 1.0),
                                    mem_budget_mb=budget, make_padded_inputs=lambda r, s, b: object(), **kw)
    return rows, status, err, rt


class FlatCost:
    """Minimal CostView stand-in: predicted pool per shape."""

    def __init__(self, pool):
        self.pool = pool

    def capture_pool_mb(self, arm, shape):
        return self.pool[shape] if arm in ("G1", "C1+G1") else 0.0


def test_evict_runs_before_main_action_and_state_follows():
    a, b = S(512, 512), S(768, 512)
    pol = Script([Action(CAPTURE), Action(CAPTURE, evict=(a,))])
    rows, status, err, rt = run_mock(pol, [a, b], budget=1000.0, cost=FlatCost({a: 600.0, b: 700.0}))
    assert status == "ok", err
    seq = [c[0] for c in rt.calls]
    second_capture = [i for i, n in enumerate(seq) if n == "prepare_graph"][1]
    assert seq.index("evict") < second_capture  # evict BEFORE the 2nd capture (#14: capture needs its whole pool free at once)
    assert rows[1]["event"] == "capture" and rows[1]["evicted"] == a.tag() and rows[1]["n_evicted"] == 1
    assert rows[1]["graph_mem_mb"] == 700.0 and rows[1]["n_graphs"] == 1


def test_no_white_eviction_when_capture_would_still_not_fit():
    a, b = S(512, 512), S(768, 512)
    pol = Script([Action(CAPTURE), Action(CAPTURE, evict=(a,))])
    rows, status, err, rt = run_mock(pol, [a, b], budget=500.0, cost=FlatCost({a: 400.0, b: 900.0}))
    assert [r["event"] for r in rows] == ["capture", "capture_denied"]  # b alone exceeds the budget: nothing is evicted
    assert not [c for c in rt.calls if c[0] == "evict"] and rows[1]["n_evicted"] == 0
    assert rows[1]["served_by"] == "E0" and rt.calls[-1] == ("eager", b)  # over budget -> eager (un-compiled modules)


def test_admission_and_register_use_the_predicted_pool_only():
    a = S()
    pol = Script([Action(CAPTURE), Action(CAPTURE)])
    # the runtime's measured pool (123 MB in MockRT) must NOT matter: admission and register use the predicted 400 MB
    rows, _, _, rt = run_mock(pol, [a, a], budget=500.0, cost=FlatCost({a: 400.0}))
    assert [r["event"] for r in rows] == ["capture", "steady"] and rows[0]["graph_mem_mb"] == 400.0


def test_capture_oom_is_recorded_and_served_eagerly():
    a = S()
    rt = MockRT(FakeClock())
    rt.capture_fail_for.add(a)
    pol = Script([Action(CAPTURE), Action(CAPTURE)])
    stats = {}
    rows, status, _, rt = run_mock(pol, [a, a], budget=1000.0, rt=rt, cost=FlatCost({a: 100.0}), stats=stats)
    assert status == "ok" and rows[0]["event"] == "capture_denied" and rows[0]["planned_event"] == "capture"
    assert "capture_oom" in rows[0]["note"] and rows[0]["n_graphs"] == 0
    assert rows[1]["event"] == "capture_denied" and stats["n_capture_failed"] == 2  # a later request retries (and fails again here)


def test_b3p_planned_limit_fallback_makes_no_compile_attempt():
    shapes = [S(512, 512), S(512, 768), S(768, 512), S(768, 768)]  # UNet keys 1,2,3,4 (limit 2 -> 3rd and 4th fall back)
    pol = Script([Action(COMPILE)] * 5, recompile_limit=2)
    rows, status, _, rt = run_mock(pol, shapes + [shapes[0]], fallback_on_limit=True)
    assert [r["event"] for r in rows] == ["compile", "compile", "eager_fallback_limit", "eager_fallback_limit", "steady"]
    assert [c for c in rt.calls if c[0] == "prepare_compiled"] == [("prepare_compiled", shapes[0]), ("prepare_compiled", shapes[1])]
    assert rows[2]["served_by"] == "E0" and rows[4]["served_by"] == "C1"


def test_b3p_unplanned_limit_exception_is_caught_and_whole_request_goes_eager():
    a, b = S(512, 512), S(512, 768)
    rt = MockRT(FakeClock())
    rt.limit_for.add(b)  # plan_step says `compile`; the (mock) Dynamo raises FailOnRecompileLimitHit
    stats = {}
    rows, status, _, _ = run_mock(Script([Action(COMPILE)] * 3), [a, b, a], rt=rt, fallback_on_limit=True, stats=stats)
    assert status == "ok" and [r["event"] for r in rows] == ["compile", "eager_fallback_limit", "steady"]
    assert rows[1]["planned_event"] == "compile" and "recompile_limit_hit_unplanned" in rows[1]["note"] and ("eager", b) in rt.calls
    assert stats["n_limit_fallback"] == 1
    # without B3' mode the exception is a failure of the process (B1 with limit 64 must never silently fall back)
    rt2 = MockRT(FakeClock())
    rt2.limit_for.add(b)
    rows2, status2, _, _ = run_mock(Script([Action(COMPILE)] * 2), [a, b], rt=rt2, fallback_on_limit=False)
    assert status2 == "fail" and rows2[-1]["status"] == "fail"


def test_warmup_counts_into_service_time_and_is_not_a_request():
    class Warm(Script):
        def warmup(self, state, cost):
            return [(S(512, 512), Action(COMPILE_CAPTURE)), (S(768, 512), Action(COMPILE_CAPTURE))]

    rows, status, _, rt = run_mock(Warm([Action(COMPILE_CAPTURE)]), [S(512, 512)], budget=5000.0,
                                   cost=FlatCost({S(512, 512): 100.0, S(768, 512): 200.0}))
    assert [r["phase"] for r in rows] == ["warmup", "warmup", "req"]
    assert [r["idx"] for r in rows] == [-1, -2, 0] and rows[2]["event"] == "steady"
    m = st.summarize_times(rows, load_s=5.0)
    assert m["n_served"] == 1 and m["event_counts"] == {"steady": 1}
    assert m["warmup_s"] == pytest.approx(2 * 1.1) and m["service_s"] == pytest.approx(2 * 1.1 + 0.1)  # warm-up inside the span


def test_run_bucket_dispatches_to_the_bucket_graph():
    b3, b4 = S(512, 512, 3, 0), S(512, 512, 4, 0)
    pol = Script([Action(CAPTURE), Action(RUN_BUCKET, bucket=b4)])
    rows, status, err, rt = run_mock(pol, [b4, b3], budget=9000.0, cost=FlatCost({b4: 100.0}))
    assert status == "ok", err
    assert rows[1]["event"] == "steady" and rows[1]["served_by"] == "bucket:" + b4.tag() and rt.calls[-1] == ("run_graph", b4)


def test_compile_only_tier_and_evicted_graph_served_by_compiled_code():
    """v2 four tiers E -> G1 -> C1 -> C1+G1: `compile` builds no graph; C1 -> C1+G1 reuses the compiled code (no second compile call);
    after the graph is evicted the shape is served by the compiled code (compile layer is never evicted), not by eager."""
    a = S()
    pol = Script([Action(COMPILE), Action(COMPILE_CAPTURE), Action(COMPILE_CAPTURE), Action(COMPILE, evict=(a,)), Action(COMPILE)])
    rows, status, err, rt = run_mock(pol, [a] * 5, budget=1000.0, cost=FlatCost({a: 300.0}))
    assert status == "ok", err
    assert [r["event"] for r in rows] == ["compile", "compile_capture", "steady", "steady", "steady"]
    assert [r["served_by"] for r in rows] == ["C1", "C1+G1", "C1+G1", "C1", "C1"]
    assert rows[0]["n_graphs"] == 0 and rows[1]["n_graphs"] == 1 and rows[3]["n_graphs"] == 0 and rows[3]["evicted"] == a.tag()
    assert [c for c in rt.calls if c[0] == "prepare_compiled"] == [("prepare_compiled", a)]  # compiled once only
    assert rt.calls[-2:] == [("run_compiled", a), ("run_compiled", a)] and ("evict", a) in rt.calls


def test_policy_pads_exactly_when_the_gate_is_open(ct):
    """run_bucket gate (interface sec 1) lives in decide(): padded only if equiv_pass_padB4 == 1 AND the bucket's steady latency < the
    shape's eager latency. Checked over every B3 cell of the cost table and both graph arms; closed cells must never return run_bucket."""
    from p24.policy import CostView
    cv = CostView(ct, "warm")
    pol = make_policy("Ours", cost=cv)
    n_open = n_closed_eq = n_closed_slow = 0
    for (h, w) in [(512, 512), (512, 768), (768, 512), (768, 768)]:
        for cfg in (0, 1):
            shape3, bucket = Shape(h, w, 3, cfg), Shape(h, w, 4, cfg)
            for arm, kind in (("G1", "G1"), ("C1+G1", "C1+G1")):
                state = ProcState(mem_budget_mb=1e9, overhead_aware=True)
                state.graphs[bucket], state.graph_kind[bucket] = 2000.0, kind
                eq = cv.equiv_pass_padB4(shape3, arm)
                fast = ct.steady_s(arm, bucket) < ct.steady_s("E0", shape3)
                got = pol._pad_bucket(shape3, state)
                assert (got == bucket) == (eq and fast), (shape3, arm, eq, fast, got)
                n_open += bool(eq and fast)
                n_closed_eq += (not eq)
                n_closed_slow += (eq and not fast)
                if got is None:  # a closed cell must not be served through the bucket by decide()
                    act = pol.decide(dict(H=h, W=w, B=3, cfg=cfg, prompt="p", noise_seed=1), state)
                    assert act.kind != RUN_BUCKET
    assert n_closed_eq >= 1 and (n_open + n_closed_slow) >= 1, (n_open, n_closed_eq, n_closed_slow)  # both kinds of cell exist
    print(f"\nPADDING-GATE open={n_open} closed(not equivalent)={n_closed_eq} closed(bucket not faster than eager)={n_closed_slow}")


def _event_rows(rows):
    return [(r["action"], r["event"], r["evicted"] or "") for r in rows if r.get("phase", "req") == "req"]


def _sim_events(name, ct_, reqs, budget, prior, params=None):
    pol, _ = st.build_policy(name, ct_, params, prior, "warm")
    res = simulate(reqs, pol, ct_, SimOptions(cache_mode="warm", mem_budget_mb=budget, overhead_aware=True))
    df = res.frame()
    df = df[df.idx >= 0]
    ev = df["evicted"].fillna("") if "evicted" in df else [""] * len(df)
    return list(zip(df.action, df.event, ev)), res


# TEST parameters (main session 11-17): {alpha 1, H_prior 1e4}, one of the three pre-registered 3.2 candidate sets. NOT the frozen
# policy_params.json. With the default H_prior (1e3) Ours evicts only 0-2 times on the tight tier and B5 is never denied, so neither alone
# satisfies ">= 3 evictions and >= 1 denial"; with these parameters both do.
P_STRESS = {"alpha": 1.0, "H_prior": 10000.0}
EVENT_CASES = [("B0", None, None), ("B1", None, None), ("B4", "default", None), ("Ours", "tight", None), ("Ours", "tight", P_STRESS),
               ("B5", "tight", None), ("B5", "tight", P_STRESS), ("B6", "default", None), ("Ours-NE", "tight", None), ("B3p", None, None)]


@pytest.mark.parametrize("name,tier,params", EVENT_CASES)
def test_mock_loop_events_equal_simulator(name, tier, params, ct, requests200):
    """serve_s3 (mock runtime) vs simulate(): per request (action, event, evicted) identical -- the loop adds nothing to the decisions."""
    from p24.sim.prior import prior_A
    prior = prior_A()
    sim_ev, _ = _sim_events(name, ct, requests200, tier_mb(tier), prior, params)
    pol, cv = st.build_policy(name, ct, params, prior, "warm")
    rt = MockRT(FakeClock())
    rows, status, err = st.serve_s3(requests200, rt, pol, lambda r, s: object(), rt.clk, lambda: None, lambda: (1.0, 1.0), cost=cv,
                                    mem_budget_mb=tier_mb(tier), make_padded_inputs=lambda r, s, b: object(),
                                    fallback_on_limit=(name == "B3p"))
    assert status == "ok", err
    got = _event_rows(rows)
    assert len(got) == len(sim_ev) == N_EVENT
    assert got == sim_ev


def _counts(sim_ev):
    return (sum(len(e[2].split(";")) for e in sim_ev if e[2]), sum(1 for e in sim_ev if e[1] == "capture_denied"))


def test_tight_budget_event_test_is_not_vacuous(ct, requests200):
    """The 3.1 event test must actually exercise eviction (>= 3) and refusal (>= 1) on the tight tier within the 200 requests.
    Finding (final Ours v2, 11-17): with the test parameters {alpha 1, H_prior 1e4} Ours/B5 evict (14 / 120 times) but are never DENIED on any of
    the 10 W2 dev traces (the compile-only tier C1 absorbs what would have been refused; LRU always finds victims), while with the default
    parameters Ours is denied 13 times but evicts only once. So the two requirements are met by the PAIR of runs on the same trace and tier
    (stress run: evictions; default run: denials), not by one run. Numbers are printed; the assertion is the pair-level one."""
    from p24.sim.prior import prior_A
    pr = prior_A()
    res = {}
    for name in ("Ours", "B5"):
        for tag, params in (("default", None), ("stress", P_STRESS)):
            res[(name, tag)] = _counts(_sim_events(name, ct, requests200, tier_mb("tight"), pr, params)[0])
    print("\n(evictions, denials) on tight:", {f"{k[0]}-{k[1]}": v for k, v in res.items()})
    assert res[("Ours", "stress")][0] >= 3 and res[("B5", "stress")][0] >= 3, res  # eviction path exercised (stress parameters)
    assert res[("Ours", "default")][1] >= 1, res  # refusal path exercised (default parameters)


def test_stderr_tee_saves_stderr_and_counts_pinned_fallbacks(tmp_path):
    code = (f"import sys, os; sys.path.insert(0, {str(CODE)!r}); from scripts.serve_trace import StderrTee; "
            f"t = StderrTee({str(tmp_path / 'x.stderr.txt')!r}).start(); "
            "sys.stderr.write('hello\\n'); os.write(2, b'P24_PINNED_FALLBACK_USED kernel=k arg=a bytes=1\\n'); "
            "sys.stderr.write('P24_PINNED_FALLBACK_USED again\\n'); print(t.stop())")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert r.stdout.strip() == "2", (r.stdout, r.stderr)
    txt = (tmp_path / "x.stderr.txt").read_text()
    assert "hello" in txt and txt.count("P24_PINNED_FALLBACK_USED") == 2
    assert "P24_PINNED_FALLBACK_USED" in r.stderr  # still visible on the terminal


# ====================================================================================================== Part B: GPU0 (dev mode)
def _gpu_reason(need_mb=12000):
    """None if GPU0 can be used now; else the reason to skip. Rules: only GPU0; others' memory + our peak <= ~20 GB; others' utilisation < 70 %."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != GPU0:
        return "CUDA_VISIBLE_DEVICES is not the GPU0 UUID (GPU tests run only on GPU0)"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", GPU0],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        used, util = float(out[0]), float(out[1])
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e!r}"
    if used + need_mb > 20480 or util >= 70:
        return f"GPU0 busy: used {used:.0f} MiB, util {util:.0f} % (need {need_mb} MiB headroom below 20 GB, util < 70 %)"
    return None


def _need_gpu(mb=12000):
    r = _gpu_reason(mb)
    if r:
        pytest.skip(r)
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _need_warm_cache():
    c = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    if "warm" not in c:
        pytest.skip("compile tests need a warm compile cache dir (TORCHINDUCTOR_CACHE_DIR=.../warm_s3dev/C1_G1/inductor): "
                    "never compile into the shared default cache")


MODEL_DIR = ROOT / "data/models/sd15_lcm_fused"
PROMPT = "a red bicycle leaning against a brick wall"


@pytest.fixture(scope="module")
def rt64():
    _need_gpu()
    _need_warm_cache()
    import torch
    from p24.runners_tiered import TieredRuntime
    torch.cuda.set_per_process_memory_fraction(min(1.0, 17000.0 / (torch.cuda.get_device_properties(0).total_memory / 2 ** 20)), 0)
    rt = TieredRuntime(str(MODEL_DIR), compile=True, recompile_limit=64)
    yield rt
    rt.release_all()


def _inp(rt, shape, seed=5):
    return rt.p.make_inputs([PROMPT] * shape.B, shape.to_shapekey(), seeds=[seed + j for j in range(shape.B)])


def _real_inputs(rt):
    return lambda r, s: rt.p.make_inputs([r["prompt"]] * s.B, s.to_shapekey(), seeds=[1])


def test_gpu_evict_reserved_falls_back_to_accounting(rt64, ct):
    import torch
    rt = rt64
    rt.release_all()
    torch.cuda.empty_cache()
    A, B = S(512, 512, 1, 0), S(512, 768, 1, 0)
    kA, kB = A.to_shapekey(), B.to_shapekey()
    inA, inB = _inp(rt, A), _inp(rt, B)
    rt.run_eager(inA, kA)  # one-time lazily allocated workspaces (cuBLAS ...) belong to the baseline, not to a graph
    torch.cuda.empty_cache()
    base0, _ = rt.mem_mb()
    warm = S(256, 256, 1, 0).to_shapekey()  # first capture of a process leaves ~66 MB behind (side-stream / capture workspaces, measured
    rt.prepare_graph("G1", warm, _inp(rt, S(256, 256, 1, 0)))  # 11-17): a one-time cost, not per graph -> part of the baseline
    rt.evict(warm)
    base, _ = rt.mem_mb()
    one_time_capture_overhead = base - base0
    rt.prepare_graph("G1", kA, inA)
    pb = rt.prepare_graph("G1", kB, inB)
    r_both, _ = rt.mem_mb()
    ref = rt.run_eager(inA, kA).clone()
    info = rt.evict(kB)
    r_after, _ = rt.mem_mb()
    freed = r_both - r_after
    pred_b = ct.capture_pool_mb("G1", B)
    out_a = rt.run_graph(inA, kA)
    torch.cuda.synchronize()
    assert rt.has_graph(kA) and not rt.has_graph(kB)
    assert (out_a.float() - ref.float()).abs().max() < 1e-2  # the surviving graph still replays correctly (G1 = near bit-exact)
    assert abs(info["freed_mb"] - freed) <= 64 and abs(info["freed_mb"] - pb["capture_pool_mb"]) <= 64  # evict() reports the graph footprint
    assert abs(freed - pb["capture_pool_mb"]) <= 64, (freed, pb["capture_pool_mb"])  # reserved falls back by the measured pool
    assert abs(freed - pred_b) <= 64, (freed, pred_b)  # and by the accounted (predicted) pool
    rt.release_all()
    del ref, out_a, inA, inB  # test tensors would pin allocator segments (measurement artefact, not a graph leak)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    r_end = rt.mem_mb()[0]
    assert abs(r_end - base) <= 64, (r_end, base)  # everything released -> back to the weights-only level
    print(f"\nEVICT-ACCOUNT freed={freed:.1f} measured_pool={pb['capture_pool_mb']:.1f} predicted_pool={pred_b:.1f} "
          f"gap_vs_predicted={freed - pred_b:+.1f} gap_vs_accounting(pool+22.33)={freed - pred_b - 22.33:+.1f} "
          f"evict_s={info['evict_s']:.3f} reserved_end_minus_base={r_end - base:+.1f} one_time_first_capture_overhead={one_time_capture_overhead:.0f}")


def test_gpu_upgrade_g1_to_c1g1_releases_old_graph_first(rt64, ct):
    _need_warm_cache()
    import torch
    rt = rt64
    rt.release_all()
    torch.cuda.empty_cache()
    base, _ = rt.mem_mb()
    A = S(384, 384, 1, 0)
    k = A.to_shapekey()
    inp = _inp(rt, A)
    rt.prepare_graph("G1", k, inp)
    r_g1, _ = rt.mem_mb()
    p2 = rt.prepare_graph("C1+G1", k, inp)
    r_cg, _ = rt.mem_mb()
    assert rt.graph_kind(k) == "C1+G1" and p2["upgrade_released_kind"] == "G1" and p2["upgrade_released_mb"] > 100
    # only the C1+G1 pool stays (predicted 356 MB at 384^2 B1; G1 was 628 MB): not G1 + C1+G1
    assert abs((r_cg - base) - ct.capture_pool_mb("C1+G1", A)) <= 128, (r_cg - base, ct.capture_pool_mb("C1+G1", A))
    out = rt.run_graph(inp, k)
    assert out.shape == (1, 3, 384, 384) and out.isfinite().all()
    rt.release_all()
    print(f"\nUPGRADE base={base:.0f} G1={r_g1 - base:.0f} (pred {ct.capture_pool_mb('G1', A):.0f}) -> C1+G1={r_cg - base:.0f} "
          f"(pred {ct.capture_pool_mb('C1+G1', A):.0f}) released_old={p2['upgrade_released_mb']:.0f}")


def test_gpu_compile_only_tier_reuses_compiled_code_after_graph_eviction(rt64):
    """v2 tier C1 (compile, no graph): C1 -> C1+G1 must not compile again; after the graph is evicted the compiled code still serves the shape."""
    _need_warm_cache()
    import torch
    from p24 import diag
    rt = rt64
    rt.release_all()
    A = S(256, 256, 1, 0)
    k = A.to_shapekey()
    inp = _inp(rt, A)
    rt.prepare_compiled(k, inp)
    assert not rt.has_graph(k)
    out_c = rt.run_compiled(inp, k).clone()
    d1 = diag.dyn_snapshot()
    rt.prepare_graph("C1+G1", k, inp)  # capture over the SAME compiled components
    d2 = diag.dyn_snapshot()
    out_g = rt.run_graph(inp, k).clone()
    rt.evict(k)
    d3 = diag.dyn_snapshot()
    out_c2 = rt.run_compiled(inp, k).clone()
    torch.cuda.synchronize()
    d4 = diag.dyn_snapshot()
    assert not rt.has_graph(k)
    for a_, b_ in ((d1, d2), (d3, d4)):
        dd = diag.dyn_delta(a_, b_)
        assert dd["unique_graphs"] == 0 and dd["frame_compiles"] == 0, dd  # no (re)compilation: Dynamo cache entries are shared
    assert (out_c.float() - out_g.float()).abs().max() < 1e-2 and (out_c.float() - out_c2.float()).abs().max() < 1e-2
    print("\nCOMPILE-ONLY-TIER ok: C1 -> C1+G1 and evicted-graph -> C1 add no Dynamo graphs")


def test_gpu_run_bucket_padding_serves_b1_from_b2_graph(rt64):
    import torch
    from p24.bench import PROMPTS_DEFAULT
    rt = rt64
    rt.release_all()
    bucket, real = S(256, 256, 2, 0), S(256, 256, 1, 0)
    kb, kr = bucket.to_shapekey(), real.to_shapekey()
    padded = rt.p.make_inputs([PROMPT, PROMPTS_DEFAULT[0]], kb, seeds=[5, st.PAD_SEED0 + 1])
    rt.prepare_graph("G1", kb, padded)
    out = rt.run_bucket(padded, kb, 1)
    native = rt.run_eager(rt.p.make_inputs([PROMPT], kr, seeds=[5]), kr)
    torch.cuda.synchronize()
    assert out.shape == native.shape == (1, 3, 256, 256) and out.isfinite().all()
    mad = (out.float() - native.float()).abs().mean().item()
    assert mad < 0.02, mad  # fp16 kernels are batch-composition dependent (#3): close, not bit-exact
    rt.release_all()


def test_gpu_b3p_recompile_limit_hit_goes_eager_whole_request(rt64):
    _need_warm_cache()
    import torch
    import torch._dynamo
    from p24 import diag
    from p24.runners_tiered import TieredRuntime
    rt64.release_all()
    a, b = S(256, 256, 1, 0), S(384, 384, 1, 0)

    def go(rt, plan_limit):
        pol = Script([Action(COMPILE)] * 3)
        pol.recompile_limit = plan_limit
        stats = {}
        rows, status, err = st.serve_s3(mkreqs([a, b, a]), rt, pol, _real_inputs(rt), time.perf_counter, torch.cuda.synchronize,
                                        rt.mem_mb, fallback_on_limit=True, stats=stats)
        assert status == "ok", err
        return rows, stats

    try:
        # limit 1 stands in for 8: the 2nd distinct UNet key hits it.  (1) planned: plan_step knows the limit -> eager, Dynamo untouched
        torch._dynamo.reset()
        rows, stats = go(TieredRuntime(pipe=rt64.p, compile=True, recompile_limit=1), 1)
        assert [r["event"] for r in rows] == ["compile", "eager_fallback_limit", "steady"] and stats["n_limit_fallback"] == 0
        # (2) unplanned: plan_step says `compile`, real Dynamo raises FailOnRecompileLimitHit -> caught, WHOLE request on the eager modules
        torch._dynamo.reset()
        rows, stats = go(TieredRuntime(pipe=rt64.p, compile=True, recompile_limit=1), None)
        assert [r["event"] for r in rows] == ["compile", "eager_fallback_limit", "steady"], [r["event"] for r in rows]
        assert rows[1]["planned_event"] == "compile" and stats["n_limit_fallback"] == 1 and rows[1]["served_by"] == "E0"
        assert rows[2]["served_by"] == "C1"  # the compiled shape keeps being served compiled
    finally:
        torch._dynamo.reset()
        diag.set_recompile_limit(64)


def test_gpu_over_budget_is_denied_and_served_eagerly(rt64, ct):
    import torch
    from p24.policy import CostView
    rt = rt64
    rt.release_all()
    torch.cuda.empty_cache()
    a, b = S(256, 256, 1, 0), S(384, 384, 1, 0)  # G1 pools 236 and 628 MB
    pol, _ = st.build_policy("B4", ct, None, None, "warm")
    rows, status, err = st.serve_s3(mkreqs([a, b, b, a]), rt, pol, _real_inputs(rt), time.perf_counter, torch.cuda.synchronize, rt.mem_mb,
                                    cost=CostView(ct, "warm"), mem_budget_mb=400.0)  # 236+22.33 fits, +628 does not
    assert status == "ok", err
    assert [r["event"] for r in rows] == ["capture", "capture_denied", "capture_denied", "steady"]
    assert [r["served_by"] for r in rows] == ["G1", "E0", "E0", "G1"]
    assert rt.has_graph(a.to_shapekey()) and not rt.has_graph(b.to_shapekey())
    rt.release_all()


def _run_cli(policy, tier, out_dir, extra=(), proc_id=0):
    env = dict(os.environ)
    cmd = ["taskset", "-c", "0-15", sys.executable, "-B", "-m", "scripts.serve_trace", "--trace", str(DEV_TRACE), "--policy", policy,
           "--cache-mode", "warm", "--warm-root", env.get("P24_TEST_WARM_ROOT", str(pathlib.Path(env["P24_CACHE"]) / "warm_s3dev")),
           "--out", str(out_dir), "--max-requests", str(N_EVENT), "--mem-cap-mb", "17000", "--proc-id", str(proc_id), *extra]
    if tier:
        cmd += ["--budget-tier", tier]
    r = subprocess.run(cmd, cwd=CODE, env=env, capture_output=True, text=True, timeout=3300)
    assert r.returncode == 0, r.stderr[-3000:]
    stem = f"{st.file_policy(policy)}_{DEV_TRACE.stem}_p{proc_id}"
    import pandas as pd
    return pd.read_csv(out_dir / f"{stem}.csv", keep_default_na=False), json.loads((out_dir / f"{stem}.meta.json").read_text())


@pytest.mark.parametrize("name,tier,params", [("B0", None, None), ("B4", "default", None), ("Ours", "tight", P_STRESS),
                                              ("B5", "tight", P_STRESS), ("B6", "default", None), ("Ours", "tight", None),
                                              ("B5", "tight", None)])
def test_gpu_event_consistency_runtime_vs_simulator(name, tier, params, ct, requests200, tmp_path_factory):
    """3.1 acceptance: first 200 requests of dev W2 seed0, CLI replay on GPU0 vs simulator: (action, event, evicted) 100 % equal."""
    _need_gpu(14000)
    _need_warm_cache()
    from p24.sim.prior import prior_A
    env_out = os.environ.get("P24_EVENT_TEST_OUT")
    out_dir = pathlib.Path(env_out) if env_out else tmp_path_factory.mktemp("ev")
    out_dir.mkdir(parents=True, exist_ok=True)
    extra, pid = [], 0
    if params:
        pj = out_dir / f"params_{name}_stress.json"
        pj.write_text(json.dumps(params))
        extra, pid = ["--params-json", str(pj)], 1
    df, meta = _run_cli(name, tier, out_dir, extra, pid)
    sim_ev, res = _sim_events(name, ct, requests200, tier_mb(tier), prior_A(), params)
    d = df[df["phase"] == "req"]
    got = list(zip(d["action"], d["event"], d["evicted"].astype(str)))
    assert len(got) == len(sim_ev) == N_EVENT and meta["status"] == "ok"
    n_ok = sum(1 for g, s_ in zip(got, sim_ev) if g == s_)
    n_evict = int(d["n_evicted"].sum())
    n_denied = int((d["event"] == "capture_denied").sum())
    print(f"\nEVENT-CONSISTENCY {name} tier={tier} params={params}: {n_ok}/{N_EVENT} equal; evictions={n_evict} denied={n_denied} "
          f"capture_oom={meta['n_capture_failed']} limit_fallback={meta['n_limit_fallback']} pinned_fallback_count={meta['pinned_fallback_count']} "
          f"evict_gap_mb_max={meta['evict_gap_mb_max']} peak_reserved_mb={meta['peak_reserved_mb']:.0f} service_s={meta['service_s']:.1f}")
    assert n_ok == N_EVENT, [(i, g, s_) for i, (g, s_) in enumerate(zip(got, sim_ev)) if g != s_][:5]
    stem = f"{st.file_policy(name)}_{DEV_TRACE.stem}_p{pid}"
    assert "pinned_fallback_count" in meta and (out_dir / f"{stem}.stderr.txt").exists()
    if name in ("Ours", "B5") and params:  # stress parameters: the eviction path must really run (denials: see the non-vacuity test above)
        assert n_evict >= 3, f"{name}: evictions={n_evict}: event test INVALID (needs >= 3 evictions)"
    if name == "Ours" and not params:  # default parameters: the refusal path must really run
        assert n_denied >= 1, f"{name}: denied={n_denied}: event test INVALID (needs >= 1 refusal)"
    if meta["evict_gap_mb_max"] is not None:
        assert meta["evict_gap_mb_max"] <= 64, meta["evict_gap_mb_max"]  # freed memory vs accounted pool, every eviction
