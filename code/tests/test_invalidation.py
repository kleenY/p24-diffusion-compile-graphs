"""Graph-invalidation catalogue v1 (S2 step 2.8): one minimal reproduction per entry of docs/capture_constraints.md.

Entries 9, 10, 12, 14, 15 are core; 11 is second tier.  (13 is deferred to BACKLOG: LoRA is fused, never swapped.)
Toy conv/attention modules (a few seconds each) unless the test name says `real_unet`.
Every test writes its observations to results/S2/invalidation/<NN>.json (merged per entry).
Compile caches live in <P24_CACHE>/invalidation/ (never warm_s2/ or matrix/).
Run on GPU0 only (set the GPU0 UUID in CUDA_VISIBLE_DEVICES in the same shell command):
    python -m pytest code/tests/test_invalidation.py -q
"""
import gc
import json
import logging
import os
import pathlib
import time

import pytest
import torch
import torch._dynamo  # noqa: F401
import torch.nn as nn

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
OUT = ROOT / "results/S2/invalidation"
CACHE = pathlib.Path("<P24_CACHE>/invalidation")
UNET_DIR = ROOT / "data/models/sd15_lcm_fused/unet"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (use GPU0 only)")
MiB = 2 ** 20


# ------------------------------------------------------------------ helpers
def record(entry: int, test: str, **kw):
    """Merge observations of one test into results/S2/invalidation/<entry>.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{entry:02d}.json"
    d = json.loads(f.read_text()) if f.exists() else {"entry": entry, "tests": {}}
    d["torch"] = torch.__version__
    d["tests"][test] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    f.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))


def reserved_mb():
    return torch.cuda.memory_reserved() / MiB


def free_cuda():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def snap():
    """Dynamo counters (torch 2.9.1): graphs actually compiled, frame attempts, recompile-limit events."""
    from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER
    from torch._dynamo.utils import counters
    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "frame_attempts": int(sum(FRAME_COMPILE_COUNTER.values())),
            "frames_total": int(counters["frames"]["total"]), "frames_ok": int(counters["frames"]["ok"]),
            "limit_events": int(counters["unimplemented"].get("recompile_limit reached", 0))}


class Toy(nn.Module):
    """Conv + self-attention block; `compiled_flag` records whether Dynamo compiled the call (the side effect is
    traced as a constant True; eager execution stores False), which tells compiled calls from eager fallback."""

    def __init__(self, c=8):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(4, c, 3, padding=1), nn.Conv2d(c, 4, 3, padding=1)
        self.attn = nn.MultiheadAttention(c, 2, batch_first=True)
        self.compiled_flag = None

    def forward(self, x):
        self.compiled_flag = torch.compiler.is_compiling()
        h = torch.relu(self.c1(x))
        b, c, hh, ww = h.shape
        t = h.flatten(2).transpose(1, 2)
        t = self.attn(t, t, t, need_weights=False)[0]
        return self.c2(t.transpose(1, 2).reshape(b, c, hh, ww))


def toy_in(hw, b=1):
    return torch.randn(b, 4, hw, hw, device="cuda", dtype=torch.half)


@pytest.fixture(autouse=True)
def _isolated_dynamo():
    """Fresh Dynamo state per test, private compile cache, config restored afterwards."""
    import torch._dynamo.config as dc
    from torch._dynamo.utils import counters
    saved_env = {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}
    (CACHE / "inductor").mkdir(parents=True, exist_ok=True)
    (CACHE / "triton").mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton")
    saved_cfg = (dc.recompile_limit, dc.fail_on_recompile_limit_hit)
    torch._dynamo.reset()
    counters.clear()
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    dc.recompile_limit, dc.fail_on_recompile_limit_hit = saved_cfg
    torch._dynamo.reset()
    free_cuda()
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _toy():
    torch.manual_seed(0)
    return Toy().cuda().half().eval()


# ------------------------------------------------------------------ entry 9: shape change
def test_09_shape_change_recompiles_and_graph_not_reusable():
    m = _toy()
    f = torch.compile(m, dynamic=False)
    shapes = [8, 10, 12]
    n_graphs = []
    with torch.inference_mode():
        for hw in shapes:
            f(toy_in(hw))
            n_graphs.append(snap()["unique_graphs"])
        f(toy_in(8))  # old shape again: cache hit
        again = snap()["unique_graphs"]
    assert n_graphs == [1, 2, 3], n_graphs  # dynamic=False: exactly one compile per distinct shape
    assert again == 3  # returning to a seen shape does not recompile
    # contrast: default automatic dynamic generalises after the 2nd distinct shape
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    g = torch.compile(m)  # dynamic=None
    with torch.inference_mode():
        for hw in [14, 16, 18, 20]:
            g(toy_in(hw))
    auto_dyn_graphs = snap()["unique_graphs"]

    # manual CUDA graph: replay is bound to the captured shapes
    with torch.inference_mode():
        static_in = toy_in(8)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(static_in)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = m(static_in)
        graph.replay()
        out_shape = tuple(static_out.shape)
        with pytest.raises(RuntimeError) as ei:
            static_in.copy_(toy_in(10))  # a different shape cannot be written into the captured input buffer
        graph.replay()
        assert tuple(static_out.shape) == out_shape == (1, 4, 8, 8)  # output stays at the captured shape
        # a new shape needs a second capture (own buffers)
        in10 = toy_in(10)
        with torch.cuda.stream(s):
            for _ in range(3):
                m(in10)
        torch.cuda.current_stream().wait_stream(s)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            out10 = m(in10)
        g2.replay()
        torch.cuda.synchronize()
        assert tuple(out10.shape) == (1, 4, 10, 10)
    record(9, "shape_change", dynamic_false_unique_graphs_after_3_shapes=n_graphs,
           recompiles_after_3_shapes=n_graphs[-1] - 1,
           revisit_adds_graphs=again - n_graphs[-1],
           automatic_dynamic_unique_graphs_for_4_shapes=auto_dyn_graphs,
           manual_graph_copy_new_shape_exception=type(ei.value).__name__,
           manual_graph_new_shape_needs_new_capture=True)


# ------------------------------------------------------------------ entry 10: guidance as Python float
def test_10_guidance_python_float_vs_tensor():
    from torch._dynamo.utils import counters

    def step(x, g):
        return x * g + 1

    x = torch.randn(16, device="cuda")
    vals = [1.0, 1.5, 2.0, 2.5, 3.0]

    def run(dynamic, as_tensor):
        torch._dynamo.reset()
        counters.clear()
        f = torch.compile(step, dynamic=dynamic)
        res = []
        for v in vals:
            f(x, torch.tensor(v, device="cuda") if as_tensor else v)
            res.append(snap()["unique_graphs"])
        return res

    pyfloat_static = run(False, False)
    pyfloat_auto = run(None, False)
    tensor0d = run(False, True)
    assert pyfloat_static == [1, 2, 3, 4, 5]  # dynamic=False: one compile per distinct float value
    assert tensor0d == [1] * 5  # 0-dim tensor: never recompiles
    assert pyfloat_auto[-1] <= 2  # default automatic dynamic: 2nd value symbolises the float -> stable at 2 graphs
    record(10, "guidance_float", values=vals, unique_graphs_python_float_dynamic_false=pyfloat_static,
           unique_graphs_python_float_automatic_dynamic=pyfloat_auto, unique_graphs_0dim_tensor=tensor0d,
           note="torch 2.9.1: a Python float is guarded by value under dynamic=False (1 compile per value); "
                "with automatic dynamic it is symbolised after the first change (2 graphs total); "
                "a 0-dim tensor never recompiles. Serving code keeps guidance as a tensor / eager step math.")


# ------------------------------------------------------------------ entry 12: recompile limit
def _shape_sweep(f, m, n_shapes):
    flags, graphs = [], []
    with torch.inference_mode():
        for i in range(n_shapes):
            m.compiled_flag = None
            f(toy_in(8 + 2 * i))
            flags.append(bool(m.compiled_flag))
            graphs.append(snap()["unique_graphs"])
    return flags, graphs


def test_12_recompile_limit_default_falls_back_to_eager():
    import torch._dynamo.config as dc
    assert dc.recompile_limit == 8, dc.recompile_limit  # torch 2.9.1 default (name: recompile_limit)
    assert dc.accumulated_recompile_limit == 256
    m = _toy()
    f = torch.compile(m, dynamic=False)
    lg = logging.getLogger("torch._dynamo.convert_frame")
    msgs = []

    class H(logging.Handler):
        def emit(self, r):
            msgs.append(r.getMessage())

    h = H(level=logging.WARNING)
    lg.addHandler(h)
    try:
        flags, graphs = _shape_sweep(f, m, 11)
    finally:
        lg.removeHandler(h)
    s = snap()
    assert graphs[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(flags[:8]) and not any(flags[8:])  # shape #9 onwards runs eagerly (no error, only a warning)
    assert graphs[8:] == [8, 8, 8]  # no new graphs
    assert s["limit_events"] >= 1  # detection 1: Dynamo counter
    assert any("recompile_limit" in x for x in msgs)  # detection 2: logger warning
    record(12, "default_limit_8", recompile_limit=dc.recompile_limit,
           accumulated_recompile_limit=dc.accumulated_recompile_limit, shapes_run=11,
           compiled_per_shape=flags, unique_graphs_per_shape=graphs, counters_final=s,
           warning_text=msgs[:1],
           observation="9th distinct shape: warning 'torch._dynamo hit config.recompile_limit (8)', then silent "
                       "eager fallback for that and all later new shapes (the frame is skipped); no exception.")


def test_12_recompile_limit_fail_on_hit_and_fullgraph():
    import torch._dynamo.config as dc
    from torch._dynamo.exc import FailOnRecompileLimitHit
    from torch._dynamo.utils import counters
    m = _toy()
    # (a) fail_on_recompile_limit_hit=True turns the silent fallback into a hard error
    dc.fail_on_recompile_limit_hit = True
    f = torch.compile(m, dynamic=False)
    with pytest.raises(FailOnRecompileLimitHit):
        _shape_sweep(f, m, 11)
    n_a = snap()["unique_graphs"]
    # (b) fullgraph=True (what our compiled runners use) with default config
    dc.fail_on_recompile_limit_hit = False
    torch._dynamo.reset()
    counters.clear()
    f2 = torch.compile(m, dynamic=False, fullgraph=True)
    exc = None
    try:
        _shape_sweep(f2, m, 11)
    except Exception as e:  # noqa: BLE001 - recorded as observation
        exc = type(e).__name__
    record(12, "fail_on_hit_and_fullgraph", fail_on_recompile_limit_hit_raises="FailOnRecompileLimitHit",
           graphs_before_raise=n_a, fullgraph_true_default_config_exception=exc,
           fullgraph_graphs=snap()["unique_graphs"])
    assert exc == "FailOnRecompileLimitHit"  # with fullgraph=True the limit hit is already a hard error


def test_12_recompile_limit_64_no_fallback():
    import torch._dynamo.config as dc
    dc.recompile_limit = 64
    m = _toy()
    f = torch.compile(m, dynamic=False)
    flags, graphs = _shape_sweep(f, m, 11)
    s = snap()
    assert all(flags) and graphs == list(range(1, 12)) and s["limit_events"] == 0
    record(12, "limit_64", recompile_limit=64, shapes_run=11, compiled_per_shape=flags,
           unique_graphs_per_shape=graphs, counters_final=s)


# ------------------------------------------------------------------ entry 14: capture under memory pressure
class Block(nn.Module):
    """Activation-heavy conv stack so that the graph private pool is tens of MB."""

    def __init__(self, c=64, n=5):
        super().__init__()
        self.net = nn.Sequential(*[nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU()) for _ in range(n)])

    def forward(self, x):
        return self.net(x)


def _capture(fn, static_in, pool=None):
    """Mirror GraphRunner.prepare(): side-stream warm-up, then capture. Returns (graph, static_out, pool_mb)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(s)
    free_cuda()
    r0 = torch.cuda.memory_reserved()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = fn(static_in)
    torch.cuda.synchronize()
    return g, out, (torch.cuda.memory_reserved() - r0) / MiB


def _release(g):
    g.reset()
    free_cuda()


def _limit_bytes(headroom_bytes):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + headroom_bytes) / total, 0)


def test_14_capture_oom_then_evict_retry_toy():
    m = Block().cuda().half().eval()
    with torch.inference_mode():
        a_in = torch.randn(8, 64, 128, 128, device="cuda", dtype=torch.half)  # bigger shape, kept resident
        b_in = torch.randn(8, 64, 112, 112, device="cuda", dtype=torch.half)
        ga, _, pool_a = _capture(m, a_in)
        gb0, _, pool_b = _capture(m, b_in)  # unconstrained measurement of B's pool
        _release(gb0)
        ref = m(b_in).clone()  # eager reference, computed before the limit is imposed
        free_cuda()
        reserved_before = reserved_mb()
        _limit_bytes(int(0.25 * pool_b * MiB))
        try:
            with pytest.raises(torch.OutOfMemoryError) as ei:
                _capture(m, b_in)
            exc_name = type(ei.value).__name__
            oom_msg = str(ei.value).splitlines()[0][:160]
            free_cuda()
            _release(ga)  # evict the other shape's graph pool
            gb, out_b, pool_b2 = _capture(m, b_in)  # same limit, retry succeeds
            gb.replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
    assert torch.allclose(out_b, ref, atol=1e-2, rtol=1e-2)
    record(14, "oom_then_evict_retry_toy", exception=exc_name, oom_message=oom_msg,
           pool_mb_A_evicted=pool_a, pool_mb_B_needed=pool_b, pool_mb_B_after_evict=pool_b2,
           headroom_mb_at_failure=0.25 * pool_b, reserved_mb_before_limit=reserved_before)


@pytest.fixture(scope="module")
def real_unet():
    if not UNET_DIR.exists():
        pytest.skip("fused UNet not available")
    free, _tot = torch.cuda.mem_get_info(0)
    if free < 9 * 2 ** 30:
        pytest.skip(f"GPU0 free {free / 2 ** 30:.1f} GiB < 9 GiB needed for real-UNet tests")
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(str(UNET_DIR), torch_dtype=torch.float16).cuda().eval()
    unet.requires_grad_(False)
    yield unet
    del unet
    free_cuda()


def _unet_bufs(h, w):
    x = torch.randn(1, 4, h // 8, w // 8, device="cuda", dtype=torch.half)
    t = torch.tensor(999, device="cuda")
    emb = torch.randn(1, 77, 768, device="cuda", dtype=torch.half)
    return x, t, emb


def test_14_real_unet_capture_oom_then_evict_retry(real_unet):
    u = real_unet
    fn = lambda xs: u(xs[0], xs[1], encoder_hidden_states=xs[2], return_dict=False)[0]  # noqa: E731
    with torch.inference_mode():
        free_cuda()
        resident_mb = reserved_mb()  # weights
        k1 = _unet_bufs(512, 768)  # resident graph (larger)
        k2 = _unet_bufs(512, 512)  # graph that has to be added
        g1, _, pool1 = _capture(fn, k1)
        g2_0, _, pool2 = _capture(fn, k2)  # unconstrained measurement
        _release(g2_0)
        _limit_bytes(int(0.25 * pool2 * MiB))
        try:
            with pytest.raises(torch.OutOfMemoryError) as ei:
                _capture(fn, k2)
            exc_name = type(ei.value).__name__
            free_cuda()
            _release(g1)
            g2, out2, pool2b = _capture(fn, k2)
            g2.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(out2).all()
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
    record(14, "real_unet_oom_then_evict_retry", unet_only=True, dtype="fp16", exception=exc_name,
           weights_resident_mb=resident_mb, pool_mb_512x768_B1_evicted=pool1, pool_mb_512x512_B1_needed=pool2,
           pool_mb_512x512_B1_after_evict=pool2b, headroom_mb_at_failure=0.25 * pool2,
           note="UNet component only (no text encoder / VAE graphs); one pool per key, as in GraphRunner._GraphSet")


# ------------------------------------------------------------------ entry 15: per-shape release
def _compile_two_shapes_reduce_overhead():
    m = Block().cuda().half().eval()
    f = torch.compile(m, mode="reduce-overhead", dynamic=False)
    free_cuda()
    base = reserved_mb()
    reserved = {}
    with torch.inference_mode():
        for hw in (128, 96):
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                f(torch.randn(8, 64, hw, hw, device="cuda", dtype=torch.half)).clone()
            torch.cuda.synchronize()
            reserved[hw] = reserved_mb()
    return m, f, base, reserved


def test_15_global_reset_releases_everything_reduce_overhead():
    m, f, base, reserved = _compile_two_shapes_reduce_overhead()
    public = [n for n in dir(torch.compiler) if not n.startswith("_")]
    per_shape_candidates = [n for n in public if any(k in n.lower() for k in ("release", "evict", "free", "drop"))]
    assert per_shape_candidates == []  # no public per-shape release API in torch.compiler
    g0 = snap()["unique_graphs"]
    torch.compiler.reset()  # the only public release: global
    free_cuda()
    after = reserved_mb()
    with torch.inference_mode():
        torch.compiler.cudagraph_mark_step_begin()
        f(torch.randn(8, 64, 128, 128, device="cuda", dtype=torch.half)).clone()  # shape 128 again
    g1 = snap()["unique_graphs"]
    assert after < base + 0.2 * (reserved[96] - base)  # the whole graph-tree memory is gone ...
    assert g1 - g0 >= 1  # ... and the shape that we wanted to keep must be recompiled (compile cache is global too)
    record(15, "global_reset_only", reserved_mb_base=base, reserved_mb_after_shape128=reserved[128],
           reserved_mb_after_shape96=reserved[96], reserved_mb_after_torch_compiler_reset=after,
           recompiled_kept_shape_after_reset=g1 - g0, public_torch_compiler_names=public,
           names_matching_release_evict_free_drop=per_shape_candidates,
           private_note="torch._inductor.cudagraph_trees.reset_cudagraph_trees() is also global; the manager "
                        "keeps one root per function id (tm.roots) but exposes no supported per-shape release.")


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="no public API releases one shape's cudagraph-tree memory (only global reset)")
def test_15_release_one_shape_reduce_overhead():
    m, f, base, reserved = _compile_two_shapes_reduce_overhead()
    g0 = snap()["unique_graphs"]
    torch.compiler.reset()  # best public attempt: nothing narrower exists
    free_cuda()
    freed = reserved[96] - reserved_mb()
    with torch.inference_mode():
        torch.compiler.cudagraph_mark_step_begin()
        f(torch.randn(8, 64, 128, 128, device="cuda", dtype=torch.half)).clone()
    kept_shape_still_cached = snap()["unique_graphs"] == g0
    # desired: memory of shape 96 freed AND shape 128 still cached/captured
    assert freed > 0 and kept_shape_still_cached


def test_15_manual_graph_per_shape_pool_release_toy():
    m = Block().cuda().half().eval()
    with torch.inference_mode():
        a_in = torch.randn(8, 64, 128, 128, device="cuda", dtype=torch.half)
        b_in = torch.randn(8, 64, 96, 96, device="cuda", dtype=torch.half)
        free_cuda()
        base = reserved_mb()
        ga, oa, pool_a = _capture(m, a_in, pool=torch.cuda.graph_pool_handle())
        gb, ob, pool_b = _capture(m, b_in, pool=torch.cuda.graph_pool_handle())
        ref_a = m(a_in).clone()  # eager reference first, so that it does not blur the memory deltas
        free_cuda()
        both = reserved_mb()
        _release(gb)
        del gb, ob
        free_cuda()
        only_a = reserved_mb()
        ga.replay()
        torch.cuda.synchronize()
        assert torch.allclose(oa, ref_a, atol=1e-2, rtol=1e-2)  # shape A's graph is untouched
    assert both - only_a > 0.8 * pool_b  # releasing B's graph returns (about) B's pool
    record(15, "manual_graph_per_shape_release_toy", reserved_mb_base=base, pool_mb_A=pool_a, pool_mb_B=pool_b,
           reserved_mb_both=both, reserved_mb_after_release_B=only_a, freed_mb=both - only_a)


def test_15_real_unet_manual_graph_per_shape_release(real_unet):
    u = real_unet
    fn = lambda xs: u(xs[0], xs[1], encoder_hidden_states=xs[2], return_dict=False)[0]  # noqa: E731
    with torch.inference_mode():
        free_cuda()
        base = reserved_mb()
        ka, kb = _unet_bufs(512, 512), _unet_bufs(512, 768)
        ga, oa, pool_a = _capture(fn, ka, pool=torch.cuda.graph_pool_handle())
        gb, ob, pool_b = _capture(fn, kb, pool=torch.cuda.graph_pool_handle())
        ref_a = fn(ka).clone()
        free_cuda()
        both = reserved_mb()
        _release(gb)
        del gb, ob
        free_cuda()
        only_a = reserved_mb()
        ga.replay()
        torch.cuda.synchronize()
        assert torch.allclose(oa, ref_a, atol=2e-2, rtol=2e-2)
    assert both - only_a > 0.8 * pool_b
    record(15, "real_unet_manual_graph_per_shape_release", unet_only=True, dtype="fp16",
           reserved_mb_base_weights_incl_static_io=base, pool_mb_512x512_B1=pool_a, pool_mb_512x768_B1=pool_b,
           reserved_mb_both=both, reserved_mb_after_release_512x768=only_a, freed_mb=both - only_a)


# ------------------------------------------------------------------ entry 11 (second tier): step count
def test_11_step_count_needs_new_unrolled_graph():
    w = torch.randn(32, 32, device="cuda", dtype=torch.half) * 0.1

    def denoise(x, n):  # Python-int step count unrolled by Dynamo, like the LCM loop in GraphRunner._whole
        for i in range(n):
            x = torch.tanh(x @ w) + 0.1 * (i + 1)
        return x

    x0 = torch.randn(4, 32, device="cuda", dtype=torch.half)
    f = torch.compile(denoise, dynamic=False)
    for n in (4, 2, 4, 3):
        f(x0, n)
    graphs = snap()["unique_graphs"]  # 4, 2, 3 -> 3 distinct step counts
    assert graphs == 3

    # manual whole-loop graph captured for n=4 cannot serve n=2
    static = x0.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        denoise(static, 4)
    torch.cuda.current_stream().wait_stream(s)
    g4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g4):
        out = denoise(static, 4)
    g4.replay()
    torch.cuda.synchronize()
    ref4, ref2 = denoise(x0, 4), denoise(x0, 2)
    assert torch.allclose(out, ref4, atol=1e-2)
    assert not torch.allclose(out, ref2, atol=1e-2)  # the replay always runs 4 steps; a 2-step request needs its own graph
    record(11, "step_count", compiled_unique_graphs_for_steps_4_2_4_3=graphs,
           graph_n4_matches_eager_n4=True, graph_n4_matches_eager_n2=False,
           max_abs_diff_graph4_vs_eager2=float((out - ref2).abs().max()))
