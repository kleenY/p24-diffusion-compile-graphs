"""End-to-end test of the recompile-limit judgement path of p24.bench (review S6, 2026-11-15).

torch 2.9.1 under fullgraph=True raises FailOnRecompileLimitHit when torch._dynamo.config.recompile_limit is exceeded (and logs
"torch._dynamo hit config.recompile_limit (N)"); it does NOT write counters["unimplemented"]["recompile_limit reached"].
This test drives the real bench path (`p24.bench.measure_guarded` -> `p24.bench.diag_columns`) with a toy compiled module
(fullgraph=True, dynamic=False, backend eager, so no Inductor/GPU compile is needed) whose input shape changes with the
request index, with recompile_limit = 1: the process must come out as status=fail, recompile_limit_hit=1, recompile_limit=1.
A control with the protocol limit (64) must stay status=ok, recompile_limit_hit=0.
Needs CUDA only because measure_key synchronises / uses CUDA events; run on GPU0 (UUID in CUDA_VISIBLE_DEVICES)."""
import os

import pytest
import torch
import torch._dynamo
import torch._dynamo.config as dc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="measure_key needs CUDA (use GPU0 only)")


class _Inp:
    def __init__(self, x):
        self.x = x


class _ToyPipe:
    """make_inputs shape depends on the request: seeds[0] // 1000 = request index r -> batch 1 + r % 3."""

    def make_inputs(self, prompts, key, seeds, noise_dtype=None):
        return _Inp(torch.zeros(1 + (int(seeds[0]) // 1000) % 3, 4))


class _ToyRunner:
    uses_compile = True
    arm = "TOY"

    def __init__(self):
        self.p = _ToyPipe()
        self.net = torch.nn.Linear(4, 4)
        self.fn = torch.compile(self.net, backend="eager", fullgraph=True, dynamic=False)

    def prepare(self, key, example):
        self.fn(example.x)
        return {}

    def run(self, inp, key, marks=None):
        return self.fn(inp.x)


def _run(limit):
    from p24 import bench, diag
    from p24.pipeline import ShapeKey
    torch._dynamo.reset()
    with dc.patch(recompile_limit=limit):
        watch = diag.install_limit_watch()
        try:
            runner = _ToyRunner()
            res = {"prepare": {}, "first_req_s": float("nan")}
            status, err = bench.measure_guarded(runner, ShapeKey(8, 8, 1, False), 2, 1, 0, res)
            cols = bench.diag_columns(runner, err, watch, "cold_diskcache", sorted(os.sched_getaffinity(0)))
            cols_no_err = bench.diag_columns(runner, "", watch, "cold_diskcache", [0])
            cols_fresh = bench.diag_columns(runner, "", diag.LimitWatch(), "cold_diskcache", [0])
        finally:
            import logging
            logging.getLogger("torch._dynamo").removeHandler(watch)
            torch._dynamo.reset()
    return status, err, watch, cols, cols_no_err, cols_fresh


def test_limit_1_is_recorded_as_fail_and_hit():
    status, err, watch, cols, cols_no_err, cols_fresh = _run(1)
    assert status == "fail", (status, err)
    assert "FailOnRecompileLimitHit" in err, err
    assert cols["recompile_limit"] == 1
    assert cols["recompile_limit_hit"] == 1, cols
    # each signal alone is enough: the exception text (fresh watch, err given) and the log handler (watch, no err)
    assert watch.hit, "log handler did not see the 'hit config.recompile_limit' warning"
    assert cols_no_err["recompile_limit_hit"] == 1
    assert cols_fresh["recompile_limit_hit"] == 0  # nothing seen by a fresh watch and no error text
    # (tier_violation is not asserted here: the Inductor counters are process-cumulative, so it depends on which tests ran
    # before in the same pytest process; its logic is covered by test_tier_violation_flag below)


def test_protocol_limit_no_false_positive():
    status, err, watch, cols, _, _ = _run(64)
    assert status == "ok", err
    assert cols["recompile_limit"] == 64
    assert cols["recompile_limit_hit"] == 0
    assert not watch.hit


@pytest.mark.parametrize("uses,tier,hit,miss,byp,expected", [
    (True, "cold_diskcache", 3, 0, 0, 0),   # the only passing case: hits, no misses, no bypasses
    (True, "cold_diskcache", 3, 1, 0, 1),   # a miss
    (True, "cold_diskcache", 3, 0, 1, 1),   # a bypass
    (True, "cold_diskcache", 0, 0, 0, 1),   # hit=0, miss=0 (review M2: used to count as fine)
    (True, "cold_empty", 0, 3, 0, 0),       # cold_empty is expected to miss
    (True, "inproc_first", 0, 3, 0, 0),
    (False, "cold_diskcache", 0, 0, 0, 0),  # E0 / G1: no compile cache involved
])
def test_tier_violation_flag(uses, tier, hit, miss, byp, expected):
    from p24.bench import tier_violation_flag
    assert tier_violation_flag(uses, tier, {"fx_hit": hit, "fx_miss": miss, "fx_bypass": byp}) == expected
