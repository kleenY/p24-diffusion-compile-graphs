"""S3 3.1b eviction -> re-capture micro-benchmark (GPU0, exclusive block S3-A part 2; NOT run in S3 prep, only written).

Question (goals/S3_detailed_goals_v2.md 3.1b, G19): what does it cost to RE-capture the CUDA graph of a shape after its graph was evicted
(release + empty_cache), and does `memory_reserved` fall back after the release? If rebuild / capture_s of the cost table > 1.5 the simulator uses
the measured rebuild seconds (CostView.rebuild_override); a fall-back difference > 64 MB is a new failure-catalogue entry (Ours-NE fallback).

Arms: G1 (manual per-component graphs on eager modules) and C1+G1 (same on torch.compile components; compiled code stays in the process, so a rebuild
re-runs only the warm-up calls and the capture). Shapes (H x W x B x cfg): 512x512x1x0, 768x512x1x0 (portrait H768 W512), 512x512x4x1 (B4 with CFG).
Per (arm, shape) 5 rounds; ONE process per arm. Round r:
    reserved0 (after empty_cache)  ->  capture #1  (runner.prepare: warm-up + capture)  ->  first request + S steady requests
    ->  release(key) + gc + empty_cache  ->  reserved_after_release  ->  capture #2 = REBUILD (prepare again)  ->  S steady requests, bitwise
    equality of the output with the pre-release output for the same input  ->  release again (start of the next round).
Recorded per round (CSV row): capture #1 / rebuild wall seconds, their capture_s and warm-up parts, steady medians, reserved before / after
capture / after release / after rebuild, pool MB, `reserved_residual_mb` (= after release - before capture), `reserved_recovered` (|residual| <= 64 MB),
release seconds, output-equal flag, status/error (an OOM or other failure is recorded and the next shape is run).

Only p24.runners / p24.pipeline / p24.bench helpers are imported (read-only); nothing in p24/ is changed. stderr lines containing
P24_PINNED_FALLBACK_USED are counted into meta `pinned_fallback_count` (the block script additionally stores the whole stderr).
Cache (C1+G1): --cache disk uses --warm-root/<arm>/ (default $P24_CACHE/warm_s2), set before torch is imported (as scripts.bench_seq).

Usage:  python -m scripts.bench_rebuild_S3 --arm G1 --out results/S3/cost_addendum/rebuild --proc-id 0 [--rounds 5 --steady 5]
        python -m scripts.bench_rebuild_S3 --dry-run            # prints the plan only (no torch import, no GPU)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time

DEFAULT_KEYS = "512x512x1x0,768x512x1x0,512x512x4x1"
ARMS = ("G1", "C1+G1")
RECOVER_TOL_MB = 64.0
COLS = ["arm", "H", "W", "B", "cfg", "proc_id", "round", "status", "error",
        "cap1_wall_s", "cap1_capture_s", "cap1_warmup_s", "cap1_pool_mb", "steady1_med_ms", "first_req1_s",
        "release_s", "reserved0_mb", "reserved_after_cap1_mb", "reserved_after_release_mb", "reserved_residual_mb", "reserved_recovered",
        "rebuild_wall_s", "rebuild_capture_s", "rebuild_warmup_s", "rebuild_pool_mb", "reserved_after_rebuild_mb", "steady2_med_ms",
        "output_equal_after_rebuild", "rebuild_over_cap1_wall", "rebuild_over_cap1_capture", "rebuild_wall_over_capture_s",
        "pinned_fallback_count_so_far", "load1_start", "load1_end", "gpu0_other_procs_start", "gpu0_other_procs_end"]


class StderrCounter:
    """Pass-through stderr proxy counting P24_PINNED_FALLBACK_USED lines."""

    def __init__(self, real):
        self.real, self.n, self._buf = real, 0, ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "P24_PINNED_FALLBACK_USED" in line:
                self.n += 1
        return self.real.write(s)

    def flush(self):
        return self.real.flush()

    def __getattr__(self, k):
        return getattr(self.real, k)


def parse_keys(s: str):
    return [tuple(int(v) for v in k.split("x")) for k in s.split(",")]


def plan(a) -> dict:
    keys = parse_keys(a.keys)
    arms = a.arm.split(",") if a.arm else list(ARMS)
    assert set(arms) <= set(ARMS), arms
    out = pathlib.Path(a.out) if a.out else None
    files = []
    for arm in arms:
        files += [f"rebuild_{arm}_p{a.proc_id}.csv", f"rebuild_{arm}_p{a.proc_id}.meta.json"]
    return dict(arms=arms, keys=keys, rounds=a.rounds, steady=a.steady, out=str(out), files=files,
                processes=len(arms), rounds_total=len(arms) * len(keys) * a.rounds,
                prepares_total=2 * len(arms) * len(keys) * a.rounds, steady_requests_total=2 * a.steady * len(arms) * len(keys) * a.rounds,
                note="one process per arm; run each arm as its own `python -m scripts.bench_rebuild_S3 --arm <arm>` (the block script does this)")


def run_arm(arm: str, a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_p = out / f"rebuild_{arm}_p{a.proc_id}.csv"
    meta_p = out / f"rebuild_{arm}_p{a.proc_id}.meta.json"
    clash = [f for f in (csv_p, meta_p) if f.exists()]
    if clash:
        print(f"REFUSING to overwrite existing {clash[0]} (use a new --out or --proc-id)", file=sys.stderr, flush=True)
        sys.exit(3)
    if a.cache == "disk":                                        # must be set before torch / inductor / triton are imported
        warm_root = pathlib.Path(a.warm_root) if a.warm_root else pathlib.Path(os.environ["P24_CACHE"]) / "warm_s2"
        cdir = warm_root / arm.replace("+", "_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cdir / "triton")
    counter = StderrCounter(sys.stderr)
    sys.stderr = counter
    import csv
    import torch
    from p24.bench import gpu0_other_procs, meta_snapshot, request_inputs_fn, median
    from p24.pipeline import ShapeKey
    from p24.runners import make_runner

    mib = 2 ** 20
    root = pathlib.Path(os.environ["PROJ_ROOT"])
    model_dir = a.model_dir or str(root / "data/models/sd15_lcm_fused")
    meta = meta_snapshot(a)
    torch.cuda.init()
    t0 = time.perf_counter()
    runner = make_runner(arm, model_dir)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    rows, shape_summary = [], []
    nan = float("nan")

    def steady_med(key, inputs, n):
        ms = []
        for r in range(n):
            inp = inputs(10 + r)
            torch.cuda.synchronize()
            w0 = time.perf_counter()
            runner.run(inp, key)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - w0) * 1e3)
        return median(ms)

    for (H, W, B, cfg) in parse_keys(a.keys):
        key = ShapeKey(H, W, B, bool(cfg))
        inputs = request_inputs_fn(runner, key)
        ex = inputs(0)
        for rnd in range(a.rounds):
            row = dict(arm=arm, H=H, W=W, B=B, cfg=cfg, proc_id=a.proc_id, round=rnd, status="ok", error="",
                       load1_start=os.getloadavg()[0], gpu0_other_procs_start=gpu0_other_procs())
            try:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                r0 = torch.cuda.memory_reserved() / mib
                tc = time.perf_counter()
                p1 = runner.prepare(key, ex)
                cap1_wall = time.perf_counter() - tc
                tq = time.perf_counter()
                runner.run(inputs(1), key)
                torch.cuda.synchronize()
                first_req1 = time.perf_counter() - tq
                r1 = torch.cuda.memory_reserved() / mib
                ref_inp = inputs(2)
                ref_out = runner.run(ref_inp, key).clone()
                s1 = steady_med(key, inputs, a.steady)
                # ---- evict: release the graph set + empty_cache ----
                tr = time.perf_counter()
                runner.release(key)
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                release_s = time.perf_counter() - tr
                r2 = torch.cuda.memory_reserved() / mib
                # ---- rebuild (capture again) ----
                tb = time.perf_counter()
                p2 = runner.prepare(key, ex)
                rebuild_wall = time.perf_counter() - tb
                torch.cuda.synchronize()
                r3 = torch.cuda.memory_reserved() / mib
                out2 = runner.run(ref_inp, key)
                torch.cuda.synchronize()
                same = bool(torch.equal(ref_out, out2))
                s2 = steady_med(key, inputs, a.steady)
                runner.release(key)                          # clean start of the next round / shape
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                c1, c2 = p1.get("capture_s", nan), p2.get("capture_s", nan)
                row.update(cap1_wall_s=cap1_wall, cap1_capture_s=c1, cap1_warmup_s=p1.get("compile_and_warmup_s", nan),
                           cap1_pool_mb=p1.get("capture_pool_mb", nan), steady1_med_ms=s1, first_req1_s=first_req1,
                           release_s=release_s, reserved0_mb=r0, reserved_after_cap1_mb=r1, reserved_after_release_mb=r2,
                           reserved_residual_mb=r2 - r0, reserved_recovered=int(abs(r2 - r0) <= RECOVER_TOL_MB),
                           rebuild_wall_s=rebuild_wall, rebuild_capture_s=c2, rebuild_warmup_s=p2.get("compile_and_warmup_s", nan),
                           rebuild_pool_mb=p2.get("capture_pool_mb", nan), reserved_after_rebuild_mb=r3, steady2_med_ms=s2,
                           output_equal_after_rebuild=int(same), rebuild_over_cap1_wall=rebuild_wall / cap1_wall,
                           rebuild_over_cap1_capture=(c2 / c1) if c1 and c1 == c1 else nan,
                           rebuild_wall_over_capture_s=(rebuild_wall / c1) if c1 and c1 == c1 else nan)
            except Exception as e:  # noqa: BLE001  (OOM etc. recorded as data; free what is held and go on with the next shape)
                row.update(status="fail", error=repr(e)[:500])
                try:
                    runner.release(key)
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            row.update(pinned_fallback_count_so_far=counter.n, load1_end=os.getloadavg()[0], gpu0_other_procs_end=gpu0_other_procs())
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k in ("arm", "H", "W", "B", "cfg", "round", "status", "cap1_capture_s", "cap1_wall_s", "rebuild_wall_s",
                                       "reserved_residual_mb", "output_equal_after_rebuild")}), flush=True)
            if row["status"] != "ok":
                break
        shape_summary.append(dict(key=f"{H}x{W}x{B}x{cfg}", rounds_done=sum(1 for r in rows if (r["H"], r["W"], r["B"], r["cfg"]) == (H, W, B, cfg))))
    with open(csv_p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, "") for c in COLS})
    meta.update(dict(arm=arm, proc_id=a.proc_id, keys=a.keys, rounds=a.rounds, steady=a.steady, load_s=load_s, cache_mode=a.cache,
                     pinned_fallback_count=counter.n, shape_summary=shape_summary, recover_tol_mb=RECOVER_TOL_MB,
                     weights_alloc_mb=torch.cuda.memory_allocated() / mib,
                     any_fail=int(any(r["status"] != "ok" for r in rows)), loadavg_end=os.getloadavg()))
    meta_p.write_text(json.dumps(meta, indent=2, default=str))
    sys.stderr = counter.real


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default=None, help="G1 or C1+G1 (default with --dry-run: both)")
    ap.add_argument("--keys", default=DEFAULT_KEYS, help="comma list HxWxBxcfg")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steady", type=int, default=5, help="steady requests timed after each capture")
    ap.add_argument("--out", default=None)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--cache", choices=["env", "disk"], default="disk", help="C1+G1 compile cache: disk = <warm-root>/C1_G1")
    ap.add_argument("--warm-root", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; no torch import, no GPU")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps(plan(a), indent=1, ensure_ascii=False))
        return
    assert a.arm in ARMS and a.out, "--arm {G1|C1+G1} and --out are required"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "the GPU0 device variable must be exported in the same command (run protocol)"
    run_arm(a.arm, a)


if __name__ == "__main__":
    main()
