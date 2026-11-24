"""One benchmark process = one arm x one shape key. Measures model load, one-time prepare
(compile / capture), first request latency and steady-state latency; writes raw rows + meta.json.

Usage: python -m p24.bench --arm E0 --H 512 --W 512 --B 1 --cfg 0 --reps 20 --out <dir> --proc-id 0
Statistics are done across processes (the process is the unit), see p24.stats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time

GPU0 = "GPU-722679a6-175b-cb56-b764-e8124d262c41"
PROMPTS_DEFAULT = [
    "a photograph of an astronaut riding a horse",
    "a red bicycle leaning against a brick wall, film photo",
    "a bowl of ramen, studio lighting, 4k",
    "an oil painting of a lighthouse at sunset",
    "a cute corgi wearing sunglasses on the beach",
    "a futuristic city skyline at night, neon lights",
    "a watercolor of mountains and a lake",
    "a close-up portrait of an old man, dramatic lighting",
]


CPU_LOAD_MAX = 16.0  # 1-min loadavg above this marks a process cpu_contended (formal runs are pinned to P-cores 0-15;
# 8 -> 16 on 2026-11-15, see STATE/AUTO_DECISIONS.md)


def smi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", GPU0],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"ERR {e}"


def gpu0_other_procs() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return -1
    pids = [l.split(",")[1].strip() for l in out.strip().splitlines() if l.startswith(GPU0)]
    return sum(1 for p in pids if p != str(os.getpid()))


def code_sha() -> dict:
    """SHA256-16 of every p24/*.py and scripts/*.py (keys "p24/<name>", "scripts/<name>"); review S5: the drivers
    (run_matrix, run_equivalence, prewarm_s2, bench_seq) are part of the measured code too."""
    pkg = pathlib.Path(__file__).resolve().parent
    out = {}
    for sub, d in (("p24", pkg), ("scripts", pkg.parent / "scripts")):
        for f in sorted(d.glob("*.py")):
            out[f"{sub}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def meta_snapshot(args) -> dict:
    import torch, diffusers, transformers
    try:
        import triton
        tv = triton.__version__
    except Exception:
        tv = None
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(), "argv": sys.argv,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": tv, "diffusers": diffusers.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "smi": smi("driver_version,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,pstate"),
        "loadavg": os.getloadavg(), "nproc": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu0_other_procs": gpu0_other_procs(),
        "env": {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                                                 "TORCHINDUCTOR_COMPILE_THREADS", "CUDA_VISIBLE_DEVICES", "P24_PINNED_FALLBACK")},
        "code_sha256_16": code_sha(), "args": vars(args),
    }


NAN = float("nan")

CSV_COLS = ["arm", "H", "W", "B", "cfg", "proc_id", "cold_tier", "status", "rep", "wall_ms", "gpu_ms",
            "first_req_s", "prep_json", "peak_alloc_mb", "reserved_mb", "contended",
            "load1_start", "load1_end", "cpu_contended",
            # ---- S2 step 2.0 additions (protocol sec 6); absent in S1 CSVs, stats.py tolerates that ----
            "row_kind", "te_ms", "denoise_ms", "vae_ms", "comp_total_ms", "comp_wall_ms",
            "resident_mb", "weights_alloc_mb", "steady_peak_alloc_mb", "steady_peak_reserved_mb",
            "alloc_after_prep_mb", "reserved_after_prep_mb",
            "compile_s", "compile_and_warmup_s", "capture_s", "capture_pool_mb", "graph_mem_mb", "static_io_mb",
            "static_in_mb", "first_call_te_s", "first_call_unet_s", "first_call_vae_s", "first_call_whole_s",
            "dyn_recompiles", "dyn_unique_graphs", "dyn_frame_compiles", "fx_cache_hit", "fx_cache_miss",
            "fx_cache_bypass", "aot_cache_hit", "aot_cache_miss", "recompile_limit", "recompile_limit_hit",
            "pcore_pinned", "tier_violation", "seq_idx",
            # ---- review 2026-11-15 (M4): request index i within the process (i=0 is the first request that is
            # part of first_req_s; steady row rep r has i = warm + r + 1). NaN for component rows. ----
            "req_idx"]


def request_inputs_fn(runner, key):
    p = runner.p
    prompts = [PROMPTS_DEFAULT[i % len(PROMPTS_DEFAULT)] for i in range(key.B)]

    def inputs(r):
        return p.make_inputs(prompts, key, seeds=[1000 * r + i for i in range(key.B)])
    return inputs


def measure_key(runner, key, reps: int, warm: int, comp_reps: int, res: dict) -> dict:
    """prepare -> first request -> `warm` warm-up -> `reps` steady requests -> `comp_reps` component-timed requests.
    Fills `res` progressively (so a failure keeps what was gathered). The steady path is exactly the S1 path
    (host perf_counter + CUDA events around runner.run(inp, key), no marks); component requests run afterwards."""
    import torch
    inputs = request_inputs_fn(runner, key)
    res.setdefault("rows", [])
    res.setdefault("comp_rows", [])
    ex = inputs(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res["prepare"] = runner.prepare(key, ex)
    runner.run(inputs(1), key)
    torch.cuda.synchronize()
    res["first_req_s"] = time.perf_counter() - t0
    res["alloc_after_prep_mb"] = torch.cuda.memory_allocated() / 2**20
    res["reserved_after_prep_mb"] = torch.cuda.memory_reserved() / 2**20
    for _ in range(warm):
        runner.run(inputs(2), key)
    torch.cuda.synchronize()
    res["peak_before_steady_mb"] = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(reps):
        inp = inputs(10 + r)
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        ev0.record()
        runner.run(inp, key)
        ev1.record()
        torch.cuda.synchronize()
        w1 = time.perf_counter()
        res["rows"].append({"rep": r, "row_kind": "steady", "wall_ms": (w1 - w0) * 1e3,
                            "gpu_ms": ev0.elapsed_time(ev1), "req_idx": warm + r + 1})
    res["steady_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["steady_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    # ---- component-timing requests (never mixed into the steady rows; whole-pipeline graphs -> NaN) ----
    for r in range(comp_reps):
        inp = inputs(30 + r)
        evs = {}

        def marks(name, _evs=evs):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _evs[name] = e
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        runner.run(inp, key, marks=marks)
        torch.cuda.synchronize()
        w1 = time.perf_counter()
        if all(n in evs for n in ("start", "te_end", "denoise_end", "vae_end")):
            te = evs["start"].elapsed_time(evs["te_end"])
            dn = evs["te_end"].elapsed_time(evs["denoise_end"])
            vae = evs["denoise_end"].elapsed_time(evs["vae_end"])
            tot = evs["start"].elapsed_time(evs["vae_end"])
        else:
            te = dn = vae = tot = NAN
        res["comp_rows"].append({"rep": r, "row_kind": "comp", "wall_ms": NAN, "gpu_ms": NAN, "te_ms": te,
                                 "denoise_ms": dn, "vae_ms": vae, "comp_total_ms": tot,
                                 "comp_wall_ms": (w1 - w0) * 1e3})
    return res


def measure_guarded(runner, key, reps: int, warm: int, comp_reps: int, res: dict):
    """measure_key with failures recorded as data (capture-constraints catalogue) -> (status, err).
    Shared by p24.bench and scripts.bench_seq; tests/test_recompile_limit.py drives it with a toy runner."""
    try:
        measure_key(runner, key, reps, warm, comp_reps, res)
        return "ok", ""
    except Exception as e:  # noqa: BLE001
        res["prepare"], res["first_req_s"] = {}, NAN
        return "fail", repr(e)[:2000]


def tier_violation_flag(uses_compile: bool, cold_tier: str, d: dict) -> int:
    """Protocol sec 5 (review M2): a compiled-arm cold_diskcache process must show FX-graph cache hits and no
    misses/bypasses; hit == 0 (nothing came from the cache) is a violation too."""
    return int(bool(uses_compile) and cold_tier == "cold_diskcache" and
               (d["fx_miss"] > 0 or d["fx_bypass"] > 0 or d["fx_hit"] == 0))


def prep_columns(prep: dict) -> dict:
    keys = ("compile_s", "compile_and_warmup_s", "capture_s", "capture_pool_mb", "graph_mem_mb", "static_io_mb",
            "static_in_mb", "first_call_te_s", "first_call_unet_s", "first_call_vae_s", "first_call_whole_s")
    return {k: prep.get(k, NAN) for k in keys}


def diag_columns(runner, err: str, limit_watch, cold_tier: str, affinity) -> dict:
    """Process-level compile diagnostics. Counters are process-cumulative (bench_seq: cumulative over the sequence)."""
    from p24 import diag
    d = diag.dyn_snapshot()
    uses = bool(getattr(runner, "uses_compile", False))
    hit = int(d["limit_reached"] > 0 or bool(limit_watch is not None and limit_watch.hit) or
              ("recompile_limit" in err) or ("FailOnRecompileLimitHit" in err))
    if uses:
        import torch._dynamo.config as dc
        limit = dc.recompile_limit
    else:
        limit = NAN
    return {"dyn_recompiles": d["recompiles"], "dyn_unique_graphs": d["unique_graphs"],
            "dyn_frame_compiles": d["frame_compiles"], "fx_cache_hit": d["fx_hit"], "fx_cache_miss": d["fx_miss"],
            "fx_cache_bypass": d["fx_bypass"], "aot_cache_hit": d["aot_hit"], "aot_cache_miss": d["aot_miss"],
            "recompile_limit": limit, "recompile_limit_hit": hit,
            "pcore_pinned": int(set(affinity) <= set(range(16))),
            # protocol sec 5: a compiled-arm cold_diskcache process must show Inductor FX-graph cache hits, no misses/bypasses
            "tier_violation": tier_violation_flag(uses, cold_tier, d)}


def host_columns(meta: dict) -> dict:
    return dict(contended=int(meta["gpu0_other_procs"] != 0 or meta["gpu0_other_procs_end"] != 0),
                load1_start=meta["loadavg"][0], load1_end=meta["loadavg_end"][0],
                # host contention: eager small-batch latency is launch-bound, so a busy CPU (other
                # jobs, SMT siblings, E-core placement on the i9-14900KF) inflates it. Threshold: see CPU_LOAD_MAX.
                cpu_contended=int(max(meta["loadavg"][0], meta["loadavg_end"][0]) > CPU_LOAD_MAX))


def write_csv(path, base: dict, res: dict):
    """base: per-(process, key) constants; steady rows then comp rows (row_kind distinguishes them)."""
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_COLS)
        wr.writeheader()
        rows = (res.get("rows") or []) + (res.get("comp_rows") or [])
        for r in rows or [{"rep": -1, "row_kind": "steady", "wall_ms": NAN, "gpu_ms": NAN}]:
            wr.writerow({**base, **r})


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else NAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--cfg", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3, help="extra steady warmup requests (not recorded)")
    ap.add_argument("--comp-reps", type=int, default=5, help="component-timing requests after the steady ones")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--cold-tier", default="na", help="label: cold_empty / cold_diskcache / na")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--profile", default=None, help="if set, write a torch.profiler trace of 2 steady requests")
    ap.add_argument("--nsys-window", type=int, default=0, help="N steady requests inside cudaProfilerStart/Stop")
    args = ap.parse_args()

    t_proc0 = time.perf_counter()
    import torch
    from p24 import diag
    from p24.pipeline import ShapeKey
    from p24.runners import make_runner

    root = pathlib.Path(os.environ["PROJ_ROOT"])
    model_dir = args.model_dir or str(root / "data/models/sd15_lcm_fused")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = ShapeKey(args.H, args.W, args.B, bool(args.cfg))
    tag = f"{args.arm}_{key.tag()}_p{args.proc_id}"
    # review S3: never overwrite raw results (an excluded process's CSV is evidence); use another --proc-id/--out
    for suffix in (".csv", ".meta.json"):
        if (out / f"{tag}{suffix}").exists():
            print(f"REFUSING to overwrite existing {out / (tag + suffix)} (use a new --out or --proc-id)",
                  file=sys.stderr, flush=True)
            sys.exit(3)
    meta = meta_snapshot(args)

    torch.cuda.init()
    t_imp = time.perf_counter()
    limit_watch = diag.install_limit_watch()
    runner = make_runner(args.arm, model_dir)
    torch.cuda.synchronize()
    t_load = time.perf_counter()
    free_b, total_b = torch.cuda.mem_get_info()
    resident_mb = (total_b - free_b) / 2**20  # device-level (all processes); valid as "ours" only when GPU0 exclusive
    weights_alloc_mb = torch.cuda.memory_allocated() / 2**20

    res = {"prepare": {}, "first_req_s": NAN}
    status, err = measure_guarded(runner, key, args.reps, args.warm, args.comp_reps, res)
    try:
        inputs = request_inputs_fn(runner, key) if status == "ok" else None
        if status == "ok" and args.nsys_window:
            # run under: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx ...
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for r in range(args.nsys_window):
                torch.cuda.nvtx.range_push(f"request{r}")
                runner.run(inputs(600 + r), key)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        if status == "ok" and args.profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for r in range(2):
                    runner.run(inputs(500 + r), key)
                torch.cuda.synchronize()
            prof.export_chrome_trace(args.profile)
            kern = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
            meta["profile_cuda_kernel_calls_2req"] = int(sum(e.count for e in kern))
    except Exception as e:  # nsys/profile extras failing is recorded as data as well (measure failures: measure_guarded)
        status, err = "fail", repr(e)[:2000]
        res["prepare"], res["first_req_s"] = {}, NAN

    prep, first_req_s = res["prepare"], res["first_req_s"]
    peak = max(res.get("peak_before_steady_mb", 0.0), torch.cuda.max_memory_allocated() / 2**20)
    reserved = torch.cuda.memory_reserved() / 2**20
    dcols = diag_columns(runner, err, limit_watch, args.cold_tier, meta["cpu_affinity"])
    rows = res.get("rows", [])
    meta.update({"status": status, "error": err, "load_s": t_load - t_imp, "proc_start_to_load_s": t_load - t_proc0,
                 "prepare": prep, "first_req_s": first_req_s, "peak_alloc_mb": peak, "reserved_mb": reserved,
                 "resident_mb": resident_mb, "weights_alloc_mb": weights_alloc_mb,
                 "steady_peak_alloc_mb": res.get("steady_peak_alloc_mb", NAN),
                 "steady_peak_reserved_mb": res.get("steady_peak_reserved_mb", NAN),
                 "alloc_after_prep_mb": res.get("alloc_after_prep_mb", NAN),
                 "reserved_after_prep_mb": res.get("reserved_after_prep_mb", NAN),
                 "diag": dcols, "recompile_limit_msgs": limit_watch.msgs,
                 "component_rows": res.get("comp_rows", []),
                 "gpu0_other_procs_end": gpu0_other_procs(), "loadavg_end": os.getloadavg(),
                 "smi_end": smi("clocks.sm,temperature.gpu,power.draw")})
    (out / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    base = dict(arm=args.arm, H=key.H, W=key.W, B=key.B, cfg=int(key.cfg), proc_id=args.proc_id,
                cold_tier=args.cold_tier, status=status, first_req_s=first_req_s,
                prep_json=json.dumps(prep), peak_alloc_mb=peak, reserved_mb=reserved,
                resident_mb=resident_mb, weights_alloc_mb=weights_alloc_mb,
                steady_peak_alloc_mb=res.get("steady_peak_alloc_mb", NAN),
                steady_peak_reserved_mb=res.get("steady_peak_reserved_mb", NAN),
                alloc_after_prep_mb=res.get("alloc_after_prep_mb", NAN),
                reserved_after_prep_mb=res.get("reserved_after_prep_mb", NAN),
                **host_columns(meta), **prep_columns(prep), **dcols)
    write_csv(out / f"{tag}.csv", base, res)
    med = median([r["wall_ms"] for r in rows])
    print(json.dumps({"tag": tag, "status": status, "first_req_s": first_req_s, "steady_med_ms": med,
                      "prepare": prep, "err": err[:300]}))


if __name__ == "__main__":
    main()
