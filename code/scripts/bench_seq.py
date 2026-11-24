"""In-process sequence benchmark (S2 step 2.0, plan 2.3 "in-process sequences"; protocol sec 5 tier `inproc_first`).

ONE process, ONE arm, a list of shape keys served in order. For every key: prepare + first request (with the per-component
first-call costs), then --warm warm-up requests and --reps steady requests (defaults 3 + 10), optionally --comp-reps
component-timed requests. Per-key CSV + one meta json per sequence. Graph sets/compiled artifacts accumulate across the
keys (unless --release-after), exactly what a long-running server would hold.

--cache env    leave TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR as in the environment (default)
--cache empty  fresh dirs $P24_CACHE/cold/<out.name>/seq_<arm>_p<id>/ , deleted when the process finishes (cold_empty)
--cache disk   per-arm warm dir (--cache-dir, default <--warm-root>/<arm with + -> _>), not cleared (cold_diskcache);
               --warm-root defaults to $P24_CACHE/warm_dev (development runs never write warm_s2; formal runs pass
               --warm-root $P24_CACHE/warm_s2 explicitly)
Raw results are never overwritten (review S3): the process exits with code 3 if any planned CSV/meta already exists.
The cold-tier label written for the first key is cold_empty / cold_diskcache / --cold-tier; later keys are `inproc_first`.
CSV compile-cache columns of key i>0 are deltas since the previous key (fx_cache_*, aot_cache_*); dyn_recompiles and
dyn_unique_graphs stay cumulative over the sequence.

Usage: python -m scripts.bench_seq --arm C1+G1 --keys 512x512x1x0,512x512x2x0,768x768x1x0 --cache empty --out results/S2/seq/x
"""
import argparse, json, os, pathlib, shutil, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--keys", required=True, help="comma list of HxWxBxcfg, served in this order")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--comp-reps", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proc-id", type=int, default=0)
    ap.add_argument("--cache", default="env", choices=["env", "empty", "disk"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--warm-root", default=None, help="parent of the per-arm warm dirs (default $P24_CACHE/warm_dev)")
    ap.add_argument("--cold-tier", default=None, help="override the label of the first key")
    ap.add_argument("--release-after", action="store_true", help="release each key's graph set after measuring it")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_keys = [tuple(int(v) for v in k.split("x")) for k in args.keys.split(",")]
    planned = [out / f"{args.arm}_seq_p{args.proc_id}.meta.json"] + [
        out / f"{args.arm}_seq{i}_{H}x{W}_b{B}_{'cfg' if cfg else 'nocfg'}_p{args.proc_id}.csv"
        for i, (H, W, B, cfg) in enumerate(seq_keys)]
    clash = [f for f in planned if f.exists()]
    if clash:  # review S3
        print(f"REFUSING to overwrite existing {clash[0]} (use a new --out or --proc-id)", file=sys.stderr, flush=True)
        sys.exit(3)
    p24 = pathlib.Path(os.environ["P24_CACHE"])
    arm_dir = args.arm.replace("+", "_")
    cleanup = None
    if args.cache == "empty":
        cdir = p24 / "cold" / out.name / f"seq_{arm_dir}_p{args.proc_id}"
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir(parents=True)
        cleanup, tier0 = cdir, "cold_empty"
    elif args.cache == "disk":
        warm_root = pathlib.Path(args.warm_root) if args.warm_root else p24 / "warm_dev"
        cdir = pathlib.Path(args.cache_dir) if args.cache_dir else warm_root / arm_dir
        cdir.mkdir(parents=True, exist_ok=True)
        tier0 = "cold_diskcache"
    else:
        cdir, tier0 = None, "na"
    if cdir is not None:  # must be set before torch/inductor/triton are imported
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cdir / "triton")
    tier0 = args.cold_tier or tier0

    import torch
    from p24 import diag
    from p24.bench import (NAN, measure_guarded, prep_columns, diag_columns, host_columns, write_csv, median,
                           meta_snapshot, gpu0_other_procs, smi, tier_violation_flag)
    from p24.pipeline import ShapeKey
    from p24.runners import make_runner

    try:
        root = pathlib.Path(os.environ["PROJ_ROOT"])
        model_dir = args.model_dir or str(root / "data/models/sd15_lcm_fused")
        keys = [ShapeKey(*(int(v) for v in k.split("x")[:3]), bool(int(k.split("x")[3]))) for k in args.keys.split(",")]
        meta = meta_snapshot(args)
        torch.cuda.init()
        limit_watch = diag.install_limit_watch()
        t0 = time.perf_counter()
        runner = make_runner(args.arm, model_dir)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        free_b, total_b = torch.cuda.mem_get_info()
        resident_mb = (total_b - free_b) / 2**20
        weights_alloc_mb = torch.cuda.memory_allocated() / 2**20
        per_key = []
        prev = diag.dyn_snapshot() if runner.uses_compile else None
        status_all = "ok"
        for idx, key in enumerate(keys):
            res = {"prepare": {}, "first_req_s": NAN}
            host0 = {"loadavg": os.getloadavg(), "gpu0_other_procs": gpu0_other_procs()}
            status, err = measure_guarded(runner, key, args.reps, args.warm, args.comp_reps, res)
            tier = tier0 if idx == 0 else "inproc_first"
            host1 = {**host0, "gpu0_other_procs_end": gpu0_other_procs(), "loadavg_end": os.getloadavg()}
            dcols = diag_columns(runner, err, limit_watch, tier, meta["cpu_affinity"])
            if prev is not None:  # per-key deltas of the cache counters
                cur = diag.dyn_snapshot()
                d = diag.dyn_delta(prev, cur)
                prev = cur
                dcols.update(fx_cache_hit=d["fx_hit"], fx_cache_miss=d["fx_miss"], fx_cache_bypass=d["fx_bypass"],
                             aot_cache_hit=d["aot_hit"], aot_cache_miss=d["aot_miss"],
                             tier_violation=tier_violation_flag(runner.uses_compile, tier, d))
            rows = res.get("rows", [])
            peak = max(res.get("peak_before_steady_mb", 0.0), torch.cuda.max_memory_allocated() / 2**20)
            reserved = torch.cuda.memory_reserved() / 2**20
            prep, first_req_s = res["prepare"], res["first_req_s"]
            base = dict(arm=args.arm, H=key.H, W=key.W, B=key.B, cfg=int(key.cfg), proc_id=args.proc_id,
                        cold_tier=tier, status=status, first_req_s=first_req_s, prep_json=json.dumps(prep),
                        peak_alloc_mb=peak, reserved_mb=reserved, resident_mb=resident_mb,
                        weights_alloc_mb=weights_alloc_mb, seq_idx=idx,
                        steady_peak_alloc_mb=res.get("steady_peak_alloc_mb", NAN),
                        steady_peak_reserved_mb=res.get("steady_peak_reserved_mb", NAN),
                        alloc_after_prep_mb=res.get("alloc_after_prep_mb", NAN),
                        reserved_after_prep_mb=res.get("reserved_after_prep_mb", NAN),
                        **host_columns(host1),
                        **prep_columns(prep), **dcols)
            tag = f"{args.arm}_seq{idx}_{key.tag()}_p{args.proc_id}"
            write_csv(out / f"{tag}.csv", base, res)
            rec = {"idx": idx, "key": key.tag(), "tier": tier, "status": status, "error": err,
                   "first_req_s": first_req_s, "prepare": prep, "steady_med_ms": median([r["wall_ms"] for r in rows]),
                   "steady_peak_alloc_mb": res.get("steady_peak_alloc_mb", NAN), "diag": dcols,
                   "component_rows": res.get("comp_rows", [])}
            per_key.append(rec)
            print(json.dumps({k: rec[k] for k in ("idx", "key", "tier", "status", "first_req_s", "steady_med_ms")}),
                  flush=True)
            if args.release_after and hasattr(runner, "release"):
                runner.release(key)
            if status != "ok":
                status_all = "fail"
                break  # state after a failed capture/compile is not trustworthy; later keys are not run
        meta.update({"status": status_all, "load_s": load_s, "resident_mb": resident_mb,
                     "weights_alloc_mb": weights_alloc_mb, "per_key": per_key, "cache_mode": args.cache,
                     "cache_dir": str(cdir) if cdir else None, "recompile_limit_msgs": limit_watch.msgs,
                     "loadavg_end": os.getloadavg(), "smi_end": smi("clocks.sm,temperature.gpu,power.draw")})
        (out / f"{args.arm}_seq_p{args.proc_id}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    finally:
        if cleanup is not None:  # empty-cache sequences leave nothing behind
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
