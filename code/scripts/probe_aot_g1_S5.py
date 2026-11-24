"""S5 (2026-11-22): feasibility probe "AOT package + CUDA Graph" (AOT+G1 on the UNet only).  Exploration level: go/no-go only, numbers are not main results.

BACKGROUND (read from results/S4P/p5/REPORT_P5.txt l.29-34, scripts/aot_side_S4P_P5.py, p24/runners.py GraphRunner, torch 2.9.1 AOTI runtime headers):
  * The P5 side probe built G1 (per-component graphs: TE, UNet, VAE) on the embedded-weights AOT modules.  Default capture mode ("global") failed with
    "operation not permitted when stream is capturing"; with capture_error_mode="relaxed" 512x512 B1 no-CFG captured (steady 72.28 ms vs AOT 74.43 ms, pool 628 MiB) but every CFG shape
    failed with "operation not permitted on an event last recorded in a capturing stream" (failure catalogue #15).
  * Cause (our reading of torch/include/torch/csrc/inductor/aoti_runtime/model_container.h + model_base.h, torch 2.9.1): every AOTInductorModelContainer::run() takes a model instance
    from available_models_; the finished instance goes to pending_models_ after run, and AOTInductorModelBase::run() ends with cudaEventRecord(run_finished_, stream).  When no instance is
    available (default num_runners=1: always, right after a call), reclaim_finished_models() calls is_finished() = cudaEventQuery(run_finished_) on the previous run's event.
      - capture mode "global": cudaEventQuery is a capture-unsafe call -> "operation not permitted when stream is capturing" (hits every capture of an already-used container).
      - capture mode "relaxed": allowed for events from earlier NON-captured runs (warm-up) -> works if the container is called ONCE per capture (UNet, VAE, and TE without CFG).
      - CFG: the text encoder container is called TWICE inside one capture (uncond half, cond half; LCMPipe.encode splits them because fp16 kernels are batch-composition dependent);
        the 2nd call finds the instance "pending" with an event recorded INSIDE the capture -> cudaEventQuery -> "...event last recorded in a capturing stream".  (The P5 probe's own
        768x768 B4 CFG note guessed the same: TE called twice in one capture.)
      - Side effect that matters for this probe: after a container has been called inside a capture its run_finished_ event stays a "captured" event; any LATER uncaptured call of that
        container fails the same way.  Hence every attempt below loads a FRESH UNet container, and the uncaptured AOT reference (outputs, timing) uses a separate container.
    This script therefore captures ONLY the UNet container, once per capture (the per-step input batch, incl. the CFG concat, is built outside), TE/VAE stay outside the graph.
  * torch 2.9.1 has a LOAD-TIME option torch._inductor.aoti_load_package(path, run_single_threaded=True) ("useful to avoid conflicts with CUDAGraphs"): the container then runs through
    AOTInductorModelContainer::run_single_threaded -> model->run_single_threaded(), which skips available/pending model bookkeeping and the run_finished_ event entirely.  No rebuild.
    This option is NOT in the task list; it is added here as M5 (declared BEFORE any run).

METHODS (each x 3 shapes: 512x512 B1 no-CFG, 512x512 B1 CFG, 768x768 B4 CFG; method-major order; each attempt = its own child process, own timeout, NO retry):
  M1  UNet-only capture, 1 warm-up call on the current stream, capture_error_mode='global' (torch default)
  M2  M1 with capture_error_mode='relaxed'
  M3a UNet-only capture, standard recipe: 3 warm-up calls on a side stream (+ wait_stream), mode 'global'
  M3b M3a with mode 'relaxed'
  M4  does the package itself offer a cudagraph option (torch.inductor config triton.cudagraphs / aot_inductor.*)?  Parent-side introspection only (CPU): config values + signature of
      aoti_load_package + grep of the package's generated wrapper.cpp for cudagraph.  Not enabled/rebuilt in any case (packages are never rebuilt here).
  M5  (added) UNet container loaded with run_single_threaded=True, side-stream 3 warm-ups, mode 'global' (torch default; the cleanest variant)
  Per attempt recorded: status ok/fail, failing stage (load/warmup/capture/replay/compare/timing), error text, steady ms (median of 20 whole requests, TE + 4 UNet steps + VAE, wall clock
  with synchronize; baseline = same inputs through the uncaptured AOT arm AOTRunnerP5c, measured alone (20) and, when the capture works, interleaved with the graphed request (20+20,
  speed-up taken from the interleaved pair)), graph pool MiB (torch reserved growth during capture; device-level delta also stored), bit-identity of (a) the first UNet eps, (b) the final image
  vs the uncaptured AOT output of the same inputs (PROMPTS_DEFAULT[:B], seeds 7000+i, noise drawn in fp16).

DECISION LINE (written before the first run; the script evaluates exactly this and writes it to summary.json / verdict.txt):
  GO    if at least one method (M1, M2, M3a, M3b or M5) succeeds on all 3 shapes, with the final image bit-identical to the uncaptured AOT output on all 3 shapes, and the steady request time
        is >= 1% faster than the uncaptured AOT arm on 512x512 B1 no-CFG (interleaved median speed-up >= 1.0%; the other two shapes only need success + bit-identity).
  NO-GO otherwise, recorded as is (which method failed where and why).  M4 is information only and never counts for GO.
  Not a main result: a GO would only open a separate, pre-registered confirmation (not part of this probe); AOT numbers elsewhere are unchanged.

ESTIMATED GPU TIME (not measured): per child ~10-15 s start-up (pipeline load + 3 package loads + first generation) + request timing: 512 B1 ~2-4 s, 512 B1 CFG ~4-6 s, 768 B4 CFG ~25 s alone
  + ~45 s interleaved (only if the capture works).  15 children -> ~8 min GPU-active, ~12-15 min wall including process start-ups.  Device memory < ~6 GB per child (one process at a time).
  CPU: ~1 core per child; no compilation; loadavg impact negligible.  Run only in a GPU0 slot the main session grants; the parent refuses to start if another compute process sits on GPU0
  (--allow-busy overrides, then the numbers do not count as timing).
SAFETY: no pinned allocation >1 GiB (none is made); any pinned-memory failure text -> child exit 7 -> parent stops and exits 7; per-attempt timeout (--timeout-s, default 480 s) kills the child.
Usage (from code/):  python -B -m scripts.probe_aot_g1_S5 [--dry-run] [--methods M1,M2,M3a,M3b,M5] [--shapes 512x512x1x0,512x512x1x1,768x768x4x1]
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve()
ROOT = pathlib.Path(os.environ.get("PROJ_ROOT") or HERE.parents[2])
P24_CACHE = pathlib.Path(os.environ.get("P24_CACHE", "cache/p24"))
GPU0 = "GPU-722679a6-175b-cb56-b764-e8124d262c41"

SHAPES_DEFAULT = "512x512x1x0,512x512x1x1,768x768x4x1"
FIRST_SHAPE = "512x512x1x0"
# method -> (warm kind, warm count, capture_error_mode, run_single_threaded)
METHODS = {
    "M1": ("default", 1, "global", False),
    "M2": ("default", 1, "relaxed", False),
    "M3a": ("side", 3, "global", False),
    "M3b": ("side", 3, "relaxed", False),
    "M5": ("side", 3, "global", True),
}
METHOD_DOC = {
    "M1": "UNet-only capture, 1 warm-up (current stream), mode global",
    "M2": "UNet-only capture, 1 warm-up (current stream), mode relaxed",
    "M3a": "UNet-only capture, 3 warm-ups on a side stream, mode global",
    "M3b": "UNet-only capture, 3 warm-ups on a side stream, mode relaxed",
    "M4": "package-provided cudagraph option? (introspection only, no rebuild)",
    "M5": "UNet container loaded with run_single_threaded=True, 3 side-stream warm-ups, mode global (added; not in the task list)",
}
REPS = 20


def key_parse(k):
    h, w, b, c = (int(v) for v in k.split("x"))
    return h, w, b, bool(c)


def key_dir(k):
    h, w, b, c = key_parse(k)
    return f"{h}x{w}_b{b}_{'cfg' if c else 'nocfg'}"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg-root", default=str(P24_CACHE / "aot_p5c" / "pkg"))
    ap.add_argument("--out", default=str(ROOT / "results/S5/probe_aot_g1"))
    ap.add_argument("--shapes", default=SHAPES_DEFAULT)
    ap.add_argument("--methods", default="M1,M2,M3a,M3b,M5")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--timeout-s", type=float, default=480.0, help="per attempt (child process); killed on expiry, recorded, not retried")
    ap.add_argument("--mem-cap-mb", type=float, default=14336)
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print plan, check paths/packages, M4 introspection on CPU; no GPU")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--key", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--method", default=None, help=argparse.SUPPRESS)
    return ap


# ---------------------------------------------------------------------------------------------------- M4 (CPU introspection)
def probe_m4(pkg_root, shapes):
    r = {"method": "M4", "doc": METHOD_DOC["M4"], "rebuild_performed": False}
    try:
        import inspect
        import zipfile
        import torch
        import torch._inductor.config as ic
        from torch._inductor import aoti_load_package
        r["torch"] = torch.__version__
        r["inductor_config_triton_cudagraphs"] = getattr(ic.triton, "cudagraphs", "n/a")
        r["inductor_config_cudagraphs_env_TORCHINDUCTOR_CUDAGRAPHS"] = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS")
        ai = getattr(ic, "aot_inductor", None)
        r["aot_inductor_config_graph_keys"] = sorted(k for k in dir(ai) if "graph" in k.lower()) if ai is not None else "n/a"
        r["aoti_load_package_signature"] = str(inspect.signature(aoti_load_package))
        found = {}
        for s in shapes:
            p = pathlib.Path(pkg_root) / key_dir(s) / "unet.pt2"
            if not p.exists():
                found[s] = "package missing"
                continue
            with zipfile.ZipFile(p) as z:
                names = [n for n in z.namelist() if n.endswith("wrapper.cpp")]
                txt = z.read(names[0]).decode("utf8", "ignore") if names else ""
            found[s] = {"wrapper_cpp_bytes": len(txt), "mentions_cudagraph": len(re.findall(r"cuda_?graph", txt, flags=re.I)),
                        "mentions_cudaStreamSynchronize": len(re.findall(r"cudaStreamSynchronize|cudaDeviceSynchronize", txt))}
        r["unet_package_wrapper_scan"] = found
        r["conclusion"] = ("cudagraphs (inductor config triton.cudagraphs / cpp-wrapper cudagraphs) is a COMPILE-TIME option baked into the exported wrapper; "
                           "it cannot be switched on for an existing package without re-export and re-compile -> NOT done (no rebuild). "
                           "A load-time, rebuild-free knob exists: aoti_load_package(..., run_single_threaded=True) -> tested as M5.")
    except Exception as e:
        r["error"] = repr(e)[:500]
    return r


# ---------------------------------------------------------------------------------------------------- plan / dry run
def check_paths(a):
    shapes = a.shapes.split(",")
    miss = []
    for s in shapes:
        for c in ("te", "unet", "vae"):
            if not (pathlib.Path(a.pkg_root) / key_dir(s) / f"{c}.pt2").exists():
                miss.append(f"{key_dir(s)}/{c}.pt2")
    for p in (ROOT / "data/models/sd15_lcm_fused", ROOT / "code/scripts/aot_runtime_S4P_P5c.py"):
        if not p.exists():
            miss.append(str(p))
    return shapes, miss


def dry_run(a):
    shapes, miss = check_paths(a)
    methods = [m for m in a.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    print("== probe_aot_g1_S5 --dry-run (no GPU) ==")
    print("pkg_root :", a.pkg_root, "(exists)" if pathlib.Path(a.pkg_root).is_dir() else "(MISSING)")
    print("out      :", a.out, "| timeout/attempt:", a.timeout_s, "s | reps:", a.reps)
    print("shapes   :", shapes, "| missing files:", miss or "none")
    for s in shapes:
        h, w, b, c = key_parse(s)
        print(f"   {s}: dir={key_dir(s)} UNet batch={b * (2 if c else 1)} latent={h // 8}x{w // 8}")
    print("methods  :", [m for m in methods if m in METHODS], "| unknown:", bad or "none", "| M4 (introspection only, always run)")
    for m in methods:
        if m in METHODS:
            print(f"   {m}: {METHOD_DOC[m]}")
    print("children (GPU attempts):", len([m for m in methods if m in METHODS]) * len(shapes))
    print("estimated GPU time: ~8 min GPU-active, ~12-15 min wall (15 child processes, 768 B4 CFG dominates); device memory < ~6 GB per child")
    m4 = probe_m4(a.pkg_root, shapes)
    print("M4 introspection:", json.dumps(m4, ensure_ascii=False)[:900])
    print("decision line: GO iff >=1 method succeeds on all 3 shapes + bit-identical image on all 3 + interleaved speed-up >= 1% on 512x512 B1 no-CFG; else NO-GO")
    return 3 if (miss or bad) else 0


# ---------------------------------------------------------------------------------------------------- child: one (shape, method) attempt
PIN_WORDS = ("pinned", "cudahostalloc", "cudahostregister", "pin_memory")


def child(a):
    key_s, mname = a.key, a.method
    warm_kind, warm_n, cmode, single = METHODS[mname]
    out = pathlib.Path(a.out) / "attempts"
    out.mkdir(parents=True, exist_ok=True)
    res_file = out / f"{key_s}__{mname}.json"
    if res_file.exists():
        sys.exit(f"refusing to overwrite {res_file}")
    sys.path.insert(0, str(ROOT / "code"))
    import torch
    from torch._inductor import aoti_load_package
    from p24 import bench
    from p24.pipeline import LCMPipe, ShapeKey
    from scripts import aot_common_S4P_P5 as C
    from scripts import aot_common_S4P_P5b as W
    from scripts import aot_runtime_S4P_P5c as R5

    res = {"key": key_s, "method": mname, "doc": METHOD_DOC[mname], "warm": [warm_kind, warm_n], "capture_error_mode": cmode, "run_single_threaded": single,
           "status": "fail", "stage": "init", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_sha16": hashlib.sha256(HERE.read_bytes()).hexdigest()[:16],
           "torch": torch.__version__, "gpu0_other_procs_start": bench.gpu0_other_procs(), "loadavg_start": os.getloadavg(),
           "smi_start": bench.smi("clocks.sm,temperature.gpu,power.draw,utilization.gpu"),
           "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "P24_PINNED_FALLBACK")}}
    rc = 0
    med = statistics.median
    try:
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_mb / (torch.cuda.get_device_properties(0).total_memory / 2**20)), 0)
        h, w, b, cfg = key_parse(key_s)
        key = ShapeKey(h, w, b, cfg)
        with torch.inference_mode():
            res["stage"] = "baseline_setup"
            pipe = LCMPipe(str(ROOT / "data/models/sd15_lcm_fused"))
            runner = R5.AOTRunnerP5c(pipe, a.pkg_root)
            inp = pipe.make_inputs(bench.PROMPTS_DEFAULT[:b], key, seeds=[7000 + i for i in range(b)], noise_dtype=torch.float16)
            runner.prepare(key, inp)                      # loads TE/UNet/VAE containers (normal mode) + first generation
            m = runner.mods[key]
            ref_img = runner._call(inp, key).clone()
            torch.cuda.synchronize()
            x0 = torch.cat([inp.latents] * 2) if cfg else inp.latents
            emb0 = runner._emb(m, inp, key)
            eps_ref = C._first(m["unet"](x0, pipe.t_bufs[0], emb0)).clone()
            # ---- baseline alone ----
            res["stage"] = "baseline_timing"

            def req_base():
                return runner._call(inp, key)

            def wall(fn):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) * 1e3
            for _ in range(3):
                req_base()
            base_alone = [wall(req_base) for _ in range(a.reps)]
            res["baseline_alone_ms"] = {"median": med(base_alone), "min": min(base_alone), "max": max(base_alone), "n": len(base_alone)}
            # ---- fresh UNet container for the capture attempt ----
            res["stage"] = "load"
            t0 = time.perf_counter()
            um = W.WFIMod(aoti_load_package(str(W.wfi_pkg_path(a.pkg_root, key, "unet")), run_single_threaded=single), runner.ws["unet"])
            torch.cuda.synchronize()
            res["load_fresh_unet_s"] = time.perf_counter() - t0
            sx = torch.empty_like(x0)
            st = pipe.t_bufs[0].clone()
            se = torch.empty_like(emb0)
            sx.copy_(x0); st.copy_(pipe.t_bufs[0]); se.copy_(emb0)
            torch.cuda.synchronize()
            res["stage"] = "warmup"
            if warm_kind == "default":
                for _ in range(warm_n):
                    C._first(um(sx, st, se))
            else:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warm_n):
                        C._first(um(sx, st, se))
                torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            res["stage"] = "capture"
            torch.cuda.empty_cache()
            r0 = torch.cuda.memory_reserved()
            f0, tot = torch.cuda.mem_get_info()
            g = torch.cuda.CUDAGraph()
            tc = time.perf_counter()
            with torch.cuda.graph(g, capture_error_mode=cmode):
                static_eps = C._first(um(sx, st, se))
            torch.cuda.synchronize()
            res["capture_s"] = time.perf_counter() - tc
            res["graph_pool_mib"] = (torch.cuda.memory_reserved() - r0) / 2**20
            res["device_delta_mib"] = (f0 - torch.cuda.mem_get_info()[0]) / 2**20
            res["capture_ok"] = 1
            # ---- replay check ----
            res["stage"] = "replay"
            sx.copy_(x0); st.copy_(pipe.t_bufs[0]); se.copy_(emb0)
            g.replay()
            torch.cuda.synchronize()
            res["eps_step0_bit_identical"] = int(torch.equal(static_eps, eps_ref))
            res["eps_step0_maxabs"] = (static_eps.float() - eps_ref.float()).abs().max().item()

            def req_graph():
                emb = runner._emb(m, inp, key)
                se.copy_(emb)
                lat = inp.latents
                for i in range(pipe.steps):
                    x = torch.cat([lat] * 2) if cfg else lat
                    sx.copy_(x)
                    st.copy_(pipe.t_bufs[i])
                    g.replay()
                    lat = pipe.step(i, static_eps, lat, cfg, inp.noises[i] if i < pipe.steps - 1 else None)
                img = C._first(m["vae"](lat / pipe.vae.config.scaling_factor))
                return (img / 2 + 0.5).clamp(0, 1)
            res["stage"] = "compare"
            img_g = req_graph().clone()
            torch.cuda.synchronize()
            res["image_bit_identical"] = int(torch.equal(img_g, ref_img))
            res["image_maxabs_vs_uncaptured_aot"] = (img_g.float() - ref_img.float()).abs().max().item()
            res["image_frac_pixels_equal"] = (img_g == ref_img).float().mean().item()
            # ---- interleaved timing ----
            res["stage"] = "timing"
            for _ in range(3):
                req_graph()
                req_base()
            tb, tg = [], []
            for _ in range(a.reps):
                tb.append(wall(req_base))
                tg.append(wall(req_graph))
            res["interleaved_baseline_ms"] = {"median": med(tb), "min": min(tb), "max": max(tb), "n": len(tb)}
            res["graphed_ms"] = {"median": med(tg), "min": min(tg), "max": max(tg), "n": len(tg)}
            res["speedup_pct"] = (med(tb) - med(tg)) / med(tb) * 100.0
            res["status"] = "ok"
            res["stage"] = "done"
    except SystemExit:
        raise
    except BaseException as e:  # record, no retry
        txt = repr(e) + " | " + traceback.format_exc()[-1500:]
        res["error"] = txt[:2500]
        low = txt.lower()
        res["error_class"] = ("captured_event" if "event last recorded in a capturing stream" in low else
                              "unsafe_during_capture" if "when stream is capturing" in low else "other")
        if any(wd in low for wd in PIN_WORDS):
            rc = 7
        try:
            torch.cuda.synchronize()
            res["context_ok_after"] = 1
        except Exception as e2:
            res["context_ok_after"] = 0
            res["context_error"] = repr(e2)[:300]
    try:
        res["gpu0_other_procs_end"] = bench.gpu0_other_procs()
        res["loadavg_end"] = os.getloadavg()
    except Exception:
        pass
    res_file.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res.get(k) for k in ("key", "method", "status", "stage", "error_class", "speedup_pct", "image_bit_identical", "graph_pool_mib")}, default=str), flush=True)
    return rc


# ---------------------------------------------------------------------------------------------------- parent
def gpu0_busy():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return -1
    return sum(1 for l in o.strip().splitlines() if l.startswith(GPU0))


def evaluate(attempts, shapes, methods):
    """Decision line (header).  attempts: {(method, key): dict}."""
    per = {}
    for mname in methods:
        rows = [attempts.get((mname, s)) for s in shapes]
        ok3 = all(r is not None and r.get("status") == "ok" for r in rows)
        bit3 = ok3 and all(r.get("image_bit_identical") == 1 for r in rows)
        r512 = attempts.get((mname, FIRST_SHAPE))
        sp = r512.get("speedup_pct") if r512 and r512.get("status") == "ok" else None
        per[mname] = {"success_all_3": int(ok3), "bit_identical_all_3": int(bit3), "speedup_pct_512x512x1x0": sp,
                      "qualifies": int(bit3 and sp is not None and sp >= 1.0),
                      "per_shape": {s: {"status": (attempts.get((mname, s)) or {}).get("status", "not_run"),
                                        "error_class": (attempts.get((mname, s)) or {}).get("error_class"),
                                        "speedup_pct": (attempts.get((mname, s)) or {}).get("speedup_pct"),
                                        "image_bit_identical": (attempts.get((mname, s)) or {}).get("image_bit_identical"),
                                        "graph_pool_mib": (attempts.get((mname, s)) or {}).get("graph_pool_mib")} for s in shapes}}
    go = any(v["qualifies"] for v in per.values())
    return {"verdict": "GO" if go else "NO-GO", "methods_qualifying": [k for k, v in per.items() if v["qualifies"]], "per_method": per}


def parent(a):
    shapes, miss = check_paths(a)
    methods = [m for m in a.methods.split(",") if m]
    if miss or any(m not in METHODS for m in methods):
        sys.exit(f"missing files or unknown methods: {miss} {[m for m in methods if m not in METHODS]}")
    out = pathlib.Path(a.out)
    if (out / "attempts").exists() and any((out / "attempts").iterdir()):
        sys.exit(f"{out}/attempts is not empty; refusing to overwrite")
    busy = gpu0_busy()
    if busy != 0 and not a.allow_busy:
        sys.exit(f"GPU0 has {busy} compute process(es) (or nvidia-smi failed); not starting (use --allow-busy only for non-timing runs)")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "attempts").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    m4 = probe_m4(a.pkg_root, shapes)
    (out / "M4_introspection.json").write_text(json.dumps(m4, indent=1, ensure_ascii=False))
    attempts = {}
    exit_code = 0
    for mname in methods:
        for s in shapes:
            log = out / "logs" / f"{s}__{mname}.log"
            cmd = [sys.executable, "-B", "-m", "scripts.probe_aot_g1_S5", "--child", "--key", s, "--method", mname, "--pkg-root", a.pkg_root,
                   "--out", str(out), "--reps", str(a.reps), "--mem-cap-mb", str(a.mem_cap_mb)]
            t0 = time.time()
            rf = out / "attempts" / f"{s}__{mname}.json"
            with open(log, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT / "code"), env=dict(os.environ))
                try:
                    rc = proc.wait(timeout=a.timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    rc = "timeout"
            print(f"[{time.strftime('%H:%M:%S')}] {mname} {s}: rc={rc} {time.time() - t0:.0f}s", flush=True)
            if rc == 7:
                exit_code = 7
            if rf.exists():
                attempts[(mname, s)] = json.loads(rf.read_text())
            else:
                tail = ""
                try:
                    tail = log.read_text()[-1200:]
                except Exception:
                    pass
                attempts[(mname, s)] = {"key": s, "method": mname, "status": "timeout" if rc == "timeout" else "crash", "error_class": "timeout_or_crash", "rc": str(rc),
                                        "log_tail": tail}
                rf.write_text(json.dumps(attempts[(mname, s)], indent=1))
            if exit_code == 7:
                break
        if exit_code == 7:
            print("pinned-memory failure -> stopping (exit 7), no retry", flush=True)
            break
    ev = evaluate(attempts, shapes, [m for m in methods])
    ev.update(M4=m4, shapes=shapes, methods=methods, wall_s=time.time() - t_start, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
              script_sha16=hashlib.sha256(HERE.read_bytes()).hexdigest()[:16],
              note="decision line from the header of scripts/probe_aot_g1_S5.py; exploration level, not a main result")
    (out / "summary.json").write_text(json.dumps(ev, indent=1, default=str, ensure_ascii=False))
    import csv
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "key", "status", "stage", "error_class", "baseline_alone_ms", "interleaved_baseline_ms", "graphed_ms", "speedup_pct", "graph_pool_mib",
                     "eps_step0_bit_identical", "image_bit_identical", "image_maxabs_vs_uncaptured_aot", "error"])
        for (mname, s), r in attempts.items():
            g = lambda k, sub="median": (r.get(k) or {}).get(sub) if isinstance(r.get(k), dict) else r.get(k)
            wr.writerow([mname, s, r.get("status"), r.get("stage"), r.get("error_class"), g("baseline_alone_ms"), g("interleaved_baseline_ms"), g("graphed_ms"),
                         r.get("speedup_pct"), r.get("graph_pool_mib"), r.get("eps_step0_bit_identical"), r.get("image_bit_identical"),
                         r.get("image_maxabs_vs_uncaptured_aot"), (r.get("error") or "")[:300].replace("\n", " ")])
    lines = [f"VERDICT: {ev['verdict']}  (qualifying methods: {ev['methods_qualifying'] or 'none'})", "decision line: see header of scripts/probe_aot_g1_S5.py", ""]
    for mname, v in ev["per_method"].items():
        lines.append(f"{mname}: success_all_3={v['success_all_3']} bit_identical_all_3={v['bit_identical_all_3']} speedup@512x512x1x0={v['speedup_pct_512x512x1x0']} qualifies={v['qualifies']}")
        for s, x in v["per_shape"].items():
            lines.append(f"    {s}: {x}")
    lines.append(f"M4: {m4.get('conclusion', m4.get('error'))}")
    (out / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))
    return exit_code


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.dry_run:
        sys.exit(dry_run(args))
    if args.child:
        sys.exit(child(args))
    sys.exit(parent(args))
