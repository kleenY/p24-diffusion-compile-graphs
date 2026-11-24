"""S5 (2026-11-22): extended output-equivalence check of the AOT arm (B1-AOT, the weight-free AOTInductor packages of P5b/P5c).  DESCRIBES, does not judge.

WHY: the P5 equivalence check (scripts/aot_equiv_S4P_P5.py -> results/S4P/p5/step1/equiv/equiv_AOT.json) covered only 2 shapes / 8 images, and the P5b/P5c quick check
(aot_equiv_S4P_P5b/_P5c) compared AOT only with C1 (coarse RMSE<=0.01 line) and, for 12 shapes, with the S2 fp32 references.  AOT and C1 are NOT bit-identical.
This script samples 10 of the 32 built packages and, for each, compares AOT with (a) fp16 eager E0, (b) C1 (torch.compile default), (c) the fp32 eager reference.

SAME CONVENTION AS P5 (frozen, S1/X1): LCMPipe from data/models/sd15_lcm_fused, 4 steps, noise drawn in fp16 (noise_dtype=float16, per-sample CPU generators, so the noise of
sample i depends only on its seed), images in [0,1] float32 on the host, p24.equiv.compare(x, e0, ref32): RMSE ratio = RMSE(x, ref32) / RMSE(E0, ref32) (pooled over the shape's
images), LPIPS(alex)(x, E0).  Tolerances are READ from code/tests/tolerances.json (c_rmse_ratio_max = 2, lpips_vs_E0_mean_max = 0.02), unchanged; the flags in the output are
descriptive (rmse_ratio_le_c / lpips_le_tol), there is NO overall verdict.  Also reported: bit-identical (whole shape and per-image rate), fraction of exactly equal pixels, max|diff|.
The AOT arm is the SERVING runner of the replay (scripts.aot_runtime_S4P_P5c.AOTRunnerP5c, weight-free packages in $P24_CACHE/aot_p5c/pkg, the one live fp16 weight copy).

PROMPTS: p24.bench.PROMPTS_DEFAULT[:4] (the same source as P5: P5 used PROMPTS_DEFAULT[:8], seed 7000+i).  SEEDS: two per prompt: 7000+i (identical to the P5 seeds
of those prompts) and 8000+i (new).  = 8 images per shape.  Batches of key.B in the order [(p0,7000),(p1,7001),(p2,7002),(p3,7003),(p0,8000),...,(p3,8003)]; when B does not divide 8
the last batch is padded by repeating items from the start (padded slots are dropped; every arm sees the identical padded batch).

SHAPE SAMPLING (fixed before any run): candidates = the packages in --pkg-root with te/unet/vae present, minus the shapes in the P5c fallback list (results/S4P/p5c/aot_fallback_shapes.csv:
512x512x3x0, equivalence-failed under the frozen criterion, served eagerly -> not an AOT-served shape; reported in the json as excluded).  The largest shape 768x768 B4 CFG is
always in (if its package exists).  Then greedy cover of {4 resolutions} x {B=1..4} x {no CFG, CFG} (all values of each factor, not the cross product) in the order of a
random.Random(20261122).shuffle() of the sorted candidate tags (each pick = first candidate in that order covering the most still-uncovered factor values), then fill up to 10
from the same shuffled order.  The result is printed by --dry-run and stored in the json.

OUTPUT: results/S5/equiv_aot_ext/equiv_aot_ext_S5{tag}.json (everything) + equiv_aot_ext_S5{tag}.csv (one row per shape x pair); refuses to overwrite.  Images are cached (rebuildable, float32 .pt,
~60 MB per shape and arm at 768x768) in --cache-dir = $P24_CACHE/s5_equiv_aot_ext -> delete after the report is signed.

PHASES (--phases, default all; each phase caches per shape so a later call resumes): ref32 -> e0 -> aot -> c1 -> metrics.  GPU0 only (CUDA_VISIBLE_DEVICES set by the caller).
ESTIMATED GPU TIME (not measured): ~1 min fp32 references + ~0.5 min E0 + ~1.5 min AOT (load ~2 s per shape + generation) + C1: ~1.5-3 min if the shared Inductor disk cache is
warm for these shapes, 8-12 min if cold (each cold shape ~45-65 s; compile uses TORCHINDUCTOR_COMPILE_THREADS=8 CPU threads) + metrics ~0.5 min + 3 model loads ~0.5 min.
=> ~6 min (warm) .. ~15 min (cold).  Device memory < ~8 GB (one pipeline at a time; the fp32 768x768 B4 CFG reference is the peak).  No pinned allocation >1 GiB;
any pinned-memory failure -> immediate exit 7, no retry.
Usage (from code/):  python -B -m scripts.equiv_AOT_ext_S5 [--dry-run] [--phases ref32,e0,aot,c1,metrics] [--skip-c1] [--tag _x]
"""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve()
ROOT = pathlib.Path(os.environ.get("PROJ_ROOT") or HERE.parents[2])
P24_CACHE = pathlib.Path(os.environ.get("P24_CACHE", "cache/p24"))

RESOLUTIONS = [(512, 512), (512, 768), (768, 512), (768, 768)]
MUST_TAG = "768x768_b4_cfg"
N_PROMPTS = 4
SEED_BASES = (7000, 8000)


def tag_parse(tag):
    """'512x768_b3_cfg' -> (512, 768, 3, True)"""
    res, b, c = tag.split("_")
    h, w = (int(v) for v in res.split("x"))
    return h, w, int(b[1:]), c == "cfg"


def tag_make(h, w, b, cfg):
    return f"{h}x{w}_b{b}_{'cfg' if cfg else 'nocfg'}"


def tag_hwbc(tag):
    h, w, b, c = tag_parse(tag)
    return f"{h}x{w}x{b}x{int(c)}"


def read_fallback(path):
    out = {}
    p = pathlib.Path(path)
    if not p.exists():
        return out
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            h, w, b, c = (int(v) for v in r["key"].strip().split("x"))
            out[tag_make(h, w, b, bool(c))] = r.get("reason", "listed")
    return out


def candidates(pkg_root, fallback):
    root = pathlib.Path(pkg_root)
    cand, excluded, incomplete = [], {}, []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir():
            continue
        try:
            tag_parse(d.name)
        except Exception:
            continue
        if not all((d / f"{c}.pt2").exists() for c in ("te", "unet", "vae")):
            incomplete.append(d.name)
        elif d.name in fallback:
            excluded[d.name] = fallback[d.name]
        else:
            cand.append(d.name)
    return cand, excluded, incomplete


def select_shapes(cand, seed, n):
    """Deterministic sample: must-include largest shape, greedy factor cover in seeded shuffle order, fill from the same order."""
    order = sorted(cand)
    random.Random(seed).shuffle(order)
    sel = [MUST_TAG] if MUST_TAG in cand else []
    cover = {"res": set(), "B": set(), "cfg": set()}
    want = {"res": set(RESOLUTIONS), "B": {1, 2, 3, 4}, "cfg": {False, True}}

    def facts(t):
        h, w, b, c = tag_parse(t)
        return {"res": (h, w), "B": b, "cfg": c}

    def add(t):
        for k, v in facts(t).items():
            cover[k].add(v)

    def gain(t):
        return sum(1 for k, v in facts(t).items() if v in want[k] and v not in cover[k])
    for t in sel:
        add(t)
    while any(want[k] - cover[k] for k in want) and len(sel) < n:
        rest = [t for t in order if t not in sel]
        if not rest:
            break
        best = max(rest, key=lambda t: (gain(t), -order.index(t)))
        if gain(best) == 0:
            break
        sel.append(best)
        add(best)
    for t in order:
        if len(sel) >= n:
            break
        if t not in sel:
            sel.append(t)
    uncovered = {k: sorted(map(str, want[k] - cover[k])) for k in want if want[k] - cover[k]}
    return sel[:n], uncovered


def make_items():
    """[(prompt_idx, seed)] x 8: seeds 7000+i then 8000+i for prompts i = 0..3."""
    return [(i, base + i) for base in SEED_BASES for i in range(N_PROMPTS)]


def batches(items, B):
    """list of (batch_items, n_real): last batch padded cyclically from the start."""
    out = []
    for g0 in range(0, len(items), B):
        real = items[g0:g0 + B]
        pad = [items[(j) % len(items)] for j in range(B - len(real))]
        out.append((real + pad, len(real)))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg-root", default=str(P24_CACHE / "aot_p5c" / "pkg"))
    ap.add_argument("--fallback-csv", default=str(ROOT / "results/S4P/p5c/aot_fallback_shapes.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/S5/equiv_aot_ext"))
    ap.add_argument("--cache-dir", default=str(P24_CACHE / "s5_equiv_aot_ext"))
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--n-shapes", type=int, default=10)
    ap.add_argument("--phases", default="ref32,e0,aot,c1,metrics")
    ap.add_argument("--skip-c1", action="store_true", help="drop the c1 phase and the AOT-vs-C1 pair")
    ap.add_argument("--mem-cap-mb", type=float, default=14336)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="print plan only; no torch import, no GPU")
    return ap


def plan(a):
    fb = read_fallback(a.fallback_csv)
    cand, excluded, incomplete = candidates(a.pkg_root, fb)
    sel, unc = select_shapes(cand, a.seed, a.n_shapes)
    return {"n_candidates": len(cand), "fallback_excluded": excluded, "incomplete_packages": incomplete, "selected": sel,
            "uncovered_factor_values": unc, "items": make_items(), "n_images_per_shape": len(make_items())}


def dry_run(a):
    pl = plan(a)
    phases = [p for p in a.phases.split(",") if p and not (a.skip_c1 and p == "c1")]
    print("== equiv_AOT_ext_S5 --dry-run (no torch, no GPU) ==")
    print("pkg_root     :", a.pkg_root, "(exists)" if pathlib.Path(a.pkg_root).is_dir() else "(MISSING)")
    print("fallback csv :", a.fallback_csv, "(exists)" if pathlib.Path(a.fallback_csv).exists() else "(MISSING)")
    print("tolerances   :", ROOT / "code/tests/tolerances.json", "(exists)" if (ROOT / "code/tests/tolerances.json").exists() else "(MISSING)")
    print("model dir    :", ROOT / "data/models/sd15_lcm_fused", "(exists)" if (ROOT / "data/models/sd15_lcm_fused").is_dir() else "(MISSING)")
    print("out dir      :", a.out, "| cache dir:", a.cache_dir)
    print("candidates   :", pl["n_candidates"], "| excluded (fallback list):", pl["fallback_excluded"], "| incomplete:", pl["incomplete_packages"])
    print(f"selected {len(pl['selected'])} shapes (seed {a.seed}):")
    for t in pl["selected"]:
        ok = all((pathlib.Path(a.pkg_root) / t / f"{c}.pt2").exists() for c in ("te", "unet", "vae"))
        print(f"   {tag_hwbc(t):14s} dir={t:16s} te/unet/vae={'ok' if ok else 'MISSING'}  batches={len(batches(pl['items'], tag_parse(t)[2]))}")
    print("uncovered factor values after sampling:", pl["uncovered_factor_values"] or "none (all 4 resolutions, B=1..4, CFG on/off covered)")
    print("images per shape:", pl["n_images_per_shape"], "(4 prompts x seeds", [f"{b}+i" for b in SEED_BASES], ")  prompts = PROMPTS_DEFAULT[:4]")
    print("phases:", phases, "| pairs: AOT-vs-E0, AOT-vs-ref32" + ("" if a.skip_c1 else ", AOT-vs-C1") + " (+ frozen-convention block, descriptive)")
    print("estimated GPU time: ~6 min (warm Inductor cache for C1) .. ~15 min (cold C1 compile, +8 CPU threads); device memory < ~8 GB")
    return 0


# ----------------------------------------------------------------------------------------------------------------------
def main(a):
    import torch
    sys.path.insert(0, str(ROOT / "code"))
    from p24 import bench, compat, equiv  # noqa: F401  (compat: P24_PINNED_FALLBACK=1 when set in the env)
    from p24.pipeline import LCMPipe, ShapeKey

    phases = [p for p in a.phases.split(",") if p and not (a.skip_c1 and p == "c1")]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(a.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    res_json, res_csv = out / f"equiv_aot_ext_S5{a.tag}.json", out / f"equiv_aot_ext_S5{a.tag}.csv"
    if "metrics" in phases and (res_json.exists() or res_csv.exists()):
        sys.exit(f"refusing to overwrite {res_json} / {res_csv}")
    pl = plan(a)
    if not pl["selected"]:
        sys.exit("no candidate package found")
    plan_file = cache / "plan.json"
    if plan_file.exists():
        if json.loads(plan_file.read_text())["selected"] != pl["selected"] or json.loads(plan_file.read_text())["items"] != [list(x) for x in pl["items"]]:
            sys.exit(f"cache dir {cache} belongs to a different plan; use another --cache-dir")
    else:
        plan_file.write_text(json.dumps({"selected": pl["selected"], "items": pl["items"]}))
    TOL = json.loads((ROOT / "code/tests/tolerances.json").read_text())
    model_dir = str(ROOT / "data/models/sd15_lcm_fused")
    keys = {t: ShapeKey(*tag_parse(t)) for t in pl["selected"]}
    items = pl["items"]
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_mb / (torch.cuda.get_device_properties(0).total_memory / 2**20)), 0)
    errors, timing = {}, {}

    def pinned_guard(e):
        s = (repr(e) + traceback.format_exc()).lower()
        if any(w in s for w in ("pinned", "cudahostalloc", "cudahostregister", "pin_memory")):
            print("PINNED-MEMORY FAILURE -> exit 7 (no retry):", repr(e)[:300], file=sys.stderr, flush=True)
            sys.exit(7)

    def gen(p, key, run_fn, prepare_fn=None):
        imgs = []
        for bi, (bitems, n_real) in enumerate(batches(items, key.B)):
            inp = p.make_inputs([bench.PROMPTS_DEFAULT[i] for i, _ in bitems], key, seeds=[s for _, s in bitems], noise_dtype=torch.float16)
            if bi == 0 and prepare_fn is not None:
                prepare_fn(inp)
            imgs.append(run_fn(inp)[:n_real].float().cpu())
        return torch.cat(imgs)

    def cpath(phase, tag):
        return cache / f"{phase}_{tag}.pt"

    def todo(phase):
        return [t for t in pl["selected"] if not cpath(phase, t).exists()]

    meta = {"script": str(HERE), "script_sha16": hashlib.sha256(HERE.read_bytes()).hexdigest()[:16], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": sys.argv, "torch": torch.__version__, "phases": phases, "pkg_root": a.pkg_root, "plan": pl,
            "tolerances": {k: TOL[k] for k in ("c_rmse_ratio_max", "c_psnr_drop_db_max", "lpips_vs_E0_mean_max")},
            "gpu0_other_procs_start": bench.gpu0_other_procs(), "loadavg_start": os.getloadavg(),
            "env": {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_VISIBLE_DEVICES", "P24_PINNED_FALLBACK")}}

    # ---- phase ref32 ----
    if "ref32" in phases and todo("ref32"):
        t0 = time.perf_counter()
        p32 = LCMPipe(model_dir, dtype=torch.float32)
        for t in todo("ref32"):
            try:
                with torch.inference_mode():
                    torch.save(gen(p32, keys[t], lambda inp, k=keys[t]: p32.run(inp, k.cfg)), cpath("ref32", t))
            except Exception as e:
                pinned_guard(e)
                errors[f"ref32:{t}"] = repr(e)[:500]
        del p32
        torch.cuda.empty_cache()
        timing["ref32_s"] = time.perf_counter() - t0
    # ---- phases e0 + aot (same fp16 pipeline object) ----
    if ("e0" in phases and todo("e0")) or ("aot" in phases and todo("aot")):
        from scripts import aot_runtime_S4P_P5c as R5
        t0 = time.perf_counter()
        p16 = LCMPipe(model_dir)
        if "e0" in phases:
            for t in todo("e0"):
                try:
                    with torch.inference_mode():
                        torch.save(gen(p16, keys[t], lambda inp, k=keys[t]: p16.run(inp, k.cfg)), cpath("e0", t))
                except Exception as e:
                    pinned_guard(e)
                    errors[f"e0:{t}"] = repr(e)[:500]
        if "aot" in phases:
            aot = R5.AOTRunnerP5c(p16, a.pkg_root)
            for t in todo("aot"):
                k = keys[t]
                try:
                    imgs = gen(p16, k, lambda inp, k=k: aot.run(inp, k), prepare_fn=lambda inp, k=k: aot.prepare(k, inp))
                    torch.save(imgs, cpath("aot", t))
                except Exception as e:
                    pinned_guard(e)
                    errors[f"aot:{t}"] = repr(e)[:500] + " | " + traceback.format_exc()[-400:]
                finally:
                    aot.release(k)
            del aot
        del p16
        torch.cuda.empty_cache()
        timing["e0_aot_s"] = time.perf_counter() - t0
    # ---- phase c1 ----
    if "c1" in phases and todo("c1"):
        from p24.runners import make_runner
        t0 = time.perf_counter()
        c1 = make_runner("C1", model_dir)
        timing["c1_per_shape_s"] = {}
        for t in todo("c1"):
            k = keys[t]
            ts = time.perf_counter()
            try:
                imgs = gen(c1.p, k, lambda inp, k=k: c1.run(inp, k), prepare_fn=lambda inp, k=k: (c1.prepare(k, inp) if not c1.has(k) else None))
                torch.save(imgs, cpath("c1", t))
            except Exception as e:
                pinned_guard(e)
                errors[f"c1:{t}"] = repr(e)[:500] + " | " + traceback.format_exc()[-400:]
            timing["c1_per_shape_s"][t] = time.perf_counter() - ts
        del c1
        torch.cuda.empty_cache()
        timing["c1_s"] = time.perf_counter() - t0
    if "metrics" not in phases:
        print(json.dumps({"phases_done": phases, "errors": errors, "timing": timing}, default=str))
        return 0

    # ---- metrics ----
    def pair(x, y):
        xs, ys = x.float(), y.float()
        per_rmse = [equiv.rmse(xs[i], ys[i]) for i in range(xs.shape[0])]
        per_lp = [equiv.lpips_mean(xs[i:i + 1].cuda(), ys[i:i + 1].cuda()) for i in range(xs.shape[0])]
        n_id = sum(int(torch.equal(xs[i], ys[i])) for i in range(xs.shape[0]))
        return {"rmse": equiv.rmse(xs, ys), "maxabs": (xs - ys).abs().max().item(), "lpips": equiv.lpips_mean(xs.cuda(), ys.cuda()),
                "bit_identical_all": int(torch.equal(xs, ys)), "n_images": int(xs.shape[0]), "n_images_bit_identical": n_id,
                "bit_identical_rate": n_id / xs.shape[0], "frac_pixels_equal": (xs == ys).float().mean().item(),
                "per_image_rmse": per_rmse, "per_image_lpips": per_lp, "max_image_lpips": max(per_lp), "max_image_rmse": max(per_rmse)}

    def frozen_block(x, e0, ref):
        m = equiv.compare(x.cuda(), e0.cuda(), ref.cuda())
        m["rmse_ratio_le_c"] = int(m["rmse_ratio"] <= TOL["c_rmse_ratio_max"])
        m["lpips_le_tol"] = int(m["lpips_vs_E0"] <= TOL["lpips_vs_E0_mean_max"])
        e0r = [equiv.rmse(e0[i], ref[i]) for i in range(e0.shape[0])]
        xr = [equiv.rmse(x[i], ref[i]) for i in range(x.shape[0])]
        m["per_image_rmse_ratio"] = [(a_ / b_ if b_ > 0 else float("nan")) for a_, b_ in zip(xr, e0r)]
        return m

    rows, by_shape = [], {}
    for t in pl["selected"]:
        got = {ph: torch.load(cpath(ph, t)) for ph in ("ref32", "e0", "aot", "c1") if cpath(ph, t).exists()}
        r = {"key": tag_hwbc(t), "dir": t, "have": sorted(got)}
        if "aot" not in got:
            r["note"] = "no AOT images (see errors)"
            by_shape[t] = r
            continue
        x = got["aot"]
        for nm, ref in (("aot_vs_e0", "e0"), ("aot_vs_ref32", "ref32"), ("aot_vs_c1", "c1")):
            if ref in got:
                r[nm] = pair(x, got[ref])
                rows.append({"key": tag_hwbc(t), "pair": nm, **{k: v for k, v in r[nm].items() if not k.startswith("per_image")}})
        if "e0" in got and "ref32" in got:
            r["frozen_convention_aot"] = frozen_block(x, got["e0"], got["ref32"])
            if "c1" in got:
                r["frozen_convention_c1"] = frozen_block(got["c1"], got["e0"], got["ref32"])
        if "e0" in got and "c1" in got:
            r["c1_vs_e0"] = pair(got["c1"], got["e0"])
        by_shape[t] = r
    meta.update(timing=timing, errors=errors, gpu0_other_procs_end=bench.gpu0_other_procs(), loadavg_end=os.getloadavg(),
                time_end=time.strftime("%Y-%m-%dT%H:%M:%S"))
    summary = {}
    fa = [v["frozen_convention_aot"] for v in by_shape.values() if "frozen_convention_aot" in v]
    if fa:
        summary["n_shapes_with_frozen_block"] = len(fa)
        summary["max_rmse_ratio"] = max(f["rmse_ratio"] for f in fa)
        summary["max_lpips_vs_E0"] = max(f["lpips_vs_E0"] for f in fa)
        summary["n_rmse_ratio_le_c"] = sum(f["rmse_ratio_le_c"] for f in fa)
        summary["n_lpips_le_tol"] = sum(f["lpips_le_tol"] for f in fa)
    for nm in ("aot_vs_e0", "aot_vs_c1", "aot_vs_ref32"):
        v = [s[nm] for s in by_shape.values() if nm in s]
        if v:
            summary[nm] = {"n_shapes": len(v), "n_shapes_bit_identical_all": sum(x_["bit_identical_all"] for x_ in v),
                           "image_bit_identical_rate": sum(x_["n_images_bit_identical"] for x_ in v) / sum(x_["n_images"] for x_ in v),
                           "max_lpips": max(x_["lpips"] for x_ in v), "max_rmse": max(x_["rmse"] for x_ in v), "max_maxabs": max(x_["maxabs"] for x_ in v)}
    res = {"meta": meta, "summary": summary, "by_shape": by_shape, "note": "descriptive only; no overall verdict (equiv_AOT_ext_S5 header)"}
    res_json.write_text(json.dumps(res, indent=1, default=str))
    keys_csv = sorted({k for r_ in rows for k in r_})
    with open(res_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_csv)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"summary": summary, "errors": errors, "out": str(res_json)}, default=str))
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(dry_run(args) if args.dry_run else main(args))
