"""S4-X X4: GenEval supplementary image generation (new file; gen_geneval_S4.py stays frozen and is only IMPORTED for the shared helpers).

Same prompt set (data/geneval/src/prompts/evaluation_metadata.jsonl, 553 prompts) x 4 samples = 2212 images per group, SAME seed rule as S4
(seed = SEED_BASE + 100*prompt_idx + sample_idx, one independent CPU torch.Generator per image), fp16, 4 steps, 512x512, every image its own request (B=1).

One group per process (--group):
  AOT      B1-AOT arm: weight-free AOTInductor packages of shape 512x512 B1 no-CFG (aot_p5c/pkg/512x512_b1_nocfg/{te,unet,vae}.pt2) driven by
           scripts.aot_runtime_S4P_P5c.AOTRunnerP5c (exactly the runner inside the B1-AOT serving arm).  guidance 1.0 (no CFG).  Reference = the S4 E0 group
           (NOT regenerated).  Process hygiene as in serve_trace_S4P_v2: PRIVATE empty Inductor/Triton cache dir + per-process TMPDIR, set before the torch import; the
           run asserts afterwards that nothing was compiled (cache files == 0, Dynamo frame_compiles delta == 0).  Shape in the equivalence fallback list or missing
           package -> refuse (an eager fallback would silently turn the group into E0).
  E0CFG    eager reference, guidance 1.5 (CFG on, ShapeKey cfg=True; LCMPipe.GUIDANCE_ON = 1.5), B=1.
  C1G1CFG  torch.compile + manual CUDA graphs per component (runner "C1+G1"), guidance 1.5, B=1.  (S2: equiv_pass = 1 for 512x512 B1 CFG.)
Not used: AOT+G1 (CFG shape capture failed), PAD.

Output layout (official GenEval, identical to S4): <out>/<group>/<idx:05d>/metadata.jsonl + samples/<sample:05d>.png, plus manifest.csv / meta.json per group.
Default <out> = results/S4X/x4/images.  Resumable (a request whose PNG exists is skipped; PNG written atomically).  Memory cap: --mem-cap-gb (default 14 GiB,
set_per_process_memory_fraction; B=1 steady peaks are ~4-5 GB).

CPU-only checks: --dry-run (no torch import, nothing written: job list, S4-seed/prompt cross-check, AOT package/manifest/fallback preflight).
  --verify-s4  with --dry-run additionally compares the planned (prompt_idx, sample_idx) -> seed of ALL 2212 images with results/S4/geneval/images/E0/manifest.csv
  and the 553 prompts with images/E0/<idx>/metadata.jsonl (the data the S4 groups were really generated with).
Real runs need GPU0 (CUDA_VISIBLE_DEVICES = GPU0 UUID in the same command, see project CLAUDE.md).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Dict, List, Optional

from scripts import gen_geneval_S4 as G   # frozen S4 generator: seed rule, prompt file, selection, png writing, SHA helpers (import only, no side effects)

ROOT = G.ROOT
GROUPS = ("AOT", "E0CFG", "C1G1CFG")
GROUP_ARM = {"AOT": "AOT", "E0CFG": "E0", "C1G1CFG": "C1+G1"}
GROUP_CFG = {"AOT": False, "E0CFG": True, "C1G1CFG": True}
GROUP_GUIDANCE = {"AOT": 1.0, "E0CFG": 1.5, "C1G1CFG": 1.5}
DEFAULT_OUT = ROOT / "results/S4X/x4/images"
P24_CACHE = pathlib.Path(os.environ.get("P24_CACHE", "cache/p24"))
AOT_PKG_ROOT = P24_CACHE / "aot_p5c/pkg"
AOT_FALLBACK_CSV = ROOT / "results/S4P/p5c/aot_fallback_shapes.csv"
AOT_SHAPE_TAG = "512x512x1x0"                    # HxWxBxcfg (aot_runtime_S4P_P5c.tag_of); package dir 512x512_b1_nocfg
AOT_COMPONENTS = ("te", "unet", "vae")
S4_E0_DIR = ROOT / "results/S4/geneval/images/E0"


def aot_pkg_dir(pkg_root: pathlib.Path = AOT_PKG_ROOT) -> pathlib.Path:
    return pathlib.Path(pkg_root) / "512x512_b1_nocfg"          # = ShapeKey(512,512,1,False).tag()


def sha16(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def aot_preflight(pkg_root: pathlib.Path = AOT_PKG_ROOT, fallback_csv: pathlib.Path = AOT_FALLBACK_CSV) -> dict:
    """CPU: the three package files exist, shape not in the equivalence fallback list, SHA16 equal to pkg_manifest.json (15 MB, hashed BEFORE any timing/run)."""
    d = aot_pkg_dir(pkg_root)
    files = {c: d / f"{c}.pt2" for c in AOT_COMPONENTS}
    missing = [c for c, p in files.items() if not p.exists()]
    fb: Dict[str, str] = {}
    if fallback_csv.exists():
        with open(fallback_csv, newline="") as f:
            fb = {r["key"].strip(): r.get("reason", "") for r in csv.DictReader(f)}
    man_path = pathlib.Path(pkg_root).parent / "pkg_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else None
    served, mismatch = {}, []
    if not missing:
        ref = ((man or {}).get("packages") or {}).get(AOT_SHAPE_TAG)
        for c, p in files.items():
            served[c] = {"sha256_16": sha16(p), "bytes": p.stat().st_size}
            if ref is None or ref[c]["sha256_16"] != served[c]["sha256_16"]:
                mismatch.append(c)
    ok = (not missing) and (AOT_SHAPE_TAG not in fb) and fallback_csv.exists() and man is not None and not mismatch
    return {"ok": bool(ok), "pkg_dir": str(d), "missing_components": missing, "in_fallback_list": AOT_SHAPE_TAG in fb,
            "fallback_csv": str(fallback_csv), "fallback_csv_exists": fallback_csv.exists(),
            "manifest": str(man_path), "manifest_exists": man is not None, "manifest_mismatch": mismatch, "served_sha": served}


def planned_seeds(prompt_idxs: List[int], n_samples: int = G.N_SAMPLES) -> Dict[tuple, int]:
    return {(i, s): G.seed_for(i, s) for i in prompt_idxs for s in range(n_samples)}


def verify_against_s4(meta_all: List[dict], s4_dir: pathlib.Path = S4_E0_DIR) -> dict:
    """CPU: planned seeds == the seeds recorded in the S4 E0 manifest for every image; prompts == the metadata.jsonl written into the S4 E0 image tree."""
    plan = planned_seeds(list(range(len(meta_all))))
    seen: Dict[tuple, int] = {}
    with open(s4_dir / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            seen[(int(r["prompt_idx"]), int(r["sample_idx"]))] = int(r["seed"])
    seed_mismatch = [k for k in plan if seen.get(k) != plan[k]]
    extra = [k for k in seen if k not in plan]
    prompt_mismatch = []
    for i, m in enumerate(meta_all):
        p = s4_dir / f"{i:05d}" / "metadata.jsonl"
        got = json.loads(p.read_text().splitlines()[0]) if p.exists() else None
        if got != m:
            prompt_mismatch.append(i)
    png_missing = [k for k in plan if not G.img_path(s4_dir.parent, "E0", *k).exists()]
    return {"n_planned": len(plan), "n_s4_manifest_rows": len(seen), "seed_mismatch": len(seed_mismatch), "extra_in_s4": len(extra),
            "prompt_metadata_mismatch": len(prompt_mismatch), "s4_e0_png_missing": len(png_missing),
            "identical": not (seed_mismatch or extra or prompt_mismatch or png_missing) and len(plan) == len(seen)}


def group_meta(group: str, args, n_images: int, real_run: bool) -> dict:
    return {
        "step": "S4-X X4 GenEval supplementary generation", "group": group, "arm": GROUP_ARM[group], "cfg": GROUP_CFG[group],
        "guidance": GROUP_GUIDANCE[group], "H": G.H, "W": G.W, "steps": 4, "dtype": "float16", "scheduler": "LCMScheduler", "served_B": 1,
        "model_dir": str(G.MODEL_DIR), "fused_weights": G.fused_sha_status(compute=real_run),
        "seed_rule": f"seed = {G.SEED_BASE} + 100*prompt_idx + sample_idx (identical to S4 gen_geneval_S4.py, imported); one independent CPU torch.Generator per image",
        "n_prompts": len(args.prompt_idxs), "n_samples": args.samples, "n_images": n_images, "limit": args.limit,
        "mem_cap_gb": args.mem_cap_gb, "torch": G.torch_version_light(), "code_sha256_16": G.code_sha(),
        "metadata_file": str(G.META_FILE), "out_dir": str(args.out / group), "cmd": sys.argv,
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", required=True, choices=list(GROUPS), help="AOT | E0CFG | C1G1CFG (one group per process)")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT, help="image root; one sub-directory per group")
    ap.add_argument("--limit", type=int, default=None, help="use N prompts evenly spaced over the 553 (smoke test; use a scratch --out)")
    ap.add_argument("--samples", type=int, default=G.N_SAMPLES)
    ap.add_argument("--mem-cap-gb", type=float, default=14.0, help="per-process GPU memory cap in GiB (0 = none)")
    ap.add_argument("--pkg-root", type=pathlib.Path, default=AOT_PKG_ROOT, help="AOT group: package root (default $P24_CACHE/aot_p5c/pkg)")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: job list + checks, no torch import, nothing written")
    ap.add_argument("--verify-s4", action="store_true", help="with --dry-run: cross-check seeds/prompts of all images against the S4 E0 manifest/metadata")
    a = ap.parse_args(argv)
    if a.verify_s4 and not a.dry_run:
        ap.error("--verify-s4 is a CPU check and needs --dry-run")
    if a.samples != G.N_SAMPLES and not a.limit:
        ap.error("the confirmatory run uses exactly 4 samples per prompt (S4 convention)")
    return a


def prepare_aot_env(tag: str) -> pathlib.Path:
    """Private EMPTY Inductor/Triton cache + per-process TMPDIR (AOTI unpacks packages there); BEFORE the torch import. Returns the cache dir."""
    p5c = P24_CACHE / "aot_p5c"
    rt = p5c / "_rt_cache" / tag
    shutil.rmtree(rt, ignore_errors=True)
    (rt / "inductor").mkdir(parents=True)
    (rt / "triton").mkdir(parents=True)
    tmpd = p5c / "_tmp" / tag
    shutil.rmtree(tmpd, ignore_errors=True)
    tmpd.mkdir(parents=True)
    import tempfile
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(rt / "inductor"), TRITON_CACHE_DIR=str(rt / "triton"), TMPDIR=str(tmpd), TMP=str(tmpd), TEMP=str(tmpd))
    tempfile.tempdir = str(tmpd)
    return rt


def run_group(group: str, args, meta_all: List[dict]) -> dict:
    """GPU part. One request per image, B=1, in prompt order x sample order (same order as S4)."""
    rt_cache: Optional[pathlib.Path] = None
    aot_pf = None
    if group == "AOT":
        aot_pf = aot_preflight(args.pkg_root)
        if not aot_pf["ok"]:
            raise SystemExit(f"AOT preflight failed (package missing / in fallback list / manifest mismatch): {json.dumps(aot_pf)[:600]}")
        rt_cache = prepare_aot_env(f"x4_{group}")
    import torch
    from p24.pipeline import LCMPipe, ShapeKey
    assert torch.cuda.is_available(), "GPU run needs CUDA_VISIBLE_DEVICES set to the GPU0 UUID"
    torch.cuda.init()
    if args.mem_cap_gb > 0:
        total_mb = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_cap_gb * 1024 / total_mb), 0)
    gdir = args.out / group
    gdir.mkdir(parents=True, exist_ok=True)
    for i in args.prompt_idxs:
        d = gdir / f"{i:05d}"
        (d / "samples").mkdir(parents=True, exist_ok=True)
        (d / "metadata.jsonl").write_text(json.dumps(meta_all[i]) + "\n")
    meta = group_meta(group, args, len(args.prompt_idxs) * args.samples, real_run=True)
    if not meta["fused_weights"]["match_model_revisions"]:
        raise SystemExit("fused weights SHA256 differs from index/model_revisions.json - abort")
    import diffusers
    meta.update(diffusers=diffusers.__version__, torch_runtime=torch.__version__, gpu=torch.cuda.get_device_name(0),
                loadavg=os.getloadavg(), start=time.strftime("%Y-%m-%dT%H:%M:%S"), status="running")
    key = ShapeKey(G.H, G.W, 1, GROUP_CFG[group])
    t = time.perf_counter()
    if group == "AOT":
        from p24 import diag
        from scripts import aot_runtime_S4P_P5c as R5
        dyn0 = diag.dyn_snapshot()
        pipe = LCMPipe(str(G.MODEL_DIR))
        runner = R5.AOTRunnerP5c(pipe, args.pkg_root)
        runner.arm = "AOT"
    else:
        from p24.runners import make_runner
        runner = make_runner(GROUP_ARM[group], str(G.MODEL_DIR))
        pipe = runner.p
    print(f"[{group}] runner {GROUP_ARM[group]} built in {time.perf_counter() - t:.1f}s", flush=True)
    assert abs(pipe.GUIDANCE_ON - 1.5) < 1e-9
    mpath = gdir / "meta.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else None
    sessions = (prev or {}).get("sessions", [])
    manifest = open(gdir / "manifest.csv", "a")
    if manifest.tell() == 0:
        manifest.write("prompt_idx,sample_idx,seed,served_B,user_B,pos_in_batch,n_filler,png_sha256_16\n")
    n_skip = n_done = 0
    prep_s = t_gen = 0.0
    n_total = len(args.prompt_idxs) * args.samples
    t_start = time.perf_counter()
    for i in args.prompt_idxs:
        for s in range(args.samples):
            if G.img_path(args.out, group, i, s).exists():
                n_skip += 1
                continue
            sd = G.seed_for(i, s)
            inp = pipe.make_inputs([meta_all[i]["prompt"]], key, seeds=[sd])
            if not runner.has(key):
                t = time.perf_counter()
                info = runner.prepare(key, inp)
                prep_s += time.perf_counter() - t
                print(f"[{group}] prepared {key.tag()} in {time.perf_counter() - t:.1f}s", flush=True)
                meta["prepare_info"] = {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str))}
            t = time.perf_counter()
            out = runner.run(inp, key)
            torch.cuda.synchronize()
            t_gen += time.perf_counter() - t
            h = G.save_png_atomic(G.to_png_array(out[0]), G.img_path(args.out, group, i, s))
            manifest.write(f"{i},{s},{sd},1,1,0,0,{h}\n")
            n_done += 1
            if n_done % 200 == 0:
                manifest.flush()
                print(f"[{group}] {n_done + n_skip}/{n_total} images, elapsed {time.perf_counter() - t_start:.0f}s, gpu {t_gen:.0f}s, "
                      f"peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    wall = time.perf_counter() - t_start
    manifest.close()
    meta.update(status="complete", end=time.strftime("%Y-%m-%dT%H:%M:%S"), n_images_skipped_resume=n_skip, n_images_done_this_session=n_done,
                prepare_s=prep_s, gpu_run_s=t_gen, session_wall_s=wall, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    if group == "AOT":
        dyn1 = diag.dyn_snapshot()
        rt_files = sum(len(fs) for _, _, fs in os.walk(rt_cache))
        after = {c: sha16(aot_pkg_dir(args.pkg_root) / f"{c}.pt2") for c in AOT_COMPONENTS}
        meta["aot"] = {"preflight": aot_pf, "rt_cache_files_after": rt_files, "dyn_before": dyn0, "dyn_after": dyn1,
                       "frame_compiles_delta": dyn1["frame_compiles"] - dyn0["frame_compiles"], "pkg_sha16_after": after,
                       "load_log": getattr(runner, "load_log", [])}
        ok = rt_files == 0 and meta["aot"]["frame_compiles_delta"] == 0 and all(after[c] == aot_pf["served_sha"][c]["sha256_16"] for c in AOT_COMPONENTS)
        meta["aot"]["no_compile_check_ok"] = bool(ok)
        print(f"[AOT] no-compile check: rt_cache_files={rt_files} frame_compiles_delta={meta['aot']['frame_compiles_delta']} ok={ok}", flush=True)
        shutil.rmtree(P24_CACHE / "aot_p5c/_tmp" / f"x4_{group}", ignore_errors=True)
    sessions.append({k_: meta[k_] for k_ in ("start", "end", "n_images_skipped_resume", "n_images_done_this_session", "prepare_s", "gpu_run_s",
                                              "session_wall_s", "loadavg", "cmd")})
    meta["sessions"] = sessions
    mpath.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[{group}] done: {n_done} images (+{n_skip} skipped) in {wall:.0f}s (gpu {t_gen:.0f}s, prepare {prep_s:.0f}s)", flush=True)
    return meta


def main(argv=None):
    args = parse_args(argv)
    meta_all = G.load_metadata()
    args.prompt_idxs = G.select_prompts(len(meta_all), args.limit)
    g = args.group
    if args.dry_run:
        n_img = len(args.prompt_idxs) * args.samples
        done = sum(G.img_path(args.out, g, i, s).exists() for i in args.prompt_idxs for s in range(args.samples))
        print(f"DRY RUN  group={g} arm={GROUP_ARM[g]} cfg={GROUP_CFG[g]} guidance={GROUP_GUIDANCE[g]} prompts={len(args.prompt_idxs)} of {len(meta_all)} "
              f"samples={args.samples} images={n_img} requests={n_img} (served B=1, shape {G.H}x{G.W}) already-done={done} -> {args.out / g}")
        for i, s in [(args.prompt_idxs[0], 0), (args.prompt_idxs[0], 3), (args.prompt_idxs[-1], 3)]:
            print(f"   prompt {i:03d} sample {s} seed={G.seed_for(i, s)} prompt={meta_all[i]['prompt'][:50]!r}")
        m = group_meta(g, args, n_img, real_run=False)
        print("fused weights:", m["fused_weights"]["status"], "match model_revisions:", m["fused_weights"]["match_model_revisions"],
              "| torch", m["torch"], "| code files hashed:", len(m["code_sha256_16"]), "| mem cap GiB:", args.mem_cap_gb)
        rc = 0
        if g == "AOT":
            pf = aot_preflight(args.pkg_root)
            print("AOT preflight:", json.dumps({k: v for k, v in pf.items() if k != "served_sha"}), "served_sha:", json.dumps(pf["served_sha"]))
            rc = 0 if pf["ok"] else 4
        if args.verify_s4:
            v = verify_against_s4(meta_all)
            print("verify vs S4 (E0 manifest + metadata, all groups share them):", json.dumps(v))
            rc = rc or (0 if v["identical"] else 5)
        return rc
    G_ok = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not G_ok:
        sys.exit("CUDA_VISIBLE_DEVICES must be set to the GPU0 UUID in the same command")
    run_group(g, args, meta_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
