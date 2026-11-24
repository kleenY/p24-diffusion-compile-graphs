"""S4 step 4.1: GenEval image generation for the three quality-validation groups (E0, C1G1, PAD).

Official GenEval prompt set (data/geneval/src/prompts/evaluation_metadata.jsonl, 553 prompts) x 4 samples each = 2212 images per group.
Fixed (project convention, not changed here): SD1.5 + LCM-LoRA fused weights (index/model_revisions.json), fp16, 4 steps,
guidance 1.0 (no CFG, ShapeKey cfg=False), 512x512 (the SD1.5 native size and the GenEval generation default).

Groups (directory names carry no '+')
  E0    eager reference (p24 runner "E0"); every image is its own request, native shape 512x512 B=1 no-CFG.
  C1G1  torch.compile + manual CUDA graphs per component (runner "C1+G1"); every image its own request, native B=1 no-CFG.
  PAD   padding path (same C1+G1 runner): each prompt is served as TWO requests,
          request A = samples 0,1,2  -> user batch B=3, padded to the B4 bucket (512x512 B4 no-CFG), keep first 3 rows;
          request B = sample 3       -> user batch B=1, padded to the same B4 bucket (3 filler rows), keep row 0
                                         (--pad-b1 native serves it from the native B1 graph instead; see REPORT_4.1prep).
        filler rows: prompt PROMPTS_DEFAULT[0], seeds PAD_SEED0 + j (same convention as scripts/run_equivalence.py and
        scripts/serve_trace.py), outputs discarded.  Only B3->B4 was equivalence-tested in S2 (equiv_pass_padB4 = 1 for C1+G1,
        512x512, CFG off and on); B1->B4 is an extension (flagged in meta.json "pad_b1_verified_in_S2": false).

Seed rule (identical for all groups): seed(prompt_idx, sample_idx) = SEED_BASE + 100*prompt_idx + sample_idx, one independent CPU
torch.Generator per image (LCMPipe.make_inputs draws latents and the 3 per-step noises from the sample's own generator, so the
initial noise of an image does not depend on its batch composition / position).

Output layout (official GenEval): <out>/<group>/<idx:05d>/metadata.jsonl (ONE json object) + samples/<sample:05d>.png.
Default <out> = results/S4/geneval/images.  Resumable: a request whose PNGs all exist is skipped; PNGs are written atomically.

CPU-only options: --dry-run (list jobs, no model, no torch import).  Real runs need GPU0 (set CUDA_VISIBLE_DEVICES to the GPU0 UUID
in the same command, see project CLAUDE.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Sequence, Tuple

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
META_FILE = ROOT / "data/geneval/src/prompts/evaluation_metadata.jsonl"
MODEL_DIR = ROOT / "data/models/sd15_lcm_fused"
REV_FILE = ROOT / "index/model_revisions.json"
FUSED_SHA_CACHE = ROOT / "index/fused_weights_sha_measured.json"
DEFAULT_OUT = ROOT / "results/S4/geneval/images"

GROUPS = ("E0", "C1G1", "PAD")
GROUP_ARM = {"E0": "E0", "C1G1": "C1+G1", "PAD": "C1+G1"}
GROUP_NO = {"E0": 0, "C1G1": 1, "PAD": 2}  # also the bootstrap group number in analyze_geneval_S4.py

H = W = 512
BUCKET_B = 4
SEED_BASE = 202612020000          # 2026-12-02 style tag; seeds stay far below 2**63 and far from PAD_SEED0
PAD_SEED0 = 99000                 # filler seeds, same as scripts/run_equivalence.py
PAD_FILLER_PROMPT = "a photograph of an astronaut riding a horse"  # = p24.bench.PROMPTS_DEFAULT[0] (tested)
N_SAMPLES = 4


def seed_for(prompt_idx: int, sample_idx: int) -> int:
    assert 0 <= sample_idx < 100 and prompt_idx >= 0
    return SEED_BASE + 100 * prompt_idx + sample_idx


@dataclasses.dataclass(frozen=True)
class Request:
    group: str
    prompt_idx: int
    prompt: str
    sample_idxs: Tuple[int, ...]   # real samples in this request
    served_B: int                  # batch size of the graph/eager call actually executed
    user_B: int                    # == len(sample_idxs)

    @property
    def n_filler(self) -> int:
        return self.served_B - self.user_B


def load_metadata(path: pathlib.Path = META_FILE) -> List[dict]:
    with open(path) as fp:
        return [json.loads(line) for line in fp if line.strip()]


def select_prompts(n_total: int, limit: int | None) -> List[int]:
    """--limit N: N prompt indices evenly spaced over the file (covers every tag), deterministic. None -> all."""
    if limit is None or limit >= n_total:
        return list(range(n_total))
    if limit <= 1:
        return [0]
    return sorted({int(round(i * (n_total - 1) / (limit - 1))) for i in range(limit)})


def plan_requests(group: str, prompt_idx: int, prompt: str, n_samples: int = N_SAMPLES, pad_b1: str = "pad") -> List[Request]:
    if group in ("E0", "C1G1"):
        return [Request(group, prompt_idx, prompt, (s,), 1, 1) for s in range(n_samples)]
    assert group == "PAD"
    assert n_samples == 4, "PAD plan is defined for 4 samples per prompt (3 + 1)"
    return [Request(group, prompt_idx, prompt, (0, 1, 2), BUCKET_B, 3),
            Request(group, prompt_idx, prompt, (3,), BUCKET_B if pad_b1 == "pad" else 1, 1)]


def pad_batch(prompt: str, real_seeds: Sequence[int], served_B: int) -> Tuple[List[str], List[int]]:
    """Prompts/seeds of the served batch: real samples first, then fillers (outputs discarded)."""
    pr, sd = [prompt] * len(real_seeds), [int(s) for s in real_seeds]
    while len(pr) < served_B:
        pr.append(PAD_FILLER_PROMPT)
        sd.append(PAD_SEED0 + len(sd))
    return pr, sd


def take_real(out, n_real: int):
    """Only the first n_real rows (real samples) of a padded output are kept."""
    return out[:n_real]


def img_path(out_root: pathlib.Path, group: str, prompt_idx: int, sample_idx: int) -> pathlib.Path:
    return out_root / group / f"{prompt_idx:05d}" / "samples" / f"{sample_idx:05d}.png"


def request_done(out_root: pathlib.Path, req: Request) -> bool:
    return all(img_path(out_root, req.group, req.prompt_idx, s).exists() for s in req.sample_idxs)


def code_sha() -> Dict[str, str]:
    """Same algorithm as p24.bench.code_sha (SHA256-16 of every code/p24/*.py and code/scripts/*.py), without importing torch."""
    base = ROOT / "code"
    out = {}
    for sub in ("p24", "scripts"):
        for f in sorted((base / sub).glob("*.py")):
            out[f"{sub}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def file_sha256(path: pathlib.Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fused_sha_status(compute: bool) -> dict:
    """Fused-weights SHA256 vs index/model_revisions.json. compute=False (dry-run): only use the size/mtime cache, never hash GBs."""
    rev = json.loads(REV_FILE.read_text())
    cache = json.loads(FUSED_SHA_CACHE.read_text()) if FUSED_SHA_CACHE.exists() else {}
    res, state = {}, "cache_match"
    for f in sorted(MODEL_DIR.rglob("*.safetensors")):
        rel, st = str(f.relative_to(MODEL_DIR)), f.stat()
        c = cache.get(rel)
        if c and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns:
            res[rel] = c["sha256"]
        elif compute:
            res[rel] = file_sha256(f)
            state = "recomputed"
        else:
            res[rel] = None
            state = "stale_cache_would_recompute"
    ok = all(v is None or v == rev["fused_files_sha256"].get(k) for k, v in res.items())
    return {"status": state, "match_model_revisions": bool(ok), "measured_sha256": res,
            "base_revision": rev["base_revision"], "lora_revision": rev["lora_revision"],
            "declared_fused_files_sha256": {k: v for k, v in rev["fused_files_sha256"].items() if k.endswith(".safetensors")}}


def torch_version_light() -> str:
    from importlib.metadata import version
    return version("torch")


def group_meta(group: str, args, n_requests: int, n_images: int, real_run: bool) -> dict:
    return {
        "step": "S4 4.1 GenEval generation", "group": group, "arm": GROUP_ARM[group], "group_no": GROUP_NO[group],
        "H": H, "W": W, "steps": 4, "guidance": 1.0, "cfg": False, "dtype": "float16", "scheduler": "LCMScheduler",
        "model_dir": str(MODEL_DIR), "fused_weights": fused_sha_status(compute=real_run),
        "seed_rule": f"seed = {SEED_BASE} + 100*prompt_idx + sample_idx; one independent CPU torch.Generator per image; "
                     "same seed for every group (identical initial latents and per-step noises across E0/C1G1/PAD)",
        "pad": ({"bucket": f"{H}x{W}_b{BUCKET_B}_nocfg", "request_A": "samples 0-2: B3 -> B4 (S2-verified: equiv_pass_padB4=1)",
                 "request_B": "sample 3: B1 -> B4" if args.pad_b1 == "pad" else "sample 3: native B1",
                 "pad_b1_verified_in_S2": False, "filler_prompt": PAD_FILLER_PROMPT, "filler_seed0": PAD_SEED0}
                if group == "PAD" else None),
        "n_prompts": len(args.prompt_idxs), "n_samples": args.samples, "n_requests": n_requests, "n_images": n_images,
        "limit": args.limit, "torch": torch_version_light(), "code_sha256_16": code_sha(),
        "metadata_file": str(META_FILE), "out_dir": str(args.out / group), "cmd": sys.argv,
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


# ----------------------------------------------------------------------------------------------------------------- GPU part
def to_png_array(x):
    """(3,H,W) float in [0,1] -> (H,W,3) uint8, identical conversion for every group."""
    import torch
    return (x.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


def save_png_atomic(arr, path: pathlib.Path) -> str:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG", compress_level=3)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_group(group: str, args, runners: dict, meta_all: List[dict]) -> dict:
    import gc
    import torch
    from p24.pipeline import ShapeKey
    from p24.runners import make_runner

    arm = GROUP_ARM[group]
    if arm not in runners:
        for k in list(runners):          # one pipeline resident at a time (E0 and C1+G1 each hold a 2 GB copy of the weights)
            runners.pop(k)
        gc.collect()
        torch.cuda.empty_cache()
        t = time.perf_counter()
        runners[arm] = make_runner(arm, str(MODEL_DIR))
        print(f"[{group}] runner {arm} built in {time.perf_counter() - t:.1f}s", flush=True)
    runner = runners[arm]
    pipe = runner.p
    gdir = args.out / group
    gdir.mkdir(parents=True, exist_ok=True)
    reqs = [r for i in args.prompt_idxs for r in plan_requests(group, i, meta_all[i]["prompt"], args.samples, args.pad_b1)]
    for i in args.prompt_idxs:  # official per-prompt metadata (a single json object)
        d = gdir / f"{i:05d}"
        (d / "samples").mkdir(parents=True, exist_ok=True)
        (d / "metadata.jsonl").write_text(json.dumps(meta_all[i]) + "\n")
    meta = group_meta(group, args, len(reqs), sum(r.user_B for r in reqs) if reqs else 0, real_run=True)
    if not meta["fused_weights"]["match_model_revisions"]:
        raise SystemExit("fused weights SHA256 differs from index/model_revisions.json - abort")
    import diffusers
    meta.update(diffusers=diffusers.__version__, torch_runtime=torch.__version__, gpu=torch.cuda.get_device_name(0),
                loadavg=os.getloadavg(), start=time.strftime("%Y-%m-%dT%H:%M:%S"), status="running")
    mpath = gdir / "meta.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else None
    sessions = (prev or {}).get("sessions", [])
    manifest = open(gdir / "manifest.csv", "a")
    if manifest.tell() == 0:
        manifest.write("prompt_idx,sample_idx,seed,served_B,user_B,pos_in_batch,n_filler,png_sha256_16\n")
    n_skip = n_done = 0
    prep_s = 0.0
    t_start = time.perf_counter()
    t_gen = 0.0
    for k, req in enumerate(reqs):
        if request_done(args.out, req):
            n_skip += 1
            continue
        key = ShapeKey(H, W, req.served_B, False)
        pr, sd = pad_batch(req.prompt, [seed_for(req.prompt_idx, s) for s in req.sample_idxs], req.served_B)
        inp = pipe.make_inputs(pr, key, seeds=sd)
        if not runner.has(key):
            t = time.perf_counter()
            runner.prepare(key, inp)
            prep_s += time.perf_counter() - t
            print(f"[{group}] prepared {key.tag()} in {time.perf_counter() - t:.1f}s", flush=True)
        t = time.perf_counter()
        out = take_real(runner.run(inp, key), req.user_B)
        torch.cuda.synchronize()
        t_gen += time.perf_counter() - t
        for row, s in enumerate(req.sample_idxs):
            h = save_png_atomic(to_png_array(out[row]), img_path(args.out, group, req.prompt_idx, s))
            manifest.write(f"{req.prompt_idx},{s},{sd[row]},{req.served_B},{req.user_B},{row},{req.n_filler},{h}\n")
        manifest.flush()
        n_done += 1
        if n_done % 50 == 0:
            el = time.perf_counter() - t_start
            print(f"[{group}] {k + 1}/{len(reqs)} requests, elapsed {el:.0f}s, gpu {t_gen:.0f}s", flush=True)
    wall = time.perf_counter() - t_start
    manifest.close()
    meta.update(status="complete", end=time.strftime("%Y-%m-%dT%H:%M:%S"), n_requests_skipped_resume=n_skip,
                n_requests_done_this_session=n_done, prepare_s=prep_s, gpu_run_s=t_gen, session_wall_s=wall)
    sessions.append({k_: meta[k_] for k_ in ("start", "end", "n_requests_skipped_resume", "n_requests_done_this_session",
                                              "prepare_s", "gpu_run_s", "session_wall_s", "loadavg", "cmd")})
    meta["sessions"] = sessions
    mpath.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[{group}] done: {n_done} requests (+{n_skip} skipped) in {wall:.0f}s (gpu {t_gen:.0f}s, prepare {prep_s:.0f}s)", flush=True)
    return meta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", default=",".join(GROUPS), help="comma list of E0,C1G1,PAD (default all)")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT, help="image root; one sub-directory per group")
    ap.add_argument("--limit", type=int, default=None, help="use N prompts evenly spaced over the 553 (smoke test)")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--pad-b1", choices=["pad", "native"], default="pad",
                    help="PAD group, 4th sample of each prompt: padded B1->B4 (default) or served by the native B1 graph")
    ap.add_argument("--mem-fraction", type=float, default=0.45, help="per-process GPU memory cap (0 = none); 0.45 = 10.8 GB")
    ap.add_argument("--dry-run", action="store_true", help="list jobs only; no model, no torch import, no files written")
    a = ap.parse_args(argv)
    a.groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    for g in a.groups:
        if g not in GROUPS:
            ap.error(f"unknown group {g}")
    return a


def main(argv=None):
    args = parse_args(argv)
    meta_all = load_metadata()
    args.prompt_idxs = select_prompts(len(meta_all), args.limit)
    if args.dry_run:
        print(f"DRY RUN  prompts={len(args.prompt_idxs)} of {len(meta_all)}  samples={args.samples}  out={args.out}")
        for g in args.groups:
            reqs = [r for i in args.prompt_idxs for r in plan_requests(g, i, meta_all[i]["prompt"], args.samples, args.pad_b1)]
            done = sum(request_done(args.out, r) for r in reqs)
            n_img = sum(r.user_B for r in reqs)
            by_B: Dict[int, int] = {}
            for r in reqs:
                by_B[r.served_B] = by_B.get(r.served_B, 0) + 1
            print(f"[{g}] arm={GROUP_ARM[g]} requests={len(reqs)} images={n_img} served-B counts={by_B} already-done={done} -> {args.out / g}")
            for r in reqs[:3] + reqs[-1:]:
                pr, sd = pad_batch(r.prompt, [seed_for(r.prompt_idx, s) for s in r.sample_idxs], r.served_B)
                print(f"   prompt {r.prompt_idx:03d} samples {r.sample_idxs} served_B={r.served_B} fillers={r.n_filler} seeds={sd} "
                      f"prompt={r.prompt[:50]!r}")
        m = group_meta(args.groups[0], args, 0, 0, real_run=False)
        print("fused weights:", m["fused_weights"]["status"], "match model_revisions:", m["fused_weights"]["match_model_revisions"],
              "| torch", m["torch"], "| code files hashed:", len(m["code_sha256_16"]))
        return 0
    import torch
    assert torch.cuda.is_available(), "GPU run needs CUDA_VISIBLE_DEVICES set to the GPU0 UUID"
    if args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    runners: dict = {}
    for g in args.groups:
        run_group(g, args, runners, meta_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
