"""Equivalence check for one arm (protocol sec 10; S1 default = 512x512 B1, cfg off and on, 8 prompts).
8 images per (arm, shape): prompt i uses seed 7000+i, noise drawn in fp16. B>1 shapes are batched in groups of B
(B=4 -> 2 requests of 4). Padding check (B=3, plan 2.7): the 8 images form groups 3,3,2 and the ARM UNDER TEST serves every
group in the B4 bucket (padded with filler prompts/seeds, padded outputs discarded); `--pad-b3 none` serves the arm natively.
Review M5 (2026-11-15): the fp32-eager reference and E0 ALWAYS run natively at B3 (the last group of 2 is filled up to 3),
only the arm under test is padded, so the padding effect is what the verdict judges; when the arm is padded, its native-B3
run is reported alongside (`by_key[<key>]["native_b3"]`, not judged; `--native-b3 off` skips it).
fp32-eager reference and E0 images are cached per shape as <out>/_<kind>_<served-shape-tag>_<native|padB4>.pt (kind =
ref32 | E0; the cache name carries the shape actually served and the padding mode; the S1 legacy files _ref32.pt / _E0.pt
are read if present - they are native).
Compiled arms run on the pre-warmed compile artefacts of `<warm-root>/<arm>/` (review S4; protocol sec 10: the judged numerics
are those of the timed kernels): --warm-root defaults to $P24_CACHE/warm_dev (development), formal runs pass
$P24_CACHE/warm_s2. Inductor FX-graph cache hit/miss counters of the run are recorded in the json.
Writes <out>/<arm><tag>.json (never overwritten) with one verdict per shape plus code SHA, tolerances SHA, model-revisions SHA, fused-weights SHA
(computed here from the safetensors files, cached in index/fused_weights_sha_measured.json and compared with
model_revisions.json), time and device. Tolerances come from tests/tolerances.json (read-only, unchanged); a failing shape is
a result (equiv_pass = 0), never re-drawn.

Usage: python -m scripts.run_equivalence --arm C2 --out results/S1/equiv   (S1 default keys)
       python -m scripts.run_equivalence --arm C1 --keys 768x768x4x1,512x512x3x0 --out results/S2/equiv \\
           --warm-root "$P24_CACHE/warm_s2"
Keys are written H x W x B x cfg (code order; portrait = 768x512x..., landscape = 512x768x...).
"""
import argparse, hashlib, json, os, pathlib, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--keys", default="512x512x1x0,512x512x1x1", help="comma list of HxWxBxcfg")
ap.add_argument("--pad-b3", default="4", choices=["4", "none"],
                help="arm under test at B=3: served in the B4 bucket (default) or natively; ref32 and E0 are always native")
ap.add_argument("--native-b3", default="on", choices=["on", "off"],
                help="when the arm is padded at B=3, also run it natively and report it alongside (not judged)")
ap.add_argument("--warm-root", default=None, help="parent of the per-arm compile caches (default $P24_CACHE/warm_dev)")
ap.add_argument("--tag", default="", help="suffix of the result file name (<arm><tag>.json); an existing file is never overwritten")
a = ap.parse_args()
_res_file = pathlib.Path(a.out) / f"{a.arm}{a.tag}.json"
if _res_file.exists():  # review S3 spirit: results (incl. failing verdicts) are never overwritten or re-drawn
    sys.exit(f"refusing to overwrite existing {_res_file} (use another --out or --tag)")

CACHE_INDEPENDENT = {"E0", "E1", "G1", "G2"}
root = pathlib.Path(os.environ["PROJ_ROOT"])
warm_root = pathlib.Path(a.warm_root) if a.warm_root else pathlib.Path(os.environ["P24_CACHE"]) / "warm_dev"
arm_cache = None
if a.arm not in CACHE_INDEPENDENT:  # must be set before torch / inductor / triton are imported
    arm_cache = warm_root / a.arm.replace("+", "_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(arm_cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(arm_cache / "triton")

import torch
from p24.pipeline import LCMPipe, ShapeKey
from p24.runners import make_runner
from p24.bench import PROMPTS_DEFAULT, code_sha
from p24 import equiv, diag

model_dir = str(root / "data/models/sd15_lcm_fused")
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
prompts = PROMPTS_DEFAULT[: a.n]
keys = [ShapeKey(int(k.split("x")[0]), int(k.split("x")[1]), int(k.split("x")[2]), bool(int(k.split("x")[3])))
        for k in a.keys.split(",")]
TOL_F = root / "code/tests/tolerances.json"
TOL = json.loads(TOL_F.read_text())
PAD_SEED0 = 99000  # filler samples (outputs discarded)
FUSED_SHA_CACHE = root / "index/fused_weights_sha_measured.json"


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def measured_fused_sha() -> dict:
    """Review S4: SHA256 of the fused *.safetensors computed from the files (not copied from model_revisions.json).
    Result cached in index/fused_weights_sha_measured.json together with size/mtime_ns of each file (recomputed when either
    changed), so the multi-GB hash is paid once."""
    fdir = root / "data/models/sd15_lcm_fused"
    cache = json.loads(FUSED_SHA_CACHE.read_text()) if FUSED_SHA_CACHE.exists() else {}
    res, dirty = {}, False
    for f in sorted(fdir.rglob("*.safetensors")):
        rel, st = str(f.relative_to(fdir)), f.stat()
        c = cache.get(rel)
        if not (c and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns):
            c = {"sha256": file_sha256(f), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                 "computed": time.strftime("%Y-%m-%dT%H:%M:%S")}
            cache[rel], dirty = c, True
        res[rel] = c["sha256"]
    if dirty:
        FUSED_SHA_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return res


def serve_key(key: ShapeKey, pad: bool) -> ShapeKey:
    """Shape actually served: B=3 goes to the B4 bucket only when `pad` (arm under test, --pad-b3 4)."""
    if pad and key.B == 3 and a.pad_b3 == "4":
        return ShapeKey(key.H, key.W, 4, key.cfg)
    return key


def gen_images(runner, key: ShapeKey, is_runner: bool, pipe=None, pad: bool = False):
    """-> (n,3,H,W) float32 CPU tensor, images in prompt order. pad=True only for the arm under test."""
    sk = serve_key(key, pad)
    p = runner.p if is_runner else pipe
    imgs = []
    for g0 in range(0, len(prompts), key.B):
        idx = list(range(g0, min(g0 + key.B, len(prompts))))
        pr = [prompts[i] for i in idx]
        sd = [7000 + i for i in idx]
        while len(pr) < sk.B:  # padding to the served batch size; padded outputs are discarded below
            pr.append(PROMPTS_DEFAULT[0])
            sd.append(PAD_SEED0 + len(sd))
        inp = p.make_inputs(pr, sk, seeds=sd, noise_dtype=torch.float16)
        if is_runner:
            if not runner.has(sk):
                runner.prepare(sk, inp)
            o = runner.run(inp, sk)
        else:
            o = p.run(inp, sk.cfg)
        imgs.append(o[: len(idx)].float().cpu())
    if is_runner:
        runner.release(sk)
    return torch.cat(imgs)


def cached(kind: str, key: ShapeKey, make):
    """Per-shape cache of the fp32 reference / E0 images (always native). The file name carries the shape actually served and
    the padding mode; reads the S1 legacy file (native) when it has this shape."""
    served = serve_key(key, False)
    f = out / f"_{kind}_{served.tag()}_native.pt"
    if f.exists():
        return torch.load(f)
    legacy = out / f"_{kind}.pt"
    if legacy.exists():
        d = torch.load(legacy)
        if key.tag() in d:
            return d[key.tag()]
    x = make()
    torch.save(x, f)
    return x


res = {"arm": a.arm, "status": "ok", "keys": [k.tag() for k in keys], "n_images": len(prompts),
       "seeds": [7000 + i for i in range(len(prompts))], "pad_b3": a.pad_b3, "native_b3": a.native_b3,
       "padding_policy": "arm under test padded B3->B4 (filler discarded); fp32 reference and E0 native B3",
       "code_sha256_16": code_sha(), "tolerances_sha256": sha256(TOL_F),
       "model_revisions_sha256": sha256(root / "index/model_revisions.json"),
       "warm_root": str(warm_root), "arm_cache_dir": str(arm_cache) if arm_cache else None,
       "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "device": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}
rev = json.loads((root / "index/model_revisions.json").read_text())
res["fused_weights_sha256"] = measured_fused_sha()
res["fused_weights_sha256_revisions"] = {k: v for k, v in rev["fused_files_sha256"].items() if k.endswith(".safetensors")}
res["fused_weights_sha256_matches_revisions"] = bool(res["fused_weights_sha256"]) and \
    res["fused_weights_sha256"] == res["fused_weights_sha256_revisions"]
if not res["fused_weights_sha256_matches_revisions"]:
    print("WARNING: measured fused-weights SHA differs from model_revisions.json", file=sys.stderr, flush=True)
res["tolerances"] = {k: TOL[k] for k in ("c_rmse_ratio_max", "c_psnr_drop_db_max", "lpips_vs_E0_mean_max")}

try:
    p32 = None
    refs, e0s = {}, {}
    for key in keys:
        def mk_ref():
            global p32
            if p32 is None:
                p32 = LCMPipe(model_dir, dtype=torch.float32)
            return gen_images(None, key, is_runner=False, pipe=p32, pad=False)
        refs[key.tag()] = cached("ref32", key, mk_ref)
        torch.cuda.empty_cache()
    if p32 is not None:
        del p32
        torch.cuda.empty_cache()
    e0_runner = None
    for key in keys:
        def mk_e0():
            global e0_runner
            if e0_runner is None:
                e0_runner = make_runner("E0", model_dir)
            return gen_images(e0_runner, key, is_runner=True, pad=False)
        e0s[key.tag()] = cached("E0", key, mk_e0)
    native_x = {}
    if a.arm == "E0":
        x = e0s
    else:
        del e0_runner
        e0_runner = None
        torch.cuda.empty_cache()
        c0 = diag.dyn_snapshot()
        runner = make_runner(a.arm, model_dir)
        x = {key.tag(): gen_images(runner, key, is_runner=True, pad=True) for key in keys}
        if a.native_b3 == "on" and a.pad_b3 == "4":
            native_x = {key.tag(): gen_images(runner, key, is_runner=True, pad=False)
                        for key in keys if key.B == 3}
        c1 = diag.dyn_snapshot()
        res["compile_cache_counters"] = {k: c1[k] - c0[k] for k in ("fx_hit", "fx_miss", "fx_bypass", "aot_hit", "aot_miss")}
    dev = "cuda"
    res["by_key"] = {}
    for key in keys:
        t = key.tag()
        m = equiv.compare(x[t].to(dev), e0s[t].to(dev), refs[t].to(dev))
        m["psnr_drop_db"] = m["psnr_E0_vs_ref32"] - m["psnr_vs_ref32"]
        m["rmse_ratio_ok"] = int(m["rmse_ratio"] <= TOL["c_rmse_ratio_max"])
        m["psnr_drop_ok"] = int(m["psnr_drop_db"] <= TOL["c_psnr_drop_db_max"] + 1e-9)
        m["lpips_ok"] = int(m["lpips_vs_E0"] <= TOL["lpips_vs_E0_mean_max"])
        m["equiv_pass"] = int(m["rmse_ratio_ok"] and m["lpips_ok"])  # verdict = RMSE ratio AND LPIPS (protocol sec 10)
        arm_served = key if a.arm == "E0" else serve_key(key, True)
        m["served_as"] = arm_served.tag()
        m["padding"] = "padded B3->B4" if arm_served.B != key.B else "native"
        m["ref32_served_as"] = m["e0_served_as"] = serve_key(key, False).tag()
        if t in native_x:  # reported alongside, never judged
            mn = equiv.compare(native_x[t].to(dev), e0s[t].to(dev), refs[t].to(dev))
            mn["psnr_drop_db"] = mn["psnr_E0_vs_ref32"] - mn["psnr_vs_ref32"]
            mn["served_as"] = key.tag()
            m["native_b3"] = mn
        res["by_key"][t] = m
except Exception as e:
    import traceback
    res.update(status="fail", error=repr(e)[:2000], traceback=traceback.format_exc()[-1500:])
_res_file.write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in res if k not in ("device", "fused_weights_sha256", "fused_weights_sha256_revisions",
                                                      "traceback", "code_sha256_16")})[:1800])
