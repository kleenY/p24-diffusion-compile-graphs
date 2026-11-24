"""S4-X X10 equivalence check of one arm on SD-Turbo (card sec 1 / sec 3): arm output vs fp32 eager, S1-frozen tolerances (tests/tolerances.json, c = 2:
RMSE ratio <= 2 AND LPIPS(arm, E0) <= 0.02; PSNR drop reported with its 6.02 dB line).  8 prompts (p24.bench.PROMPTS_DEFAULT), seed 7000+i, noise drawn in fp16.
NOT a wrapper: scripts/run_equivalence.py hard-codes the SD1.5 fused model dir, LCMPipe and the SD1.5 weight-SHA bookkeeping, so the measurement logic is
COPIED here (source scripts/run_equivalence.py, lines in brackets) with the model loading replaced by scripts.x10_common.SDTurboPipe:
  - arm compile-cache redirect before torch import  [run_equivalence.py 42-49]            - gen_images [110-133] without B=3 padding (no B3 key in X10)
  - per-shape cache of the fp32 reference and E0 images `_ref32_<tag>_native.pt` / `_E0_<tag>_native.pt` [135-150]
  - fp32 reference = SDTurboPipe(dtype=float32) on the same fp16 weights, noise drawn in fp16 [171-183]   - E0 / arm images and verdicts [186-225]
  - verdict = RMSE ratio AND LPIPS (protocol sec 10), p24.equiv.compare unchanged, never re-drawn, an existing result file is never overwritten.
Extra (reported only): `--diffusers-check` (use with --arm E0) compares the eager SDTurboPipe against the OFFICIAL diffusers call
(num_inference_steps=1, guidance_scale=0.0) for the first B=1 key: bit-exact or max-abs (tolerances.json: E0_vs_diffusers expectation, <= 1/255).
Compiled arms run on the pre-warmed compile artefacts of <warm-root>/<arm with '+' -> '_'>/ (same as S2 formal equivalence).
Usage (from code/): python -B -m scripts.x10_equiv --arm C1 --keys 512x512x1x0,384x384x1x0,512x512x4x0 --out ../results/S4X/x10/equiv --warm-root "$P24_CACHE/x10/warm_s2"
"""
import argparse, hashlib, json, os, pathlib, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--keys", default="512x512x1x0,384x384x1x0,512x512x4x0", help="comma list of HxWxBxcfg (cfg must be 0)")
ap.add_argument("--warm-root", default=None, help="parent of the per-arm compile caches (default $P24_CACHE/x10/warm_dev)")
ap.add_argument("--diffusers-check", action="store_true", help="E0 only: eager SDTurboPipe vs the official diffusers call (reported, not judged)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
_res_file = pathlib.Path(a.out) / f"{a.arm}{a.tag}.json"
if _res_file.exists():
    sys.exit(f"refusing to overwrite existing {_res_file} (use another --out or --tag)")

CACHE_INDEPENDENT = {"E0", "E1", "G1", "G2"}
root = pathlib.Path(os.environ["PROJ_ROOT"])
warm_root = pathlib.Path(a.warm_root) if a.warm_root else pathlib.Path(os.environ["P24_CACHE"]) / "x10" / "warm_dev"
arm_cache = None
if a.arm not in CACHE_INDEPENDENT:  # must be set before torch / inductor / triton are imported  [run_equivalence.py 44-49]
    arm_cache = warm_root / a.arm.replace("+", "_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(arm_cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(arm_cache / "triton")

from scripts import x10_common as X  # noqa: E402

keys_spec = [tuple(int(v) for v in k.split("x")) for k in a.keys.split(",")]
assert all(k[3] == 0 for k in keys_spec), "X10: no CFG keys"
assert all(k[2] != 3 for k in keys_spec), "X10: no B=3 key (padding logic not ported)"

import torch  # noqa: E402
from p24.pipeline import ShapeKey  # noqa: E402
from p24.bench import PROMPTS_DEFAULT, code_sha  # noqa: E402
from p24 import equiv, diag  # noqa: E402

R = X.patch_runners()
SDTurboPipe = X.sd_turbo_pipe_class()
model_dir = X.model_dir()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
prompts = PROMPTS_DEFAULT[: a.n]
keys = [ShapeKey(h, w, b, bool(c)) for (h, w, b, c) in keys_spec]
TOL_F = root / "code/tests/tolerances.json"
TOL = json.loads(TOL_F.read_text())


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def gen_images(runner, key: ShapeKey, is_runner: bool, pipe=None):
    """-> (n,3,H,W) float32 CPU tensor in prompt order  [run_equivalence.py gen_images, padding removed]."""
    p = runner.p if is_runner else pipe
    imgs = []
    for g0 in range(0, len(prompts), key.B):
        idx = list(range(g0, min(g0 + key.B, len(prompts))))
        pr = [prompts[i] for i in idx]
        sd = [7000 + i for i in idx]
        assert len(pr) == key.B, "n_prompts must be a multiple of B"
        inp = p.make_inputs(pr, key, seeds=sd, noise_dtype=torch.float16)
        if is_runner:
            if not runner.has(key):
                runner.prepare(key, inp)
            o = runner.run(inp, key)
        else:
            o = p.run(inp, key.cfg)
        imgs.append(o[: len(idx)].float().cpu())
    if is_runner:
        runner.release(key)
    return torch.cat(imgs)


def cached(kind: str, key: ShapeKey, make):
    f = out / f"_{kind}_{key.tag()}_native.pt"
    if f.exists():
        return torch.load(f)
    x = make()
    torch.save(x, f)
    return x


rev = json.loads(X.REV_X10.read_text())["sd_turbo"]
res = {"arm": a.arm, "model": "stabilityai/sd-turbo", "model_revision": rev["revision"], "model_dir": model_dir, "status": "ok",
       "keys": [k.tag() for k in keys], "n_images": len(prompts), "seeds": [7000 + i for i in range(len(prompts))],
       "code_sha256_16": code_sha(), "tolerances_sha256": sha256(TOL_F), "model_revisions_x10_sha256": sha256(X.REV_X10),
       "warm_root": str(warm_root), "arm_cache_dir": str(arm_cache) if arm_cache else None, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
       "device": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
       "tolerances": {k: TOL[k] for k in ("c_rmse_ratio_max", "c_psnr_drop_db_max", "lpips_vs_E0_mean_max")}}
try:
    p32 = None
    refs, e0s = {}, {}
    for key in keys:  # fp32 eager reference  [run_equivalence.py 171-183]
        def mk_ref():
            global p32
            if p32 is None:
                p32 = SDTurboPipe(model_dir, dtype=torch.float32)
            return gen_images(None, key, is_runner=False, pipe=p32)
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
                e0_runner = R.make_runner("E0", model_dir)
            return gen_images(e0_runner, key, is_runner=True)
        e0s[key.tag()] = cached("E0", key, mk_e0)
    if a.arm == "E0":
        x = e0s
    else:
        del e0_runner
        e0_runner = None
        torch.cuda.empty_cache()
        c0 = diag.dyn_snapshot()
        runner = R.make_runner(a.arm, model_dir)
        x = {key.tag(): gen_images(runner, key, is_runner=True) for key in keys}
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
        m["served_as"] = t
        res["by_key"][t] = m
    if a.diffusers_check:
        assert a.arm == "E0", "--diffusers-check only with --arm E0"
        key = next((k for k in keys if k.B == 1), None)
        if key is not None:
            del x
            torch.cuda.empty_cache()
            pp = SDTurboPipe(model_dir)
            diffs = []
            for i, pr in enumerate(prompts):
                inp = pp.make_inputs([pr], key, seeds=[7000 + i], noise_dtype=torch.float16)
                mine = pp.run(inp, key.cfg).float().cpu()
                ref = pp.diffusers_reference([pr], key, [7000 + i]).float().cpu()
                diffs.append((mine - ref).abs().max().item())
            res["diffusers_check"] = {"key": key.tag(), "maxabs_per_image": diffs, "maxabs": max(diffs), "bit_exact": bool(max(diffs) == 0.0),
                                      "within_1_over_255": bool(max(diffs) <= 1 / 255), "judged": False}
except Exception as e:
    import traceback
    res.update(status="fail", error=repr(e)[:2000], traceback=traceback.format_exc()[-1500:])
_res_file.write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in res if k not in ("device", "traceback", "code_sha256_16")})[:1800])
