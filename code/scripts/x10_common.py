"""S4-X X10 (SD-Turbo mechanism check) shared helpers.  Prereg card: goals/S4X_X10_second_model_mechanism_prereg_v1.md.

Nothing signed-off is modified.  The SD1.5 + LCM-LoRA measurement code (p24/runners.py, p24/bench.py, scripts/run_matrix.py,
scripts/prewarm_s2.py) is reused UNCHANGED; only the model class is swapped:

  SDTurboPipe(LCMPipe)   same interface as p24.pipeline.LCMPipe (encode / unet_eps / step / decode / make_inputs / run / t_bufs / steps /
                         consts), but with the SD-Turbo default scheduler (diffusers EulerDiscreteScheduler from the repo's scheduler_config.json,
                         timestep_spacing=trailing), 1 step, guidance 0 (= no CFG, cfg=False only), fp16 weights (variant "fp16").
                         Math = op-for-op diffusers StableDiffusionPipeline + EulerDiscreteScheduler (diffusers 0.35.2 scheduling_euler_discrete.py:
                         scale_model_input l.295-317, step l.~400-470, init_noise_sigma l.262-268), see the method docstrings.
                         DECISION (for the main session to confirm): scale_model_input (sample / sqrt(sigma^2+1)) is folded into the UNet component
                         (ScaledUNet wrapper), because p24.runners feeds the latent straight into pipe.unet; in E0 it is one eager elementwise
                         kernel, in C1 it is inside the compiled UNet, in G1 inside the UNet graph.  It is the same op diffusers runs before the UNet.
  patch_runners()        p24.runners.LCMPipe = SDTurboPipe, so p24.runners.make_runner(arm, model_dir) builds E0 / G1 / C1 / C1+G1 on SD-Turbo.
  install_subprocess_guard()  for the drivers run_matrix.py / prewarm_s2.py (run through runpy by scripts/x10_run_matrix.py, x10_prewarm.py):
                         every child `python -m p24.bench ...` is redirected to `python -m scripts.x10_bench ...` (which patches the model class and
                         then calls p24.bench.main() unchanged), with per-process gates and records like the X8 window:
                         MemAvailable >= X10_MEM_MIN_MB before each process (wait <= 600 s), stderr of EVERY process saved as <tag>.stderr.txt,
                         pinned_fallback_count (P24_PINNED_FALLBACK_USED lines) saved as <tag>.pinned_fallback_count.txt, host meta
                         <tag>.hostmeta.json, STOP (exit 7) on rc != 0, pinned count > 0 or meta status != ok, leak guard (MemAvailable falls
                         > X10_MEM_DROP_MB below the first reading and does not recover within 60 s -> exit 5).  Driver code is not edited.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
REV_X10 = ROOT / "index/model_revisions_x10.json"
X10_RESULT = ROOT / "results/S4X/x10"
SD_TURBO = "stabilityai/sd-turbo"


def model_dir() -> str:
    """Snapshot dir of the locked SD-Turbo revision (HF cache under $HF_HOME, no spaces); env X10_MODEL_DIR overrides (tests)."""
    if os.environ.get("X10_MODEL_DIR"):
        return os.environ["X10_MODEL_DIR"]
    return json.loads(REV_X10.read_text())["sd_turbo"]["snapshot_dir"]


# ---------------------------------------------------------------------------------------------------------------- model class
def _make_classes():
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
    from diffusers.utils.torch_utils import randn_tensor

    from p24.pipeline import LCMPipe, RequestInputs, ShapeKey  # noqa: F401

    class ScaledUNet(torch.nn.Module):
        """UNet component = scale_model_input + UNet (diffusers: sample / ((sigma**2 + 1) ** 0.5), sigma 0-dim fp32; fp16 sample stays fp16)."""

        def __init__(self, unet, denom):
            super().__init__()
            self.unet = unet
            self.register_buffer("denom", denom, persistent=False)

        @property
        def config(self):
            return self.unet.config

        def forward(self, x, t, encoder_hidden_states=None, return_dict=False, **kw):
            return self.unet(x / self.denom, t, encoder_hidden_states=encoder_hidden_states, return_dict=return_dict, **kw)

    class SDTurboPipe(LCMPipe):
        """SD-Turbo (SD2.1 distilled), 1 step, guidance 0 (no CFG), default Euler scheduler; interface of p24.pipeline.LCMPipe."""

        def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.float16, steps: int = 1,
                     channels_last: bool = False, fuse_qkv: bool = False):
            assert steps == 1 and not channels_last and not fuse_qkv, "X10: SD-Turbo 1 step, no E1 options"
            pipe = StableDiffusionPipeline.from_pretrained(
                model_dir, torch_dtype=dtype, variant="fp16", safety_checker=None, feature_extractor=None,
                requires_safety_checker=False)
            # default scheduler of the repo (EulerDiscreteScheduler, timestep_spacing=trailing); NOT replaced by LCMScheduler
            assert pipe.scheduler.__class__.__name__ == "EulerDiscreteScheduler", pipe.scheduler.__class__.__name__
            pipe.set_progress_bar_config(disable=True)
            pipe.to(device)
            self.pipe, self.device, self.dtype, self.steps = pipe, torch.device(device), dtype, steps
            self.tok, self.te, self.vae = pipe.tokenizer, pipe.text_encoder, pipe.vae
            self._build_schedule()
            self.unet = ScaledUNet(pipe.unet, self.consts[0]["denom"])
            for m in (self.te, self.unet, self.vae):
                m.eval().requires_grad_(False)

        def _build_schedule(self):
            s = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            s.set_timesteps(self.steps, device=self.device)
            self.timesteps = s.timesteps                      # device, e.g. tensor([999.])
            self.init_noise_sigma = float(s.init_noise_sigma)  # trailing -> max sigma
            sig = s.sigmas.to(torch.float32)                  # cpu, len steps+1, last = 0 (final_sigmas_type zero)
            consts = []
            for i in range(self.steps):
                sigma, nxt = sig[i], sig[i + 1]
                consts.append(dict(sigma=sigma.to(self.device), dt=(nxt - sigma).to(self.device),
                                   denom=((sigma ** 2 + 1) ** 0.5).to(self.device)))
            self.consts = consts
            self.t_bufs = [t.clone() for t in self.timesteps]  # static 0-dim timestep buffers (stable addresses for graph capture)

        @torch.inference_mode()
        def make_inputs(self, prompts, key, seeds, noise_dtype=None):
            """diffusers prepare_latents: per-sample CPU generators, randn in `noise_dtype` (default = pipeline dtype), latents * init_noise_sigma."""
            assert len(prompts) == len(seeds) == key.B and not key.cfg, "X10: no CFG"
            h, w = key.latent_hw
            nd = noise_dtype or self.dtype
            gens = [torch.Generator("cpu").manual_seed(int(sd)) for sd in seeds]
            lat = randn_tensor((key.B, 4, h, w), generator=gens, device=self.device, dtype=nd).to(self.dtype) * self.init_noise_sigma
            return RequestInputs(ids=self.tokenize(prompts, False).to(self.device), latents=lat,
                                 noises=lat.new_zeros((0,) + lat.shape))

        def step(self, i, eps, latents, cfg, noise):
            """EulerDiscreteScheduler.step, prediction_type epsilon, s_churn 0 (gamma 0 -> sigma_hat = sigma), op-for-op."""
            assert not cfg
            k = self.consts[i]
            sample = latents.to(torch.float32)               # "Upcast to avoid precision issues"
            pred_x0 = sample - k["sigma"] * eps
            derivative = (sample - pred_x0) / k["sigma"]
            prev = sample + derivative * k["dt"]
            return prev.to(eps.dtype)

        @torch.inference_mode()
        def diffusers_reference(self, prompts, key, seeds):
            """Official diffusers call (SD-Turbo model card: num_inference_steps=1, guidance_scale=0.0) on the unmodified pipeline."""
            gens = [torch.Generator("cpu").manual_seed(int(sd)) for sd in seeds]
            out = self.pipe(prompt=list(prompts), height=key.H, width=key.W, num_inference_steps=1, guidance_scale=0.0,
                            generator=gens, output_type="pt", return_dict=True)
            return out.images

    return SDTurboPipe, ScaledUNet


_CLS = None


def sd_turbo_pipe_class():
    global _CLS
    if _CLS is None:
        _CLS = _make_classes()[0]
    return _CLS


def patch_runners():
    """p24.runners.make_runner builds its pipelines through the module global LCMPipe -> swap it."""
    import p24.runners as R
    R.LCMPipe = sd_turbo_pipe_class()
    return R


# ---------------------------------------------------------------------------------------------------------------- subprocess guard
def _avail_mb() -> int:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) // 1024
    return -1


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def install_subprocess_guard(log_dir: str | None = None):
    import subprocess
    orig = subprocess.run
    log_dir = pathlib.Path(log_dir or os.environ.get("X10_LOG_DIR") or (X10_RESULT / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    mem_min = int(os.environ.get("X10_MEM_MIN_MB", "25600"))
    mem_drop = int(os.environ.get("X10_MEM_DROP_MB", "12288"))
    state = {"a0": None}

    def wait_mem():
        t0 = time.time()
        while _avail_mb() < mem_min:
            if time.time() - t0 > 600:
                print(f"X10_STOP MemAvailable={_avail_mb()}MB < {mem_min}MB for 600 s", flush=True)
                raise SystemExit(3)
            time.sleep(30)

    def run(cmd, *a, **k):
        if not (isinstance(cmd, (list, tuple)) and "p24.bench" in cmd):
            return orig(cmd, *a, **k)
        cmd = list(cmd)
        cmd[cmd.index("p24.bench")] = "scripts.x10_bench"
        wait_mem()
        if state["a0"] is None:
            state["a0"] = _avail_mb()
        arm, H, W, B, cfg, pid = (_flag(cmd, "--arm"), _flag(cmd, "--H") or "512", _flag(cmd, "--W") or "512", _flag(cmd, "--B") or "1",
                                  _flag(cmd, "--cfg") or "0", _flag(cmd, "--proc-id") or "0")
        out = pathlib.Path(_flag(cmd, "--out"))
        tag = f"{arm}_{H}x{W}_b{B}_{'cfg' if int(cfg) else 'nocfg'}_p{pid}"
        ltag = f"{out.name}__{tag}" if out.name.startswith("_warmup_") else tag
        a_before, t0 = _avail_mb(), time.strftime("%FT%T")
        r = orig(cmd, *a, **k)
        err = r.stderr if isinstance(r.stderr, str) else (r.stderr or b"").decode("utf-8", "replace")
        (log_dir / f"{ltag}.stderr.txt").write_text(err)
        cnt = err.count("P24_PINNED_FALLBACK_USED")
        (log_dir / f"{ltag}.pinned_fallback_count.txt").write_text(f"{cnt}\n")
        (log_dir / f"{ltag}.hostmeta.json").write_text(json.dumps(dict(
            tag=ltag, start=t0, end=time.strftime("%FT%T"), exit_code=r.returncode, pinned_fallback_count=cnt,
            memavailable_mb_before=a_before, memavailable_mb_after=_avail_mb(), load1_after=os.getloadavg()[0],
            out=str(out)), indent=1))
        status = None
        mf = out / f"{tag}.meta.json"
        if mf.exists():
            status = json.loads(mf.read_text()).get("status")
        why = None
        if cnt > 0:
            why = f"pinned_fallback_count={cnt}"
        elif r.returncode != 0:
            why = f"rc={r.returncode}"
        elif status != "ok":
            why = f"meta status={status!r}"
        if why:
            print(f"X10_STOP process {ltag}: {why} (no retry; finished processes stay in {out})", flush=True)
            raise SystemExit(7)
        for i in (1, 2):  # leak guard
            drop = state["a0"] - _avail_mb()
            if drop <= mem_drop:
                break
            if i == 1:
                time.sleep(60)
            else:
                print(f"X10_STOP leak_guard avail drop {drop}MB > {mem_drop}MB after {ltag}", flush=True)
                raise SystemExit(5)
        return r

    subprocess.run = run
    return run


# ---------------------------------------------------------------------------------------------------------------- CPU config check (DRY_RUN)
def check_config(mdir: str | None = None) -> int:
    """CPU only: read the config.json files of the locked snapshot (NO weights loaded), build the default scheduler and print the 1-step schedule the pipeline will use."""
    d = pathlib.Path(mdir or model_dir())
    rd = lambda rel: json.loads((d / rel).read_text())  # noqa: E731
    mi, uc, sc, tc, vc = rd("model_index.json"), rd("unet/config.json"), rd("scheduler/scheduler_config.json"), rd("text_encoder/config.json"), rd("vae/config.json")
    for rel in ("text_encoder/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"):
        assert (d / rel).exists(), f"missing {rel}"
    print(f"snapshot {d}")
    print(f"model_index _class_name={mi.get('_class_name')} scheduler={mi.get('scheduler')} safety_checker={mi.get('safety_checker')}")
    print(f"unet: in_channels={uc.get('in_channels')} cross_attention_dim={uc.get('cross_attention_dim')} attention_head_dim={uc.get('attention_head_dim')} "
          f"use_linear_projection={uc.get('use_linear_projection')} sample_size={uc.get('sample_size')} block_out_channels={uc.get('block_out_channels')}")
    print(f"text_encoder: {tc.get('architectures')} hidden_size={tc.get('hidden_size')} num_hidden_layers={tc.get('num_hidden_layers')} max_position_embeddings={tc.get('max_position_embeddings')}")
    print(f"vae: scaling_factor={vc.get('scaling_factor')} latent_channels={vc.get('latent_channels')}")
    print(f"scheduler_config: {sc.get('_class_name')} prediction_type={sc.get('prediction_type')} timestep_spacing={sc.get('timestep_spacing')} steps_offset={sc.get('steps_offset')} "
          f"beta_schedule={sc.get('beta_schedule')} final_sigmas_type={sc.get('final_sigmas_type', '(default zero)')}")
    assert sc.get("_class_name") == "EulerDiscreteScheduler" and sc.get("prediction_type") == "epsilon", "unexpected scheduler (card: default scheduler, 1 step)"
    from diffusers import EulerDiscreteScheduler
    import torch
    s = EulerDiscreteScheduler.from_pretrained(str(d), subfolder="scheduler")
    s.set_timesteps(1, device="cpu")
    sig = s.sigmas.to(torch.float32)
    print(f"1-step schedule: timesteps={s.timesteps.tolist()} dtype={s.timesteps.dtype} sigmas={sig.tolist()} init_noise_sigma={float(s.init_noise_sigma):.4f} "
          f"scale_model_input denominator={float((sig[0] ** 2 + 1) ** 0.5):.4f} dt={float(sig[1] - sig[0]):.4f}")
    assert len(s.timesteps) == 1 and float(sig[1]) == 0.0
    print("CONFIG_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(check_config(sys.argv[1] if len(sys.argv) > 1 else None))
