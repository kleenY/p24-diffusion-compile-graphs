"""Explicit, capture-friendly SD1.5 + LCM-LoRA (fused) pipeline.

The math mirrors diffusers' StableDiffusionPipeline + LCMScheduler op-for-op (same 0-dim scalar tensors,
same op order, same RNG call order) so that eager outputs can be checked bit-exactly against diffusers.
Everything that varies per request (token ids, initial latents, per-step noise) is an explicit tensor
input, so the same code path can run eagerly, under torch.compile, or inside a CUDA graph.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


@dataclasses.dataclass(frozen=True)
class ShapeKey:
    """Request shape key: resolution, user batch size, classifier-free guidance on/off."""
    H: int
    W: int
    B: int
    cfg: bool

    @property
    def unet_batch(self) -> int:
        return self.B * (2 if self.cfg else 1)

    @property
    def latent_hw(self):
        return self.H // 8, self.W // 8

    def tag(self) -> str:
        return f"{self.H}x{self.W}_b{self.B}_{'cfg' if self.cfg else 'nocfg'}"


@dataclasses.dataclass
class RequestInputs:
    """All per-request tensors (already on device)."""
    ids: torch.Tensor          # (B*(1+cfg), 77) int64, order [uncond..., cond...] when cfg
    latents: torch.Tensor      # (B, 4, h, w)
    noises: torch.Tensor       # (steps-1, B, 4, h, w)


class LCMPipe:
    GUIDANCE_ON = 1.5  # LCM-LoRA README: guidance disabled (1.0) or within [1.0, 2.0]

    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.float16, steps: int = 4,
                 channels_last: bool = False, fuse_qkv: bool = False):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=dtype, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        if fuse_qkv:
            pipe.fuse_qkv_projections()
        if channels_last:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
        self.pipe, self.device, self.dtype, self.steps = pipe, torch.device(device), dtype, steps
        self.tok, self.te, self.unet, self.vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
        for m in (self.te, self.unet, self.vae):
            m.eval().requires_grad_(False)
        self._build_schedule()

    # ---------------- schedule constants (identical ops/values to LCMScheduler.step) ----------------
    def _build_schedule(self):
        s = LCMScheduler.from_config(self.pipe.scheduler.config)
        s.set_timesteps(self.steps, device=self.device)
        self.timesteps = s.timesteps  # device int64, e.g. [999, 759, 499, 259]
        consts = []
        for i, t in enumerate(self.timesteps):
            prev_t = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else t
            a_t = s.alphas_cumprod[t]
            a_prev = s.alphas_cumprod[prev_t] if prev_t >= 0 else s.final_alpha_cumprod
            b_t, b_prev = 1 - a_t, 1 - a_prev
            c_skip, c_out = s.get_scalings_for_boundary_condition_discrete(t)
            consts.append(dict(sqrt_b_t=b_t.sqrt(), sqrt_a_t=a_t.sqrt(), c_skip=c_skip, c_out=c_out,
                               sqrt_a_prev=a_prev.sqrt(), sqrt_b_prev=b_prev.sqrt()))
        self.consts = consts
        # one static 0-dim timestep buffer per step (stable addresses -> safe for graph capture)
        self.t_bufs = [t.clone() for t in self.timesteps]

    # ---------------- host-side input preparation (NOT timed as GPU work; done before submission) ---------
    def tokenize(self, prompts: Sequence[str], cfg: bool) -> torch.Tensor:
        def ids(p):
            return self.tok(list(p), padding="max_length", max_length=self.tok.model_max_length,
                            truncation=True, return_tensors="pt").input_ids
        cond = ids(prompts)
        if cfg:
            cond = torch.cat([ids([""] * len(prompts)), cond])
        return cond

    # Inputs are created as inference tensors (2026-11-15, S2 2.0): a normal tensor fed to a compiled UNet that
    # otherwise sees inference tensors fails Dynamo's dispatch-key guard (ADInplaceOrView) and forces a second
    # graph for the same shape (catalogue item 17). Values are unchanged.
    @torch.inference_mode()
    def make_inputs(self, prompts: Sequence[str], key: ShapeKey, seeds: Sequence[int],
                    noise_dtype=None) -> RequestInputs:
        """Per-sample CPU generators -> noise of sample i depends only on seeds[i] (padding-invariant).
        noise_dtype: dtype the random numbers are drawn in (CPU randn draws DIFFERENT values for fp16 vs fp32),
        then cast to the pipeline dtype. The fp32 reference uses noise_dtype=float16 so it sees the same noise."""
        assert len(prompts) == len(seeds) == key.B
        h, w = key.latent_hw
        nd = noise_dtype or self.dtype
        gens = [torch.Generator("cpu").manual_seed(int(sd)) for sd in seeds]
        lat = randn_tensor((key.B, 4, h, w), generator=gens, device=self.device, dtype=nd).to(self.dtype)
        noises = [randn_tensor((key.B, 4, h, w), generator=gens, device=self.device, dtype=nd).to(self.dtype)
                  for _ in range(self.steps - 1)]
        return RequestInputs(ids=self.tokenize(prompts, key.cfg).to(self.device), latents=lat,
                             noises=torch.stack(noises) if noises else lat.new_zeros((0,) + lat.shape))

    # ---------------- components (pure tensor functions) ----------------
    def encode(self, ids: torch.Tensor, cfg: bool = False, te=None) -> torch.Tensor:
        """Like diffusers.encode_prompt: with CFG the uncond and cond halves are encoded in two separate
        calls (fp16 kernels are batch-composition dependent, so one concatenated call is NOT bit-identical)."""
        te = te or self.te
        if cfg:
            n = ids.shape[0] // 2
            return torch.cat([te(ids[:n], attention_mask=None)[0], te(ids[n:], attention_mask=None)[0]])
        return te(ids, attention_mask=None)[0]

    def unet_eps(self, x: torch.Tensor, t: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t, encoder_hidden_states=emb, return_dict=False)[0]

    def step(self, i: int, eps: torch.Tensor, latents: torch.Tensor, cfg: bool, noise: Optional[torch.Tensor]):
        if cfg:
            u, c = eps.chunk(2)
            eps = u + self.GUIDANCE_ON * (c - u)
        k = self.consts[i]
        x0 = (latents - k["sqrt_b_t"] * eps) / k["sqrt_a_t"]
        den = k["c_out"] * x0 + k["c_skip"] * latents
        if i != self.steps - 1:
            return k["sqrt_a_prev"] * den + k["sqrt_b_prev"] * noise
        return den

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        return (img / 2 + 0.5).clamp(0, 1)

    # ---------------- end-to-end (eager reference structure) ----------------
    def denoise_step(self, i: int, latents, emb, cfg: bool, noise):
        x = torch.cat([latents] * 2) if cfg else latents
        eps = self.unet_eps(x, self.t_bufs[i], emb)
        return self.step(i, eps, latents, cfg, noise)

    @torch.inference_mode()
    def run(self, inp: RequestInputs, cfg: bool, marks=None) -> torch.Tensor:
        """marks: optional callback marks(name) with name in start / te_end / denoise_end / vae_end (the caller
        records a CUDA event). marks=None -> exactly the original code path (nothing added to the timed path)."""
        if marks is not None:
            return self._run_marked(inp, cfg, marks)
        emb = self.encode(inp.ids, cfg)
        lat = inp.latents
        for i in range(self.steps):
            lat = self.denoise_step(i, lat, emb, cfg, inp.noises[i] if i < self.steps - 1 else None)
        return self.decode(lat)

    def _run_marked(self, inp: RequestInputs, cfg: bool, marks) -> torch.Tensor:
        marks("start")
        emb = self.encode(inp.ids, cfg)
        marks("te_end")
        lat = inp.latents
        for i in range(self.steps):
            lat = self.denoise_step(i, lat, emb, cfg, inp.noises[i] if i < self.steps - 1 else None)
        marks("denoise_end")
        out = self.decode(lat)
        marks("vae_end")
        return out

    # ---------------- diffusers reference (for bit-exact eager check) ----------------
    @torch.inference_mode()
    def diffusers_reference(self, prompts: List[str], key: ShapeKey, seeds: Sequence[int]) -> torch.Tensor:
        gens = [torch.Generator("cpu").manual_seed(int(sd)) for sd in seeds]
        out = self.pipe(prompt=list(prompts), height=key.H, width=key.W, num_inference_steps=self.steps,
                        guidance_scale=self.GUIDANCE_ON if key.cfg else 1.0, generator=gens,
                        output_type="pt", return_dict=True)
        return out.images  # (B,3,H,W) in [0,1]
