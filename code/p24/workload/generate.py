"""Workload scenario generator (S2 2.4): W1-W5 and validation streams V1-V3.  Deterministic: byte-identical for the same (scenario, N, seed, half).

Request fields: req_id, prompt, H, W, B, cfg(0/1), noise_seed, src_row.  Closed-loop arrival order (no arrival times).
Declared distribution (declared, not measured): B in {1,2,3,4} ~ 0.55/0.20/0.10/0.15, CFG on with probability 0.3, independent of the shape.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

from .split import RESOLUTIONS, RES_SQUARE_S, map_resolutions, res_index

B_VALUES = [1, 2, 3, 4]
B_PROBS = [0.55, 0.20, 0.10, 0.15]
CFG_PROB = 0.3
# all 32 shapes (H, W, B, cfg)
ALL_SHAPES = [(H, W, b, c) for (H, W) in RESOLUTIONS for b in B_VALUES for c in (0, 1)]
SCENARIOS = ["W1", "W2", "W3", "W4", "W5"]
DEV_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = list(range(100, 110))
VALIDATION = {"V1": 0, "V2": 1, "V3": 2}   # validation stream -> development seed
VALIDATION_N = {"V1": 400, "V2": 300, "V3": 200}


@dataclass
class HalfPool:
    """Row pool of one half (A or B): global row ids, prompts, mapped shapes."""
    half: str
    rows: np.ndarray       # global row ids int64 (ascending)
    prompts: np.ndarray    # object array
    hw: np.ndarray         # (n,2) int16, mapped (H,W)

    def __len__(self):
        return len(self.rows)


def make_pool(df, half_arr: np.ndarray, half: str) -> HalfPool:
    code = {"A": 0, "B": 1}[half]
    sel = np.flatnonzero(half_arr == code)
    sub = df.iloc[sel]
    return HalfPool(
        half=half,
        rows=sel.astype(np.int64),
        prompts=sub["prompt"].fillna("").to_numpy(dtype=object),
        hw=map_resolutions(sub["width"].to_numpy(), sub["height"].to_numpy()),
    )


def _rng(tag: str, n: int, seed: int, half: str) -> np.random.Generator:
    s = int(hashlib.sha256(f"p24-s2|{tag}|{n}|{seed}|{half}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(s)


def _draw_b(rng, n):
    return rng.choice(B_VALUES, size=n, p=B_PROBS)


def _draw_cfg(rng, n):
    return (rng.random(n) < CFG_PROB).astype(np.int64)


def _mk(tag, half, seed, pool, idx, H, W, B, cfg, noise):
    out = []
    for i in range(len(idx)):
        j = int(idx[i])
        out.append({
            "req_id": f"{tag}-{half}-s{seed}-{i:05d}",
            "prompt": str(pool.prompts[j]),
            "H": int(H[i]), "W": int(W[i]), "B": int(B[i]), "cfg": int(cfg[i]),
            "noise_seed": int(noise[i]),
            "src_row": int(pool.rows[j]),
        })
    return out


def _swap_orient(H, W):
    """Swap portrait/landscape (square images unchanged)."""
    H = np.asarray(H); W = np.asarray(W)
    return np.where(H == W, H, W), np.where(H == W, W, H)


def generate_scenario(pool: HalfPool, scenario: str, n: int, seed: int) -> list[dict]:
    """W1-W5.  half is taken from pool.half."""
    rng = _rng(scenario, n, seed, pool.half)
    idx = rng.integers(0, len(pool), size=n)           # row sampling (with replacement)
    noise = rng.integers(0, 2**31 - 1, size=n)
    if scenario == "W1":
        H = np.full(n, RES_SQUARE_S[0]); W = np.full(n, RES_SQUARE_S[1])
        B = np.ones(n, dtype=np.int64); cfg = np.zeros(n, dtype=np.int64)
    elif scenario in ("W2", "W4", "W5"):
        H = pool.hw[idx, 0].astype(np.int64); W = pool.hw[idx, 1].astype(np.int64)
        B = _draw_b(rng, n); cfg = _draw_cfg(rng, n)
        if scenario == "W4":
            n1 = n // 2
            H2, W2 = _swap_orient(H[n1:], W[n1:])
            H = np.concatenate([H[:n1], H2]); W = np.concatenate([W[:n1], W2])
        elif scenario == "W5":
            k = max(32, math.ceil(0.05 * n))
            assert n >= 32
            perm = rng.permutation(len(ALL_SHAPES))
            for pos in range(32):
                h_, w_, b_, c_ = ALL_SHAPES[int(perm[pos])]
                H[pos], W[pos], B[pos], cfg[pos] = h_, w_, b_, c_
            # the first k requests (k>=32) cover all 32 shapes; k is recorded in W5_PREFIX
    elif scenario == "W3":
        ri = rng.integers(0, len(RESOLUTIONS), size=n)
        H = np.array([RESOLUTIONS[r][0] for r in ri]); W = np.array([RESOLUTIONS[r][1] for r in ri])
        B = rng.integers(1, 5, size=n); cfg = _draw_cfg(rng, n)
    else:
        raise ValueError(scenario)
    return _mk(scenario, pool.half, seed, pool, idx, H, W, B, cfg, noise)


def w5_prefix_len(n: int) -> int:
    return max(32, math.ceil(0.05 * n))


def shape_probs(pool: HalfPool) -> dict[tuple, float]:
    """A's mapped distribution x declared distribution (B and cfg independent of the shape) -> probabilities of the 32 shapes."""
    ri = res_index(pool.hw)
    pres = np.bincount(ri, minlength=len(RESOLUTIONS)) / len(ri)
    probs = {}
    for r, (H, W) in enumerate(RESOLUTIONS):
        for b, pb in zip(B_VALUES, B_PROBS):
            for c, pc in ((0, 1 - CFG_PROB), (1, CFG_PROB)):
                probs[(H, W, b, c)] = float(pres[r] * pb * pc)
    return probs


def select_top_shapes(probs: dict[tuple, float], k: int) -> list[tuple]:
    """The k most probable shapes; if B=3 is absent, the k-th is replaced by the most probable B=3 shape.  Ties are ordered by the shape tuple."""
    order = sorted(probs, key=lambda s: (-probs[s], s))
    top = order[:k]
    if not any(s[2] == 3 for s in top):
        b3 = next(s for s in order if s[2] == 3)
        top[-1] = b3
    return top


def generate_validation(pool: HalfPool, name: str) -> tuple[list[dict], list[tuple], list[float]]:
    """Validation streams V1-V3 (from A only, development seeds).  Returns (requests, selected shapes, normalized probabilities)."""
    assert pool.half == "A"
    seed = VALIDATION[name]; n = VALIDATION_N[name]
    if name == "V1":
        return generate_scenario(pool, "W1", n, seed), [(512, 512, 1, 0)], [1.0]
    k = 8 if name == "V2" else 12
    probs = shape_probs(pool)
    shapes = select_top_shapes(probs, k)
    p = np.array([probs[s] for s in shapes]); p = p / p.sum()
    rng = _rng(name, n, seed, pool.half)
    idx = rng.integers(0, len(pool), size=n)
    noise = rng.integers(0, 2**31 - 1, size=n)
    pick = rng.choice(len(shapes), size=n, p=p)
    if name == "V3":
        pick[:k] = rng.permutation(k)                  # the first k requests each cover one shape
    H = np.array([shapes[i][0] for i in pick]); W = np.array([shapes[i][1] for i in pick])
    B = np.array([shapes[i][2] for i in pick]); cfg = np.array([shapes[i][3] for i in pick])
    return _mk(name, pool.half, seed, pool, idx, H, W, B, cfg, noise), shapes, [float(x) for x in p]


def stream_bytes(reqs: list[dict]) -> bytes:
    """Canonical byte string (jsonl, sorted keys, UTF-8, non-ASCII not escaped)."""
    return ("".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for r in reqs)).encode("utf-8")


def stream_sha256(reqs: list[dict]) -> str:
    return hashlib.sha256(stream_bytes(reqs)).hexdigest()
