"""A/B split and resolution mapping (S2 2.4).  Pure functions, no IO.

Split: h = int(sha256(("p24-s2|" + user_name).encode()).hexdigest()[:16], 16); even -> A, odd -> B.
Missing/empty user_name: split the same way by "p24-s2|prompt|" + prompt.
Alternative split: all rows by "p24-s2|prompt|" + prompt, used only for the comparison in the report.
The shape key is always (H, W).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SALT = "p24-s2|"
PROMPT_SALT = "p24-s2|prompt|"

# the 4 mapped resolutions, keyed by (H, W)
RES_SQUARE_S = (512, 512)
RES_LANDSCAPE = (512, 768)   # H512 x W768 (a>1.1)
RES_PORTRAIT = (768, 512)    # H768 x W512 (a<0.9)
RES_SQUARE_L = (768, 768)
RESOLUTIONS = [RES_SQUARE_S, RES_LANDSCAPE, RES_PORTRAIT, RES_SQUARE_L]
SQUARE_AREA_LIMIT = 640 * 640


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def map_resolution(width: int, height: int) -> tuple[int, int]:
    """Raw (width, height) -> shape key (H, W)."""
    a = width / height
    if a < 0.9:
        return RES_PORTRAIT
    if a > 1.1:
        return RES_LANDSCAPE
    return RES_SQUARE_S if width * height <= SQUARE_AREA_LIMIT else RES_SQUARE_L


def map_resolutions(width, height) -> np.ndarray:
    """Vectorized mapping; returns an int16 array of shape (n, 2), columns (H, W)."""
    w = np.asarray(width, dtype=np.int64)
    h = np.asarray(height, dtype=np.int64)
    a = w / h
    out = np.empty((len(w), 2), dtype=np.int16)
    sq = ~((a < 0.9) | (a > 1.1))
    small = (w * h) <= SQUARE_AREA_LIMIT
    out[:] = RES_SQUARE_L
    out[sq & small] = RES_SQUARE_S
    out[a < 0.9] = RES_PORTRAIT
    out[a > 1.1] = RES_LANDSCAPE
    return out


def res_index(hw: np.ndarray) -> np.ndarray:
    """(H,W) pairs of shape (n,2) -> index into RESOLUTIONS."""
    idx = np.full(len(hw), -1, dtype=np.int8)
    for i, (H, W) in enumerate(RESOLUTIONS):
        idx[(hw[:, 0] == H) & (hw[:, 1] == W)] = i
    assert (idx >= 0).all()
    return idx


def _hash_parity(strings: pd.Series, salt: str) -> np.ndarray:
    """Hash parity of a string column computed per unique value; returns uint8 (0=A, 1=B)."""
    codes, uniques = pd.factorize(strings)
    par = np.fromiter((_h(salt + u) & 1 for u in uniques), dtype=np.uint8, count=len(uniques))
    return par[codes]


def user_missing_mask(df: pd.DataFrame) -> np.ndarray:
    u = df["user_name"]
    return (u.isna() | (u.astype(str) == "")).to_numpy()


def half_by_user(df: pd.DataFrame) -> np.ndarray:
    """Primary split: by user_name; missing/empty ones by prompt.  Returns uint8, 0=A, 1=B."""
    miss = user_missing_mask(df)
    by_prompt = _hash_parity(df["prompt"].fillna(""), PROMPT_SALT)
    out = by_prompt.copy()
    ok = ~miss
    if ok.any():
        out[ok] = _hash_parity(df.loc[ok, "user_name"].reset_index(drop=True), SALT)
    return out


def half_by_prompt(df: pd.DataFrame) -> np.ndarray:
    """Alternative split: all rows by prompt."""
    return _hash_parity(df["prompt"].fillna(""), PROMPT_SALT)
