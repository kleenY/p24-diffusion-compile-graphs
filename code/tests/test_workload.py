"""S2 2.4 workload tests (synthetic data + real split file).  Pure CPU."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p24.workload import generate as G
from p24.workload.split import (RESOLUTIONS, half_by_prompt, half_by_user, map_resolution,
                                map_resolutions)

ROOT = Path(".")


def synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [(512, 512), (512, 768), (768, 512), (640, 640), (1024, 1024), (448, 832), (832, 448), (768, 768)]
    sel = rng.integers(0, len(sizes), n)
    users = [f"u{int(x)}" for x in rng.integers(0, 300, n)]
    for i in range(0, n, 97):
        users[i] = None if i % 2 == 0 else ""
    return pd.DataFrame({
        "prompt": [f"p{i}" for i in range(n)],
        "width": [sizes[s][0] for s in sel], "height": [sizes[s][1] for s in sel],
        "user_name": users,
    })


@pytest.fixture(scope="module")
def pools():
    df = synth()
    h = half_by_user(df)
    return df, h, G.make_pool(df, h, "A"), G.make_pool(df, h, "B")


def test_deterministic(pools):
    _, _, A, _ = pools
    for sc in G.SCENARIOS:
        a = G.stream_sha256(G.generate_scenario(A, sc, 500, 3))
        b = G.stream_sha256(G.generate_scenario(A, sc, 500, 3))
        c = G.stream_sha256(G.generate_scenario(A, sc, 500, 4))
        assert a == b and a != c
    for v in G.VALIDATION:
        assert G.stream_sha256(G.generate_validation(A, v)[0]) == G.stream_sha256(G.generate_validation(A, v)[0])


def test_split_disjoint_cover(pools):
    df, h, A, B = pools
    assert set(A.rows).isdisjoint(set(B.rows))
    assert len(A) + len(B) == len(df)
    assert set(h.tolist()) <= {0, 1}
    # the same user never crosses halves; a missing user is split by prompt
    u = df["user_name"]
    ok = u.notna() & (u != "")
    for name, g in df[ok].groupby("user_name"):
        assert len(set(h[g.index])) == 1
    exp = int(hashlib.sha256(b"p24-s2|u5").hexdigest()[:16], 16) & 1
    assert h[df.index[df.user_name == "u5"][0]] == exp
    miss = ~ok
    exp_p = np.array([int(hashlib.sha256(("p24-s2|prompt|" + p).encode()).hexdigest()[:16], 16) & 1
                      for p in df.loc[miss, "prompt"]])
    assert (h[miss.to_numpy()] == exp_p).all()
    assert (half_by_prompt(df)[miss.to_numpy()] == exp_p).all()


def test_mapping_boundaries():
    assert map_resolution(9, 10) == (512, 512)           # a=0.9 not portrait
    assert map_resolution(899, 1000) == (768, 512)       # a<0.9 portrait
    assert map_resolution(11, 10) == (512, 512)           # a=1.1 not landscape
    assert map_resolution(1101, 1000) == (512, 768)       # a>1.1 landscape
    assert map_resolution(640, 640) == (512, 512)        # area exactly 640^2
    assert map_resolution(641, 640) == (768, 768)        # square, area slightly above 640^2
    assert map_resolution(639, 640) == (512, 512)
    assert map_resolution(768, 768) == (768, 768)
    assert map_resolution(512, 768) == (768, 512)        # portrait: H768 W512
    assert map_resolution(768, 512) == (512, 768)
    w = np.array([9, 899, 11, 1101, 640, 642, 512, 768]); h = np.array([10, 1000, 10, 1000, 640, 640, 768, 512])
    vec = map_resolutions(w, h)
    for i in range(len(w)):
        assert tuple(vec[i]) == map_resolution(int(w[i]), int(h[i]))


def test_w5_prefix_covers_32(pools):
    _, _, A, _ = pools
    for n in (200, 1000):
        reqs = G.generate_scenario(A, "W5", n, 0)
        k = G.w5_prefix_len(n)
        assert k >= 32
        assert {(r["H"], r["W"], r["B"], r["cfg"]) for r in reqs[:k]} == set(G.ALL_SHAPES)
        assert len(G.ALL_SHAPES) == 32


def test_scenario_semantics(pools):
    _, _, A, _ = pools
    w1 = G.generate_scenario(A, "W1", 300, 0)
    assert {(r["H"], r["W"], r["B"], r["cfg"]) for r in w1} == {(512, 512, 1, 0)}
    w3 = G.generate_scenario(A, "W3", 4000, 0)
    assert {r["B"] for r in w3} == {1, 2, 3, 4}
    # W2 shapes come from the row mapping of that row
    w2 = G.generate_scenario(A, "W2", 300, 1)
    rowpos = {int(r): i for i, r in enumerate(A.rows)}
    for r in w2[:50]:
        assert (r["H"], r["W"]) == tuple(A.hw[rowpos[r["src_row"]]])
    # W4: first half as the row mapping of W2, second half swaps portrait/landscape
    w4 = G.generate_scenario(A, "W4", 200, 1)
    for i, r in enumerate(w4):
        H0, W0 = map(int, A.hw[rowpos[r["src_row"]]])
        exp = (H0, W0) if i < 100 else ((H0, W0) if H0 == W0 else (W0, H0))
        assert (r["H"], r["W"]) == exp


def test_validation_has_b3(pools):
    _, _, A, _ = pools
    for name, k in (("V2", 8), ("V3", 12)):
        reqs, shapes, p = G.generate_validation(A, name)
        assert len(shapes) == k and any(s[2] == 3 for s in shapes)
        assert abs(sum(p) - 1) < 1e-9
        assert {(r["H"], r["W"], r["B"], r["cfg"]) for r in reqs} <= set(shapes)
        assert any(r["B"] == 3 for r in reqs)
    reqs, shapes, _ = G.generate_validation(A, "V3")
    assert {(r["H"], r["W"], r["B"], r["cfg"]) for r in reqs[:12]} == set(shapes)
    assert len(G.generate_validation(A, "V1")[0]) == 400


def test_select_top_replaces_with_b3():
    probs = {s: 0.0 for s in G.ALL_SHAPES}
    for i, s in enumerate(G.ALL_SHAPES):
        probs[s] = 1.0 / (1 + i) if s[2] != 3 else 1e-6 / (1 + i)
    top = G.select_top_shapes(probs, 8)
    assert any(s[2] == 3 for s in top) and len(set(top)) == 8


REAL = ROOT / "results/S2/workload/split_A_B.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="split file does not exist")
def test_real_split():
    s = pd.read_parquet(REAL)
    assert list(s.columns) == ["row", "half"]
    assert len(s) == 2_000_000 and (s["row"].to_numpy() == np.arange(2_000_000)).all()
    assert set(s["half"].unique()) == {"A", "B"}
    sha = (REAL.parent / "split_A_B.parquet.sha256").read_text().split()[0]
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == sha
    assert 0.4 < (s["half"] == "A").mean() < 0.6
