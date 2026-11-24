"""S4-P P3-a: generate the NEW real-validation streams (W2/W3/W5, N=300, seed 101, half B) for the P3 true-GPU validation block. CPU only.

Pre-registration: goals/S4P_uplift_line_prereg_v1.md (P3: "new-seed real validation streams of half B (W2/W3/W5, N=300, seed 101)").
Same generator (p24.workload.generate, frozen in S2), same B-half pool and split_A_B.parquet as scripts/gen_traces_S3.py; only the seed differs from S3 3.4
(seed 100).  Output (0444): results/S4P/workload/real_validation/{W2,W3,W5}_N300_seed101_B.jsonl + sha256_manifest_realval.txt (0444).
Checks: row count == N; every src_row belongs to half B; req_id prefix B; the stream differs from the S3 seed-100 stream and from the first 300 rows of
the S3 eval trace (seed 101, N=1000) -> reported (the generator seeds its RNG with N, so they are not prefixes of each other).
Existing files: identical content -> skipped, different -> error (never overwritten).
Usage (from the project root): export CUDA_VISIBLE_DEVICES=''; nice -n 10 "$PROJ_PYTHON" -B code/scripts/gen_realval_S4P.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.workload import generate as G  # noqa: E402

S2 = ROOT / "results/S2/workload"
OUT = ROOT / "results/S4P/workload/real_validation"
S3W = ROOT / "results/S3/workload"
REAL_N = 300
REAL_SEED = 101
REAL_SCEN = ["W2", "W3", "W5"]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_ro(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(data).hexdigest()
    if path.exists():
        assert sha_file(path) == sha, f"exists with different content, refusing to overwrite: {path}"
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o444)
        tmp.rename(path)
    os.chmod(path, 0o444)
    return sha


def main():
    stats = json.loads((S2 / "workload_stats.json").read_text())
    meta = ROOT / "data/diffusiondb/metadata.parquet"
    assert sha_file(meta) == stats["metadata_sha256"], "metadata sha differs from S2"
    sp = S2 / "split_A_B.parquet"
    assert sha_file(sp) == (S2 / "split_A_B.parquet.sha256").read_text().split()[0], "split sha differs"
    df = pd.read_parquet(meta, columns=["prompt", "width", "height", "user_name"])
    split = pd.read_parquet(sp)
    assert len(split) == len(df) and (split["row"].to_numpy() == np.arange(len(df))).all()
    half_arr = np.where(split["half"].to_numpy() == "A", 0, 1)
    pool = G.make_pool(df, half_arr, "B")
    lines, info = [], {}
    for sc in REAL_SCEN:
        reqs = G.generate_scenario(pool, sc, REAL_N, REAL_SEED)
        assert len(reqs) == REAL_N
        rows = np.array([r["src_row"] for r in reqs])
        assert (half_arr[rows] == 1).all(), f"{sc}: src_row outside half B"
        assert all(r["req_id"].split("-")[1] == "B" for r in reqs)
        data = G.stream_bytes(reqs)
        assert data.count(b"\n") == REAL_N
        rel = f"{sc}_N{REAL_N}_seed{REAL_SEED}_B.jsonl"
        sha = write_ro(OUT / rel, data)
        # relation to existing S3 streams (informative)
        old = (S3W / f"real_validation/{sc}_N{REAL_N}_seed100_B.jsonl").read_text().splitlines()
        ev = (S3W / f"eval_traces/{sc}_N1000_seed{REAL_SEED}_B.jsonl").read_text().splitlines()
        new = data.decode().splitlines()
        shapes = sorted({(r["H"], r["W"], r["B"], r["cfg"]) for r in reqs})
        info[rel] = dict(sha256=sha, n=REAL_N, n_shapes=len(shapes), same_as_seed100=int(new == old),
                         prefix_of_eval_N1000_seed101=int(new == ev[:REAL_N]),
                         n_src_rows_shared_with_seed100=int(len({r["src_row"] for r in reqs} & {json.loads(l)["src_row"] for l in old})))
        lines.append(f"{sha}  {rel}")
    # frozen inputs (relative to this dir) for sha256sum -c
    refs = {"../../../S2/cost_table/merged/cost_table.csv": ROOT / "results/S2/cost_table/merged/cost_table.csv",
            "../../../S2/workload/split_A_B.parquet": sp,
            "../../../../code/p24/workload/generate.py": ROOT / "code/p24/workload/generate.py"}
    for rel, p in refs.items():
        lines.append(f"{sha_file(p)}  {rel}")
    text = "\n".join(lines) + "\n"
    mp = OUT / "sha256_manifest_realval.txt"
    if mp.exists():
        assert mp.read_text() == text, "manifest exists with different content"
        os.chmod(mp, 0o444)
    else:
        mp.write_text(text)
        os.chmod(mp, 0o444)
    print(json.dumps(info, indent=1))
    print("manifest", mp, "manifest_sha16", sha_file(mp)[:16])


if __name__ == "__main__":
    main()
