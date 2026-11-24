"""S3 3.0: generate evaluation / real-validation / development traces and write sha256_manifest.txt (CPU, deterministic).

Usage (project root = PROJ_ROOT, nice 10, single process):
    export CUDA_VISIBLE_DEVICES=''
    nice -n 10 "$PROJ_PYTHON" -B code/scripts/gen_traces_S3.py
Rules:
  * evaluation traces (W1-W5 x seeds 100-109 x N in {1e2,1e3,1e4,1e5}) are drawn from half B only;
    real validation (W2/W3/W5, seed 100, N=REAL_N) is also drawn from half B only;
    development traces (W1-W5 x seeds 0-4 x N in {1e3,1e4}) are drawn from half A only.
  * Reuses the generator p24.workload.generate and split_A_B.parquet frozen in S2 (read only); the N=1000 development traces must be byte-identical to S2 (SHA check).
  * Existing files: skipped if the content is identical, an error if different (never overwritten).  Generated files and the manifest are set to 0444.
  * Every trace has N lines; every src_row belongs to the corresponding half (checked against split_A_B.parquet).
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
OUT = ROOT / "results/S3/workload"
EVAL_N = [100, 1000, 10000, 100000]
DEV_N = [1000, 10000]
REAL_N = 300                       # real-validation N (same order as the S2 validation stream V2; the plan says N~300-400, fixed at 300 here)
REAL_SCEN = ["W2", "W3", "W5"]
REAL_SEED = 100


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_ro(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = sha_bytes(data)
    if path.exists():
        assert sha_file(path) == sha, f"already exists with different content, refusing to overwrite: {path}"
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o444)
        tmp.rename(path)
    os.chmod(path, 0o444)
    return sha


def main():
    seeds = json.loads((S2 / "seeds.json").read_text())
    assert seeds["dev_seeds"]["values"] == G.DEV_SEEDS and seeds["eval_seeds"]["values"] == G.EVAL_SEEDS
    stats = json.loads((S2 / "workload_stats.json").read_text())
    meta = ROOT / "data/diffusiondb/metadata.parquet"
    assert sha_file(meta) == stats["metadata_sha256"], "metadata sha differs from S2"
    sp = S2 / "split_A_B.parquet"
    assert sha_file(sp) == (S2 / "split_A_B.parquet.sha256").read_text().split()[0], "split sha mismatch"

    df = pd.read_parquet(meta, columns=["prompt", "width", "height", "user_name"])
    split = pd.read_parquet(sp)
    assert len(split) == len(df) and (split["row"].to_numpy() == np.arange(len(df))).all()
    half_arr = np.where(split["half"].to_numpy() == "A", 0, 1)
    assert int((half_arr == 0).sum()) == stats["split_by_user"]["A"]["rows"]
    assert int((half_arr == 1).sum()) == stats["split_by_user"]["B"]["rows"]
    pools = {h: G.make_pool(df, half_arr, h) for h in "AB"}
    rowhalf = {h: set() for h in "AB"}  # lazy: check directly with half_arr
    del rowhalf

    manifest: dict[str, str] = {}
    log = []

    def emit(rel: str, scen: str, n: int, seed: int, half: str):
        reqs = G.generate_scenario(pools[half], scen, n, seed)
        assert len(reqs) == n, (rel, len(reqs), n)
        rows = np.array([r["src_row"] for r in reqs])
        want = 0 if half == "A" else 1
        assert (half_arr[rows] == want).all(), f"{rel}: src_row out of range (not in half {half})"
        assert all(r["req_id"].split("-")[1] == half for r in reqs)
        data = G.stream_bytes(reqs)
        assert data.count(b"\n") == n
        manifest[rel] = write_ro(OUT / rel, data)
        log.append((rel, n, half, manifest[rel]))

    # evaluation (half B)
    for sc in G.SCENARIOS:
        for seed in G.EVAL_SEEDS:
            for n in EVAL_N:
                emit(f"eval_traces/{sc}_N{n}_seed{seed}_B.jsonl", sc, n, seed, "B")
    # real validation (half B)
    for sc in REAL_SCEN:
        emit(f"real_validation/{sc}_N{REAL_N}_seed{REAL_SEED}_B.jsonl", sc, REAL_N, REAL_SEED, "B")
    # development (half A)
    s2man = {l.split()[1]: l.split()[0] for l in (S2 / "sha256_manifest.txt").read_text().splitlines() if l.strip()}
    for sc in G.SCENARIOS:
        for seed in G.DEV_SEEDS:
            for n in DEV_N:
                rel = f"dev_traces/{sc}_N{n}_seed{seed}_A.jsonl"
                emit(rel, sc, n, seed, "A")
                if n == 1000:   # byte-identical to the N=1000 development trace frozen in S2 (reuse)
                    assert manifest[rel] == s2man[f"dev_traces/{sc}_N1000_seed{seed}_A.jsonl"], f"{rel} differs from S2"

    # SHA of the frozen input files (path relative to OUT, checkable with sha256sum -c)
    refs = {
        "../../S2/cost_table/merged/cost_table.csv": ROOT / "results/S2/cost_table/merged/cost_table.csv",
        "../../S2/workload/seeds.json": S2 / "seeds.json",
        "../../S2/workload/split_A_B.parquet": sp,
        "../../S2/workload/workload_stats.json": S2 / "workload_stats.json",
        "../../S2/memory_budget.json": ROOT / "results/S2/memory_budget.json",
        "../../../code/docs/workload_declaration.md": ROOT / "code/docs/workload_declaration.md",
        "../../../code/p24/workload/generate.py": ROOT / "code/p24/workload/generate.py",
    }
    for rel, p in refs.items():
        manifest[rel] = sha_file(p)
    text = "".join(f"{v}  {k}\n" for k, v in sorted(manifest.items(), key=lambda kv: (kv[0].startswith(".."), kv[0])))
    mp = OUT / "sha256_manifest.txt"
    if mp.exists():
        assert mp.read_text() == text, "manifest already exists with different content"
        os.chmod(mp, 0o444)
    else:
        mp.write_text(text); os.chmod(mp, 0o444)
    print("traces", len(log), "manifest_lines", len(manifest), "manifest_sha256", sha_file(mp))
    print("cost_table_sha16", manifest["../../S2/cost_table/merged/cost_table.csv"][:16])


if __name__ == "__main__":
    main()
