"""Rebuild the request traces that are not shipped (N > 1000) and check every trace against the frozen manifest.

Public-repository helper (added at export time, 2026-11-23). It uses the same frozen generator as
`code/scripts/gen_traces_S3.py` (`p24.workload.generate`, same scenarios, seeds and halves) but checks only the
trace lines of `results/S3/workload/sha256_manifest.txt`. The input-file lines of that manifest ("../" entries) hash the
internal, untranslated source files and therefore do not match the translated copies in this repository
(see "Provenance of hashes" in README.md).

Run from the repository root, after placing the DiffusionDB metadata at data/diffusiondb/metadata.parquet:
    PYTHONPATH=code python -m scripts.rebuild_traces            # writes missing traces, verifies all
    PYTHONPATH=code python -m scripts.rebuild_traces --check    # verify only, write nothing
Exit code 0 only if every trace listed in the manifest exists (or was written) with the listed SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from p24.workload import generate as G

ROOT = Path(os.environ.get("PROJ_ROOT", "."))
S2 = ROOT / "results/S2/workload"
OUT = ROOT / "results/S3/workload"
META = ROOT / "data/diffusiondb/metadata.parquet"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse(rel: str):
    """'eval_traces/W2_N10000_seed100_B.jsonl' -> ('W2', 10000, 100, 'B')"""
    stem = Path(rel).stem
    sc, n, seed, half = stem.split("_")
    return sc, int(n[1:]), int(seed[4:]), half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not generate missing traces")
    a = ap.parse_args()
    man = {}
    for line in (OUT / "sha256_manifest.txt").read_text().splitlines():
        if line.strip() and not line.split()[1].startswith(".."):
            sha, rel = line.split()
            man[rel] = sha
    missing = [r for r in man if not (OUT / r).exists()]
    print(f"manifest traces: {len(man)}; present: {len(man) - len(missing)}; missing: {len(missing)}", flush=True)
    if missing and not a.check:
        stats = json.loads((S2 / "workload_stats.json").read_text())
        assert META.exists(), f"place the DiffusionDB metadata at {META} (see docs/DATASET.md)"
        assert sha_file(META) == stats["metadata_sha256"], "metadata.parquet differs from the one used in the study"
        sp = S2 / "split_A_B.parquet"
        assert sha_file(sp) == (S2 / "split_A_B.parquet.sha256").read_text().split()[0], "split file hash mismatch"
        df = pd.read_parquet(META, columns=["prompt", "width", "height", "user_name"])
        split = pd.read_parquet(sp)
        assert len(split) == len(df) and (split["row"].to_numpy() == np.arange(len(df))).all()
        half_arr = np.where(split["half"].to_numpy() == "A", 0, 1)
        pools = {h: G.make_pool(df, half_arr, h) for h in "AB"}
        for rel in sorted(missing):
            sc, n, seed, half = parse(rel)
            reqs = G.generate_scenario(pools[half], sc, n, seed)
            assert len(reqs) == n
            rows = np.array([r["src_row"] for r in reqs])
            assert (half_arr[rows] == (0 if half == "A" else 1)).all(), f"{rel}: rows outside half {half}"
            data = G.stream_bytes(reqs)
            p = OUT / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(data)
            os.chmod(tmp, 0o444)
            tmp.rename(p)
    bad = [r for r in man if not (OUT / r).exists() or sha_file(OUT / r) != man[r]]
    print(f"verified: {len(man) - len(bad)} / {len(man)} traces match the manifest" + (f"; mismatched or missing: {bad[:5]}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
