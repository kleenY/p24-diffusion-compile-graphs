"""S4-P P5c step 2: generate the NEW evaluation traces (half B, seeds 120-129, W1-W5 x N in {1e2,1e3,1e4,1e5}) and write the sha256 manifest (CPU, deterministic).
Copy of scripts/gen_traces_S4P.py (NOT modified) with the seeds changed to 120-129 and extra guards (preregistration goals/S4P_P5c_AOT_confirmatory_prereg_v1.md sec 3 + 8, 0444, SHA16 c14507708a968699).

Rules:
  * refuses to run unless results/S4P/p5c/cost_table_v2_aot.csv exists and is read-only (0444) -- the evaluation traces are generated only AFTER the AOT cost table is frozen (sec 8: cost_table_v2_aot.csv 0444 -> Dev-2 -> traces);
  * refuses to run unless the preregistration file is 0444 with the frozen SHA16, and the S4-P frozen v3 params (results/S4P/v3/policy_params_v3.json) are 0444 (as the original);
  * same generator and declared distributions as S3 / S4-P (p24.workload.generate, S2 split_A_B.parquet, B half only); seeds 120-129 are asserted disjoint from the dev seeds 0-4, the S3 evaluation seeds 100-109, the S4-P
    evaluation seeds 110-119 and the real-machine stream seed 101 (also from the seeds recorded in S2 seeds.json and p24.workload.generate DEV_SEEDS / EVAL_SEEDS);
  * output results/S4P/p5c/eval_traces/<W>_N<N>_seed<s>_B.jsonl (0444) + results/S4P/p5c/sha256_manifest_eval.txt (0444); an existing file with identical content is skipped, a different one is an error (no overwrite);
  * after generation NOTHING is read from the traces: this script prints counts and SHAs only.
`--selftest` (no guard, writes nothing): regenerates two S3 DEV traces (A half) through the same code path and asserts byte equality with the S3 files (does NOT generate any evaluation seed).
`--dry-run` checks the guards and lists the files that would be written (`--cost-table` may point at another path ONLY together with --dry-run: guard tests).
Usage: export CUDA_VISIBLE_DEVICES=''; nice -n 10 "$PROJ_PYTHON" -B code/scripts/gen_traces_S4P_P5c.py [--selftest|--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.workload import generate as G  # noqa: E402

S2 = ROOT / "results/S2/workload"
S3W = ROOT / "results/S3/workload"
OUT = ROOT / "results/S4P/p5c"
V3_FROZEN = ROOT / "results/S4P/v3/policy_params_v3.json"
COST_TABLE_AOT = OUT / "cost_table_v2_aot.csv"
PREREG = ROOT / "goals/S4P_P5c_AOT_confirmatory_prereg_v1.md"
PREREG_SHA16 = "c14507708a968699"
SEEDS = list(range(120, 130))
FORBIDDEN = set(range(0, 5)) | set(range(100, 120)) | {101}          # dev, S3 eval, S4-P eval, real-machine stream (preregistration sec 3)
EVAL_N = [100, 1000, 10000, 100000]
MANIFEST = "sha256_manifest_eval.txt"


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
        assert sha_file(path) == sha, f"exists with different content, refusing to overwrite: {path}"
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o444)
        tmp.rename(path)
    os.chmod(path, 0o444)
    return sha


def load_pools(need=("A", "B")):
    stats = json.loads((S2 / "workload_stats.json").read_text())
    meta = ROOT / "data/diffusiondb/metadata.parquet"
    assert sha_file(meta) == stats["metadata_sha256"], "metadata sha differs from S2"
    sp = S2 / "split_A_B.parquet"
    assert sha_file(sp) == (S2 / "split_A_B.parquet.sha256").read_text().split()[0], "split sha differs"
    df = pd.read_parquet(meta, columns=["prompt", "width", "height", "user_name"])
    split = pd.read_parquet(sp)
    assert len(split) == len(df) and (split["row"].to_numpy() == np.arange(len(df))).all()
    half_arr = np.where(split["half"].to_numpy() == "A", 0, 1)
    assert int((half_arr == 0).sum()) == stats["split_by_user"]["A"]["rows"] and int((half_arr == 1).sum()) == stats["split_by_user"]["B"]["rows"]
    return {h: G.make_pool(df, half_arr, h) for h in need}, half_arr, sp


def selftest():
    pools, half_arr, _ = load_pools(("A",))
    for (sc, n, seed) in [("W2", 1000, 0), ("W5", 10000, 3)]:
        reqs = G.generate_scenario(pools["A"], sc, n, seed)
        rows = np.array([r["src_row"] for r in reqs])
        assert (half_arr[rows] == 0).all()
        data = G.stream_bytes(reqs)
        ref = (S3W / f"dev_traces/{sc}_N{n}_seed{seed}_A.jsonl").read_bytes()
        assert data == ref, f"{sc} N{n} seed{seed}: regenerated dev trace differs from the S3 file"
        print(f"selftest ok: {sc} N={n} seed={seed} (A half) regenerated == S3 dev file (sha16 {sha_bytes(data)[:16]})")


def guard(cost_table: Path = COST_TABLE_AOT, prereg: Path = PREREG, v3_frozen: Path = V3_FROZEN, seeds=None):
    """Raises SystemExit('REFUSED: ...') / AssertionError on the first violated condition."""
    seeds = list(SEEDS if seeds is None else seeds)
    if not Path(cost_table).is_file():
        sys.exit(f"REFUSED: {cost_table} does not exist - the AOT cost table is not frozen yet; evaluation traces must be generated AFTER it is 0444.")
    mode = stat.S_IMODE(Path(cost_table).stat().st_mode)
    if mode != 0o444:
        sys.exit(f"REFUSED: {cost_table} must be 0444 (is {oct(mode)}).")
    if not Path(prereg).is_file() or stat.S_IMODE(Path(prereg).stat().st_mode) != 0o444 or sha_file(Path(prereg))[:16] != PREREG_SHA16:
        sys.exit(f"REFUSED: preregistration {prereg} missing / not 0444 / SHA16 differs from {PREREG_SHA16}.")
    if not Path(v3_frozen).is_file():
        sys.exit(f"REFUSED: {v3_frozen} does not exist - v3 is not frozen.")
    if stat.S_IMODE(Path(v3_frozen).stat().st_mode) != 0o444:
        sys.exit(f"REFUSED: {v3_frozen} must be 0444.")
    assert not (set(seeds) & FORBIDDEN), f"seeds overlap dev 0-4 / S3 100-109 / S4-P 110-119 / real-machine 101: {sorted(set(seeds) & FORBIDDEN)}"
    s2seeds = json.loads((S2 / "seeds.json").read_text())
    assert not (set(seeds) & (set(s2seeds["dev_seeds"]["values"]) | set(s2seeds["eval_seeds"]["values"]))), "new seeds overlap S2/S3 seeds"
    assert not (set(seeds) & (set(G.DEV_SEEDS) | set(G.EVAL_SEEDS))), "new seeds overlap the generator's DEV_SEEDS / EVAL_SEEDS"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cost-table", default=str(COST_TABLE_AOT), help="guard tests only, together with --dry-run")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if Path(a.cost_table).resolve() != COST_TABLE_AOT.resolve() and not a.dry_run:
        sys.exit("--cost-table may differ from the official path only together with --dry-run")
    guard(Path(a.cost_table))
    plan = [(sc, n, s) for sc in G.SCENARIOS for s in SEEDS for n in EVAL_N]
    if a.dry_run:
        print(f"guards ok ({a.cost_table}, {PREREG.name}, {V3_FROZEN.name}); would write {len(plan)} traces to {OUT}/eval_traces + {MANIFEST}")
        return
    pools, half_arr, sp = load_pools(("B",))
    manifest: dict[str, str] = {}
    for sc, n, seed in plan:
        rel = f"eval_traces/{sc}_N{n}_seed{seed}_B.jsonl"
        reqs = G.generate_scenario(pools["B"], sc, n, seed)
        assert len(reqs) == n, (rel, len(reqs))
        rows = np.array([r["src_row"] for r in reqs])
        assert (half_arr[rows] == 1).all(), f"{rel}: src_row outside the B half"
        assert all(r["req_id"].split("-")[1] == "B" for r in reqs)
        data = G.stream_bytes(reqs)
        assert data.count(b"\n") == n
        manifest[rel] = write_ro(OUT / rel, data)
        del reqs, data
    refs = {
        "../../S2/cost_table/merged/cost_table.csv": ROOT / "results/S2/cost_table/merged/cost_table.csv",
        "../../S3/cost_addendum/cost_addendum.csv": ROOT / "results/S3/cost_addendum/cost_addendum.csv",
        "../../S3/policy_params.json": ROOT / "results/S3/policy_params.json",
        "../../S3/budgets.json": ROOT / "results/S3/budgets.json",
        "../../S2/workload/seeds.json": S2 / "seeds.json",
        "../../S2/workload/split_A_B.parquet": sp,
        "../../S2/workload/workload_stats.json": S2 / "workload_stats.json",
        "../v3/policy_params_v3.json": V3_FROZEN,
        "../v3/params_prereg.json": ROOT / "results/S4P/v3/params_prereg.json",
        "../../../code/docs/workload_declaration.md": ROOT / "code/docs/workload_declaration.md",
        "../../../code/p24/workload/generate.py": ROOT / "code/p24/workload/generate.py",
        "../../../code/p24/policy/ours_v3.py": ROOT / "code/p24/policy/ours_v3.py",
        "../../../code/p24/policy/factory_v3.py": ROOT / "code/p24/policy/factory_v3.py",
        "../../../code/scripts/gen_traces_S4P_P5c.py": Path(__file__).resolve(),
        "cost_table_v2_aot.csv": COST_TABLE_AOT,
        "cost_table_v2_aot_meta.json": OUT / "cost_table_v2_aot_meta.json",
        "aot_fallback_shapes.csv": OUT / "aot_fallback_shapes.csv",
        "../../../goals/S4P_P5c_AOT_confirmatory_prereg_v1.md": PREREG,
    }
    for rel, p in refs.items():
        manifest[rel] = sha_file(p)
    text = "".join(f"{v}  {k}\n" for k, v in sorted(manifest.items(), key=lambda kv: (kv[0].startswith(".."), kv[0])))
    mp = OUT / MANIFEST
    if mp.exists():
        assert mp.read_text() == text, "manifest exists with different content"
        os.chmod(mp, 0o444)
    else:
        mp.write_text(text)
        os.chmod(mp, 0o444)
    print("traces", len(plan), "manifest_lines", len(manifest), "manifest_sha16", sha_file(mp)[:16])


if __name__ == "__main__":
    main()
