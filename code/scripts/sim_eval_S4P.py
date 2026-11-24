"""S4-P P1 batch runner of the request-level simulator for policy v3 (CPU only, <=8 worker processes, nice 10).

Reuses the S3 3.3 machinery without modifying any S3 file: sim_eval_S3 (trace store + manifest check, cold levels, S2-table imputation of the
missing cold_empty cells, generated dev traces) and the frozen simulator / cost table / 3.1b addendum. Policies come from p24.policy.factory_v3
(S3 factory + "Ours-v3").  One raw-CSV row = one simulation of ONE process (variant, workload, N, seed, cold level).

  --split dev   half A, seeds 0-4 (S3 dev traces, N 1e3/1e4; --dev-extra-N generates more N in memory from the A half pool, flagged
                trace_kind=gen_default).  v3 variants: --v3-group g (a group of results/S4P/v3/params_prereg.json) or --v3-all-groups.
  --split eval  half B, NEW seeds 110-119, W1-W5 x N in {1e2,1e3,1e4,1e5}: results/S4P/workload/eval_traces (manifest results/S4P/workload/sha256_manifest.txt).
                Needs results/S4P/v3/policy_params_v3.json to be read-only (0444); its SHAs (ours_v3.py, factory_v3.py, cost table, addendum, S3 params)
                must match the files; the v3 variant takes its params from it.  NEVER run before the main session orders it.
  Variants of the `main` experiment: all S3 policies (B0-B7, Ours = v2, Ours-NE; S3 frozen params results/S3/policy_params.json) + "Ours-v3".
  Cold levels: empty / disk / hot as sim_eval_S3 (empty = cache_mode empty on the imputed table, imputed=1; hot = warm with the first shape at the in-process tier).

--dry-run lists the guards, the task grid and the file checks WITHOUT reading any trace content and without simulating.
Usage (project root, after `source ops/project_env.sh`):
  export CUDA_VISIBLE_DEVICES=''; nice -n 10 "$PROJ_PYTHON" -B code/scripts/sim_eval_S4P.py --split dev --v3-group g3_K30_a0.5 --tiny --out results/S4P/v3/dev_smoke
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import pathlib
import stat
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.policy import CostView, shape_of  # noqa: E402
from p24.policy.factory_v3 import NAMES, make_policy  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402
from p24.sim import simulator as SIM  # noqa: E402
from p24.sim.costs import CostTable, MissingCost  # noqa: E402
from scripts import sim_eval_S3 as S3  # noqa: E402

S4 = ROOT / "results/S4P"
PREREG = S4 / "v3/params_prereg.json"
V3_FROZEN = S4 / "v3/policy_params_v3.json"
EVAL_DIR = S4 / "workload"
EVAL_MANIFEST = EVAL_DIR / "sha256_manifest.txt"
EVAL_SEEDS = list(range(110, 120))
EVAL_N = [100, 1000, 10000, 100000]
SPLITS = {"dev": dict(S3.SPLITS["dev"]), "eval": dict(half="B", dir="eval_traces", suffix="B", seeds=EVAL_SEEDS, N=EVAL_N)}
sha16 = S3.sha16
W: Dict[str, object] = {}


@dataclass
class Variant:
    label: str
    policy: str
    params: dict = field(default_factory=dict)


# ====================================================================================================== params / variants
def v2_params() -> dict:
    return S3.base_params(S3.load_params())              # S3 frozen v2 params (tiers as tuple)


def prereg_candidates() -> dict:
    return json.loads(PREREG.read_text())["candidates"]


def frozen_v3_params() -> dict:
    return json.loads(V3_FROZEN.read_text())["selected"]["params"]


def build_variants(P2: dict, v3: Dict[str, dict], names: Optional[List[str]] = None) -> List[Variant]:
    """S3 policies with the S3 frozen params (Ours = v2) + one 'Ours-v3[:group]' variant per entry of `v3` (group -> extra params)."""
    out = [Variant(n, n, dict(P2)) for n in S3.NAMES]
    for g, extra in v3.items():
        out.append(Variant("Ours-v3" if g == "" else f"Ours-v3:{g}", "Ours-v3", {**P2, **extra}))
    if names:
        out = [v for v in out if v.label in names]
    return out


# ====================================================================================================== guards
def _code_shas() -> Dict[str, str]:
    base = ROOT / "code/p24/policy"
    return {k: sha16(base / k) for k in ("ours_v3.py", "factory_v3.py", "ours.py", "factory.py", "base.py", "baselines.py", "costview.py")}


def eval_guards() -> dict:
    """Eval split: the frozen v3 params must exist (0444) and every SHA it records must match the files on disk."""
    if not V3_FROZEN.is_file():
        sys.exit(f"ERROR: {V3_FROZEN} does not exist: policy v3 is not frozen yet (eval split refused).")
    mode = stat.S_IMODE(V3_FROZEN.stat().st_mode)
    assert mode == 0o444, f"{V3_FROZEN} must be 0444 (is {oct(mode)})"
    fj = json.loads(V3_FROZEN.read_text())
    cur = dict(ours_v3_py=sha16(ROOT / "code/p24/policy/ours_v3.py"), factory_v3_py=sha16(ROOT / "code/p24/policy/factory_v3.py"),
               cost_table=sha16(S3.CT_PATH), cost_addendum=sha16(S3.ADDENDUM_PATH), s3_params=sha16(S3.PARAMS_PATH),
               prereg=sha16(ROOT / fj["prereg_file"]))
    for k, v in cur.items():
        assert fj["sha16"].get(k) == v, f"frozen v3 params record {k}={fj['sha16'].get(k)} but the file has {v}"
    return dict(v3_params_sha16=sha16(V3_FROZEN), **cur)


def dev_guards(pj: dict) -> dict:
    info = dict(cost_table_sha16=sha16(S3.CT_PATH), cost_addendum_sha16=sha16(S3.ADDENDUM_PATH), s3_manifest_sha16=sha16(S3.MANIFEST_PATH),
                s3_params_sha16=sha16(S3.PARAMS_PATH), prereg_sha16=sha16(PREREG), code_sha16=_code_shas())
    assert info["cost_table_sha16"] == pj["cost_table_sha16"] and info["s3_manifest_sha16"] == pj["manifest_sha16"]
    assert stat.S_IMODE(PREREG.stat().st_mode) == 0o444, "params_prereg.json must be read-only (pre-registration)"
    return info


# ====================================================================================================== worker
def _init_worker(cfg: dict):
    try:
        os.nice(10)
    except OSError:
        pass
    W.update(cfg)
    sp = SPLITS[cfg["split"]]
    if cfg["split"] == "dev":
        W["store"] = S3.TraceStore("dev", cfg["manifest"])
    else:
        W["store"] = None
    ADD.install_simulator_hook()
    add_df = ADD.read_addendum(S3.ADDENDUM_PATH)
    W["ct"] = ADD.load_with_addendum(S3.impute_empty(pd.read_csv(S3.CT_PATH))[0], add_df)     # S3 main evaluation: imputed table for every cold level
    W["views"] = {}


def _get_trace(w, n, seed, tkind):
    if W["split"] == "dev":
        return W["store"].get(w, n, seed, tkind)
    rel = f"eval_traces/{w}_N{n}_seed{seed}_B.jsonl"
    p = EVAL_DIR / rel
    assert p.exists(), p
    assert hashlib.sha256(p.read_bytes()).hexdigest() == W["manifest"][rel], f"manifest SHA mismatch {rel}"
    reqs = S3.read_trace_file(p)
    assert len(reqs) == n
    return reqs


def run_task(task):
    w, n, seed, tkind = task
    ct = W["ct"]
    reqs = _get_trace(w, n, seed, tkind)
    oc = dict(collections.Counter(shape_of(q) for q in reqs))
    rows = []
    for cold in W["colds"]:
        cm, hot = S3.COLDS[cold]
        S3._set_hot(hot)
        cv = W["views"].setdefault(cm, CostView(ct, cm, rebuild_override=getattr(ct, "rebuild_override", None)))
        opt = SIM.SimOptions(cache_mode=cm, mem_budget_mb=W["budget_mb"], overhead_aware=True)
        for v in W["variants"]:
            row = dict(split=W["split"], variant=v.label, policy=v.policy, workload=w, N=n, seed=seed, budget="default", budget_mb=W["budget_mb"],
                       cold=cold, trace_kind=tkind, imputed=int(cm == "empty"))
            t0 = time.perf_counter()
            try:
                pol = make_policy(v.policy, cost=cv, params=v.params, prior=W["prior"], oracle_counts=oc)
                s = SIM.simulate(reqs, pol, ct, opt).summary
                row.update(status="ok", error="", service_s=s["service_s"], warmup_s=s["warmup_s"], peak_graph_mb=s["peak_graph_mb"],
                           peak_reserved_est_mb=s["peak_reserved_est_mb"], n_requests=s["n_requests"], n_shapes=s["n_shapes"],
                           n_capture=s.get("n_capture", 0), n_compile=s.get("n_compile", 0), n_compile_capture=s.get("n_compile_capture", 0),
                           n_capture_denied=s.get("n_capture_denied", 0), n_eager_first=s.get("n_eager_first", 0),
                           n_eager_fallback_limit=s.get("n_eager_fallback_limit", 0), n_evicted=s["n_evicted_total"], n_denied=s["n_denied_total"],
                           n_steady=s.get("n_steady", 0), n_events_with_fallback=s["n_events_with_fallback"])
            except MissingCost as e:
                row.update(status="missing_cost", error=str(e)[:160])
            except Exception as e:  # noqa: BLE001  (recorded; the summary counts them)
                row.update(status="error", error=f"{type(e).__name__}: {str(e)[:160]}")
            row["sim_wall_s"] = time.perf_counter() - t0
            rows.append(row)
    S3._set_hot(False)
    return rows


def run_grid(split: str, workloads, Ns, seeds, colds, variants: List[Variant], procs: int, dev_extra_N=(), manifest=None, label="") -> pd.DataFrame:
    """Simulate the full grid; returns the raw frame (sorted). `dev_extra_N` (dev only) are generated in memory from the A half pool."""
    bj = json.loads(S3.BUDGETS_PATH.read_text())["tiers"]
    specs = []
    for w in workloads:
        for n in Ns:
            for s in seeds:
                kind = "gen_default" if (split == "dev" and n in dev_extra_N and n not in SPLITS["dev"]["N"]) else "file"
                specs.append((w, n, s, kind))
    if split == "dev" and any(k != "file" for *_, k in specs):
        S3.pool_arrays("A")                       # build the npz cache once in the parent
    cfg = dict(split=split, manifest=manifest or S3.read_manifest(), colds=list(colds), variants=variants, budget_mb=float(bj["default"]["mb"]),
               prior=S3.make_priors(False)["A"])
    t0 = time.time()
    rows = []
    # largest traces first (load balance)
    specs.sort(key=lambda t: -t[1])
    print(f"[{label or split}] tasks={len(specs)} variants={len(variants)} colds={list(colds)} sims~{len(specs)*len(variants)*len(colds)} procs={procs}", flush=True)
    if procs == 1:
        _init_worker(cfg)
        for t in specs:
            rows += run_task(t)
    else:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, specs, chunksize=1)):
                rows += r
                if (i + 1) % max(1, len(specs) // 10) == 0:
                    print(f"[{label or split}] {i+1}/{len(specs)} tasks {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["cold", "workload", "N", "seed", "variant"]).reset_index(drop=True)
    print(f"[{label or split}] rows={len(df)} status={df.status.value_counts().to_dict()} wall={time.time()-t0:.0f}s", flush=True)
    return df


# ====================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--out", help="output directory (refuses to overwrite main_raw.csv)")
    ap.add_argument("--workloads", type=lambda s: s.split(","))
    ap.add_argument("--N", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--colds", type=lambda s: s.split(","), default=["empty", "disk", "hot"])
    ap.add_argument("--policies", type=lambda s: s.split(","), help="variant labels, e.g. B0,B1,Ours,Ours-v3")
    ap.add_argument("--v3-group", help="dev only: group of params_prereg.json candidates (comma list allowed) -> variant 'Ours-v3:<group>'")
    ap.add_argument("--v3-all-groups", action="store_true", help="dev only: all four prereg candidates")
    ap.add_argument("--prereg", help="dev only: pre-registration json holding the candidate groups (default results/S4P/v3/params_prereg.json)")
    ap.add_argument("--dev-extra-N", type=lambda s: [int(x) for x in s.split(",")], default=[])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tiny", action="store_true", help="1 seed x 1 N x 1 workload x 1 cold: plumbing check")
    ap.add_argument("--dry-run", action="store_true", help="guards + task grid + file existence only; no trace content is read, nothing is simulated")
    a = ap.parse_args()
    assert 1 <= a.procs <= 8, "at most 8 worker processes"
    sp = SPLITS[a.split]
    if a.prereg:
        globals()["PREREG"] = pathlib.Path(a.prereg).resolve()
    pj = S3.load_params()
    P2 = v2_params()
    if a.split == "eval":
        info = eval_guards()
        v3 = {"": dict(frozen_v3_params())}                # full frozen param dict (v2 params + K_pay + alpha_c1)
        assert not (a.v3_group or a.v3_all_groups or a.dev_extra_N), "eval split uses the frozen v3 params only"
        manifest = {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in EVAL_MANIFEST.read_text().splitlines() if l.strip()}
    else:
        info = dev_guards(pj)
        cands = prereg_candidates()
        groups = list(cands) if a.v3_all_groups else (a.v3_group.split(",") if a.v3_group else [])
        for g in groups:
            assert g in cands, g
        v3 = {g: dict(cands[g]) for g in groups}
        manifest = S3.read_manifest()
    seeds, Ns, wls = a.seeds or sp["seeds"], a.N or sp["N"], a.workloads or S3.G.SCENARIOS
    colds = a.colds
    if a.tiny:
        seeds, Ns, wls, colds = seeds[:1], (a.N or [1000]), (a.workloads or ["W2"]), (a.colds if a.colds != ["empty", "disk", "hot"] else ["disk"])
    if a.split == "eval":
        assert set(seeds) <= set(EVAL_SEEDS) and set(Ns) <= set(EVAL_N)
    else:
        assert set(seeds) <= set(sp["seeds"]), seeds
        assert set(Ns) <= set(sp["N"]) | set(a.dev_extra_N), "dev N must be 1e3/1e4 (+ --dev-extra-N)"
    variants = build_variants(P2, v3, a.policies)
    assert variants, "no variants selected"
    print("guards:", json.dumps(info, ensure_ascii=False))
    print(f"split={a.split} workloads={wls} N={Ns} seeds={seeds} colds={colds} variants={[v.label for v in variants]}")
    if a.dry_run:
        n_files = 0
        for w in wls:
            for n in Ns:
                for s in seeds:
                    if a.split == "eval":
                        rel = f"eval_traces/{w}_N{n}_seed{s}_B.jsonl"
                        ok = (EVAL_DIR / rel).exists() and rel in manifest
                    else:
                        rel = f"dev_traces/{w}_N{n}_seed{s}_A.jsonl"
                        ok = ((S3.WL_DIR / rel).exists() and rel in manifest) or n in a.dev_extra_N
                    assert ok, f"missing trace file or manifest entry: {rel}"
                    n_files += 1
        print(f"DRY RUN ok: {n_files} trace files present and in the manifest, {n_files*len(colds)*len(variants)} simulations would run. Nothing was read or simulated.")
        return
    assert a.out, "--out required"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "main_raw.csv"
    if csv.exists():
        sys.exit(f"REFUSING to overwrite {csv} (use a new --out)")
    df = run_grid(a.split, wls, Ns, seeds, colds, variants, a.procs, a.dev_extra_N, manifest, "main")
    df.to_csv(csv, index=False)
    meta = dict(args={k: v for k, v in vars(a).items()}, n_rows=len(df), status_counts=df.status.value_counts().to_dict(), guards=info,
                variants=[asdict(v) for v in variants], script_sha16=sha16(__file__), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                cold_definition="empty=cache_mode empty (imputed table); disk=warm; hot=warm + first shape at inproc tier")
    (out / "main_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    if a.split == "eval":
        os.chmod(csv, 0o444)
        os.chmod(out / "main_meta.json", 0o444)
    print(f"-> {csv}")


if __name__ == "__main__":
    main()
