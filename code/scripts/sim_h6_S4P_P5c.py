"""S4-P P5c step 3: request-level simulation of B1-AOT and v2-AOT next to the old policies (CPU only, no torch; <=8 worker processes, default 1, nice 10).
Preregistration goals/S4P_P5c_AOT_confirmatory_prereg_v1.md (0444, SHA16 c14507708a968699) sec 3.  Builds on the S4-P simulator (scripts/sim_eval_S4P.py machinery, imported, not modified) and
the world table cost_table_v2_aot.csv (+ addendum_rebuild_v2.csv); ALL arms share that one table.
  B1-AOT  = policy B1 on the table in which every C1 row carries the AOT row's steady/one-time cells (aot_tables_S4P_P5c.swap_c1); B1-AOT has no parameters.
  v2-AOT  = policy Ours (S3 frozen v2 params) on the same swapped table (policy sees the AOT prices); C1+G1 / G1 / E0 rows are NOT swapped.
  Fallback list (results/S4P/p5c/aot_fallback_shapes.csv, 0444): those shapes use the E0 steady/one-time cells of the same table (E0 service, as on the real machine).
  Old policies (H6c "best old", prereg sec 3): B0, B1, B2, B3, B3p, B3pp, B4, B5, B6, B6p, Ours (= v2), Ours-v3 (frozen v3 params), all on the unswapped table.
  Sensitivities (reported, not judged): B1-AOT-nofb (fallback shapes priced as AOT = "as if the package had passed"), B1-AOT-sens (empty-cache inproc tier: build wall-clock x C1's own inproc/cold ratio),
  B1-excl / B1-AOT-excl (requests of the fallback shapes removed from the trace = "fallback shapes removed"); --world s3 = frozen S2 table + S3 addendum + the AOT rows (prereg: "S3 frozen table + AOT rows").
Splits: --split dev  = Dev-2 (half A, seeds 0-4, S3 dev traces N 1e3/1e4, --dev-extra-N generates more N in memory, flagged trace_kind=gen_default), run BEFORE the evaluation traces exist;
        --split eval = half B, seeds 120-129, W1-W5 x N in {1e2,1e3,1e4,1e5}: <eval-dir>/eval_traces + manifest (0444, every trace 0444, SHA checked per task); needs cost_table_v2_aot.csv 0444,
        fallback list 0444, preregistration SHA, S4-P frozen v3 params guards (scripts.sim_eval_S4P.eval_guards).  Output csv/meta are 0444 after an eval run.
Cold levels as sim_eval_S3: empty (cache_mode empty, imputed table), disk (warm), hot (warm + first shape at the in-process tier).
Consistency (--repro-p5-step2): the P5 step-2 in-memory tables are rebuilt with swap_c1 + the P5 mapping (results/S4P/p5/step2/step2_prereg.json) and the simulations of B0/B1/Ours/B1-AOT/-scale/-card50/v2-AOT/-scale are compared
  with results/S4P/p5/step2/step2_raw.csv: service_s must be EQUAL bit for bit (also n_shapes and the compile count).  Exit code 1 otherwise.
--synthetic (selftest): inputs may be anywhere, output must NOT be under results/, eval-dir must not be the official one; real evaluation traces are never touched.
Usage (project root, after `source ops/project_env.sh`; `export CUDA_VISIBLE_DEVICES=''; nice -n 10`):
  "$PROJ_PYTHON" -B -m scripts.sim_h6_S4P_P5c --split dev  --out ../results/S4P/p5c/dev2 [--dev-extra-N 100,100000] [--procs 4]
  "$PROJ_PYTHON" -B -m scripts.sim_h6_S4P_P5c --split eval --out ../results/S4P/p5c/eval_sim [--procs 8]
  "$PROJ_PYTHON" -B -m scripts.sim_h6_S4P_P5c --repro-p5-step2 --out ../results/S4P/p5c/repro_p5_step2
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
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.policy import CostView, shape_of  # noqa: E402
from p24.policy.factory_v3 import make_policy  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402
from p24.sim import simulator as SIM  # noqa: E402
from p24.sim.costs import MissingCost  # noqa: E402
from scripts import aot_tables_S4P_P5c as T  # noqa: E402
from scripts import sim_eval_S3 as S3  # noqa: E402
from scripts import sim_eval_S4P as S4P  # noqa: E402

P5C = T.P5C
ADD_V2 = ROOT / "results/S4P/p3/cost_v2/addendum_rebuild_v2.csv"
EVAL_DIR = P5C                                           # traces in <dir>/eval_traces/<W>_N<N>_seed<s>_B.jsonl
EVAL_MANIFEST = P5C / "sha256_manifest_eval.txt"
EVAL_N = [100, 1000, 10000, 100000]
P5_STEP2 = ROOT / "results/S4P/p5/step2"
W: Dict[str, object] = {}
OLD_LABELS = list(T.OLD_SET)                             # B0, B1..B6p, Ours, Ours-v3


@dataclass
class Var:
    label: str
    policy: str
    params: dict = field(default_factory=dict)
    table: str = "base"          # key into the table dict
    trace: str = "full"          # full | excl (fallback-shape requests removed)


# ====================================================================================================== tables / variants
def load_world(world: str, table_csv, addendum_csv):
    """-> (imputed DataFrame (before the addendum overlay), addendum DataFrame, ct0 = AddendumCostTable of the unswapped world)."""
    if world == "v2":
        df = pd.read_csv(table_csv)
        add_df = ADD.read_addendum(addendum_csv)
    else:                                               # s3: frozen S2 table + S3 addendum (+ the AOT rows of table_csv, when given)
        df = pd.read_csv(S3.CT_PATH)
        add_df = ADD.read_addendum(S3.ADDENDUM_PATH)
        if table_csv:
            aot = pd.read_csv(table_csv)
            aot = aot[aot.arm == T.AOT_ARM]
            df = pd.concat([df, aot], ignore_index=True, sort=False)
    df_imp = S3.impute_empty(df)[0]
    ct0 = ADD.load_with_addendum(df_imp, add_df)
    return df_imp, add_df, ct0


def make_tables(ct0, fallback) -> Dict[str, ADD.AddendumCostTable]:
    tabs = {"base": ct0}
    if (ct0.df.arm == T.AOT_ARM).any():
        tabs["aot"] = T.swap_c1(ct0, T.aot_provider(ct0), fallback)
        tabs["aot_nofb"] = T.swap_c1(ct0, T.aot_provider(ct0), ())
        tabs["aot_sens"] = T.swap_c1(ct0, T.aot_provider(ct0, resid_col_override={"inproc_empty": "aot_resid_s__inproc_empty_sens"}), fallback)
    return tabs


def build_vars(P2: dict, v3_params: dict, labels: Optional[List[str]] = None) -> List[Var]:
    old = [Var(v.label, v.policy, v.params) for v in S4P.build_variants(P2, {"": v3_params}, OLD_LABELS)]
    new = [Var("B1-AOT", "B1", dict(P2), "aot"), Var("v2-AOT", "Ours", dict(P2), "aot"), Var("B1-AOT-nofb", "B1", dict(P2), "aot_nofb"),
           Var("B1-AOT-sens", "B1", dict(P2), "aot_sens"), Var("B1-excl", "B1", dict(P2), "base", "excl"), Var("B1-AOT-excl", "B1", dict(P2), "aot", "excl")]
    out = old + new
    if labels:
        out = [v for v in out if v.label in labels]
    return out


# ====================================================================================================== worker
def _init_worker(cfg: dict):
    try:
        os.nice(10)
    except OSError:
        pass
    W.update(cfg)
    ADD.install_simulator_hook()
    W["store"] = S3.TraceStore("dev", cfg["manifest"]) if cfg["split"] == "dev" else None
    W["tabs"] = cfg["tabs"] if cfg.get("tabs") is not None else make_tables(cfg["ct0"], cfg["fallback"])
    W["views"] = {}


def _get_trace(w, n, seed, tkind):
    if W["split"] == "dev":
        return W["store"].get(w, n, seed, tkind)
    rel = f"eval_traces/{w}_N{n}_seed{seed}_B.jsonl"
    p = pathlib.Path(W["eval_dir"]) / rel
    assert p.exists(), p
    assert hashlib.sha256(p.read_bytes()).hexdigest() == W["manifest"][rel], f"manifest SHA mismatch {rel}"
    reqs = S3.read_trace_file(p)
    assert len(reqs) == n, (rel, len(reqs))
    return reqs


def sim_one(v: Var, reqs, oc, cm: str, base_row: dict) -> dict:
    tab = W["tabs"].get(v.table)
    row = dict(base_row, variant=v.label, policy=v.policy, table=v.table, trace=v.trace)
    t0 = time.perf_counter()
    if tab is None:
        row.update(status="missing_table", error=f"table {v.table} not built (no AOT rows?)")
        return row
    try:
        key = (v.table, cm)
        cv = W["views"].get(key)
        if cv is None:
            cv = W["views"][key] = CostView(tab, cm, rebuild_override=getattr(tab, "rebuild_override", None))
        rq = reqs if v.trace == "full" else [q for q in reqs if shape_of(q) not in W["fallback"]]
        pol = make_policy(v.policy, cost=cv, params=v.params, prior=W["prior"], oracle_counts=oc)
        opt = SIM.SimOptions(cache_mode=cm, mem_budget_mb=W["budget_mb"], overhead_aware=True)
        s = SIM.simulate(rq, pol, tab, opt).summary
        row.update(status="ok", error="", service_s=s["service_s"], warmup_s=s["warmup_s"], peak_graph_mb=s["peak_graph_mb"], peak_reserved_est_mb=s["peak_reserved_est_mb"],
                   n_requests=s["n_requests"], n_shapes=s["n_shapes"], n_compile=s.get("n_compile", 0), n_compile_capture=s.get("n_compile_capture", 0), n_capture=s.get("n_capture", 0),
                   n_eager_first=s.get("n_eager_first", 0), n_capture_denied=s.get("n_capture_denied", 0), n_evicted=s["n_evicted_total"], n_denied=s["n_denied_total"],
                   n_steady=s.get("n_steady", 0), n_events_with_fallback=s["n_events_with_fallback"])
    except MissingCost as e:
        row.update(status="missing_cost", error=str(e)[:160])
    except Exception as e:  # noqa: BLE001  (recorded; the summary counts them)
        row.update(status="error", error=f"{type(e).__name__}: {str(e)[:160]}")
    row["sim_wall_s"] = time.perf_counter() - t0
    return row


def run_task(task):
    w, n, seed, tkind = task
    reqs = _get_trace(w, n, seed, tkind)
    oc = dict(collections.Counter(shape_of(q) for q in reqs))
    rows = []
    for cold in W["colds"]:
        cm, hot = S3.COLDS[cold]
        S3._set_hot(hot)
        base = dict(split=W["split"], workload=w, N=n, seed=seed, budget="default", cold=cold, trace_kind=tkind, imputed=int(cm == "empty"))
        for v in W["variants"]:
            rows.append(sim_one(v, reqs, oc, cm, base))
    S3._set_hot(False)
    return rows


def run_grid(split, wls, Ns, seeds, colds, variants, tabs, ct0, fallback, procs, manifest, eval_dir, dev_extra_N=(), label="") -> pd.DataFrame:
    bj = json.loads(S3.BUDGETS_PATH.read_text())["tiers"]
    specs = []
    for w in wls:
        for n in Ns:
            for s in seeds:
                kind = "gen_default" if (split == "dev" and n in dev_extra_N and n not in S3.SPLITS["dev"]["N"]) else "file"
                specs.append((w, n, s, kind))
    if split == "dev" and any(k != "file" for *_, k in specs):
        S3.pool_arrays("A")
    cfg = dict(split=split, manifest=manifest, colds=list(colds), variants=variants, budget_mb=float(bj["default"]["mb"]), prior=S3.make_priors(False)["A"], tabs=tabs, ct0=ct0,
               fallback=set(fallback), eval_dir=str(eval_dir) if eval_dir else None)
    t0 = time.time()
    rows = []
    specs.sort(key=lambda t: -t[1])
    print(f"[{label or split}] tasks={len(specs)} variants={len(variants)} colds={list(colds)} sims~{len(specs) * len(variants) * len(colds)} procs={procs}", flush=True)
    if procs == 1:
        _init_worker(cfg)
        for i, t in enumerate(specs):
            rows += run_task(t)
            if (i + 1) % max(1, len(specs) // 10) == 0:
                print(f"[{label or split}] {i + 1}/{len(specs)} tasks {time.time() - t0:.0f}s", flush=True)
    else:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, specs, chunksize=1)):
                rows += r
                if (i + 1) % max(1, len(specs) // 10) == 0:
                    print(f"[{label or split}] {i + 1}/{len(specs)} tasks {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["cold", "workload", "N", "seed", "variant"]).reset_index(drop=True)
    print(f"[{label or split}] rows={len(df)} status={df.status.value_counts().to_dict()} wall={time.time() - t0:.0f}s", flush=True)
    return df


# ====================================================================================================== P5 step-2 reproduction
P5_VARIANTS = [("B0", "B0", "ref"), ("B1", "B1", "ref"), ("Ours", "Ours", "ref"), ("B1-AOT", "B1", "add"), ("B1-AOT-scale", "B1", "scale"),
               ("B1-AOT-card50", "B1", "card50"), ("v2-AOT", "Ours", "add"), ("v2-AOT-scale", "Ours", "scale")]


def p5_tables(ct0, m: dict) -> Dict[str, ADD.AddendumCostTable]:
    """The P5 step-2 in-memory tables, built with the SAME swap function: only the disk-tier one-time cells of the C1 rows change, steady cells stay."""
    tiers = m["tiers"]

    def add(i, row, sh):
        return None, {t: (m["c0_s"] + float(row["steady_ms"]) / 1e3, None) for t in tiers}

    def scale(f):
        def prov(i, row, sh):
            vals = {}
            for t in tiers:
                ot = ct0.one_time("C1", sh, t)
                vals[t] = (f * ot.resid_s, {c: f * ot.fc[c] for c in ("te", "unet", "vae")})
            return None, vals
        return prov

    return {"ref": ct0, "add": T.swap_c1(ct0, add), "scale": T.swap_c1(ct0, scale(m["r_max"])), "card50": T.swap_c1(ct0, scale(m["card50"]))}


def cmd_repro_p5(a) -> int:
    out = pathlib.Path(a.out)
    T.check_not_official_output(out, a.synthetic)
    m = json.loads((P5_STEP2 / "step2_prereg.json").read_text())
    assert T.is_ro(P5_STEP2 / "step2_prereg.json")
    ref = pd.read_csv(P5_STEP2 / "step2_raw.csv")
    pj = S3.load_params()
    info = S4P.dev_guards(pj)
    df_imp, add_df, ct0 = load_world("s3", None, None)
    tabs = p5_tables(ct0, m)
    P2 = S4P.v2_params()
    base_v = {v.label: v for v in S4P.build_variants(P2, {}, ["B0", "B1", "Ours"])}
    variants = [Var(lab, base_v[pol].policy, base_v[pol].params, tab) for lab, pol, tab in P5_VARIANTS]
    df = run_grid("dev", m["workloads"], m["N"], m["seeds"], ["disk"], variants, tabs, ct0, (), 1, S3.read_manifest(), None, label="repro-p5")
    mg = df.merge(ref, on=["workload", "N", "seed", "variant"], suffixes=("", "_p5"), how="left")
    assert len(mg) == len(df) and mg.service_s_p5.notna().all(), "step2_raw.csv misses rows of the repro grid"
    mg["abs_diff"] = (mg.service_s - mg.service_s_p5).abs()
    mg["bit_equal"] = (mg.service_s == mg.service_s_p5) & (mg.n_shapes == mg.n_shapes_p5) & (mg.n_compile == mg.n_c1)
    res = dict(n_compared=int(len(mg)), n_bit_equal=int(mg.bit_equal.sum()), max_abs_diff=float(mg.abs_diff.max()), variants=sorted(mg.variant.unique()), grid=dict(W=m["workloads"], N=m["N"], seeds=m["seeds"]),
               step2_raw_sha16=T.sha16(P5_STEP2 / "step2_raw.csv"), step2_prereg_sha16=T.sha16(P5_STEP2 / "step2_prereg.json"), script_sha16=T.sha16(__file__),
               aot_tables_sha16=T.sha16(T.ROOT / "code/scripts/aot_tables_S4P_P5c.py"), guards=info, passed=bool(mg.bit_equal.all()), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    out.mkdir(parents=True, exist_ok=True)
    mg[["workload", "N", "seed", "variant", "service_s", "service_s_p5", "abs_diff", "bit_equal", "n_shapes", "n_shapes_p5", "n_compile", "n_c1"]].to_csv(out / "repro_p5_step2_check.csv", index=False)
    (out / "repro_p5_step2_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: res[k] for k in ("n_compared", "n_bit_equal", "max_abs_diff", "passed")}))
    return 0 if res["passed"] else 1


# ====================================================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "eval"])
    ap.add_argument("--repro-p5-step2", action="store_true")
    ap.add_argument("--out", help="output directory (refuses to overwrite main_raw.csv)")
    ap.add_argument("--table-csv", default=str(T.COST_TABLE_AOT))
    ap.add_argument("--addendum-csv", default=str(ADD_V2))
    ap.add_argument("--fallback-csv", default=str(T.FALLBACK_CSV))
    ap.add_argument("--world", choices=["v2", "s3"], default="v2", help="v2 (main): cost_table_v2_aot.csv + addendum_rebuild_v2.csv; s3 (sensitivity): frozen S2 table + S3 addendum + the AOT rows")
    ap.add_argument("--workloads", type=lambda s: s.split(","))
    ap.add_argument("--N", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--colds", type=lambda s: s.split(","), default=["empty", "disk", "hot"])
    ap.add_argument("--policies", type=lambda s: s.split(","), help="variant labels to run (default all)")
    ap.add_argument("--dev-extra-N", type=lambda s: [int(x) for x in s.split(",")], default=[])
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--eval-manifest", default=str(EVAL_MANIFEST))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="guards + grid + file checks only (no trace content read, nothing simulated)")
    a = ap.parse_args(argv)
    assert 1 <= a.procs <= 8, "at most 8 worker processes"
    if a.repro_p5_step2:
        assert a.out, "--out required"
        return cmd_repro_p5(a)
    assert a.split, "--split dev|eval (or --repro-p5-step2) required"
    sp = S4P.SPLITS[a.split] if a.split == "eval" else S3.SPLITS["dev"]
    eval_seeds = T.EVAL_SEEDS
    assert not (set(eval_seeds) & T.FORBIDDEN_SEEDS), "P5c evaluation seeds overlap dev / S3 / S4-P / real-machine seeds"
    # ---- guards
    T.check_prereg()
    if not a.synthetic:
        assert T.is_ro(a.table_csv), f"{a.table_csv} must be 0444 (frozen cost table) before any simulation"
        assert T.is_ro(a.fallback_csv), f"{a.fallback_csv} must be 0444"
        assert pathlib.Path(a.eval_dir).resolve() == EVAL_DIR.resolve() and pathlib.Path(a.eval_manifest).resolve() == EVAL_MANIFEST.resolve(), "official run: default eval dir / manifest only"
    elif a.split == "eval":
        assert pathlib.Path(a.eval_dir).resolve() != EVAL_DIR.resolve(), "--synthetic must not point at the official eval dir"
    if a.out:
        T.check_not_official_output(a.out, a.synthetic)
    pj = S3.load_params()
    info = dict(prereg_sha16=T.PREREG_SHA16, table_csv=dict(path=a.table_csv, sha16=T.sha16(a.table_csv), mode=oct(T.mode_of(a.table_csv))), fallback_csv=dict(path=a.fallback_csv, sha16=T.sha16(a.fallback_csv)),
                world=a.world)
    if a.world == "v2":
        info["addendum_csv"] = dict(path=a.addendum_csv, sha16=T.sha16(a.addendum_csv))
    if a.split == "eval":
        info.update(S4P.eval_guards())
        manifest = {l.split(None, 1)[1].strip(): l.split(None, 1)[0] for l in pathlib.Path(a.eval_manifest).read_text().splitlines() if l.strip()}
        assert T.is_ro(a.eval_manifest), "eval trace manifest must be 0444"
        info["eval_manifest_sha16"] = T.sha16(a.eval_manifest)
    else:
        info.update(S4P.dev_guards(pj))
        manifest = S3.read_manifest()
    seeds, Ns, wls = a.seeds or (eval_seeds if a.split == "eval" else sp["seeds"]), a.N or sp["N"], a.workloads or S3.G.SCENARIOS
    if a.split == "eval":
        assert set(seeds) <= set(eval_seeds) and set(Ns) <= set(EVAL_N), (seeds, Ns)
        assert not (set(seeds) & T.FORBIDDEN_SEEDS)
    else:
        assert set(seeds) <= set(sp["seeds"]), seeds
        assert set(Ns) <= set(sp["N"]) | set(a.dev_extra_N), "dev N must be 1e3/1e4 (+ --dev-extra-N)"
    fallback = T.read_fallback(a.fallback_csv)
    df_imp, add_df, ct0 = load_world(a.world, a.table_csv, a.addendum_csv)
    tabs = make_tables(ct0, fallback)
    P2 = S4P.v2_params()
    variants = build_vars(P2, S4P.frozen_v3_params(), a.policies)
    assert variants, "no variants selected"
    print("guards:", json.dumps(info, ensure_ascii=False, default=str))
    print(f"split={a.split} world={a.world} workloads={wls} N={Ns} seeds={seeds} colds={a.colds} variants={[v.label for v in variants]} fallback={[s.tag() for s in fallback]}")
    if a.dry_run:
        n = 0
        for w in wls:
            for nn in Ns:
                for s in seeds:
                    if a.split == "eval":
                        rel = f"eval_traces/{w}_N{nn}_seed{s}_B.jsonl"
                        p = pathlib.Path(a.eval_dir) / rel
                        assert p.exists() and rel in manifest, f"missing trace file or manifest entry: {rel}"
                        assert T.is_ro(p), f"trace file not 0444: {rel}"
                    else:
                        rel = f"dev_traces/{w}_N{nn}_seed{s}_A.jsonl"
                        assert ((S3.WL_DIR / rel).exists() and rel in manifest) or nn in a.dev_extra_N, f"missing trace file or manifest entry: {rel}"
                    n += 1
        print(f"DRY RUN ok: {n} traces present, {n * len(a.colds) * len(variants)} simulations would run. Nothing was read or simulated.")
        return 0
    assert a.out, "--out required"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "main_raw.csv"
    if csv.exists():
        sys.exit(f"REFUSING to overwrite {csv} (use a new --out)")
    df = run_grid(a.split, wls, Ns, seeds, a.colds, variants, tabs, ct0, fallback, a.procs, manifest, a.eval_dir if a.split == "eval" else None, a.dev_extra_N, "main")
    df.to_csv(csv, index=False)
    aot_rows = ct0.df[ct0.df.arm == T.AOT_ARM]
    c1_prewarm = None
    if len(aot_rows):
        try:
            c1_prewarm = float(sum(ct0.one_time("C1", T.Shape(int(r.H), int(r.W), int(r.B), int(r.cfg)), "inproc_empty").total_s for r in aot_rows.itertuples()))
        except MissingCost:
            c1_prewarm = None
    meta = dict(args={k: v for k, v in vars(a).items()}, n_rows=len(df), status_counts=df.status.value_counts().to_dict(), guards=info, variants=[asdict(v) for v in variants],
                fallback_shapes=sorted(s.tag() for s in fallback), script_sha16=T.sha16(__file__), aot_tables_sha16=T.sha16(T.ROOT / "code/scripts/aot_tables_S4P_P5c.py"),
                time=time.strftime("%Y-%m-%dT%H:%M:%S"), synthetic=a.synthetic, cold_definition="empty=cache_mode empty (imputed table); disk=warm; hot=warm + first shape at inproc tier",
                offline=dict(aot_build_total_s=float(aot_rows.aot_build_wall_s.sum()) if len(aot_rows) and "aot_build_wall_s" in aot_rows else None,
                             c1_prewarm_total_s_inproc_empty=c1_prewarm,
                             definition="aot_build_total_s = sum of the (k-applied) build wall-clocks of the AOT rows; c1_prewarm_total_s = sum over the same shapes of the C1 one-time cost at tier inproc_empty "
                                        "(executor's definition of 'C1 pre-warm flow', not fixed in the preregistration)"))
    (out / "main_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    if a.split == "eval" and not a.synthetic:
        os.chmod(csv, 0o444)
        os.chmod(out / "main_meta.json", 0o444)
    print(f"-> {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
