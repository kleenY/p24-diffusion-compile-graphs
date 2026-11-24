"""S4-X regime map (descriptive, CPU only, no torch): when does the reuse-aware policy beat 'compile on first sight' (B1) and all-eager (B0)?
Spec (written before the run, 0444): results/S4X/regime/SPEC_regime.md.  Nothing frozen is modified; only existing, frozen simulators / parameters are called.

World S3 = scripts/sim_eval_S3.py worker (frozen S3 table + addendum, --missing impute for the empty-cache tier), policies B0 / B1 / Ours.
World V2 = scripts/s4x_x3_common.World (cost_table_v2_aot + rebuild addendum; same predictor as predict_S4P_P5d / predict_S4X_X3), policies B0 / B1 / Ours / B1-AOT / B1-AOT-K.

Stages (from code/):  nice -n 10 python -B -m scripts.regime_S4X run | check | analyze | plot
  run     generate N=300/3000 traces (seeds 100-109, B half) -> results/S4X/regime/workload ; simulate both worlds -> regime_seeds.csv
  check   reproduction checks of SPEC sec 5
  analyze regime_grid.csv, regime_best.csv, real-point table
  plot    fig_regime.pdf
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from p24.policy import shape_of
from p24.sim.costs import MissingCost
from scripts import sim_eval_S3 as SS
from scripts import s4p_p3_common_v2 as C
from scripts import s4x_x3_common as X

ROOT = C.ROOT
OUT = ROOT / "results/S4X/regime"
WLD = OUT / "workload"
SEEDS = list(range(100, 110))
NS = [100, 300, 1000, 3000, 10000]
WLS = ["W1", "W2", "W3", "W5"]
FILE_NS = {100, 1000, 10000}                   # S3 frozen eval trace files
BUDGETS = ["wide", "tight", "default"]         # default only for the reproduction check / real-point reference
COLDS = ["empty", "disk"]
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]

# ---------------------------------------------------------------------------------------------- traces
def gen_new_traces():
    WLD.mkdir(parents=True, exist_ok=True)
    man = {}
    for w in WLS:
        for n in (300, 3000):
            for s in SEEDS:
                p = WLD / f"{w}_N{n}_seed{s}_B.jsonl"
                reqs = SS.gen_trace("B", w, n, s, "gen_default")
                data = "".join(json.dumps(r) + "\n" for r in reqs).encode()
                sha = hashlib.sha256(data).hexdigest()
                if p.exists():
                    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, f"differs, refusing to overwrite {p}"
                else:
                    p.write_bytes(data)
                    os.chmod(p, 0o444)
                man[p.name] = sha
    mp_ = WLD / "sha256_manifest_regime.txt"
    if not mp_.exists():
        mp_.write_text("".join(f"{v}  {k}\n" for k, v in sorted(man.items())))
        os.chmod(mp_, 0o444)
    return man


class Store:
    def __init__(self):
        self.manifest = SS.read_manifest()
        self.cache = {}

    def get(self, w, n, s, tkind="file"):
        key = (w, n, s)
        if key not in self.cache:
            if n in FILE_NS:
                rel = f"eval_traces/{w}_N{n}_seed{s}_B.jsonl"
                p = SS.WL_DIR / rel
                assert hashlib.sha256(p.read_bytes()).hexdigest() == self.manifest[rel], f"manifest SHA mismatch {rel}"
            else:
                p = WLD / f"{w}_N{n}_seed{s}_B.jsonl"
            reqs = SS.read_trace_file(p)
            assert len(reqs) == n
            self.cache[key] = reqs
            while len(self.cache) > 4:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


# ---------------------------------------------------------------------------------------------- world S3 worker
def _init_s3(cfg):
    SS._init_worker(cfg)
    SS.W["store"] = Store()


def _task_s3(task):
    return SS.run_task(task)


# ---------------------------------------------------------------------------------------------- world V2 worker
V = {}


def _init_v2(budgets):
    try:
        os.nice(10)
    except OSError:
        pass
    V["w"] = X.World()
    V["K"] = X.k_shapes(X.read_k())
    V["store"] = Store()
    V["budgets"] = budgets


def _row(base, res, w):
    sm = res.summary
    base.update(status="ok", error="", service_s=float(sm["service_s"]), n_requests=int(sm["n_requests"]), n_shapes=int(sm["n_shapes"]), n_compile=int(sm.get("n_compile", 0)),
                n_capture=int(sm.get("n_capture", 0)), n_eager_first=int(sm.get("n_eager_first", 0)), n_steady=int(sm.get("n_steady", 0)), n_evicted=sm.get("n_evicted_total"),
                n_denied=sm.get("n_denied_total"), peak_graph_mb=sm.get("peak_graph_mb"))
    return base


def _task_v2(task):
    wl, n, seed = task
    w = V["w"]
    reqs = V["store"].get(wl, n, seed)
    rows = []

    def one(policy, budget, bmb, runner, build_built=0.0, build_used=0.0):
        base = dict(world="V2", workload=wl, N=n, seed=seed, policy=policy, budget=budget, budget_mb=bmb, cold="disk", imputed=0,
                    build_built_s=build_built, build_used_s=build_used)
        t0 = time.perf_counter()
        try:
            row = _row(base, runner(), w)
        except MissingCost as e:
            base.update(status="missing_cost", error=str(e)[:160])
            row = base
        except Exception as e:  # noqa: BLE001
            base.update(status="error", error=f"{type(e).__name__}: {str(e)[:160]}")
            row = base
        row["sim_wall_s"] = time.perf_counter() - t0
        rows.append(row)

    dflt = w.budget
    one("B0", "all", None, lambda: C.simulate_run(w.ct_v2, reqs, "B0", w.P, w.prior, dflt, "warm", ct_policy=w.ct_v1))
    one("B1", "all", None, lambda: C.simulate_run(w.ct_v2, reqs, "B1", w.P, w.prior, dflt, "warm", ct_policy=w.ct_v1))
    one("B1-AOT", "all", None, lambda: w.simulate("B1-AOT", reqs), w.build_charge("B1-AOT"), w.build_used("B1-AOT", reqs))
    one("B1-AOT-K", "all", None, lambda: w.simulate("B1-AOT-K", reqs, V["K"]), w.build_charge("B1-AOT-K", V["K"]), w.build_used("B1-AOT-K", reqs, V["K"]))
    for b, mb in V["budgets"].items():
        one("Ours", b, mb, lambda mb=mb: C.simulate_run(w.ct_v2, reqs, "Ours", w.P, w.prior, mb, "warm", ct_policy=w.ct_v1))
    return rows


# ---------------------------------------------------------------------------------------------- run
def cmd_run(a):
    pj = SS.load_params()
    SS.guards("eval", pj)                                  # frozen code SHAs + params 0444 (asserts)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds_csv = OUT / "regime_seeds.csv"
    if seeds_csv.exists():
        sys.exit(f"REFUSING to overwrite {seeds_csv}")
    man = gen_new_traces()
    print(f"generated/verified {len(man)} new traces (N=300/3000)", flush=True)
    P = SS.base_params(pj)
    bj = json.loads(SS.BUDGETS_PATH.read_text())["tiers"]
    budgets = {k: float(bj[{"wide": "wide", "tight": "tight", "default": "default"}[k]]["mb"]) for k in BUDGETS}
    variants = [SS.Variant("B0", "B0", dict(P)), SS.Variant("B1", "B1", dict(P)), SS.Variant("Ours", "Ours", dict(P))]
    ex = SS.Exp("regime", WLS, NS, SEEDS, BUDGETS, COLDS, variants)
    priors = SS.make_priors(False)
    cfg = dict(split="eval", manifest=SS.read_manifest(), exp=ex, budgets=budgets, priors=priors, impute=True, need_procs=False, addendum=str(SS.ADDENDUM_PATH))
    specs = [(w, n, s, "file") for w in WLS for n in NS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, initializer=_init_s3, initargs=(cfg,)) as pool:
        r3 = []
        for i, r in enumerate(pool.imap_unordered(_task_s3, [(-1, [sp]) for sp in specs], chunksize=1)):
            r3 += r
            if (i + 1) % 40 == 0:
                print(f"[S3] {i+1}/{len(specs)} {time.time()-t0:.0f}s", flush=True)
    d3 = pd.DataFrame(r3)
    d3["world"] = "S3"
    print(f"[S3] rows={len(d3)} status={d3.status.value_counts().to_dict()} {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    tasks = [(w, n, s) for w in WLS for n in NS for s in SEEDS]
    with mp.get_context("fork").Pool(a.procs, initializer=_init_v2, initargs=(budgets,)) as pool:
        r2 = []
        for i, r in enumerate(pool.imap_unordered(_task_v2, tasks, chunksize=1)):
            r2 += r
            if (i + 1) % 40 == 0:
                print(f"[V2] {i+1}/{len(tasks)} {time.time()-t1:.0f}s", flush=True)
    d2 = pd.DataFrame(r2)
    print(f"[V2] rows={len(d2)} status={d2.status.value_counts().to_dict()} {time.time()-t1:.0f}s", flush=True)
    # input time per trace (same input model as predict_S3_4 / World.input_s)
    inp = SS.__dict__.get("_INP") or C.P34.InputModel(C.P34.INPUT_PATH)
    st = Store()
    inrows = []
    for w in WLS:
        for n in NS:
            for s in SEEDS:
                reqs = st.get(w, n, s)
                inp.n_extrap = 0
                tot = sum(inp.ms(shape_of(r)) for r in reqs) / 1e3
                inrows.append(dict(workload=w, N=n, seed=s, input_s=tot, n_input_extrapolated=inp.n_extrap))
    din = pd.DataFrame(inrows)
    d3 = d3.drop(columns=["policy"]).rename(columns={"variant": "policy"})
    keep3 = ["world", "workload", "N", "seed", "policy", "budget", "budget_mb", "cold", "imputed", "status", "error", "service_s", "n_requests", "n_shapes", "n_compile", "n_capture", "n_eager_first",
             "n_steady", "n_evicted", "n_denied", "peak_graph_mb", "sim_wall_s"]
    d3 = d3[keep3]
    d3["build_built_s"] = 0.0
    d3["build_used_s"] = 0.0
    d = pd.concat([d3, d2], ignore_index=True, sort=False).merge(din, on=["workload", "N", "seed"], how="left")
    d["total_s"] = d["service_s"] + d["input_s"]
    d.to_csv(seeds_csv, index=False)
    os.chmod(seeds_csv, 0o444)
    meta = dict(time=time.strftime("%FT%T"), argv=sys.argv, seeds=SEEDS, Ns=NS, workloads=WLS, budgets=budgets, colds=COLDS, n_rows=len(d), status=d.status.value_counts().to_dict(),
                traces_new_manifest_sha16=sha16(WLD / "sha256_manifest_regime.txt"), policy_params_sha16=sha16(SS.PARAMS_PATH), script_sha16=sha16(__file__),
                K=X.read_k(), s3_cost_table_sha16=sha16(SS.CT_PATH), v2_cost_table_sha16=sha16(X.T.COST_TABLE_AOT), wall_s=time.time() - t0)
    (OUT / "regime_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
    os.chmod(OUT / "regime_meta.json", 0o444)
    print("done", seeds_csv, meta["status"], flush=True)


# ---------------------------------------------------------------------------------------------- check
def cmd_check(a):
    d = pd.read_csv(OUT / "regime_seeds.csv")
    ok = True
    # S3: vs main_raw (default budget, N in file Ns)
    m = pd.read_csv(ROOT / "results/S3/sim_eval/main_raw.csv")
    m = m[(m.variant.isin(["B0", "B1", "Ours"])) & (m.budget == "default") & (m.N.isin(sorted(FILE_NS))) & (m.cold.isin(COLDS)) & (m.workload.isin(WLS))]
    x = d[(d.world == "S3") & (d.budget == "default") & (d.N.isin(sorted(FILE_NS)))]
    j = x.merge(m, left_on=["policy", "workload", "N", "seed", "cold"], right_on=["variant", "workload", "N", "seed", "cold"], suffixes=("", "_m"))
    n_exp = len(m)
    diff = (j.service_s - j.service_s_m).abs().max()
    print(f"CHECK S3 vs main_raw: matched {len(j)}/{n_exp} rows, max|diff service_s| = {diff:.3e}")
    ok &= (len(j) == n_exp) and diff < 1e-9
    # V2: reproduce pred_P5d (seed 130) and pred_X3 (seed 131) with the same World
    w = X.World()
    K = X.k_shapes(X.read_k())
    P = w.P
    mx = 0.0
    nchk = 0
    pr = pd.read_csv(ROOT / "results/S4P/p5d/pred_P5d.csv")
    for wl in ("W2", "W5"):
        stem = f"{wl}_N1000_seed130_B"
        reqs = C.SIM.read_trace(ROOT / f"results/S4P/p5d/workload/{stem}.jsonl")
        for pol in ("B0", "B1", "Ours"):
            r = C.simulate_run(w.ct_v2, reqs, pol, P, w.prior, w.budget, "warm", ct_policy=w.ct_v1).summary["service_s"]
            q = pr[(pr.stream == stem) & (pr.policy == pol)].pred_excl_input_s.iloc[0]
            mx = max(mx, abs(r - q))
            nchk += 1
        r = w.simulate("B1-AOT", reqs).summary["service_s"]
        mx = max(mx, abs(r - pr[(pr.stream == stem) & (pr.policy == "B1-AOT")].pred_excl_input_s.iloc[0]))
        nchk += 1
    px = pd.read_csv(ROOT / "results/S4X/x3/pred_X3.csv")
    for wl in ("W2", "W5"):
        stem = f"{wl}_N1000_seed131_B"
        reqs = C.SIM.read_trace(ROOT / f"results/S4X/x3/workload/{stem}.jsonl")
        for arm in ("B0", "B1-AOT", "B1-AOT-K"):
            r = w.simulate(arm, reqs, K if arm == "B1-AOT-K" else None).summary["service_s"]
            mx = max(mx, abs(r - px[(px.stream == stem) & (px.policy == arm)].pred_excl_input_s.iloc[0]))
            nchk += 1
    print(f"CHECK V2 vs pred_P5d / pred_X3: {nchk} comparisons, max|diff service_s| = {mx:.3e}")
    ok &= mx < 1e-9
    # B0 difference between worlds
    b = d[(d.policy == "B0") & (d.budget.isin(["default", "all"])) & (d.cold.isin(["disk"]))]
    p = b.pivot_table(index=["workload", "N", "seed"], columns="world", values="total_s")
    rr = (p["V2"] / p["S3"]).groupby(level=["workload", "N"]).median().unstack("N")
    print("B0 total_s ratio V2/S3 (median over seeds):")
    print(rr.round(4).to_string())
    bad = d[d.status != "ok"]
    print("non-ok rows:", len(bad), bad.groupby(["world", "policy", "status"]).size().to_dict() if len(bad) else "")
    print("CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------- analyze
def per_seed_ratios(d):
    """long frame of (world, workload, N, cache, budget, policy) with per-seed total & ratios to B1 and B0 of the same world/cache/seed."""
    rows = []
    d = d[d.status == "ok"].copy()
    # S3 world: B0/B1 have rows for every budget (budget has no effect on them in principle) -> take the matching budget row
    for (world, wl, n), g in d.groupby(["world", "workload", "N"]):
        for cache in COLDS:
            for budget in BUDGETS:
                if world == "S3":
                    gg = g[(g.cold == cache) & (g.budget == budget)]
                    tot = {p: gg[gg.policy == p].set_index("seed").total_s for p in ("B0", "B1", "Ours")}
                    ext = {}
                    pols = ["B0", "B1", "Ours"]
                else:
                    gd = g[g.cold == "disk"]
                    tot, ext = {}, {}
                    for p in ("B0", "B1", "B1-AOT", "B1-AOT-K"):
                        x = gd[(gd.policy == p) & (gd.budget == "all")].set_index("seed")
                        tot[p] = x.total_s
                        ext[p] = x
                    tot["Ours"] = gd[(gd.policy == "Ours") & (gd.budget == budget)].set_index("seed").total_s
                    pols = ["B0", "B1", "Ours", "B1-AOT", "B1-AOT-K"]
                    if cache == "empty":
                        for p in ("B1-AOT", "B1-AOT-K"):
                            x = ext[p]
                            tot[p + "|cold"] = x.total_s + x.build_built_s
                            tot[p + "|cold_used"] = x.total_s + x.build_used_s
                        pols = ["B0", "B1-AOT|cold", "B1-AOT-K|cold", "B1-AOT|cold_used", "B1-AOT-K|cold_used"]
                        if budget != "default":
                            pols = pols     # B0 / AOT arms do not depend on the budget
                for p in pols:
                    if p not in tot or len(tot[p]) == 0:
                        continue
                    t = tot[p]
                    r1 = (t / tot["B1"]) if ("B1" in tot and len(tot["B1"]) and not (world == "V2" and cache == "empty")) else None
                    r0 = (t / tot["B0"]) if "B0" in tot and len(tot["B0"]) else None
                    rows.append(dict(world=world, workload=wl, N=n, cache=cache, budget=budget, policy=p, n_seeds=len(t),
                                     total_median_s=float(t.median()), total_min_s=float(t.min()), total_max_s=float(t.max()),
                                     ratio_B1_median=float(r1.median()) if r1 is not None else np.nan, ratio_B1_min=float(r1.min()) if r1 is not None else np.nan,
                                     ratio_B1_max=float(r1.max()) if r1 is not None else np.nan,
                                     ratio_B0_median=float(r0.median()) if r0 is not None else np.nan, ratio_B0_min=float(r0.min()) if r0 is not None else np.nan,
                                     ratio_B0_max=float(r0.max()) if r0 is not None else np.nan))
    return pd.DataFrame(rows)


def cmd_analyze(a):
    d = pd.read_csv(OUT / "regime_seeds.csv")
    g = per_seed_ratios(d)
    # xworld rows for the empty cache: B1/Ours from world S3 (imputed), AOT/K from world V2 (cold = build charged)
    g["cross_world"] = ((g.cache == "empty") & (g.world == "V2")).astype(int)
    g.to_csv(OUT / "regime_grid.csv", index=False)
    print("grid rows", len(g))
    # ---- best policy table
    TIE = 0.02
    rows = []
    for wl in WLS:
        for n in NS:
            for budget in ("wide", "tight"):
                for cache in COLDS:
                    if cache == "disk":
                        v2 = g[(g.world == "V2") & (g.workload == wl) & (g.N == n) & (g.cache == "disk") & (g.budget == budget)].set_index("policy")
                        s3 = g[(g.world == "S3") & (g.workload == wl) & (g.N == n) & (g.cache == "disk") & (g.budget == budget)].set_index("policy")
                        sets = {"V2_all(B0,B1,Ours,AOT,K)": v2.total_median_s, "S3_online(B0,B1,Ours)": s3.total_median_s}
                        sets["V2_online(B0,B1,Ours)"] = v2.total_median_s[["B0", "B1", "Ours"]]
                    else:
                        s3 = g[(g.world == "S3") & (g.workload == wl) & (g.N == n) & (g.cache == "empty") & (g.budget == budget)].set_index("policy")
                        v2 = g[(g.world == "V2") & (g.workload == wl) & (g.N == n) & (g.cache == "empty") & (g.budget == budget)].set_index("policy")
                        sets = {"S3_online(B0,B1,Ours)": s3.total_median_s}
                        x = pd.concat([s3.ratio_B0_median[["B0", "B1", "Ours"]], v2.ratio_B0_median.drop("B0", errors="ignore")])
                        sets["xworld_ratio_to_B0(B0,B1,Ours,AOT-cold,K-cold)"] = x[[i for i in x.index if not i.endswith("cold_used")]]
                    for sname, ser in sets.items():
                        ser = ser.dropna().sort_values()
                        best = ser.index[0]
                        second = ser.index[1]
                        gap = ser.iloc[1] / ser.iloc[0] - 1
                        tied = [i for i in ser.index if ser[i] / ser.iloc[0] - 1 < TIE]
                        rows.append(dict(workload=wl, N=n, budget=budget, cache=cache, comparison=sname, best=best, second=second, gap_second_pct=100 * gap,
                                         within2pct="+".join(tied), best_value=float(ser.iloc[0]), is_ratio=int(sname.startswith("xworld"))))
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "regime_best.csv", index=False)
    print(b[b.budget == "wide"].pivot_table(index=["comparison", "cache", "workload"], columns="N", values="best", aggfunc="first").to_string())
    # ---- post-hoc arithmetic: number of processes over which the offline build must be amortised for the AOT arms to pay off (per cell, median over seeds)
    v = d[(d.world == "V2") & (d.budget == "all")]
    b0 = v[v.policy == "B0"].set_index(["workload", "N", "seed"])
    am = []
    for pol in ("B1-AOT", "B1-AOT-K"):
        aa = v[v.policy == pol].set_index(["workload", "N", "seed"])
        sav = b0.total_s - aa.total_s
        nproc = (aa.build_built_s / sav).where(sav > 0)
        for (wl, n), x in nproc.groupby(level=["workload", "N"]):
            am.append(dict(policy=pol, workload=wl, N=n, build_built_s=float(aa.build_built_s.iloc[0]), saving_per_process_s=float(sav.loc[wl, n].median()), processes_to_amortise_median=float(x.median())))
    pd.DataFrame(am).to_csv(OUT / "regime_amortise.csv", index=False)
    # ---- real-machine points
    rp = real_points()
    # scan range (10 seeds, default budget, disk) of the same ratio in both worlds
    def scan(row, world):
        pol = row.num
        col = "ratio_B0_median" if row.den == "B0" else "ratio_B1_median"
        lo = "ratio_B0_min" if row.den == "B0" else "ratio_B1_min"
        hi = "ratio_B0_max" if row.den == "B0" else "ratio_B1_max"
        x = g[(g.world == world) & (g.cache == "disk") & (g.workload == row.workload) & (g.N == row.N) & (g.policy == pol) & (g.budget == "default")]
        if len(x) == 0 or x[col].isna().all():
            return (np.nan, np.nan, np.nan)
        return (float(x[lo].iloc[0]), float(x[col].iloc[0]), float(x[hi].iloc[0]))
    for world in ("S3", "V2"):
        v = [scan(r, world) for r in rp.itertuples()]
        rp[f"scan{world}_min"], rp[f"scan{world}_median"], rp[f"scan{world}_max"] = zip(*v)
    rp["scanV2_same_side_as_measured"] = ((rp.scanV2_median - 1) * (rp.ratio_measured - 1) > 0).astype(int)
    rp.loc[rp.scanV2_median.isna(), "scanV2_same_side_as_measured"] = -1
    rp.to_csv(OUT / "regime_realpoints.csv", index=False)
    print(rp.round(3).to_string(index=False))
    print("pred same side:", int(rp.same_side_of_1.sum()), "/", len(rp), "; V2 scan median same side:", int((rp.scanV2_same_side_as_measured == 1).sum()), "/", int((rp.scanV2_same_side_as_measured >= 0).sum()))


def med_proc(path, col="service_s", polcol="policy"):
    d = pd.read_csv(path)
    d = d[d.status == "ok"]
    return d.groupby(["stream", polcol])[col].median()


def real_points():
    """Measured (3-process median, incl. input) vs frozen predictions for the real streams (descriptive). Ratios Ours/B1, Ours/B0, B1/B0, B1-AOT/B0, K/B0 when both arms exist."""
    srcs = [("S3 3.4 real (seed100, N=300)", ROOT / "results/S3/real_validation/measured_processes.csv", ROOT / "results/S3/real_validation/pred_S3_4.csv", 300),
            ("S4-P block C (seed101, N=300)", ROOT / "results/S4P/p3/real_validation/measured_processes.csv", ROOT / "results/S4P/p3/real_validation/pred_S4P_P5c.csv", 300),
            ("S4-P P5d (seed130, N=1000)", ROOT / "results/S4P/p5d/verdict/d123_processes.csv", ROOT / "results/S4P/p5d/pred_P5d.csv", 1000),
            ("S4-X X3 (seed131, N=1000)", ROOT / "results/S4X/x3/verdict/k1234_processes.csv", ROOT / "results/S4X/x3/pred_X3.csv", 1000)]
    out = []
    for name, mpth, ppth, n in srcs:
        m = pd.read_csv(mpth)
        m = m[m.status == "ok"]
        if "cache_mode" in m:
            m = m[m.cache_mode == "warm"]
        med = m.groupby(["stream", "policy"]).service_s.median()
        pr = pd.read_csv(ppth)
        if "cache_mode" in pr:
            pr = pr[pr.cache_mode == "warm"]
        pm = pr.groupby(["stream", "policy"]).pred_total_s.first()
        for stream in sorted(set(med.index.get_level_values(0))):
            wl = stream.split("_")[0]
            for num, den in (("Ours", "B1"), ("Ours", "B0"), ("B1", "B0"), ("B1-AOT", "B0"), ("B1-AOT-K", "B0"), ("B1-AOT", "B1")):
                if (stream, num) in med.index and (stream, den) in med.index and (stream, num) in pm.index and (stream, den) in pm.index:
                    meas = med[(stream, num)] / med[(stream, den)]
                    pred = pm[(stream, num)] / pm[(stream, den)]
                    out.append(dict(source=name, stream=stream, workload=wl, N=n, num=num, den=den, ratio_measured=meas, ratio_pred=pred, meas_over_pred=meas / pred,
                                    same_side_of_1=int((meas - 1) * (pred - 1) > 0)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- plot
def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 7
    g = pd.read_csv(OUT / "regime_grid.csv")
    rp = pd.read_csv(OUT / "regime_realpoints.csv")
    COL = {"wide": "#1f77b4", "default": "k", "tight": "#d62728"}
    LAB = {"wide": "wide budget", "default": "default budget", "tight": "tight budget"}

    def series(world, cache, wl, policy, budget, col):
        x = g[(g.world == world) & (g.cache == cache) & (g.workload == wl) & (g.policy == policy) & (g.budget == budget)].sort_values("N")
        return x

    def panels(title):
        fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=True, sharey=False)
        fig.suptitle(title, fontsize=9)
        return fig, axs

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    def fin(fig, axs, ylab, ticks):
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                tk = ticks[cache]
                ax.axhline(1.0, color="0.5", lw=0.8, ls="-", zorder=0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(tk[0] / 1.15, tk[-1] * 1.15)
                ax.yaxis.set_major_locator(FixedLocator(tk))
                ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_title(f"{wl}, {'empty cache' if cache == 'empty' else 'disk cache'}", fontsize=7.5)
                ax.set_xticks(NS)
                ax.set_xticklabels([str(n) if n < 1000 else f"{n//1000}k" for n in NS], fontsize=6)
                ax.minorticks_off()
                if j == 0:
                    ax.set_ylabel(ylab)
                if i == 1:
                    ax.set_xlabel("requests per process N")
                ax.grid(alpha=0.25, lw=0.4)
        h, l = axs[1][0].get_legend_handles_labels()
        seen = {}
        for ax in axs.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(ll, hh)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, fontsize=6.5, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    with PdfPages(OUT / "fig_regime.pdf") as pdf:
        # ---------- page 1: Ours / B1
        fig, axs = panels("Ours (reuse-aware, frozen S3 policy) / B1 (compile on first sight): total service time ratio; <1 = Ours faster")
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                for b in ("wide", "default", "tight"):
                    x = series("S3", cache, wl, "Ours", b, "ratio_B1_median")
                    ax.plot(x.N, x.ratio_B1_median, color=COL[b], lw=1.3 if b == "default" else 0.9, ls="-", marker="o", ms=2.5, label=f"sim S3 world, {LAB[b]}")
                    if b == "default":
                        ax.fill_between(x.N, x.ratio_B1_min, x.ratio_B1_max, color=COL[b], alpha=0.15, lw=0)
                if cache == "disk":
                    for b in ("wide", "default", "tight"):
                        x = series("V2", "disk", wl, "Ours", b, "ratio_B1_median")
                        ax.plot(x.N, x.ratio_B1_median, color=COL[b], lw=1.0 if b == "default" else 0.7, ls="--", marker="s", ms=2.0, label=f"sim V2 world (real-calibrated), {LAB[b]}")
                    pts = rp[(rp.workload == wl) & (rp.num == "Ours") & (rp.den == "B1")]
                    for src, mk in (("S3 3.4 real (seed100, N=300)", "o"), ("S4-P block C (seed101, N=300)", "D")):
                        q = pts[pts.source == src]
                        if len(q):
                            ax.scatter(q.N, q.ratio_measured, facecolors="none", edgecolors="m", marker=mk, s=28, lw=1.2, zorder=5, label=f"real 4090: {src.split(' (')[0]}")
        fin(fig, axs, "Ours / B1", {"empty": [0.04, 0.1, 0.2, 0.5, 1.0, 1.2], "disk": [0.2, 0.3, 0.5, 0.7, 1.0, 1.2]})
        pdf.savefig(fig)
        plt.close(fig)
        # ---------- page 2: everything / B0
        fig, axs = panels("Total service time relative to all-eager B0 (<1 = faster than B0).\nEmpty-cache row: B1/Ours from S3 world (imputed); AOT arms from V2 world with offline build time charged (cross-world, order of magnitude only).\nDisk row: B1 above 1.5 is off-scale (up to 5x at N=100)")
        STY = {"B1": ("#8c564b", "^", "-"), "Ours": ("k", "o", "-"), "B1-AOT": ("#2ca02c", "s", "-"), "B1-AOT-K": ("#ff7f0e", "D", "-"),
               "B1-AOT|cold": ("#2ca02c", "s", "-"), "B1-AOT-K|cold": ("#ff7f0e", "D", "-")}
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                if cache == "disk":
                    for p in ("B1", "Ours", "B1-AOT", "B1-AOT-K"):
                        x = series("V2", "disk", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim V2: {p}")
                    for p in ("B1", "Ours"):
                        x = series("S3", "disk", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.0, lw=0.7, ls="--", label=f"sim S3: {p}")
                    pts = rp[(rp.workload == wl) & (rp.den == "B0")]
                    for num, mk in (("B1", "^"), ("Ours", "o"), ("B1-AOT", "s"), ("B1-AOT-K", "D")):
                        q = pts[pts.num == num]
                        if len(q):
                            ax.scatter(q.N, q.ratio_measured, facecolors="none", edgecolors=STY[num][0], marker=mk, s=28, lw=1.2, zorder=5, label=f"real 4090: {num}")
                else:
                    for p in ("B1", "Ours"):
                        x = series("S3", "empty", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim S3 (imputed): {p}")
                    for p in ("B1-AOT|cold", "B1-AOT-K|cold"):
                        x = series("V2", "empty", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim V2: {p.split('|')[0]} + build time")
        fin(fig, axs, "policy / B0", {"empty": [0.8, 1, 2, 5, 10, 20, 50, 100, 200], "disk": [0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]})
        pdf.savefig(fig)
        plt.close(fig)
    print("wrote", OUT / "fig_regime.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["run", "check", "analyze", "plot"])
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    assert 1 <= a.procs <= 8
    try:
        os.nice(10)
    except OSError:
        pass
    sys.exit({"run": cmd_run, "check": cmd_check, "analyze": cmd_analyze, "plot": cmd_plot}[a.stage](a) or 0)


if __name__ == "__main__":
    main()
