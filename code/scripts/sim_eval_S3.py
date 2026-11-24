"""S3 3.3 batch runner of the request-level simulator (CPU only, <=8 worker processes, nice 10).

One raw-CSV row = one simulation of ONE process: (policy variant, workload, N, seed, budget, cold level [, prior kind, trace kind,
cost-table bootstrap replicate]). Experiments (all are presets of this one script; `--experiment a,b,...`):

  main       W1-W5 x N x seeds x 3 cold levels x policies B0-B7, Ours, Ours-NE (frozen params; default budget)      -> main figure
  budget     same grid, cold=disk, 4 budget tiers (tight/mid/default/wide) from budgets.json
  prior      prior mismatch: prior kind {A (default), B (matched to the evaluation half), uniform, swapped orientation} for the
             policies that use a prior, on W2/W3/W4 (W4 = mid-run drift) ; default budget, cold=disk
  ablation   Ours v1_spec / fixed horizon / three tiers / two tiers / no prior / no padding / no component-key credit / LRU (=B5) /
             Ours-NE / alpha in {0.5,1,2}, plus the baselines as references; W2/W5, default budget, cold=disk
  bootstrap  cost-table PROCESS bootstrap: --boots (200) tables are rebuilt from cost_table_procs.csv (processes of every cell
             resampled with replacement, then the same B3-interpolation / C2,C5,C6 extrapolation as build_cost_table.py),
             W2/W3 N=1e4, all policies re-run on every table (policy and simulator see the same resampled table); boot=-1 is the original
  declared   declared-distribution sensitivity: the TRACES are regenerated from the same half with CFG probability 0.25/0.5/0.75 and
             two alternative B distributions (uniform; heavy: .10/.20/.20/.50); the policy prior and budgets stay at the declared
             defaults (the operator's declaration is what is wrong). W2/W3 N=1e4, cold=disk.

Cold levels (SimOptions.cache_mode): empty = cache_mode 'empty'; disk = 'warm' (pre-warmed disk cache; cold_diskcache for the first shape,
inproc_diskcache for later ones); hot = 'warm' with the first shape ALSO charged at the in-process tier (the process is already
running: wrapper around simulator.tier_for, no file changed) -> "in-process warm".

Splits: --split dev reads only workload/dev_traces (half A, seeds 0-4, N 1e3/1e4) and may generate extra dev N from the A half pool
(--dev-extra-N, in memory, flagged trace_kind=gen_default). --split eval reads workload/eval_traces (half B, seeds 100-109) and needs
results/S3/policy_params.json to be read-only (0444), its cost-table SHA and the policy-code SHAs to match; dev never touches eval_traces.
Every file trace is checked against workload/sha256_manifest.txt.

Missing costs: the S2 cost table has no measured cold_empty one-time cost for most compile shapes. Default `--missing error`: such a
simulation is recorded as status=missing_cost (nothing is guessed). `--missing impute` fills the missing cold_empty residual /
first-call seconds of the compile arms with (disk-cache value x median empty/disk ratio of the measured cells of that arm) and flags every
row with imputed=1 (main-session decision needed before it may be used for results).

Usage (from project root, after `source ops/project_env.sh`; CPU only):
  export CUDA_VISIBLE_DEVICES=''; nice -n 10 "$PROJ_PYTHON" -B code/scripts/sim_eval_S3.py --split dev --experiment main --out results/S3/sim_dev/eval_tools
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import pathlib
import stat
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from p24.policy import CostView, NAMES, make_policy, shape_of  # noqa: E402
from p24.policy.base import Shape  # noqa: E402
from p24.policy.ours import V1_SPEC_PARAMS  # noqa: E402
from p24.sim import simulator as SIM  # noqa: E402
from p24.sim.costs import COMPILE_ARMS, COMPONENTS, CostTable, MissingCost, tier_for as _TIER_FOR  # noqa: E402
from p24.sim import addendum as ADD  # noqa: E402   (3.1b cost addendum overlay)
from p24.sim.prior import prior_A  # noqa: E402
from p24.workload import generate as G  # noqa: E402
from p24.workload import split as SPL  # noqa: E402

CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"
PROCS_PATH = ROOT / "results/S2/cost_table/merged/cost_table_procs.csv"
ADDENDUM_PATH = ROOT / "results/S3/cost_addendum/cost_addendum.csv"   # default --addendum (S3-A measurements); `--addendum none` switches it off
PARAMS_PATH = ROOT / "results/S3/policy_params.json"
BUDGETS_PATH = ROOT / "results/S3/budgets.json"
MANIFEST_PATH = ROOT / "results/S3/workload/sha256_manifest.txt"
WL_DIR = ROOT / "results/S3/workload"
META_PARQUET = ROOT / "data/diffusiondb/metadata.parquet"
SPLIT_PARQUET = ROOT / "results/S2/workload/split_A_B.parquet"
CACHE_DIR = pathlib.Path(os.environ.get("P24_CACHE", "cache/p24")) / "sim_eval"

COLDS = {"empty": ("empty", False), "disk": ("warm", False), "hot": ("warm", True)}   # name -> (cache_mode, first shape at in-process tier)
SPLITS = {
    "dev": dict(half="A", dir="dev_traces", suffix="A", seeds=[0, 1, 2, 3, 4], N=[1000, 10000]),
    "eval": dict(half="B", dir="eval_traces", suffix="B", seeds=list(range(100, 110)), N=[100, 1000, 10000, 100000]),
}
PRIOR_USERS = {"B3pp", "B5", "B6", "B6p", "Ours", "Ours-NE"}
B_ALT = {"B_unif": [0.25, 0.25, 0.25, 0.25], "B_heavy": [0.10, 0.20, 0.20, 0.50]}   # declared-distribution sensitivity (alternatives to .55/.20/.10/.15)
CFG_ALT = {"cfg0.25": 0.25, "cfg0.5": 0.5, "cfg0.75": 0.75}
sha16 = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


# ====================================================================================================== variants / experiments
@dataclass
class Variant:
    label: str                 # name in the CSV `variant` column
    policy: str                # make_policy name (p24.policy.NAMES)
    params: dict = field(default_factory=dict)
    prior: str = "A"           # prior kind: A | B | uniform | swapped


def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text())


def base_params(pj: dict) -> dict:
    p = dict(pj["ours"]["params"])
    p["tiers"] = tuple(p["tiers"])
    return p


def v_main(P: dict) -> List[Variant]:
    return [Variant(n, n, dict(P)) for n in NAMES]


def v_ablation(P: dict) -> List[Variant]:
    out = [Variant("Ours", "Ours", dict(P)), Variant("Ours-NE", "Ours-NE", dict(P)), Variant("B5", "B5", dict(P))]   # B5 = LRU eviction
    out += [Variant("Ours:v1_spec", "Ours", {**P, **V1_SPEC_PARAMS}),
            Variant("Ours:fixed_horizon", "Ours", {**P, "horizon": "fixed"}),
            Variant("Ours:three_tiers", "Ours", {**P, "tiers": ("E", "G1", "C1+G1")}),
            Variant("Ours:two_tiers", "Ours", {**P, "tiers": ("E", "C1+G1")}),
            Variant("Ours:no_prior", "Ours", {**P, "prior_admit": False}),
            Variant("Ours:no_padding", "Ours", {**P, "pad": False}),
            Variant("Ours:no_keydecomp", "Ours", {**P, "key_decomp": False}),
            Variant("Ours:alpha0.5", "Ours", {**P, "alpha": 0.5}),
            Variant("Ours:alpha2", "Ours", {**P, "alpha": 2.0})]
    out += [Variant(n, n, dict(P)) for n in NAMES if n not in ("Ours", "Ours-NE", "B5")]
    return out


def v_prior(P: dict, kinds: Sequence[str]) -> List[Variant]:
    out = []
    for n in NAMES:
        if n in PRIOR_USERS:
            out += [Variant(n if k == "A" else f"{n}@{k}", n, dict(P), k) for k in kinds]
        else:
            out.append(Variant(n, n, dict(P)))
    return out


@dataclass
class Exp:
    name: str
    workloads: List[str]
    Ns: List[int]
    seeds: List[int]
    budgets: List[str]
    colds: List[str]
    variants: List[Variant]
    trace_kinds: List[str] = field(default_factory=lambda: ["file"])
    boots: Optional[int] = None


def build_exp(name: str, split: str, P: dict, a) -> Exp:
    sp = SPLITS[split]
    seeds = a.seeds or sp["seeds"]
    Ns_all = a.N or sp["N"]
    colds = a.colds or None
    wl_all = a.workloads or G.SCENARIOS
    if name == "main":
        v = v_main(P)
        if a.policies:
            v = [x for x in v if x.label in a.policies]
        return Exp(name, wl_all, Ns_all, seeds, a.budgets or ["default"], colds or ["empty", "disk", "hot"], v)
    if name == "budget":
        v = v_main(P)
        if a.policies:
            v = [x for x in v if x.label in a.policies]
        return Exp(name, wl_all, a.N or [1000, 10000], seeds, a.budgets or ["tight", "mid", "default", "wide"], colds or ["disk"], v)
    if name == "prior":
        return Exp(name, a.workloads or ["W2", "W3", "W4"], a.N or [1000, 10000], seeds, a.budgets or ["default"], colds or ["disk"],
                   v_prior(P, ["A", "B", "uniform", "swapped"]))
    if name == "ablation":
        return Exp(name, a.workloads or ["W2", "W5"], a.N or [1000, 10000], seeds, a.budgets or ["default"], colds or ["disk"], v_ablation(P))
    if name == "bootstrap":
        v = [x for x in v_main(P)]
        return Exp(name, a.workloads or ["W2", "W3"], a.N or [10000], seeds, a.budgets or ["default"], colds or ["disk"], v, boots=a.boots)
    if name == "declared":
        return Exp(name, a.workloads or ["W2", "W3"], a.N or [10000], seeds, a.budgets or ["default"], colds or ["disk"], v_main(P),
                   trace_kinds=["gen_default"] + list(CFG_ALT) + list(B_ALT))
    raise KeyError(name)


# ====================================================================================================== traces
def _slim(r: dict) -> dict:
    return {"req_id": r.get("req_id", ""), "H": r["H"], "W": r["W"], "B": r["B"], "cfg": r["cfg"]}


def read_trace_file(path: pathlib.Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [_slim(json.loads(l)) for l in f if l.strip()]


def pool_arrays(half: str):
    """(rows, hw) of half A/B from metadata.parquet (width/height only) + split_A_B.parquet; cached as npz on the data disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"pool_{half}.npz"
    if f.exists():
        z = np.load(f)
        return z["rows"], z["hw"]
    df = pd.read_parquet(META_PARQUET, columns=["width", "height"])
    sp = pd.read_parquet(SPLIT_PARQUET)
    assert len(sp) == len(df) and (sp["row"].to_numpy() == np.arange(len(df))).all()
    for h in "AB":
        sel = np.flatnonzero(sp["half"].to_numpy() == h)
        hw = SPL.map_resolutions(df["width"].to_numpy()[sel], df["height"].to_numpy()[sel])
        tmp = CACHE_DIR / f"pool_{h}.tmp.npz"
        np.savez(tmp, rows=sel.astype(np.int64), hw=hw)
        os.replace(tmp, CACHE_DIR / f"pool_{h}.npz")
    z = np.load(f)
    return z["rows"], z["hw"]


_POOLS: Dict[str, G.HalfPool] = {}


def get_pool(half: str) -> G.HalfPool:
    if half not in _POOLS:
        rows, hw = pool_arrays(half)
        _POOLS[half] = G.HalfPool(half=half, rows=rows, prompts=np.zeros(len(rows), dtype=object), hw=hw)   # prompts are irrelevant to the simulator
    return _POOLS[half]


@contextlib.contextmanager
def declared(b_probs=None, cfg_prob=None):
    """Temporarily change the declared B / CFG distribution of the generator (module constants read at call time)."""
    ob, oc = G.B_PROBS, G.CFG_PROB
    try:
        if b_probs is not None:
            G.B_PROBS = list(b_probs)
        if cfg_prob is not None:
            G.CFG_PROB = float(cfg_prob)
        yield
    finally:
        G.B_PROBS, G.CFG_PROB = ob, oc


def gen_trace(half: str, w: str, n: int, seed: int, tkind: str) -> List[dict]:
    pool = get_pool(half)
    kw = {}
    if tkind in CFG_ALT:
        kw["cfg_prob"] = CFG_ALT[tkind]
    elif tkind in B_ALT:
        kw["b_probs"] = B_ALT[tkind]
    elif tkind != "gen_default":
        raise KeyError(tkind)
    with declared(**kw):
        return [_slim(r) for r in G.generate_scenario(pool, w, n, seed)]


class TraceStore:
    """Loads (and caches the last few) traces; verifies file traces against the S3 manifest."""

    def __init__(self, split: str, manifest: Dict[str, str]):
        self.split, self.manifest, self.cache = split, manifest, collections.OrderedDict()

    def rel(self, w, n, s):
        sp = SPLITS[self.split]
        return f"{sp['dir']}/{w}_N{n}_seed{s}_{sp['suffix']}.jsonl"

    def get(self, w, n, s, tkind):
        key = (w, n, s, tkind)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        sp = SPLITS[self.split]
        if tkind == "file":
            rel = self.rel(w, n, s)
            assert (self.split == "eval") == ("eval_traces" in rel), rel            # dev never touches eval traces
            p = WL_DIR / rel
            assert p.exists(), p
            assert hashlib.sha256(p.read_bytes()).hexdigest() == self.manifest[rel], f"manifest SHA mismatch {rel}"
            reqs = read_trace_file(p)
            assert len(reqs) == n, (rel, len(reqs))
        else:
            assert s in sp["seeds"], f"generated traces only for the split's own seeds ({sp['seeds']})"
            reqs = gen_trace(sp["half"], w, n, s, tkind)
        self.cache[key] = reqs
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return reqs


# ====================================================================================================== cost tables
def impute_empty(df: pd.DataFrame):
    """Fill missing cold_empty one-time columns of the compile arms from the disk-cache columns x median measured empty/disk ratio."""
    df = df.copy()
    meas = df.source == "measured"
    log = {}
    for pref in ["resid_s"] + [f"fc_{c}_s" for c in COMPONENTS]:
        ce, cd = f"{pref}__cold_empty", f"{pref}__cold_diskcache"
        allm = df.arm.isin(COMPILE_ARMS) & meas & df[ce].notna() & df[cd].notna() & (df[cd] > 0)
        pooled = float(np.median((df[ce] / df[cd])[allm])) if allm.any() else np.nan
        for arm in sorted(COMPILE_ARMS & set(df.arm)):
            m = df.arm == arm
            both = m & allm
            ratio = float(np.median((df[ce] / df[cd])[both])) if both.any() else pooled
            miss = m & df[ce].isna() & df[cd].notna()
            if np.isfinite(ratio) and miss.any():
                df.loc[miss, ce] = df.loc[miss, cd] * ratio
            log[f"{arm}/{pref}"] = dict(ratio=ratio, n_filled=int(miss.sum()), n_pairs=int(both.sum()), pooled_ratio=pooled)
    return df, log


def bootstrap_table(procs: pd.DataFrame, orig: pd.DataFrame, b: int):
    """One process-bootstrap cost table: processes of every (arm,H,W,B,cfg) cell resampled with replacement (seed 1000+b), then the same cell
    aggregation / B3 interpolation / C2,C5,C6 extrapolation as scripts/build_cost_table.py; transient / equivalence columns are copied from
    the original table. Identity resampling reproduces the original table to 1e-9 (checked in --selftest)."""
    from p24 import stats
    from scripts import build_cost_table as BCT
    rng = np.random.default_rng(1000 + b)
    parts = []
    for _, g in sorted(procs.groupby(BCT.KEY), key=lambda kv: tuple(kv[0])):
        parts.append(g.iloc[rng.integers(0, len(g), len(g))] if b >= 0 else g)
    pp = pd.concat(parts, ignore_index=True)
    ob = stats.boot_ci
    stats.boot_ci = lambda x, n=0, seed=0: (np.nan, np.nan)           # CI columns are not used by the simulator
    try:
        ct = BCT.cells_from_procs(pp)
        ct, _ = BCT.interpolate_b3(ct)
        ct = BCT.extrapolate(ct, ["C2", "C5", "C6"])
    finally:
        stats.boot_ci = ob
    extra = ["transient_excess_s", "equiv_pass", "equiv_pass_padB4"]
    ct = ct.drop(columns=[c for c in extra if c in ct.columns]).merge(orig[BCT.KEY + extra], on=BCT.KEY, how="left")
    return ct.sort_values(["H", "W", "B", "cfg", "arm"]).reset_index(drop=True)


# ====================================================================================================== priors
def prior_from_res_dist(pres: Dict[tuple, float]) -> Dict[Shape, float]:
    pr = {}
    for (H, W), p in pres.items():
        for b, pb in zip(G.B_VALUES, G.B_PROBS):
            for c, pc in ((0, 1 - G.CFG_PROB), (1, G.CFG_PROB)):
                pr[Shape(H, W, b, c)] = p * pb * pc
    s = sum(pr.values())
    return {k: v / s for k, v in pr.items()}


def make_priors(need_B: bool) -> Dict[str, Dict[Shape, float]]:
    A = prior_A()
    out = {"A": A, "uniform": {s: 1.0 / len(A) for s in A}}
    out["swapped"] = {Shape(s.W, s.H, s.B, s.cfg) if s.H != s.W else s: p for s, p in A.items()}
    if need_B:
        rows, hw = pool_arrays("B")
        ri = SPL.res_index(hw)
        pres = np.bincount(ri, minlength=len(SPL.RESOLUTIONS)) / len(ri)
        out["B"] = prior_from_res_dist({r: float(pres[i]) for i, r in enumerate(SPL.RESOLUTIONS)})
    return out


# ====================================================================================================== worker
W: Dict[str, object] = {}


def _init_worker(cfg: dict):
    try:
        os.nice(10)
    except OSError:
        pass
    W.update(cfg)
    W["store"] = TraceStore(cfg["split"], cfg["manifest"])
    if cfg.get("addendum"):                       # overlay of the S3-A measurements; the simulator's own CostView must see the rebuild override too
        W["add_df"] = ADD.read_addendum(cfg["addendum"])
        ADD.install_simulator_hook()
    W["ct0"] = _mk_table(pd.read_csv(CT_PATH))
    if cfg["impute"]:
        W["ct0i"] = _mk_table(impute_empty(pd.read_csv(CT_PATH))[0])
    W["procs"] = pd.read_csv(PROCS_PATH) if cfg["need_procs"] else None
    W["orig_df"] = pd.read_csv(CT_PATH) if cfg["need_procs"] else None
    W["tabs"] = {}


def _mk_table(df: pd.DataFrame) -> CostTable:
    """Cost table from a frame; with an addendum only the cells it covers are replaced (p24.sim.addendum.apply_addendum)."""
    return ADD.load_with_addendum(df, W["add_df"]) if W.get("add_df") is not None else CostTable(df)


def _table(boot: int):
    if boot < 0:
        return W["ct0i"] if W["impute"] else W["ct0"]
    if boot not in W["tabs"]:
        W["tabs"].clear()
        df = bootstrap_table(W["procs"], W["orig_df"], boot)
        W["tabs"][boot] = _mk_table(impute_empty(df)[0] if W["impute"] else df)
    return W["tabs"][boot]


def _set_hot(flag: bool):
    SIM.tier_for = (lambda cm, first: _TIER_FOR(cm, False)) if flag else _TIER_FOR


def run_task(task):
    """task = (boot, [(w, n, seed, tkind), ...]) -> list of raw rows."""
    boot, traces = task
    ex: Exp = W["exp"]
    ct = _table(boot)
    rows = []
    views: Dict[str, CostView] = {}
    for (w, n, seed, tkind) in traces:
        reqs = W["store"].get(w, n, seed, tkind)
        oc = dict(collections.Counter(shape_of(q) for q in reqs))
        for budget in ex.budgets:
            mb = W["budgets"][budget]
            for cold in ex.colds:
                cm, hot = COLDS[cold]
                _set_hot(hot)
                cv = views.setdefault(cm, CostView(ct, cm, rebuild_override=getattr(ct, "rebuild_override", None)))
                opt = SIM.SimOptions(cache_mode=cm, mem_budget_mb=mb, overhead_aware=True)
                for v in ex.variants:
                    row = dict(experiment=ex.name, split=W["split"], variant=v.label, policy=v.policy, workload=w, N=n, seed=seed,
                               budget=budget, budget_mb=mb, cold=cold, prior_kind=v.prior, trace_kind=tkind, boot=boot,
                               imputed=int(W["impute"] and cm == "empty"))
                    t0 = time.perf_counter()
                    try:
                        pol = make_policy(v.policy, cost=cv, params=v.params, prior=W["priors"][v.prior], oracle_counts=oc)
                        s = SIM.simulate(reqs, pol, ct, opt).summary
                        row.update(status="ok", error="", service_s=s["service_s"], load_s=s["load_s"], warmup_s=s["warmup_s"],
                                   total_incl_load_s=s["total_incl_load_s"], peak_graph_mb=s["peak_graph_mb"],
                                   peak_reserved_est_mb=s["peak_reserved_est_mb"], n_requests=s["n_requests"], n_shapes=s["n_shapes"],
                                   n_capture=s.get("n_capture", 0), n_compile=s.get("n_compile", 0),
                                   n_compile_capture=s.get("n_compile_capture", 0), n_capture_denied=s.get("n_capture_denied", 0),
                                   n_eager_first=s.get("n_eager_first", 0), n_eager_fallback_limit=s.get("n_eager_fallback_limit", 0),
                                   n_evicted=s["n_evicted_total"], n_denied=s["n_denied_total"], n_steady=s.get("n_steady", 0),
                                   hit_rate=s.get("n_steady", 0) / max(1, s["n_requests"]), n_events_with_fallback=s["n_events_with_fallback"])
                    except MissingCost as e:
                        row.update(status="missing_cost", error=str(e)[:160])
                    except Exception as e:  # noqa: BLE001  (recorded, run continues; the summary counts them)
                        row.update(status="error", error=f"{type(e).__name__}: {str(e)[:160]}")
                    row["sim_wall_s"] = time.perf_counter() - t0
                    rows.append(row)
    _set_hot(False)
    return rows


# ====================================================================================================== guards / main
def read_manifest() -> Dict[str, str]:
    d = {}
    for l in MANIFEST_PATH.read_text().splitlines():
        if l.strip():
            sha, rel = l.split(None, 1)
            d[rel.strip()] = sha
    return d


def code_shas() -> Dict[str, str]:
    base = ROOT / "code/p24"
    return {k: sha16(base / k) for k in ("policy/base.py", "policy/ours.py", "policy/baselines.py", "policy/factory.py", "policy/costview.py",
                                          "sim/simulator.py", "sim/prior.py")}


def guards(split: str, pj: dict) -> dict:
    mode = stat.S_IMODE(PARAMS_PATH.stat().st_mode)
    info = dict(params_mode=oct(mode), params_sha16=sha16(PARAMS_PATH), cost_table_sha16=sha16(CT_PATH),
                manifest_sha16=sha16(MANIFEST_PATH), code_sha16=code_shas())
    assert info["cost_table_sha16"] == pj["cost_table_sha16"], "cost table differs from the one frozen in policy_params.json"
    assert info["manifest_sha16"] == pj["manifest_sha16"], "workload manifest differs from policy_params.json"
    frozen_code = pj["policy_code"]
    info["code_matches_freeze"] = all(frozen_code.get(k) == v for k, v in info["code_sha16"].items())
    if split == "eval":
        assert mode == 0o444, f"eval split requires {PARAMS_PATH} to be read-only 0444 (is {oct(mode)})"
        assert info["code_matches_freeze"], f"policy/sim code differs from the frozen SHAs: {info['code_sha16']} vs {pj['policy_code']}"
    elif not info["code_matches_freeze"]:
        print("WARNING: policy/sim code differs from the SHAs recorded in policy_params.json (dev mode: continuing)", file=sys.stderr)
    return info


def selftest():
    """Tool self-checks (CPU, seconds): identity bootstrap == original table; generated default traces == dev files; hot/disk/empty wiring."""
    pj = load_params()
    guards("dev", pj)
    orig = pd.read_csv(CT_PATH)
    procs = pd.read_csv(PROCS_PATH)
    from scripts import build_cost_table as BCT
    ident = bootstrap_table(procs, orig, -1).set_index(BCT.KEY).sort_index()
    ref = orig.set_index(BCT.KEY).sort_index()
    bad = []
    for c in ref.columns:
        if c in ("steady_ci_lo", "steady_ci_hi", "block", "source", "fc_source"):
            continue
        if not np.allclose(ident[c].reindex(ref.index).astype(float), ref[c].astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
            bad.append(c)
    assert not bad, f"identity bootstrap differs in {bad}"
    print("selftest: identity bootstrap reproduces cost_table.csv (all numeric columns)")
    store = TraceStore("dev", read_manifest())
    for (w, n, s) in [("W2", 1000, 0), ("W4", 1000, 1), ("W5", 10000, 2)]:
        f = store.get(w, n, s, "file")
        g = gen_trace("A", w, n, s, "gen_default")
        assert [(x["H"], x["W"], x["B"], x["cfg"]) for x in f] == [(x["H"], x["W"], x["B"], x["cfg"]) for x in g], (w, n, s)
    print("selftest: in-memory generated dev traces reproduce the dev files (shape sequence)")
    ct = CostTable.from_csv(CT_PATH)
    reqs = store.get("W2", 1000, 0, "file")
    res = {}
    for cold, (cm, hot) in COLDS.items():
        _set_hot(hot)
        res[cold] = SIM.simulate(reqs, make_policy("B0"), ct, SIM.SimOptions(cache_mode=cm)).summary["service_s"]
    _set_hot(False)
    print("selftest: B0 W2 N1e3 seed0 service_s by cold level:", {k: round(v, 2) for k, v in res.items()})
    assert res["hot"] <= res["disk"] + 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--experiment", default="main", help="comma list of: main,budget,prior,ablation,bootstrap,declared (or all)")
    ap.add_argument("--out", help="output directory for <experiment>_raw.csv / <experiment>_meta.json")
    ap.add_argument("--workloads", type=lambda s: s.split(","))
    ap.add_argument("--N", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--colds", type=lambda s: s.split(","), help="subset of empty,disk,hot")
    ap.add_argument("--budgets", type=lambda s: s.split(","), help="subset of tight,mid,default,wide")
    ap.add_argument("--policies", type=lambda s: s.split(","), help="variant labels (main/budget only), e.g. B0,B1,Ours")
    ap.add_argument("--boots", type=int, default=200)
    ap.add_argument("--dev-extra-N", type=lambda s: [int(x) for x in s.split(",")], default=[],
                    help="dev only: extra N generated in memory from the A half pool (e.g. 100,100000) for tool checks")
    ap.add_argument("--addendum", default=str(ADDENDUM_PATH),
                    help="S3-A cost addendum csv (default results/S3/cost_addendum/cost_addendum.csv; a missing file is an ERROR); 'none' = S2 table only")
    ap.add_argument("--missing", choices=["error", "impute"], default="error")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tiny", action="store_true", help="1 seed x 1 N x 1 workload x 1 cold: plumbing check")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-lock", action="store_true", help="do not chmod 0444 the outputs (dev only)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    assert a.out, "--out required"
    assert 1 <= a.procs <= 8, "at most 8 worker processes"
    pj = load_params()
    info = guards(a.split, pj)
    addendum = None if str(a.addendum).lower() == "none" else pathlib.Path(a.addendum)
    if addendum is not None:
        if not addendum.is_file():
            sys.exit(f"ERROR: cost addendum not found: {addendum} (the S3-A block has not produced it yet; pass --addendum none to run on the S2 table only)")
        rep_ = ADD.load_with_addendum(CT_PATH, addendum).addendum_report
        info["addendum"] = dict(path=str(addendum), sha16=sha16(addendum), n_inproc_replaced=rep_["n_inproc_replaced"],
                                n_inproc_skipped_no_table_row=len(rep_["inproc_skipped_no_table_row"]), n_override=rep_["n_override"])
        print(f"cost addendum: {info['addendum']}", flush=True)
    else:
        info["addendum"] = None
    P = base_params(pj)
    sp = SPLITS[a.split]
    if a.tiny:
        a.seeds, a.N, a.workloads, a.colds = a.seeds or sp["seeds"][:1], a.N or [1000], a.workloads or ["W2"], a.colds or ["disk"]
        a.boots = min(a.boots, 3)
    if a.seeds:
        assert set(a.seeds) <= set(sp["seeds"]), f"{a.split} seeds must be within {sp['seeds']}"
    if a.dev_extra_N:
        assert a.split == "dev", "--dev-extra-N is a dev-only tool check"
    if a.N and a.split == "dev":
        assert set(a.N) <= set(sp["N"]) | set(a.dev_extra_N), f"dev N must be in {sp['N']} (+ --dev-extra-N)"
    names = ["main", "budget", "prior", "ablation", "bootstrap", "declared"] if a.experiment == "all" else a.experiment.split(",")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bj = json.loads(BUDGETS_PATH.read_text())["tiers"]
    budgets = {k: float(v["mb"]) for k, v in bj.items()}
    manifest = read_manifest()
    for name in names:
        ex = build_exp(name, a.split, P, a)
        assert set(ex.colds) <= set(COLDS) and set(ex.budgets) <= set(budgets), (ex.colds, ex.budgets)
        csv_path = out / f"{name}_raw.csv"
        if csv_path.exists():
            sys.exit(f"REFUSING to overwrite {csv_path} (use a new --out)")
        need_B = any(v.prior == "B" for v in ex.variants)
        need_gen = any(k != "file" for k in ex.trace_kinds) or bool(a.dev_extra_N)
        if need_B or need_gen:
            pool_arrays("A")           # build the npz cache once in the parent
        priors = make_priors(need_B)
        # --- task list
        Ns = list(ex.Ns)
        specs = []
        for w in ex.workloads:
            for n in Ns:
                for s in ex.seeds:
                    for tk in ex.trace_kinds:
                        kind = tk
                        if tk == "file" and a.split == "dev" and n in a.dev_extra_N and n not in sp["N"]:
                            kind = "gen_default"
                        specs.append((w, n, s, kind))
        if ex.boots:
            boots = [-1] + list(range(ex.boots))
            tasks = [(b, specs) for b in boots]
        else:
            tasks = [(-1, [sp_]) for sp_ in specs]
        cfg = dict(split=a.split, manifest=manifest, exp=ex, budgets=budgets, priors=priors, impute=(a.missing == "impute"),
                   need_procs=bool(ex.boots), addendum=str(addendum) if addendum else None)
        if a.missing == "impute":
            print("NOTE: --missing impute is a cost-table extension (flagged imputed=1 on empty-cache rows); needs main-session approval.")
        t0 = time.time()
        rows = []
        n_sims = len(tasks) * (len(specs) if ex.boots else 1) * len(ex.budgets) * len(ex.colds) * len(ex.variants)
        print(f"[{name}] split={a.split} tasks={len(tasks)} sims~{n_sims} procs={a.procs}", flush=True)
        if a.procs == 1:
            _init_worker(cfg)
            for i, t in enumerate(tasks):
                rows += run_task(t)
        else:
            with mp.get_context("fork").Pool(a.procs, initializer=_init_worker, initargs=(cfg,)) as pool:
                for i, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
                    rows += r
                    if (i + 1) % max(1, len(tasks) // 10) == 0:
                        print(f"[{name}] {i+1}/{len(tasks)} tasks {time.time()-t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        key = ["experiment", "split", "variant", "workload", "N", "seed", "budget", "cold", "prior_kind", "trace_kind", "boot"]
        df = df.sort_values(key).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        st = df.status.value_counts().to_dict()
        miss = df[df.status != "ok"].groupby(["variant", "cold"]).size().to_dict()
        meta = dict(experiment=name, args={k: v for k, v in vars(a).items()}, n_rows=len(df), status_counts=st,
                    not_ok_by_variant_cold={f"{k[0]}|{k[1]}": v for k, v in miss.items()}, wall_s=time.time() - t0,
                    variants=[asdict(v) for v in ex.variants], colds=ex.colds, budgets={b: budgets[b] for b in ex.budgets},
                    guards=info, script_sha16=sha16(__file__), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    trace_manifest_checked=True, cold_definition="empty=cache_mode empty; disk=warm; hot=warm + first shape at inproc tier",
                    prior_kinds=sorted(priors), missing_mode=a.missing)
        (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, default=str, ensure_ascii=False))
        if a.split == "eval" or not a.no_lock:
            os.chmod(csv_path, 0o444)
            os.chmod(out / f"{name}_meta.json", 0o444)
        print(f"[{name}] rows={len(df)} status={st} wall={time.time()-t0:.0f}s -> {csv_path}", flush=True)
        if st.get("error"):
            print("ERRORS present:", df[df.status == "error"].error.value_counts().head(5).to_dict(), file=sys.stderr)


if __name__ == "__main__":
    main()
