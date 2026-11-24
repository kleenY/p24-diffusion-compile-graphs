"""S4-P P3 shared helpers (CPU only; imports the frozen S3 modules, changes nothing in them).

Streams: S3 3.4 real-validation streams (seed 100) and the new S4-P streams (seed 101), both half B, N=300.
Table loading: `load_v1` = frozen S2 table + S3 cost addendum exactly as scripts/predict_S3_4.py; `load_v2` = S4-P table v2 (+ rebuild-only addendum).
`make_plan` lists the (stream, policy, cache mode) runs of the S4-P validation (B0, B1, B4, Ours, v3 + W3 B3').
Policy "v3" is a PLACEHOLDER name: p24.policy.factory.make_policy has no `v3` until P1 freezes it; `make_policy_v3` below tries make_policy("v3")
and, if the name is unknown, returns None (callers then skip the policy and log it) -- unless --v3-stand-in <name> is given (dry run only).
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib

import pandas as pd

from p24.policy import CostView, make_policy, shape_of
from p24.sim import addendum as ADD
from p24.sim import simulator as SIM
from p24.sim.costs import CostTable, MissingCost
from p24.sim.prior import prior_A

from scripts import predict_S3_4 as P34

ROOT = P34.ROOT
R3 = ROOT / "results/S3"
R4 = ROOT / "results/S4P"
RV3 = R3 / "workload/real_validation"                        # seed 100 (S3 3.4)
RV4 = R4 / "workload/real_validation"                        # seed 101 (S4-P, gen_realval_S4P.py)
WLS = ["W2", "W3", "W5"]
STREAMS3 = {w: f"{w}_N300_seed100_B" for w in WLS}
STREAMS4 = {w: f"{w}_N300_seed101_B" for w in WLS}
CT_V1 = P34.CT_PATH
ADD_V1 = P34.ADDENDUM_PATH
ARMS = ["E0", "G1", "C1", "C1+G1"]                           # arms used by B0/B4/B1(B3')/Ours (B3'=C1, v3 uses the same four)
TIER_IN = "inproc_diskcache"
TIER_COLD = "cold_diskcache"
sha16 = P34.sha16


def stream_path(w: str, seed: int) -> pathlib.Path:
    return (RV3 / f"{STREAMS3[w]}.jsonl") if seed == 100 else (RV4 / f"{STREAMS4[w]}.jsonl")


def stream_shapes(path) -> collections.Counter:
    c = collections.Counter()
    for ln in open(path):
        r = json.loads(ln)
        c[(r["H"], r["W"], r["B"], r["cfg"])] += 1
    return c


def first_shape(path):
    r = json.loads(open(path).readline())
    return (r["H"], r["W"], r["B"], r["cfg"])


def load_v1():
    ct = ADD.load_with_addendum(CT_V1, ADD_V1)
    ADD.install_simulator_hook()
    return ct


def load_v2(table_csv, addendum_csv):
    ct = ADD.load_with_addendum(table_csv, addendum_csv)
    ADD.install_simulator_hook()
    return ct


def make_plan(policies=("B0", "B1", "B4", "Ours", "v3"), wls=WLS):
    """[(stream_key, policy, cache_mode)]; W3 gets B3' in addition (S4-P pre-registration: strategies B0, B1, B4, S3 Ours, v3, W3 + B3')."""
    runs = []
    for w in wls:
        for p in policies:
            runs.append((w, p, "warm"))
        if w == "W3":
            runs.append((w, "B3′", "warm"))
    return runs


def make_policy_v3(cost, params, prior, stand_in=None):
    for nm in (["v3"] if not stand_in else [stand_in]):
        try:
            return make_policy(nm, cost=cost, params=dict(params), prior=prior)
        except (KeyError, ValueError):
            continue
    return None


def simulate_run(ct, reqs, pol, P, prior, budget, mode="warm", v3_stand_in=None, ct_policy=None):
    """-> SimResult or None (policy v3 not available). Same call as predict_S3_4.main.
    ct = the WORLD table (charges one-time costs); ct_policy = the table the policy's CostView reads (default = ct). The real runtime (scripts/serve_trace.py) builds its
    policy from the frozen S2 table unless --cost-table is given, so the S4-P prediction default is world = v2, policy view = v1 (calibration 'perspective i' of S3)."""
    bud = budget if pol in P34.BUDGETED or pol == "v3" else None
    cp = ct_policy if ct_policy is not None else ct
    cv = CostView(cp, mode, rebuild_override=getattr(cp, "rebuild_override", None))
    if pol == "v3":
        po = make_policy_v3(cv, P, prior, v3_stand_in)
        if po is None:
            return None
    else:
        po = make_policy(P34.SIM_NAME.get(pol, pol), cost=cv, params=dict(P), prior=prior)
    return SIM.simulate(reqs, po, ct, SIM.SimOptions(cache_mode=mode, mem_budget_mb=bud, overhead_aware=True))
