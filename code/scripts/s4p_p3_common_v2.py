"""S4-P P3 shared helpers, version 2 (2026-11-18): the frozen-name original scripts/s4p_p3_common.py is NOT modified.
Fix: the original `make_policy_v3` calls p24.policy.make_policy("v3") (the S3 factory, which does not know v3), so the v3 row was always omitted.
Here `make_policy_v3` builds the policy through p24.policy.factory_v3.make_policy("Ours-v3", cost, params, prior) -- the same constructor path as scripts/serve_trace_S4P.py.
Everything else is re-exported unchanged from the original module; `simulate_run` is re-defined here (identical body) only so that it binds the v2 `make_policy_v3`.
"""
from __future__ import annotations

from scripts.s4p_p3_common import *                                           # noqa: F401,F403  (re-export: ROOT, P34, SIM, load_v1/2, make_plan, stream_path, sha16, ...)
from scripts.s4p_p3_common import P34, SIM, CostView, make_policy             # noqa: F401  (names the star import would already give; explicit for clarity)
from p24.policy.factory_v3 import make_policy as make_policy_factory_v3

V3_NAME = "Ours-v3"


def make_policy_v3(cost, params, prior, stand_in=None):
    """v3 = OursV3 via factory_v3 (params: frozen flat dict, tiers list -> tuple as predict does).  --v3-stand-in <name> (dry run only) keeps the old stand-in behaviour."""
    p = dict(params)
    if "tiers" in p:
        p["tiers"] = tuple(p["tiers"])
    if stand_in:
        return make_policy_factory_v3(stand_in, cost=cost, params=p, prior=prior)
    return make_policy_factory_v3(V3_NAME, cost=cost, params=p, prior=prior)


def simulate_run(ct, reqs, pol, P, prior, budget, mode="warm", v3_stand_in=None, ct_policy=None):
    """Identical to s4p_p3_common.simulate_run except that policy v3 goes through the v2 `make_policy_v3` (factory_v3)."""
    bud = budget if pol in P34.BUDGETED or pol == "v3" else None
    cp = ct_policy if ct_policy is not None else ct
    cv = CostView(cp, mode, rebuild_override=getattr(cp, "rebuild_override", None))
    if pol == "v3":
        po = make_policy_v3(cv, P, prior, v3_stand_in)
    else:
        po = make_policy(P34.SIM_NAME.get(pol, pol), cost=cv, params=dict(P), prior=prior)
    return SIM.simulate(reqs, po, ct, SIM.SimOptions(cache_mode=mode, mem_budget_mb=bud, overhead_aware=True))
