"""S4-X X3 shared simulator helpers (CPU only, no torch).  Extends -- does not modify -- the frozen predictor machinery of block C / P5d
(scripts/s4p_p3_common_v2.py, scripts/aot_tables_S4P_P5c.py, scripts/predict_S4P_P5c.py / predict_S4P_P5d.py).

New arm B1-AOT-K (preregistration goals/S4X_X3_finite_bucket_AOT_prereg_v1.md sec 3): policy B1 (compile every new shape) on the world table in which
  * the C1 rows of the shapes in K carry the AOT prices (steady + one-time cells of the AOT rows of cost_table_v2_aot.csv, exactly as B1-AOT),
  * every other shape (AOT shapes outside K, plus the equivalence-fallback list) carries the E0 prices of the SAME table (steady E0, one-time E0 at each tier) = eager service,
    the same mechanism aot_tables_S4P_P5c.swap_c1 already uses for the fallback list.
So B1-AOT-K with K = all non-fallback AOT shapes is B1-AOT, and with K = {} it is B0 (the self-test checks both bit for bit, scripts/selftest_S4X_X3.py).
Arms: B0 (policy B0 on the world table), B1-AOT (policy B1 on the AOT-swapped table, fallback shapes E0), B1-AOT-K.  The policy view is the frozen S2/S3 table (v1) as in predict_S4P_P5c/P5d.
Build charge for the "empty cache" variant (preregistration sec 4): every package of the arm's built set costs its build wall-clock b_s (build_summary.csv process_total_s,
once per shape): B1-AOT 32 packages, B1-AOT-K |K| packages, B0 none; added to the process' total service time (disk-tier simulation: the packages exist on disk).
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Set

from p24.policy import shape_of
from p24.policy.base import Shape
from scripts import aot_tables_S4P_P5c as T
from scripts import s4p_p3_common_v2 as C
from scripts import select_buckets_S4X_X3 as SEL

P34 = C.P34
ROOT = C.ROOT
X3 = ROOT / "results/S4X/x3"
K_JSON = X3 / "K.json"
ADDENDUM = SEL.ADDENDUM
ARMS = ("B0", "B1-AOT", "B1-AOT-K")


def read_k(path=K_JSON) -> List[str]:
    """Shape keys 'HxWxBxcfg' of the K set of K.json (selection order)."""
    return list(json.loads(pathlib.Path(path).read_text())["K"])


def k_shapes(keys: Iterable[str]) -> Set[Shape]:
    return {T.shape_from_key(k) for k in keys}


class World:
    """Everything the predictor loads, once: params, budget, prior, input model, v1 (policy view) and v2 (world) tables, the fallback list, the build wall-clocks b_s."""

    def __init__(self, table_csv=None, addendum_csv=None, fallback_csv=None):
        table_csv = str(table_csv or T.COST_TABLE_AOT)
        addendum_csv = str(addendum_csv or ADDENDUM)
        fallback_csv = str(fallback_csv or T.FALLBACK_CSV)
        assert T.is_ro(table_csv) and T.is_ro(fallback_csv), "cost table / fallback list must be 0444"
        assert (P34.PARAMS_PATH.stat().st_mode & 0o222) == 0, "policy_params.json must be read-only (0444)"
        self.P = P34.load_params()
        pj = json.loads(P34.PARAMS_PATH.read_text())
        assert P34.sha16(C.CT_V1) == pj["cost_table_sha16"], "frozen cost table differs from policy_params.json"
        self.budget = float(json.loads(P34.BUDGETS_PATH.read_text())["tiers"]["default"]["mb"])
        self.ct_v1 = C.load_v1()                                   # same call order as predict_S4P_P5d.predict
        self.ct_v2 = C.load_v2(table_csv, addendum_csv)
        self.prior = C.prior_A()
        self.inp = P34.InputModel(P34.INPUT_PATH)
        self.fallback = T.read_fallback(fallback_csv)
        self.aot_all = T.aot_shapes(self.ct_v2.df)                 # the 32 packaged shapes
        self.b = {T.shape_from_key(k): v for k, v in SEL.build_wall().items()}
        assert set(self.b) == self.aot_all, "build_summary shapes differ from the AOT rows of the cost table"
        self.paths = dict(table_csv=table_csv, addendum_csv=addendum_csv, fallback_csv=fallback_csv)
        self._aot_tab = T.swap_c1(self.ct_v2, T.aot_provider(self.ct_v2), self.fallback)       # B1-AOT world table (as predict_S4P_P5d)
        self._k_tabs: Dict[frozenset, object] = {}

    # ---------------------------------------------------------------- tables
    def aot_table(self):
        return self._aot_tab

    def k_table(self, K: Iterable[Shape]):
        """World table of arm B1-AOT-K: AOT prices for the shapes of K, E0 prices for every other packaged shape and for the fallback list."""
        K = frozenset(K)
        bad = K - self.aot_all
        assert not bad, f"K contains shapes without an AOT package: {sorted(s.tag() for s in bad)}"
        assert not (K & self.fallback), f"K contains fallback-list shapes: {sorted(s.tag() for s in K & self.fallback)}"
        if K not in self._k_tabs:
            outside = (self.aot_all - K) | self.fallback
            self._k_tabs[K] = T.swap_c1(self.ct_v2, T.aot_provider(self.ct_v2), outside)
        return self._k_tabs[K]

    # ---------------------------------------------------------------- simulation
    def simulate(self, arm: str, reqs: Sequence[dict], K: Optional[Iterable[Shape]] = None, mode: str = "warm"):
        """-> SimResult of one arm on `reqs` (disk-tier, 'warm' cache mode; same call as predict_S4P_P5d.predict)."""
        if arm == "B0":
            return C.simulate_run(self.ct_v2, reqs, "B0", self.P, self.prior, self.budget, mode, ct_policy=self.ct_v1)
        if arm == "B1-AOT":
            return C.simulate_run(self._aot_tab, reqs, "B1", self.P, self.prior, self.budget, mode, ct_policy=self.ct_v1)
        if arm == "B1-AOT-K":
            assert K is not None, "arm B1-AOT-K needs K"
            return C.simulate_run(self.k_table(K), reqs, "B1", self.P, self.prior, self.budget, mode, ct_policy=self.ct_v1)
        raise ValueError(arm)

    def input_s(self, reqs: Sequence[dict]) -> float:
        self.inp.n_extrap = 0
        return sum(self.inp.ms(shape_of(r)) for r in reqs) / 1e3

    def build_charge(self, arm: str, K: Optional[Iterable[Shape]] = None) -> float:
        """Sum of the build wall-clocks b_s of the packages the arm built (all 32 for B1-AOT, K for B1-AOT-K, none for B0)."""
        if arm == "B0":
            return 0.0
        if arm == "B1-AOT":
            return float(sum(self.b.values()))
        return float(sum(self.b[s] for s in K))

    def build_used(self, arm: str, reqs: Sequence[dict], K: Optional[Iterable[Shape]] = None) -> float:
        """Sensitivity: only the packages whose shape actually occurs in the trace."""
        if arm == "B0":
            return 0.0
        seen = {shape_of(r) for r in reqs}
        built = self.aot_all if arm == "B1-AOT" else set(K)
        return float(sum(self.b[s] for s in built & seen))


def run_row(world: World, arm: str, reqs: Sequence[dict], K: Optional[Iterable[Shape]] = None) -> dict:
    """One simulation -> flat row: service (excl. input), input, total (T of the preregistration: service + input), counts, build charge, cold total."""
    res = world.simulate(arm, reqs, K)
    sm = res.summary
    tot_in = world.input_s(reqs)
    n_out = n_req = 0
    if arm == "B1-AOT-K":
        Ks = set(K)
        n_out = sum(1 for r in reqs if shape_of(r) not in Ks)
    elif arm == "B1-AOT":
        n_out = sum(1 for r in reqs if shape_of(r) in world.fallback)
    n_req = len(reqs)
    b_built = world.build_charge(arm, K)
    return dict(arm=arm, n_requests=n_req, n_shapes=int(sm["n_shapes"]), service_s=float(sm["service_s"]), input_s=tot_in, total_s=float(sm["service_s"]) + tot_in,
                n_compile=int(sm.get("n_compile", 0)), n_eager_first=int(sm.get("n_eager_first", 0)), n_steady=int(sm.get("n_steady", 0)), n_eager_served_requests=n_out,
                n_input_extrapolated=world.inp.n_extrap, build_built_s=b_built, build_used_s=world.build_used(arm, reqs, K),
                n_built=(0 if arm == "B0" else (len(world.aot_all) if arm == "B1-AOT" else len(set(K)))),
                total_cold_s=float(sm["service_s"]) + tot_in + b_built, total_cold_used_s=float(sm["service_s"]) + tot_in + world.build_used(arm, reqs, K))
