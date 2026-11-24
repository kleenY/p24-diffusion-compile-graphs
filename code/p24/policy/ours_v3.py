"""Ours v3 = "cost-aware admission" (S4-P P1, 2026-11-18; pre-registration goals/S4P_uplift_line_prereg_v1.md P1; candidates and selection rules in
results/S4P/v3/params_prereg.json (first pass: payback gate) and params_prereg_rev2.json (second pass: affordability regime)).
Subclass of the S3 Ours v2 (p24/policy/ours.py is NOT modified; v2 = this class with every switch off, checked bit-for-bit in
tests/test_policy_v3_S4P.py).  The first-pass version of this file is kept in the internal tree (SHA16 5fc37b46c6080510).

Motivation (S3 mechanism, results/S3/sim_eval/ANALYSIS_3.3_opus.txt): with a pre-warmed disk cache the one-time cost of C1 is small (about 4-7 s per new shape,
~0.1 s when the component keys are already compiled) while the steady saving is large, and the online ski-rental of v2 loses about one one-time cost per
compiled shape while it "watches".  With an empty cache the one-time cost is ~10x larger and watching pays.  Whether a compile is cheap is a property of
the cost table in the CURRENT cache state (CostView cache mode + component keys this process already compiled), measured against the work already done.

Switches (all off = v2; everything else, incl. graph-layer admission / eviction / padding, is v2):
  (1) K_pay   payback gate (first pass).  n_star(s) = U_C1(s | cache mode, compiled keys) / (steady_E0 - steady_C1) = predicted requests of the shape
      needed to repay compiling it now; n_star <= K_pay -> Action(COMPILE) at the first request (memory-free tier).  0 = off.
  (2) afford_rho  affordability regime (second pass).  W(t) = predicted work done so far = sum over served requests of the cost-table steady time of the
      path that served them (E0/C1/G1/C1+G1/bucket; no measured timing).  A shape is "affordable" when U_C1(s | state) <= afford_rho * W(t): the premium of
      compiling it is at most that fraction of the work this process has already done (so a process that ends soon never pays a large premium: a short
      stream of one shape never reaches the regime; with an empty cache U is ~8x larger and the regime starts ~8x later).  In the regime
        (a) the admission uses the POSTERIOR shape frequency p_hat = (count + w x prior) / (n_served + w) (v2: the prior only) x H_eff, so a stream that
            concentrates on a few shapes (single-shape workload, drift, mismatched declaration) is recognised from the served counts;
        (b) the ski-rental fraction of the memory-free C1 tier is alpha_c1 (None = alpha).
      Outside the regime (incl. t = 0, W = 0) the policy is exactly v2.  None = off.
  (3) alpha_c1 without afford_rho (first pass): constant ski-rental fraction of the C1 tier.

The policy never reads the true N, future requests, measured timings, system files or GPU state: inputs are the injected CostView (predicted numbers),
the prior, the process state and counters of served requests, exactly as v2 (interface: code/docs/policy_interface_S3.md).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import CAPTURE, COMPILE, COMPILE_CAPTURE, EAGER, RUN_BUCKET, Action, ProcState, Shape, shape_of
from .ours import C1, C1G1, E, G1, RANK, Ours

V3_DEFAULTS = dict(
    K_pay=0.0,          # payback-gate threshold in requests of the shape (0 = off)
    alpha_c1=None,      # ski-rental fraction for the C1 tier (None = alpha, i.e. v2)
    afford_rho=None,    # affordability regime threshold (None = off): U_C1 <= afford_rho x W(t)
)


def split_v3_params(params: Optional[dict]):
    """-> (v3 dict, remaining v2 params). Unknown keys still raise in Ours.merge_params."""
    p = dict(params or {})
    v3 = {k: p.pop(k, d) for k, d in V3_DEFAULTS.items()}
    assert v3["K_pay"] >= 0.0, v3
    assert v3["alpha_c1"] is None or v3["alpha_c1"] >= 0.0, v3
    assert v3["afford_rho"] is None or v3["afford_rho"] > 0.0, v3
    return v3, p


class OursV3(Ours):
    name, arm = "Ours-v3", None

    def __init__(self, cost, params: Optional[dict] = None, prior: Optional[Dict[Shape, float]] = None, name: Optional[str] = None):
        self.v3, rest = split_v3_params(params)
        super().__init__(cost, rest, prior, name)
        if self.v3["K_pay"] > 0 or self.v3["alpha_c1"] is not None or self.v3["afford_rho"] is not None:
            assert C1 in self.ladder, "v3 additions need the C1 tier in the ladder"
        self._work = 0.0     # predicted work done so far (s): cost-table steady time of the path that served each request

    # ---------------------------------------------------------------- cost-aware quantities (predicted numbers only)
    def payback_requests(self, s: Shape, state: ProcState) -> float:
        """Predicted requests of shape s needed to repay compiling it (C1) NOW, in the current cache state; +inf if it never pays."""
        sav = self._steady(E, s) - self._steady(C1, s)
        if not sav > 0:
            return float("inf")
        u = self._U(C1, s, state)           # cost table, cache mode of the CostView, key-sharing credit of state.compiled_keys
        return float("inf") if u == float("inf") else u / sav

    def affordable(self, s: Shape, state: ProcState) -> bool:
        """Affordability regime: the premium of compiling s now (predicted one-time cost) is at most afford_rho x the work already done."""
        rho = self.v3["afford_rho"]
        return rho is not None and self._U(C1, s, state) <= rho * self._work

    # ---------------------------------------------------------------- decision
    def decide(self, request: dict, state: ProcState) -> Action:
        k = self.v3["K_pay"]
        if k > 0:
            s = shape_of(request)
            if self._tier(s, state) == E and self.payback_requests(s, state) <= k:
                return Action(COMPILE)
        return super().decide(request, state)

    def _targets(self, s: Shape, cur: str, state: ProcState) -> List[str]:
        """Ours._targets with (a) posterior admission and (b) alpha_c1 in the affordability regime (identical to v2 when the switches are off)."""
        a_c1, rho = self.v3["alpha_c1"], self.v3["afford_rho"]
        if a_c1 is None and rho is None:
            return super()._targets(s, cur, state)
        cands = self._above(cur)
        if not cands:
            return []
        aff = self.affordable(s, state)
        alpha_c1 = (a_c1 if a_c1 is not None else self.p["alpha"]) if (aff or rho is None) else self.p["alpha"]
        x = self._steady(cur, s)
        out: List[str] = []
        if self.p["prior_admit"] and s in self.prior:      # prior admission (v2); posterior frequency in the affordability regime
            pr = self._p_hat(s, state) if aff else self.prior[s]
            nh = pr * self._h_eff(state)
            scored = []
            for t in cands:
                sav = x - self._steady(t, s)
                if sav <= 0:
                    continue
                net = nh * sav - self._U(t, s, state)
                if net > 0:
                    scored.append((-net, RANK[t], t))
            out += [t for _, _, t in sorted(scored)]
        if out:
            return out
        thr = self.p["alpha"] * (2.0 ** self._fails.get(s, 0))
        sv = self._sv.get(s, {})
        for t in reversed(cands):
            if sv.get(t, 0.0) >= (alpha_c1 if t == C1 else thr) * self._U(t, s, state):
                out.append(t)
        return out

    # ---------------------------------------------------------------- bookkeeping after the event
    def on_event(self, request: dict, act: Action, event: str, state: ProcState) -> None:
        if self.v3["afford_rho"] is not None:
            self._work += self._served_steady(shape_of(request), act, event, state)
        super().on_event(request, act, event, state)

    def _served_steady(self, s: Shape, act: Action, event: str, state: ProcState) -> float:
        """Predicted steady seconds of the path that served this request (cost table only)."""
        k = act.kind
        if event == "capture_denied" or k == EAGER:
            return self._steady(E, s)
        if k == COMPILE_CAPTURE:
            return self._steady(C1G1, s)
        if k == CAPTURE:
            return self._steady(G1, s)
        if k == COMPILE:
            return self._steady(C1, s)
        if k == RUN_BUCKET:
            return self.cost.steady_s("G1" if state.graph_kind.get(act.bucket) == "G1" else "C1+G1", act.bucket)
        return self._steady(E, s)
