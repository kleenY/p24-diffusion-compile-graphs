"""p24.stats S2 additions (review M4, 2026-11-15): one-time costs split by cold tier, request-index windows, transient excess.
Synthetic per-process CSV-like frames; CPU only."""
import numpy as np
import pandas as pd

from p24 import stats


def _proc(arm, tier, pid, walls, compile_s, first_req_s, warm=0, tv=0, req_idx=True):
    n = len(walls)
    d = dict(arm=arm, H=512, W=512, B=1, cfg=0, proc_id=pid, cold_tier=tier, status="ok", rep=np.arange(n), row_kind="steady",
             wall_ms=walls, gpu_ms=walls, first_req_s=first_req_s, prep_json="{}", peak_alloc_mb=1.0, contended=0,
             load1_start=1.0, load1_end=1.0, cpu_contended=0, pcore_pinned=1, recompile_limit_hit=0, tier_violation=tv,
             compile_s=compile_s)
    if req_idx:
        d["req_idx"] = np.arange(n) + warm + 1
    return pd.DataFrame(d)


def _frame():
    base = np.full(43, 40.0)
    cold_w = base.copy()
    cold_w[:11] = 100.0  # transient: first 11 recorded requests (i = 1..11) are 60 ms slower
    parts = []
    for pid in range(3):
        parts.append(_proc("C1", "cold_empty", pid, cold_w + pid, 40.0 + pid, 50.0 + pid))
        parts.append(_proc("C1", "cold_diskcache", pid, base, 2.0 + pid * 0.1, 8.0 + pid))
    return pd.concat(parts, ignore_index=True)


def test_one_time_costs_split_by_tier():
    s = stats.summarize(_frame())
    r = s.iloc[0]
    assert r["compile_s_median_cold_empty"] == 41.0
    assert abs(r["compile_s_median_cold_diskcache"] - 2.1) < 1e-9
    assert r["first_req_s_cold_empty"] == 51.0 and r["first_req_s_diskcache"] == 9.0
    assert "compile_s_median" not in s.columns  # the mixed-tier median is gone


def test_tier_violation_dropped_from_one_time_only():
    f = _frame()
    f.loc[(f.cold_tier == "cold_diskcache") & (f.proc_id == 0), "tier_violation"] = 1
    r = stats.summarize(f).iloc[0]
    assert abs(r["compile_s_median_cold_diskcache"] - 2.15) < 1e-9  # median of procs 1, 2 only
    assert r["n_tier_violation"] == 1 and r["n_proc"] == 6


def test_request_windows_per_process_median():
    r = stats.summarize(_frame(), windows="4-23,24-43").iloc[0]
    # cold_empty processes: i=4..11 are 100(+pid), i=12..23 are 40(+pid) -> 8 slow of 20 -> median of 20 = 40+pid (12 fast)
    assert r["steady_wall_ms_i4_23_cold_empty"] == 41.0
    assert r["steady_wall_ms_i24_43_cold_empty"] == 41.0
    assert r["steady_wall_ms_i4_23_cold_diskcache"] == 40.0
    assert r["steady_wall_ms_i24_43"] == 40.0  # all six processes: median of {40,40,40,40,41,42}


def test_window_index_derived_without_req_idx_column():
    # S1 / 2.0 CSVs have no req_idx: rep r means request i = r + 4 (--warm 3)
    f = pd.concat([_proc("E0", "na", p, np.arange(20) + 100.0, 0.0, 1.0, warm=3, req_idx=False) for p in range(2)],
                  ignore_index=True)
    r = stats.summarize(f, windows="4-13,14-23").iloc[0]
    assert r["steady_wall_ms_i4_13"] == 104.5 and r["steady_wall_ms_i14_23"] == 114.5


def test_transient_excess_matches_hand_value():
    te = stats.transient_excess(_frame(), n_boot=200).iloc[0]
    # each cold process: 11 requests x 60 ms excess + pid*43 ms offset; median over pid=0,1,2 is pid=1: 0.66 + 0.043
    assert abs(te["excess_s_median"] - (11 * 0.060 + 43 * 0.001)) < 1e-9
    assert te["n_cold"] == 3 and te["n_ref"] == 3 and te["i_lo"] == 1 and te["i_hi"] == 43
    assert te["ci_lo"] <= te["excess_s_median"] <= te["ci_hi"]


def test_transient_excess_needs_both_tiers_and_full_range():
    f = _frame()
    f = f[f.cold_tier == "cold_empty"]
    te = stats.transient_excess(f, n_boot=50).iloc[0]
    assert np.isnan(te["excess_s_median"]) and te["n_ref"] == 0
    f2 = _frame()
    f2 = f2[~((f2.cold_tier == "cold_empty") & (f2.proc_id == 0) & (f2.req_idx == 20))]  # missing request in proc 0
    te2 = stats.transient_excess(f2, n_boot=50).iloc[0]
    assert te2["n_cold"] == 2
