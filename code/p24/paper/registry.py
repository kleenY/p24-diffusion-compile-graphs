"""Number registry for the paper (S5 item 5.1, first version).

Only numbers from FROZEN S2/S3(/S4 GenEval) outputs are registered here.  Blocks A/B/C and the H200 reproduction
are NOT registered (their outputs are not frozen yet): add them AFTER their SHA16 are logged, either by appending
entries below or, without touching this file, by dropping a CSV with columns
    macro,source_path,selector,fmt,used_in,script
into  paper/paper/registry_extra/*.csv  (loaded automatically by build.py; see PENDING_BLOCKS).

Selector syntax (applied to source_path, paths relative to the project root)
  JSON : dotted path, e.g.  per_N.10000.H4a_W2.mean_rel_diff   (list index = integer token); optional  #len
  CSV  : COLUMN?col=val&col=val[#agg]   filters are exact string matches on the cell text; without #agg exactly one
         row must match; agg in {min,max,mean,median,sum,count} runs over all matched rows (COLUMN still required).
Format (fmt)
  s                       verbatim string
  [*K:|/K:]<spec>         optional scale (multiply/divide by K), then a Python format spec, e.g.  *100:.1f  /1024:.1f
                          .3f  +.4f  d (value must be integer-valued after scaling)
Macro names: letters only (TeX), prefix N, digits spelled out (H4 -> Hfour, 512 -> FiveTwelve, E0 -> EZero).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, NamedTuple


class Reg(NamedTuple):
    macro: str
    source_path: str
    selector: str
    fmt: str
    used_in: str = ""
    script: str = ""


# ---- sources (all relative to the project root) ----
H4 = "results/S3/sim_eval/h4_verdict.json"
H4_SCRIPT = "code/scripts/h4_verdict_S3.py"
MFT = "results/S3/sim_eval/figs_v2_annotated/main_figure_table.csv"
MFT_SCRIPT = "code/p24/reproduce_main_figure.py"
RV = "results/S3/real_validation/verdict.json"
ET = "results/S3/real_validation/error_table.csv"
RV_SCRIPT = "code/scripts/compare_S3_4.py"
GE = "results/S4/geneval/analysis/geneval_S4_summary.csv"
GE_SCRIPT = "code/scripts/analyze_geneval_S4.py"
HS2 = "results/S2/cost_table/hs2_r_table.csv"
CT = "results/S2/cost_table/merged/cost_table.csv"
CT_SCRIPT = "code/scripts/build_cost_table.py"
BE = "results/S2/breakeven/breakeven.csv"
BE_SCRIPT = "code/p24/sim/breakeven.py"
MB = "results/S2/memory_budget.json"
MBB = "results/S2/memory_budget_buckets.csv"
MB_SCRIPT = "code/scripts/build_memory_budget.py"

_W2 = "stream=W2_N300_seed100_B"
_W3 = "stream=W3_N300_seed100_B"
_SH = "H=512&W=512&B=1&cfg=0"  # the 512x512, batch 1, no-CFG reference shape

REGISTRY: List[Reg] = [
    # ---- S3 H4 verdict (primary: N=10^4, disk-cache cold tier, default budget, 10 eval seeds) ----
    Reg("NHfourAGapPctTenK", H4, "per_N.10000.H4a_W2.mean_rel_diff", "*100:.1f", "abstract;IV;Tab.II", H4_SCRIPT),
    Reg("NHfourABootLoPctTenK", H4, "per_N.10000.H4a_bootstrap.ci95.0", "*100:.1f", "IV;Tab.II", H4_SCRIPT),
    Reg("NHfourABootHiPctTenK", H4, "per_N.10000.H4a_bootstrap.ci95.1", "*100:.1f", "IV;Tab.II", H4_SCRIPT),
    Reg("NHfourAOursSecTenK", H4, "per_N.10000.H4a_W2.mean_ours_s", ".0f", "IV", H4_SCRIPT),
    Reg("NHfourABestSecTenK", H4, "per_N.10000.H4a_W2.mean_best_s", ".0f", "IV", H4_SCRIPT),
    Reg("NHfourABestNameTenK", H4, "per_N.10000.H4a_W2.best_online", "s", "IV;Tab.II", H4_SCRIPT),
    Reg("NHfourASeedsHigherTenK", H4, "per_N.10000.H4a_W2.seeds_ours_higher_by_2pct", "d", "IV", H4_SCRIPT),
    Reg("NHfourSeedsTotal", H4, "per_N.10000.H4a_W2.n_seeds", "d", "IV;Tab.II", H4_SCRIPT),
    Reg("NHfourBRatioTenK", H4, "per_N.10000.H4b_W3.mean_ratio", ".3f", "abstract;IV;Tab.II", H4_SCRIPT),
    Reg("NHfourCRatioTenK", H4, "per_N.10000.H4c_W1.mean_ratio", ".3f", "abstract;IV;Tab.II", H4_SCRIPT),
    # ---- S3 Tab.II base table (mean total service time relative to B0, N=10^4, disk tier) ----
    Reg("NTabWtwoOursTenK", MFT, "mean?workload=W2&cold=disk&variant=Ours&N=10000", ".3f", "Tab.II", MFT_SCRIPT),
    Reg("NTabWtwoBOneTenK", MFT, "mean?workload=W2&cold=disk&variant=B1&N=10000", ".3f", "Tab.II", MFT_SCRIPT),
    Reg("NTabWtwoBSevenTenK", MFT, "mean?workload=W2&cold=disk&variant=B7&N=10000", ".3f", "Tab.II", MFT_SCRIPT),
    Reg("NTabWoneOursTenK", MFT, "mean?workload=W1&cold=disk&variant=Ours&N=10000", ".3f", "Tab.II", MFT_SCRIPT),
    # ---- S3 real validation (3.4; the simulator-validation FAILED the 10% line: report as is) ----
    Reg("NRealJudged", RV, "n_judged", "d", "IV;Tab.II", RV_SCRIPT),
    Reg("NRealMaxErrPct", RV, "max_error", "*100:.1f", "abstract;IV;Tab.II", RV_SCRIPT),
    Reg("NRealMemViolations", RV, "mem_line_violations#len", "d", "IV", RV_SCRIPT),
    Reg("NRealErrOursWtwoPct", ET, f"error?{_W2}&policy=Ours&cache_mode=warm", "*100:.1f", "Tab.II", RV_SCRIPT),
    Reg("NRealErrOursWthreePct", ET, f"error?{_W3}&policy=Ours&cache_mode=warm", "*100:.1f", "Tab.II", RV_SCRIPT),
    Reg("NRealErrBOneWtwoPct", ET, f"error?{_W2}&policy=B1&cache_mode=warm", "*100:.1f", "IV;Tab.II", RV_SCRIPT),
    # ---- S4 GenEval (paired bootstrap on Delta-Overall vs E0; margin +-0.02 frozen) ----
    Reg("NGenEvalEZeroOverall", GE, "abs_score?group=E0&metric=Overall", ".3f", "IV;Tab.II", GE_SCRIPT),
    Reg("NGenEvalEZeroCiLo", GE, "ci_lo?group=E0&metric=Overall", ".3f", "IV", GE_SCRIPT),
    Reg("NGenEvalEZeroCiHi", GE, "ci_hi?group=E0&metric=Overall", ".3f", "IV", GE_SCRIPT),
    Reg("NGenEvalCOneGOneDelta", GE, "delta?group=C1G1&metric=Overall", "+.4f", "IV;Tab.II", GE_SCRIPT),
    Reg("NGenEvalCOneGOneCiLo", GE, "ci_lo?group=C1G1&metric=Overall", "+.4f", "IV;Tab.II", GE_SCRIPT),
    Reg("NGenEvalCOneGOneCiHi", GE, "ci_hi?group=C1G1&metric=Overall", "+.4f", "IV;Tab.II", GE_SCRIPT),
    Reg("NGenEvalPadDelta", GE, "delta?group=PAD&metric=Overall", "+.4f", "IV;Tab.II", GE_SCRIPT),
    # ---- S2 steady-state speed-up r = (e0-g1)/e0 over the 26 workload shapes (Fig.2a) ----
    Reg("NSTwoRFiveTwelvePct", HS2, f"r?{_SH}", "*100:.1f", "IV;Fig.2", "unknown"),
    Reg("NSTwoRMaxPct", HS2, "r#max", "*100:.1f", "abstract;IV;Fig.2", "unknown"),
    Reg("NSTwoShapes", HS2, "r#count", "d", "III;IV;Fig.2", "unknown"),
    # ---- S2 cost table, 512x512 B=1 no-CFG (Tab.I) ----
    Reg("NSTwoCostEZeroSteadyMs", CT, f"steady_ms?arm=E0&{_SH}", ".1f", "IV;Tab.I", CT_SCRIPT),
    Reg("NSTwoCostCOneGOneSteadyMs", CT, f"steady_ms?arm=C1+G1&{_SH}", ".1f", "IV;Tab.I", CT_SCRIPT),
    Reg("NSTwoCostCOneFirstEmptyS", CT, f"first_req_s__cold_empty?arm=C1&{_SH}", ".1f", "IV;Tab.I", CT_SCRIPT),
    Reg("NSTwoCostCOneFirstDiskS", CT, f"first_req_s__cold_diskcache?arm=C1&{_SH}", ".1f", "IV;Tab.I", CT_SCRIPT),
    # ---- S2 break-even counts, 512x512 B=1 no-CFG (Tab.I) ----
    Reg("NBreakCaptureFiveTwelve", BE, f"n_capture?{_SH}&tier=cold_diskcache", ".0f", "IV;Tab.I", BE_SCRIPT),
    Reg("NBreakCompileDiskFiveTwelve", BE, f"n_compile?{_SH}&tier=cold_diskcache", ".0f", "IV;Tab.I", BE_SCRIPT),
    Reg("NBreakCompileEmptyFiveTwelve", BE, f"n_compile?{_SH}&tier=cold_empty", ".0f", "IV;Tab.I", BE_SCRIPT),
    # ---- S2 memory budget (Fig.3); source units MiB, shown as GB via /1024 ----
    Reg("NMemCapGB", MB, "cap_mb", "/1024:.1f", "III;IV;Fig.3", MB_SCRIPT),
    Reg("NMemGraphBudgetGB", MB, "graph_budget_mb", "/1024:.1f", "III;IV;Fig.3", MB_SCRIPT),
    Reg("NPoolGOneFiveTwelveBOneMB", MBB, "pool_median?arm=G1&H=512&W=512&B=1&cfg=0", "d", "IV;Fig.3", MB_SCRIPT),
    Reg("NPoolGOneFiveTwelveBFourMB", MBB, "pool_median?arm=G1&H=512&W=512&B=4&cfg=0", "d", "IV;Fig.3", MB_SCRIPT),
]

# Interface for sources that are not frozen yet (NOT registered in the first version).
PENDING_BLOCKS = {
    "blockA": "cost-table supplement (S2 addendum): register after SHA16 is logged",
    "blockB": "forward/reverse-order memory raw (Fig.3 peak markers): register after SHA16 is logged",
    "blockC": "simulator-validation v2 error table + verdict, recipe-R serving row (Tab.II): after SHA16 is logged",
    "H200": "reproduction numbers (user submits): after SHA16 is logged",
}

EXTRA_DIR = "paper/paper/registry_extra"
EXTRA_COLUMNS = ("macro", "source_path", "selector", "fmt", "used_in", "script")


def load_extra(csv_path: Path) -> List[Reg]:
    """Read an extra registry CSV (columns macro,source_path,selector,fmt[,used_in,script])."""
    out: List[Reg] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Reg(row["macro"].strip(), row["source_path"].strip(), row["selector"].strip(),
                           row["fmt"].strip(), (row.get("used_in") or "").strip(), (row.get("script") or "").strip()))
    return out
