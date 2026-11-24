"""S4-X ceiling accounting (CPU only, read-only on every existing result; new files only).

Q1  why compile gives only ~16% steady speed-up and CUDA Graph <=~5% on top of compile:
    kernel classification of existing Nsight sqlite files + "big ops unchanged" ceiling + idle-time ceiling.
Q2  why no selective online compile policy can beat "compile on first sight" (B1) by much on a long request stream:
    one-time-cost cap (re-derived from the frozen cost table + eval traces, checked against main_raw.csv B1) and an
    oracle (known counts) selective-compile estimate; N dependence.

Usage (from project root):  python -B code/scripts/ceiling_S4X.py            -> results/S4X/ceiling/{kernel_breakdown.csv,ceiling_accounting.csv,fig_kernel_breakdown.pdf}
No simulator loop is run (only CostTable lookups and arithmetic).
"""
from __future__ import annotations

import collections
import pathlib
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
OUT = ROOT / "results/S4X/ceiling"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------------
# 1. kernel classification (explicit, ordered rules on the lower-cased demangled name; first match wins)
# ----------------------------------------------------------------------------------------------------------------------
CATS = ["conv", "gemm", "attention", "triton_fused", "layout_convert", "norm_reduce", "elementwise_copy", "other"]
BIG = ("conv", "gemm", "attention")  # "vendor-library big ops" assumed unchanged by compile
RULES = [
    ("layout_convert", ["nchwtonhwc", "nhwctonchw"]),
    ("triton_fused", ["triton_"]),                       # torch.compile / Inductor generated (name starts with triton_)
    ("attention", ["flash", "fmha", "efficient_attention", "attention_kernel", "sdpa"]),
    ("conv", ["fprop", "dgrad", "wgrad", "implicit_gemm", "cudnn::", "convolution", "conv2d", "winograd"]),
    ("gemm", ["gemm", "cutlass", "cublas", "xmma", "gemv", "sgemm", "splitk", "wmma"]),
    ("norm_reduce", ["rowwisemoments", "layer_norm", "group_norm", "batch_norm", "layernorm", "groupnorm", "softmax",
                     "reduce_kernel", "moments"]),
    ("elementwise_copy", ["elementwise", "silu", "gelu", "copy", "catarray", "index_kernel", "fill", "vectorized",
                          "unrolled", "upsample", "transpose", "permute", "cat_"]),
]


def classify(name: str) -> str:
    n = name.lower()
    for cat, keys in RULES:
        if n.startswith("triton_") and cat != "triton_fused":
            continue
        if any(k in n for k in keys):
            return cat
    return "other"


# (label, arm, shape-tag, cfg, B, H, sqlite path relative to ROOT)
Q2D, QD, S2D = "results/S1/nsys_q2", "results/S1/nsys_q", "results/S2/cost_table/S-1/nsys"
TRACES = []
for cfg in (0, 1):
    for arm in ("E0", "C1", "C2", "G1", "G2"):
        TRACES.append((arm, 512, 1, cfg, f"{Q2D}/{arm}_b1_cfg{cfg}.sqlite"))
    TRACES.append(("C1+G2", 512, 1, cfg, f"{QD}/C1+G2_b1_cfg{cfg}.sqlite"))
TRACES += [("E0", 256, 1, 0, f"{S2D}/E0_256x256_b1_cfg0.sqlite"), ("G1", 256, 1, 0, f"{S2D}/G1_256x256_b1_cfg0.sqlite"),
           ("E0", 768, 4, 1, f"{S2D}/E0_768x768_b4_cfg1.sqlite"), ("G1", 768, 4, 1, f"{S2D}/G1_768x768_b4_cfg1.sqlite")]


def union_len(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def load_trace(path: pathlib.Path):
    """-> dict(per-request kernel records, GPU busy union, request walls). Only events inside NVTX request<i> ranges."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    reqs = con.execute("select start, end from NVTX_EVENTS where end is not null and text like 'request%' order by start").fetchall()
    assert reqs, f"no NVTX request ranges in {path}"
    lo, hi = min(s for s, _ in reqs), max(e for _, e in reqs)
    kern = con.execute("""select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                          join StringIds s on k.demangledName = s.id where k.start >= ? and k.end <= ?""", (lo, hi)).fetchall()
    other = []
    for tab in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            other += con.execute(f"select start, end from {tab} where start >= ? and end <= ?", (lo, hi)).fetchall()
        except sqlite3.Error:
            pass
    busy = sum(union_len([(s, e) for s, e in [(a, b) for a, b, _ in kern] + other if rs <= s and e <= re_])
               for rs, re_ in reqs)
    return dict(n_req=len(reqs), kern=kern, busy_ms=busy / 1e6 / len(reqs),
                wall_ms=float(np.mean([(e - s) / 1e6 for s, e in reqs])))


def kernel_breakdown():
    rows, unclassified, summ = [], [], {}
    for arm, H, B, cfg, rel in TRACES:
        p = ROOT / rel
        if not p.exists():
            print("MISSING", p)
            continue
        t = load_trace(p)
        n = t["n_req"]
        tot = sum(e - s for s, e, _ in t["kern"])
        per = collections.defaultdict(lambda: [0, 0])  # cat -> [ns, count]
        names = collections.defaultdict(lambda: [0, 0])
        for s, e, nm in t["kern"]:
            c = classify(nm)
            per[c][0] += e - s
            per[c][1] += 1
            if c == "other":
                names[nm][0] += e - s
                names[nm][1] += 1
        key = (arm, H, B, cfg)
        summ[key] = dict(total_ms=tot / 1e6 / n, busy_ms=t["busy_ms"], wall_ms=t["wall_ms"], n_req=n,
                         **{c: per[c][0] / 1e6 / n for c in CATS})
        for c in CATS:
            rows.append(dict(arm=arm, H=H, W=H, B=B, cfg=cfg, n_requests=n, category=c,
                             n_kernels_per_req=per[c][1] / n, gpu_ms_per_req=per[c][0] / 1e6 / n,
                             frac_of_kernel_time=per[c][0] / tot if tot else float("nan"),
                             kernel_time_total_ms_per_req=tot / 1e6 / n, gpu_busy_union_ms_per_req=t["busy_ms"],
                             traced_request_wall_ms=t["wall_ms"], source=rel))
        for nm, (ns, cnt) in sorted(names.items(), key=lambda x: -x[1][0])[:8]:
            unclassified.append((arm, H, B, cfg, ns / tot, cnt / n, nm[:120]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kernel_breakdown.csv", index=False)
    return df, summ, unclassified


# ----------------------------------------------------------------------------------------------------------------------
# 2. Q1 accounting
# ----------------------------------------------------------------------------------------------------------------------
CT_PATH = ROOT / "results/S2/cost_table/merged/cost_table.csv"


def steady(ct, arm, H, B, cfg):
    r = ct[(ct.arm == arm) & (ct.H == H) & (ct.W == H) & (ct.B == B) & (ct.cfg == cfg)]
    assert len(r) == 1, (arm, H, B, cfg, len(r))
    return float(r.steady_ms.iloc[0]), str(r.source.iloc[0])


def q1_accounting(summ):
    ct = pd.read_csv(CT_PATH)
    acc = []

    def add(section, shape, quantity, value, unit, source, note=""):
        acc.append(dict(section=section, workload="", N="", shape=shape, quantity=quantity, value=value, unit=unit,
                        source=source, note=note))

    cells = [(256, 1, 0), (512, 1, 0), (512, 1, 1), (768, 1, 0), (768, 1, 1), (768, 4, 1)]
    for H, B, cfg in cells:
        sh = f"{H}x{H}_b{B}_cfg{cfg}"
        src = f"{CT_PATH.relative_to(ROOT)} (steady_ms)"
        w = {a: steady(ct, a, H, B, cfg)[0] for a in ("E0", "C1", "C1+G1", "G1")}
        for a, v in w.items():
            add("Q1_measured_untraced", sh, f"steady_ms_{a}", v, "ms", src)
        add("Q1_measured_untraced", sh, "saving_compile_vs_eager_(E0-C1)/E0", (w["E0"] - w["C1"]) / w["E0"] * 100, "%", src)
        add("Q1_measured_untraced", sh, "saving_graph_on_eager_(E0-G1)/E0", (w["E0"] - w["G1"]) / w["E0"] * 100, "%", src)
        add("Q1_measured_untraced", sh, "saving_graph_on_compile_(C1-C1+G1)/C1", (w["C1"] - w["C1+G1"]) / w["C1"] * 100, "%", src)
        add("Q1_measured_untraced", sh, "saving_graph_on_compile_(C1-C1+G1)/E0", (w["C1"] - w["C1+G1"]) / w["E0"] * 100, "%", src)
        e = summ.get(("E0", H, B, cfg))
        if not e:
            continue
        nb_E = e["total_ms"] - sum(e[c] for c in BIG)
        add("Q1_ceiling_compile", sh, "E0_kernel_time_per_req", e["total_ms"], "ms", "kernel_breakdown.csv")
        add("Q1_ceiling_compile", sh, "E0_big_ops(conv+gemm+attn)_per_req", sum(e[c] for c in BIG), "ms", "kernel_breakdown.csv")
        add("Q1_ceiling_compile", sh, "E0_big_ops_fraction_of_kernel_time", sum(e[c] for c in BIG) / e["total_ms"] * 100, "%", "kernel_breakdown.csv")
        add("Q1_ceiling_compile", sh, "E0_layout_convert_fraction", e["layout_convert"] / e["total_ms"] * 100, "%", "kernel_breakdown.csv")
        add("Q1_ceiling_compile", sh, "E0_other_nonbig_fraction(norm+elementwise+other)",
            (e["norm_reduce"] + e["elementwise_copy"] + e["other"]) / e["total_ms"] * 100, "%", "kernel_breakdown.csv")
        add("Q1_ceiling_compile", sh, "ceiling_if_big_ops_unchanged_and_all_rest_vanished_(nonbig_ms/E0_wall)",
            nb_E / w["E0"] * 100, "%", "kernel_breakdown.csv + cost table",
            "hard upper bound of compile saving under the assumption; traced kernel time is not inflated by tracing (results/S2/nsight/README_nsight.txt)")
        add("Q1_ceiling_compile", sh, "implied_fraction_of_nonbig_kernel_time_removed_by_compile_=(E0-C1)/nonbig_E0",
            (w["E0"] - w["C1"]) / nb_E * 100, "%", "cost table + kernel_breakdown.csv",
            "ASSUMES big ops unchanged (verified at 512 within +-2%; no C1 trace at 256/768 so unverified there)")
        c = summ.get(("C1", H, B, cfg))
        if c:
            dtot = e["total_ms"] - c["total_ms"]
            add("Q1_ceiling_compile", sh, "C1_kernel_time_per_req", c["total_ms"], "ms", "kernel_breakdown.csv")
            add("Q1_ceiling_compile", sh, "kernel_time_reduction_E0_to_C1", dtot, "ms", "kernel_breakdown.csv")
            add("Q1_ceiling_compile", sh, "kernel_time_reduction_E0_to_C1_pct_of_E0", dtot / e["total_ms"] * 100, "%", "kernel_breakdown.csv")
            for cat in CATS:
                add("Q1_decomposition_E0_to_C1", sh, f"delta_{cat}_(E0-C1)", e[cat] - c[cat], "ms", "kernel_breakdown.csv",
                    f"share of total reduction {((e[cat] - c[cat]) / dtot * 100):.1f}%")
            bigE, bigC = sum(e[x] for x in BIG), sum(c[x] for x in BIG)
            add("Q1_decomposition_E0_to_C1", sh, "big_ops_E0", bigE, "ms", "kernel_breakdown.csv")
            add("Q1_decomposition_E0_to_C1", sh, "big_ops_C1", bigC, "ms", "kernel_breakdown.csv")
            add("Q1_decomposition_E0_to_C1", sh, "big_ops_change_pct", (bigC - bigE) / bigE * 100, "%", "kernel_breakdown.csv")
            add("Q1_decomposition_E0_to_C1", sh, "big_ops_share_of_C1_kernel_time", bigC / c["total_ms"] * 100, "%", "kernel_breakdown.csv")
            add("Q1_decomposition_E0_to_C1", sh, "compile_nonbig_remaining_C1_ms_(layout+norm+elem+triton+other)",
                c["total_ms"] - bigC, "ms", "kernel_breakdown.csv")
        # idle ceiling (derived idle = untraced wall - traced GPU busy union; plain traced idle is inflated by tracing)
        for arm in ("E0", "C1"):
            t = summ.get((arm, H, B, cfg))
            if not t:
                continue
            wall = w[arm]
            add("Q1_ceiling_graph", sh, f"{arm}_derived_idle_ms_(untraced_wall-traced_busy)", wall - t["busy_ms"], "ms", "cost table + trace busy",
                "mixes two runs (untraced steady median vs one Nsight window); can be slightly off if GPU clocks differ")
            add("Q1_ceiling_graph", sh, f"{arm}_derived_idle_pct_of_wall", (wall - t["busy_ms"]) / wall * 100, "%", "cost table + trace busy")
            add("Q1_ceiling_graph", sh, f"{arm}_traced_idle_pct_(inflated_by_tracing)", (1 - t["busy_ms"] / t["wall_ms"]) * 100, "%", "trace only",
                "S1 summary sec. 5 numbers (E0 9.8/101 ms)")
        add("Q1_ceiling_graph", sh, "measured_graph_on_eager_saving_pct_of_E0_(for_comparison)", (w["E0"] - w["G1"]) / w["E0"] * 100, "%", src)
    return acc


# ----------------------------------------------------------------------------------------------------------------------
# 3. Q2 accounting
# ----------------------------------------------------------------------------------------------------------------------
def q2_accounting():
    from p24.policy.base import component_keys, shape_of
    from p24.sim import addendum as ADD
    from p24.sim.simulator import read_trace
    ct = ADD.load_with_addendum(pd.read_csv(CT_PATH), ADD.read_addendum(ROOT / "results/S3/cost_addendum/cost_addendum.csv"))
    raw = pd.read_csv(ROOT / "results/S3/sim_eval/main_raw.csv")
    raw = raw[(raw.cold == "disk") & (raw.status == "ok")]
    acc = []

    def add(workload, N, quantity, value, unit, source, note=""):
        acc.append(dict(section="Q2", workload=workload, N=N, shape="", quantity=quantity, value=value, unit=unit, source=source, note=note))

    _ot = {}
    _orig = ct.one_time

    def _cached(arm, shape, tier, need_components=None):
        k = (arm, shape, tier)
        if k not in _ot:
            _ot[k] = _orig(arm, shape, tier, need_components)
        return _ot[k]
    ct.one_time = _cached

    def b1_parts(sh):
        """B1 = compile on first sight (C1): steady of C1 + one-time (resid + first-call of component keys not yet compiled);
        first shape of the process at tier cold_diskcache, later ones inproc_diskcache (same as p24.sim.simulator.step)."""
        seen, keys_done, one, st = set(), set(), 0.0, 0.0
        per_shape = {}
        for i, s in enumerate(sh):
            st += ct.steady_s("C1", s)
            if s not in seen:
                seen.add(s)
                ot = ct.one_time("C1", s, "cold_diskcache" if i == 0 else "inproc_diskcache")
                d = ot.resid_s
                for c, k in component_keys(s).items():
                    if k not in keys_done:
                        keys_done.add(k)
                        d += ot.fc[c]
                per_shape[s] = d
                one += d
        return st, one, per_shape

    def total_selective(counts, order, comp_set, first_shape):
        """Total time when shapes in comp_set are compiled (C1) and the rest served eagerly (E0), known counts."""
        tot, keys_done = 0.0, set()
        for s in order:                        # order of first appearance (compile-key sharing is charged to the first compiled shape)
            n = counts[s]
            tier = "cold_diskcache" if s == first_shape else "inproc_diskcache"
            if s in comp_set:
                ot = ct.one_time("C1", s, tier)
                tot += n * ct.steady_s("C1", s) + ot.resid_s
                for c, k in component_keys(s).items():
                    if k not in keys_done:
                        keys_done.add(k)
                        tot += ot.fc[c]
            else:
                tot += n * ct.steady_s("E0", s) + ct.one_time("E0", s, tier).total_s
        return tot

    def oracle(counts, order, first_shape):
        full = set(order)
        best_set, best = set(full), total_selective(counts, order, full, first_shape)
        for start in (set(full), set()):       # coordinate descent from all-compile and from all-eager
            cur = set(start)
            cur_t = total_selective(counts, order, cur, first_shape)
            improved = True
            while improved:
                improved = False
                for s in order:
                    cand = cur ^ {s}
                    t = total_selective(counts, order, cand, first_shape)
                    if t < cur_t - 1e-12:
                        cur, cur_t, improved = cand, t, True
            if cur_t < best:
                best_set, best = cur, cur_t
        return best, best_set

    summary_rows = []
    for wl in ("W1", "W2", "W3", "W4", "W5"):
        for N in (100, 1000, 10000, 100000):
            if wl != "W2" and N not in (1000, 10000):
                continue
            rows = []
            for seed in range(100, 110):
                p = ROOT / f"results/S3/workload/eval_traces/{wl}_N{N}_seed{seed}_B.jsonl"
                sh = [shape_of(r) for r in read_trace(p)]
                counts = collections.Counter(sh)
                order = list(dict.fromkeys(sh))
                st, one, per_shape = b1_parts(sh)
                x = raw[(raw.workload == wl) & (raw.N == N) & (raw.seed == seed)].set_index("variant").service_s
                b1_sim, b0_sim, ours = float(x["B1"]), float(x["B0"]), float(x["Ours"])
                recon_err = (st + one) - b1_sim
                orc, oset = oracle(counts, order, sh[0])
                rows.append(dict(seed=seed, b1_sim=b1_sim, b0_sim=b0_sim, ours=ours, steady=st, onetime=one, recon_err=recon_err,
                                 oracle=orc, n_shapes=len(order), n_eager_in_oracle=len(order) - len(oset),
                                 n_unprofitable_alone=sum(1 for s in order if counts[s] * (ct.steady_s("E0", s) - ct.steady_s("C1", s))
                                                          + ct.one_time("E0", s, "inproc_diskcache").total_s < per_shape[s])))
            d = pd.DataFrame(rows)
            m = d.mean()
            src = "results/S3/sim_eval/main_raw.csv (cold=disk) + eval_traces + cost table/addendum"
            add(wl, N, "B1_total_mean", m.b1_sim, "s", src, "main_raw service_s, mean of 10 seeds")
            add(wl, N, "B0_total_mean", m.b0_sim, "s", src)
            add(wl, N, "Ours_total_mean", m.ours, "s", src)
            add(wl, N, "B1_steady_part_(reconstructed)", m.steady, "s", src)
            add(wl, N, "B1_onetime_part_(reconstructed)", m.onetime, "s", src)
            add(wl, N, "reconstruction_error_max_abs_(steady+onetime-B1_sim)", float(d.recon_err.abs().max()), "s", src,
                "must be ~0: check that the arithmetic reproduces the simulator's B1")
            add(wl, N, "CAP_max_saving_vs_B1_=onetime/B1_total", m.onetime / m.b1_sim * 100, "%", src,
                "any policy that never does worse than eager on a shape and has compiled-steady<=eager-steady cannot save more than the B1 one-time cost")
            add(wl, N, "B1_vs_B0_ratio", m.b1_sim / m.b0_sim, "ratio", src)
            add(wl, N, "Ours_vs_B1_ratio", m.ours / m.b1_sim, "ratio", src)
            add(wl, N, "ORACLE_selective_compile_total_(known_counts,coord-descent)", m.oracle, "s", src,
                "compile-or-eager per shape, known counts, no graphs/memory limit; local optimum so it is a lower bound of the true oracle saving, approximate tier choice")
            add(wl, N, "ORACLE_saving_vs_B1", (m.b1_sim - m.oracle) / m.b1_sim * 100, "%", src)
            add(wl, N, "oracle_vs_B0_ratio", m.oracle / m.b0_sim, "ratio", src)
            best = min(m.b0_sim, m.b1_sim)
            add(wl, N, "ORACLE_saving_vs_best_of(B0,B1)", (best - m.oracle) / best * 100, "%", src,
                "at N<=1e3 the oracle simply never compiles (= B0), so there is nothing to gain over the better fixed baseline")
            add(wl, N, "Ours_vs_best_of(B0,B1)_ratio", m.ours / best, "ratio", src)
            if wl == "W2":
                add(wl, N, "B1_vs_B0_break_even_N_estimate_=onetime/((B0-B1steady)/N)", m.onetime / ((m.b0_sim - m.steady) / N), "requests", src,
                    "rough: assumes both totals scale linearly with N and the same shape mix")
            add(wl, N, "mean_n_shapes", m.n_shapes, "count", src)
            add(wl, N, "mean_shapes_served_eagerly_by_oracle", m.n_eager_in_oracle, "count", src)
            summary_rows.append((wl, N, m.b1_sim, m.onetime, m.oracle, m.b0_sim, m.ours, float(d.recon_err.abs().max())))
    # per-shape one-time cost vs offline AOT (numbers from existing reports, not recomputed)
    sh = [shape_of(r) for r in read_trace(ROOT / "results/S3/workload/eval_traces/W2_N10000_seed100_B.jsonl")]
    order = list(dict.fromkeys(sh))
    ots = [ct.one_time("C1", s, "inproc_diskcache").total_s for s in order]
    add("W2", 10000, "mean_C1_onetime_per_new_shape_inproc_diskcache_(no_key_sharing,32_shapes_seed100)", float(np.mean(ots)), "s",
        "cost table + addendum", "reports_internal/S3_summary.md line 79 quotes ~4.4 s for a single C1 shape (disk cache)")
    add("W2", 10000, "sum_C1_onetime_without_key_sharing_seed100", float(np.sum(ots)), "s", "cost table + addendum", "reports_internal/S3_summary.md line 78: unshared 215.2 s")
    add("W2", 10000, "AOT_first_request_median_(quoted)", 0.6, "s", "reports_internal/S4X_summary_DRAFT_v1_20261120.md line 5 (D3 0.57-0.60 s)",
        "quoted, not recomputed; includes the steady part of the first request (~0.1-0.2 s)")
    add("W2", 10000, "AOT_one_time_32_shapes_assumed_0.5s_each_(ASSUMPTION)", 32 * 0.5, "s", "assumption: 0.6 s first request minus ~0.1 s steady",
        "assumption, ~1 % of B1 total at N=1e4")
    st1e4 = float(np.mean([r["value"] for r in acc if r["workload"] == "W2" and r["N"] == 10000 and r["quantity"] == "B1_steady_part_(reconstructed)"]))
    add("W2", 10000, "B1-AOT_total_(steady+16s_assumed_one_time)", st1e4 + 16.0, "s", "reconstructed steady + assumption", "")
    add("W2", 10000, "CAP_max_saving_vs_B1-AOT_=16/(steady+16)", 16.0 / (st1e4 + 16.0) * 100, "%", "reconstructed steady + assumption",
        "after offline AOT the remaining one-time cost that a selective policy could still save is ~1 % (assumption-based)")
    return acc


# ----------------------------------------------------------------------------------------------------------------------
# 4. figure
# ----------------------------------------------------------------------------------------------------------------------
def figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    order = [(("E0", 256, 1, 0), "256$^2$ B1 E0"), (("G1", 256, 1, 0), "256$^2$ B1 G1"),
             (("E0", 512, 1, 0), "512$^2$ B1 E0"), (("C1", 512, 1, 0), "512$^2$ B1 C1"), (("G1", 512, 1, 0), "512$^2$ B1 G1"),
             (("E0", 512, 1, 1), "512$^2$ B1 CFG E0"), (("C1", 512, 1, 1), "512$^2$ B1 CFG C1"), (("G1", 512, 1, 1), "512$^2$ B1 CFG G1"),
             (("E0", 768, 4, 1), "768$^2$ B4 CFG E0"), (("G1", 768, 4, 1), "768$^2$ B4 CFG G1")]
    order = [(k, l) for k, l in order if k in summ]
    cols = {"conv": "#1f77b4", "gemm": "#2ca02c", "attention": "#17becf", "triton_fused": "#d62728",
            "layout_convert": "#ff7f0e", "norm_reduce": "#9467bd", "elementwise_copy": "#8c564b", "other": "#7f7f7f"}
    lab = {"conv": "conv (cuDNN)", "gemm": "GEMM", "attention": "attention", "triton_fused": "Triton fused (Inductor)",
           "layout_convert": "NCHW/NHWC convert", "norm_reduce": "norm / reduce", "elementwise_copy": "elementwise / copy", "other": "other"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, mode in zip(axs, ("abs", "share")):
        left = np.zeros(len(order))
        for c in CATS:
            v = np.array([summ[k][c] for k, _ in order])
            tot = np.array([summ[k]["total_ms"] for k, _ in order])
            vv = v if mode == "abs" else v / tot * 100
            ax.barh(y, vv, left=left, color=cols[c], label=lab[c], edgecolor="white", linewidth=0.3)
            left += vv
        ax.set_yticks(y)
        ax.set_yticklabels([l for _, l in order], fontsize=8)
        ax.set_xlabel("GPU kernel time per request (ms)" if mode == "abs" else "share of kernel time (%)")
        ax.grid(axis="x", alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=8, frameon=False)
    fig.suptitle("Kernel-time breakdown per request (Nsight, 3 steady requests; E0 eager / C1 compile / G1 CUDA Graph)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUT / "fig_kernel_breakdown.pdf")
    plt.close(fig)


if __name__ == "__main__":
    kdf, summ, uncl = kernel_breakdown()
    acc = q1_accounting(summ)
    acc += q2_accounting()
    pd.DataFrame(acc).to_csv(OUT / "ceiling_accounting.csv", index=False)
    figure(summ)
    # console digest
    print("== unclassified (category 'other') kernels, top per trace ==")
    for arm, H, B, cfg, fr, cnt, nm in uncl:
        if fr > 0.005:
            print(f"{arm:6s} {H}^2 B{B} cfg{cfg}  {fr * 100:5.2f}% {cnt:6.1f}/req  {nm}")
    print("== other fraction per trace ==")
    for k, v in summ.items():
        print(k, f"total {v['total_ms']:.1f} ms busy {v['busy_ms']:.1f} wall {v['wall_ms']:.1f} other {v['other'] / v['total_ms'] * 100:.2f}%")
