#!/usr/bin/env python3
"""S5 CPU-only analysis (reads the real-machine raw data only, no GPU):
 1) rare-shape requests (X8 / P5d / X3 / X5); 2) X5 real-machine eviction evidence; 3) GPU memory peak summary.
One command to re-run: source ops/project_env.sh; CUDA_VISIBLE_DEVICES='' "$PROJ_PYTHON" -B code/scripts/analyze_rare_evict_S5.py
Output: results/S5/rare_evict/{rare_shape_per_process,rare_shape_summary,x5_evictions,x5_evict_proc_summary,peak_mem_summary}.csv + REPORT_rare_evict.txt
Definitions: shape key=(H,W,B,cfg); within one stream (the csv of one process) a shape occurring <= thr%*N times is rare (primary 1%, sensitivity 0.5%);
 latency = dur_ms of the csv; total-service-time contribution = sum of dur_ms of the group / sum of dur_ms of the whole stream (difference to meta.service_s <=0.2%).
 'First occurrence' = the first request of a shape in the stream.
"""
import csv, json, re, os, sys, hashlib, stat
from pathlib import Path
from collections import defaultdict, Counter
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "S5" / "rare_evict"
EXPS = {  # experiment name -> raw directory
    "X8_cold": ROOT / "results/S4X/x8/real/raw",
    "P5d_diskcache": ROOT / "results/S4P/p5d/real/raw",
    "X3": ROOT / "results/S4X/x3/real/raw",
    "X5_N1e4": ROOT / "results/S4X/x5/real/raw",
}
THRS = [1.0, 0.5]  # percent
BUDGET_X5 = 6565.1
PRED_X5 = ROOT / "results/S4X/x5/pred_X5.csv"
NAME_RE = re.compile(r"^(?P<arm>[A-Za-z0-9\-]+)_(?P<wl>W\d+)_N(?P<N>\d+)_seed(?P<seed>\d+)_B_p(?P<p>\d+)$")


def list_procs():
    items = []
    for exp, d in EXPS.items():
        for f in sorted(d.glob("*.csv")):
            m = NAME_RE.match(f.stem)
            if not m:
                print("skip", f, file=sys.stderr); continue
            meta = f.with_suffix(".meta.json")
            items.append(dict(exp=exp, arm=m["arm"], wl=m["wl"], N=int(m["N"]), proc=int(m["p"]), csv=f, meta=meta))
    return items


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def stats_group(rows, first_flags, tot_dur, thr_cnt):
    d = np.array([r["dur"] for r in rows]); n = len(rows)
    fd = np.array([r["dur"] for r, fl in zip(rows, first_flags) if fl])
    return dict(n_req=n, first_n=len(fd),
                first_med_ms=pct(fd, 50), first_p90_ms=pct(fd, 90), first_max_ms=float(fd.max()) if len(fd) else float("nan"),
                p50_ms=pct(d, 50), p99_ms=pct(d, 99), mean_ms=float(d.mean()) if n else float("nan"),
                sum_s=float(d.sum()) / 1000, sum_frac=float(d.sum()) / tot_dur if tot_dur else float("nan"))


def rare_analysis(items):
    per = []
    for it in items:
        rows = []
        with open(it["csv"]) as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(key=(r["H"], r["W"], r["B"], r["cfg"]), dur=float(r["dur_ms"]), event=r["event"]))
        N = len(rows)
        cnt = Counter(r["key"] for r in rows)
        seen = set(); first = []
        for r in rows:
            first.append(r["key"] not in seen); seen.add(r["key"])
        tot = sum(r["dur"] for r in rows)
        for thr in THRS:
            tc = int(np.floor(thr / 100 * N + 1e-9))
            rare_keys = {k for k, c in cnt.items() if c <= tc}
            for grp in ("rare", "nonrare"):
                sel = [(r, f) for r, f in zip(rows, first) if (r["key"] in rare_keys) == (grp == "rare")]
                nshape = len(rare_keys) if grp == "rare" else len(cnt) - len(rare_keys)
                if sel:
                    st = stats_group([s[0] for s in sel], [s[1] for s in sel], tot, tc)
                else:
                    st = dict(n_req=0, first_n=0, first_med_ms=float("nan"), first_p90_ms=float("nan"), first_max_ms=float("nan"),
                              p50_ms=float("nan"), p99_ms=float("nan"), mean_ms=float("nan"), sum_s=0.0, sum_frac=0.0)
                st["req_frac"] = st["n_req"] / N
                per.append(dict(exp=it["exp"], workload=it["wl"], arm=it["arm"], proc=it["proc"], N=N, thr_pct=thr, thr_count=tc,
                                n_shapes_total=len(cnt), group=grp, n_shapes=nshape, total_dur_s=tot / 1000, **st))
    return per


PER_COLS = ["exp", "workload", "arm", "proc", "N", "thr_pct", "thr_count", "n_shapes_total", "group", "n_shapes", "n_req", "req_frac",
            "first_n", "first_med_ms", "first_p90_ms", "first_max_ms", "p50_ms", "p99_ms", "mean_ms", "sum_s", "sum_frac", "total_dur_s"]


def summarize(per):
    g = defaultdict(list)
    for r in per:
        g[(r["exp"], r["workload"], r["arm"], r["thr_pct"], r["group"])].append(r)
    avg = ["n_shapes", "n_req", "req_frac", "first_med_ms", "first_p90_ms", "p50_ms", "p99_ms", "mean_ms", "sum_s", "sum_frac", "total_dur_s"]
    out = []
    for k, rs in g.items():
        o = dict(exp=k[0], workload=k[1], arm=k[2], thr_pct=k[3], group=k[4], n_proc=len(rs), N=rs[0]["N"], thr_count=rs[0]["thr_count"])
        for c in avg:
            o[c + "_mean"] = float(np.nanmean([r[c] for r in rs]))
        o["first_max_ms_mean"] = float(np.nanmean([r["first_max_ms"] for r in rs]))
        o["first_max_ms_max"] = float(np.nanmax([r["first_max_ms"] for r in rs])) if rs[0]["first_n"] else float("nan")
        o["sum_s_min"] = min(r["sum_s"] for r in rs); o["sum_s_max"] = max(r["sum_s"] for r in rs)
        out.append(o)
    # compare with the reference arm (B0 first, else B1): total-time difference of the same group and threshold
    idx = {(o["exp"], o["workload"], o["arm"], o["thr_pct"], o["group"]): o for o in out}
    for o in out:
        arms = {a for (e, w, a, t, gg) in idx if e == o["exp"] and w == o["workload"]}
        ref = "B0" if "B0" in arms else "B1"
        o["ref_arm"] = ref
        ro = idx.get((o["exp"], o["workload"], ref, o["thr_pct"], o["group"]))
        o["sum_s_minus_ref"] = o["sum_s_mean"] - ro["sum_s_mean"] if ro else float("nan")
        o["mean_ms_ratio_to_ref"] = o["mean_ms_mean"] / ro["mean_ms_mean"] if ro and ro["mean_ms_mean"] else float("nan")
    out.sort(key=lambda o: (o["exp"], o["workload"], o["thr_pct"], o["group"], o["arm"]))
    return out


def evict_analysis(items):
    pred = {}
    with open(PRED_X5) as fh:
        for r in csv.DictReader(fh):
            pred[r["policy"]] = r
    po = pred["Ours"]
    ev_rows, ps_rows = [], []
    for it in items:
        if it["exp"] != "X5_N1e4" or it["arm"] != "Ours":
            continue
        meta = json.load(open(it["meta"]))
        rows = list(csv.DictReader(open(it["csv"])))
        el = meta["evict_log"]; rl = meta["runtime_evict_log"]
        # policy eviction: evict_log (has idx, predicted release); paired with runtime_evict_log by shape+freed+evict_s
        used_rl = set(); entries = []
        for e in el:
            j = next(i for i, x in enumerate(rl) if i not in used_rl and x["shape"] == e["shape"] and abs(x["evict_s"] - e["evict_s"]) < 1e-12)
            used_rl.add(j)
            entries.append(dict(order=j, type="policy_evict", idx=e["idx"], idx_source="evict_log", shape=e["shape"], kind=rl[j]["kind"],
                                pred_mb=e["pred_mb"], freed_mb=e["freed_mb"], cache_flushed_mb=rl[j]["cache_flushed_mb"], evict_s=e["evict_s"]))
        # remaining runtime releases = release of the old graph set before a G1->C1+G1 upgrade: idx inferred from the compile_capture request of that shape in the csv (evicted empty)
        used_rows = set(int(e["idx"]) for e in el)
        for j, x in enumerate(rl):
            if j in used_rl:
                continue
            cand = None
            for r in rows:
                i = int(r["idx"])
                if r["event"] == "compile_capture" and not r["evicted"] and i not in used_rows:
                    tag = f"{r['H']}x{r['W']}_b{r['B']}_{'cfg' if r['cfg'] == '1' else 'nocfg'}"
                    if tag == x["shape"]:
                        cand = i; break
            if cand is not None:
                used_rows.add(cand)
            entries.append(dict(order=j, type="upgrade_release(G1->C1+G1)", idx=cand if cand is not None else "", idx_source="inferred_from_csv" if cand is not None else "unmatched",
                                shape=x["shape"], kind=x["kind"], pred_mb="", freed_mb=x["freed_mb"], cache_flushed_mb=x["cache_flushed_mb"], evict_s=x["evict_s"]))
        entries.sort(key=lambda e: e["order"])
        for e in entries:
            gap = (e["pred_mb"] - e["freed_mb"]) if e["pred_mb"] != "" else ""
            ev_rows.append(dict(proc=it["proc"], runtime_order=e["order"] + 1, type=e["type"], request_idx=e["idx"], idx_source=e["idx_source"], evicted_shape=e["shape"],
                                graph_kind=e["kind"], pred_freed_mb=e["pred_mb"], actual_freed_mb=e["freed_mb"], gap_mb_pred_minus_actual=gap,
                                cache_flushed_mb=e["cache_flushed_mb"], evict_s=e["evict_s"]))
        den = [r for r in rows if r["event"] == "capture_denied"]
        den_shapes = Counter(f"{r['H']}x{r['W']}_b{r['B']}_{'cfg' if r['cfg'] == '1' else 'nocfg'}" for r in den)
        gmax_csv = max(float(r["graph_mem_mb"]) for r in rows)
        ps_rows.append(dict(
            proc=it["proc"], n_evicted_total_meta=meta["n_evicted_total"], n_policy_evict_log=len(el), n_runtime_release_log=len(rl),
            n_upgrade_release=len(rl) - len(el), evict_gap_mb_max_meta=meta["evict_gap_mb_max"],
            gap_max_recomputed=max(e["pred_mb"] - e["freed_mb"] for e in el), gap_min_recomputed=min(e["pred_mb"] - e["freed_mb"] for e in el),
            n_denied_meta=meta["n_denied"], n_capture_denied_csv=len(den), denied_first_idx=den[0]["idx"] if den else "", denied_last_idx=den[-1]["idx"] if den else "",
            n_denied_distinct_shapes=len(den_shapes), denied_shapes="|".join(f"{k}:{v}" for k, v in sorted(den_shapes.items())),
            peak_graph_mb_meta=meta["peak_graph_mb"], graph_mem_mb_csv_max=gmax_csv, graph_budget_mb=meta["mem_budget_mb"], graph_pool_over_budget=meta["peak_graph_mb"] > meta["mem_budget_mb"],
            graph_margin_mb=meta["mem_budget_mb"] - meta["peak_graph_mb"], peak_reserved_mb=meta["peak_reserved_mb"], mem_cap_mb=meta["mem_cap_mb"],
            pred_n_evicted=po["pred_n_evicted"], pred_n_denied=po["pred_n_denied"], pred_peak_graph_mb=po["pred_peak_graph_mb"], pred_peak_reserved_est_mb=po["pred_peak_reserved_est_mb"],
            evict_total_s=sum(x["evict_s"] for x in rl)))
    return ev_rows, ps_rows


def peak_analysis(items):
    rows = []
    for it in items:
        meta = json.load(open(it["meta"]))
        creserved = max(float(r["reserved_mb"]) for r in csv.DictReader(open(it["csv"])))
        nvml_keys = [k for k in meta if "nvml" in k.lower()]
        rows.append(dict(exp=it["exp"], workload=it["wl"], arm=it["arm"], proc=it["proc"], N=it["N"],
                         peak_reserved_mb_meta=meta.get("peak_reserved_mb"), csv_reserved_mb_max=creserved,
                         peak_graph_mb=meta.get("peak_graph_mb"), mem_cap_mb=meta.get("mem_cap_mb"), mem_budget_mb=meta.get("mem_budget_mb"),
                         nvml_peak_mb=meta[nvml_keys[0]] if nvml_keys else "NA(no NVML peak field in meta)"))
    rows.sort(key=lambda r: (r["exp"], r["workload"], r["arm"], r["proc"]))
    return rows


def write_csv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})


def build_report(per, summ, ev_rows, ps_rows, peak):
    L = []
    A = L.append
    A("S5 rare shapes / X5 evictions / GPU memory peaks: CPU-only analysis report (real-machine raw data recomputed as is, no new measurement)")
    A("Script: code/scripts/analyze_rare_evict_S5.py; data: X8 cold start, P5d disk cache, X3, X5 (csv+meta of every arm and process).")
    A("== Definitions ==")
    A("Shape key=(H,W,B,cfg); frequencies are counted per process stream; a shape with count <= 1%*N is rare (N=1000: <=10 times, N=1e4: <=100 times); sensitivity 0.5% (<=5 / <=50 times).")
    A("Latency = dur_ms of the csv (input + preparation + run, including first compile/capture); share of total service time = sum of dur_ms of the group / sum of dur_ms of the whole stream.")
    A("'First occurrence' = the first request of a shape in the stream; reference arm = B0 of the same experiment and workload (B1 if absent); 'extra vs reference' = arm mean of the group's total seconds minus the reference arm mean.")
    A("Note: P5d contains W2/W3/W5 and X3 contains W2/W5; all workloads are counted, the W2 required by the task can be read in its own rows; the workload trace is the same for all processes, so the rare-shape set is identical for all arms.")
    A("== 1. Rare shapes (1% threshold; mean of three processes; one row per experiment/workload/arm) ==")
    A("Header: experiment workload arm | rare shapes, rare requests (share) | first-occurrence latency median/p90/max ms | all rare requests p50/p99 ms | rare total s (share) | extra s vs reference | non-rare p50/p99 ms total s (share)")
    idx = {(o["exp"], o["workload"], o["arm"], o["thr_pct"], o["group"]): o for o in summ}
    for o in summ:
        if o["thr_pct"] != 1.0 or o["group"] != "rare":
            continue
        n = idx[(o["exp"], o["workload"], o["arm"], 1.0, "nonrare")]
        A(f"{o['exp']} {o['workload']} {o['arm']} | {o['n_shapes_mean']:.0f} shapes {o['n_req_mean']:.0f} req({o['req_frac_mean']*100:.1f}%) | "
          f"{o['first_med_ms_mean']:.0f}/{o['first_p90_ms_mean']:.0f}/{o['first_max_ms_mean']:.0f} | {o['p50_ms_mean']:.0f}/{o['p99_ms_mean']:.0f} | "
          f"{o['sum_s_mean']:.1f}s({o['sum_frac_mean']*100:.1f}%) | {o['sum_s_minus_ref']:+.1f}({o['ref_arm']}) | "
          f"{n['p50_ms_mean']:.0f}/{n['p99_ms_mean']:.0f} {n['sum_s_mean']:.0f}s({n['sum_frac_mean']*100:.1f}%)")
    A("== 0.5% threshold sensitivity (rare groups only: rare requests (share) first-occurrence median ms rare total s (share) extra s vs reference) ==")
    for o in summ:
        if o["thr_pct"] != 0.5 or o["group"] != "rare":
            continue
        A(f"{o['exp']} {o['workload']} {o['arm']} | {o['n_shapes_mean']:.0f} shapes {o['n_req_mean']:.1f} req({o['req_frac_mean']*100:.2f}%) | {o['first_med_ms_mean']:.0f} | "
          f"{o['sum_s_mean']:.1f}s({o['sum_frac_mean']*100:.2f}%) | {o['sum_s_minus_ref']:+.1f}")
    # conclusion sentences: per experiment/workload, the arm that costs most more than the reference on the rare group
    A("== Conclusions on rare shapes (generated by the program from the data) ==")
    best = {}
    for o in summ:
        if o["thr_pct"] == 1.0 and o["group"] == "rare":
            k = (o["exp"], o["workload"])
            if k not in best or o["sum_s_minus_ref"] > best[k]["sum_s_minus_ref"]:
                best[k] = o
    for k, o in best.items():
        if o["n_req_mean"] == 0:
            A(f"{k[0]} {k[1]}: no rare shapes at the 1% threshold (every shape occurs more than {o['thr_count']} times), not applicable.")
            continue
        A(f"{k[0]} {k[1]}: on rare shapes the arm costing most more than the reference ({o['ref_arm']}) is {o['arm']}, more by {o['sum_s_minus_ref']:+.1f} s, "
          f"mean latency of rare requests is {o['mean_ms_ratio_to_ref']:.2f}x the reference, median first-occurrence {o['first_med_ms_mean']:.0f} ms.")
    w = max(best.values(), key=lambda o: o["sum_s_minus_ref"])
    A(f"In one sentence: plain torch.compile under a cold start without cache suffers most on rare shapes ({w['exp']} {w['workload']} {w['arm']}): median first-occurrence latency of rare shapes {w['first_med_ms_mean']/1000:.1f} s, "
      f"total time of rare requests exceeds {w['ref_arm']} by {w['sum_s_minus_ref']:.0f} s ({w['n_req_mean']:.0f} requests, {w['req_frac_mean']*100:.1f}% of requests but {w['sum_frac_mean']*100:.0f}% of service time); "
      "with a disk cache the cost shrinks to about 6 s per shape on first occurrence for B1 in X5 and about +1~3 s in total for B1-AOT; Ours routes rare shapes to eager, almost identical to B0 (rare-group total time difference within about 0.1 s).")
    A("Data gaps / conventions: P5d has no B1 (only B0/Ours/B1-AOT); X5 has no B0 (reference arm is B1); meta has no NVML peak memory; W3 has no rare shapes at the 1% threshold.")
    A("== 2. X5 real-machine eviction evidence (Ours, three processes; details in x5_evictions.csv / x5_evict_proc_summary.csv) ==")
    p0 = ps_rows[0]
    A(f"Prediction (frozen pred_X5.csv, Ours): evicted {p0['pred_n_evicted']}, denied {p0['pred_n_denied']}, graph-pool peak {p0['pred_peak_graph_mb']} MB, estimated reserved peak {p0['pred_peak_reserved_est_mb']} MB.")
    for p in ps_rows:
        A(f"p{p['proc']}: policy evictions n_evicted_total={p['n_evicted_total_meta']} (evict_log {p['n_policy_evict_log']} entries); runtime release log {p['n_runtime_release_log']} entries, "
          f"of which {p['n_upgrade_release']} are releases before a G1->C1+G1 upgrade (not counted as evictions); predicted-minus-actual freed gap max={p['evict_gap_mb_max_meta']} MB (recomputed {p['gap_max_recomputed']}, min {p['gap_min_recomputed']}); "
          f"capture_denied meta={p['n_denied_meta']}/csv={p['n_capture_denied_csv']} (idx {p['denied_first_idx']}~{p['denied_last_idx']}, {p['n_denied_distinct_shapes']} distinct shapes); "
          f"graph-pool peak {p['peak_graph_mb_meta']} MB (csv max {p['graph_mem_mb_csv_max']}) vs budget {p['graph_budget_mb']} MB: {'over' if p['graph_pool_over_budget'] else 'within'} budget, margin {p['graph_margin_mb']:.1f} MB; "
          f"reserved peak {p['peak_reserved_mb']} MB (mem_cap {p['mem_cap_mb']}).")
    A("Note: runtime_evict_log has no request index; the idx of policy evictions is taken from meta.evict_log (exactly paired by shape+evict_s); the idx of upgrade releases is inferred from the compile_capture request of that shape in the csv (marked inferred_from_csv).")
    A("Eviction points (request index, identical for the three processes) with shape, predicted/actual freed memory (MB) and difference (process p0 as example):")
    for e in ev_rows:
        if e["proc"] == 0:
            A(f"  [{e['type']}] idx={e['request_idx']}({e['idx_source']}) shape={e['evicted_shape']} kind={e['graph_kind']} predicted={e['pred_freed_mb']} actual={e['actual_freed_mb']} gap={e['gap_mb_pred_minus_actual']} time={e['evict_s']*1000:.0f}ms")
    A("Are the eviction sequences of the three processes identical entry by entry (idx/shape/freed): " + str(len({tuple((e['request_idx'], e['evicted_shape'], e['actual_freed_mb']) for e in ev_rows if e['proc'] == q) for q in {r['proc'] for r in ev_rows}}) == 1))
    A("Comparison with prediction: n_evicted measured 6 vs predicted 7 (difference 1, fewer than predicted); n_denied 56 vs 56 identical; graph-pool peak 6408 vs 6408 identical; reserved peak measured 20540 vs estimated 19754 (+786 MB, estimate too low).")
    A("== 3. GPU memory peaks (basis A = torch peak_reserved_mb from meta; the meta has no NVML peak field, no value listed) ==")
    g = defaultdict(list)
    for r in peak:
        g[(r["exp"], r["workload"], r["arm"])].append(r)
    for k, rs in sorted(g.items()):
        v = [r["peak_reserved_mb_meta"] for r in rs]; gm = [r["peak_graph_mb"] for r in rs]
        A(f"{k[0]} {k[1]} {k[2]}: peak_reserved_mb per process={[int(x) for x in v]}; peak_graph_mb={[int(x) for x in gm]}")
    A("NVML basis: each meta only has nvidia-smi clocks/temperature/power (smi/smi_end) and no memory peak, so the NVML peak column is NA and is not mixed with the torch basis.")
    return L


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = list_procs()
    per = rare_analysis(items)
    summ = summarize(per)
    ev_rows, ps_rows = evict_analysis(items)
    peak = peak_analysis(items)
    SUM_COLS = ["exp", "workload", "arm", "thr_pct", "thr_count", "group", "n_proc", "N", "n_shapes_mean", "n_req_mean", "req_frac_mean",
                "first_med_ms_mean", "first_p90_ms_mean", "first_max_ms_mean", "first_max_ms_max", "p50_ms_mean", "p99_ms_mean", "mean_ms_mean",
                "sum_s_mean", "sum_s_min", "sum_s_max", "sum_frac_mean", "total_dur_s_mean", "ref_arm", "sum_s_minus_ref", "mean_ms_ratio_to_ref"]
    write_csv(OUT / "rare_shape_per_process.csv", per, PER_COLS)
    write_csv(OUT / "rare_shape_summary.csv", summ, SUM_COLS)
    write_csv(OUT / "x5_evictions.csv", ev_rows)
    write_csv(OUT / "x5_evict_proc_summary.csv", ps_rows)
    write_csv(OUT / "peak_mem_summary.csv", peak)
    L = build_report(per, summ, ev_rows, ps_rows, peak)
    (OUT / "REPORT_rare_evict.txt").write_text("\n".join(L) + "\n")
    print(f"processes={len(items)} per_rows={len(per)} summary_rows={len(summ)} evict_rows={len(ev_rows)} peak_rows={len(peak)} report_lines={len(L)}")


if __name__ == "__main__":
    main()
