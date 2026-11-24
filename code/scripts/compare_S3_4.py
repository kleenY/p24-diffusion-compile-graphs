"""S3-B verdict (goals/S3_detailed_goals_v2.md 3.4 / G19; criteria = S2 sec 9.9 as in scripts/compare_S2_6.py). CPU only.
To be fixed BEFORE any S3-B measurement is seen (this file is written in S3 prep; its SHA goes into the log before the block).

Inputs : frozen prediction CSV from scripts/predict_S3_4.py (pred_total_s = simulated one-time + steady + per-shape input_s),
         serve_trace metas  <raw>/<policy>_<trace stem>_p<id>.meta.json   (disk-cache tier)  and  <raw_empty>/  (empty-cache run, reported only).
Measured total service time = meta `service_s` (first item started -> last request finished, includes input preparation and B6/B6' warm-up);
the load-inclusive `total_incl_load_s` is reported alongside.
Rules (S2 9.9, S3 3.4):
  time    : error = |sim - median(measured over the ok processes)| / median ; pass iff error <= 10 % for EVERY judged combination
            (every warm row of the prediction: 3 streams x {B0, B1, B4, Ours, BEST} + W3 x B3'); the W5 x Ours empty-cache run is reported, not judged.
  ranking : delta = max over the two policies of (max - min)/median of their process values; policies a, b of one stream are tied when
            |Ta - Tb| / min(Ta, Tb) < max(2 %, 2 * delta); for every NON-tied pair (measured medians) the simulated order must equal the measured order.
  memory  : whole-process peak check of 3.1: max over processes of meta peak_reserved_mb <= predicted peak_reserved_est_mb + 1,024 MB
            (reported per combination; NOT part of `pass`, which is time + ranking; `mem_line_all_pass` is reported separately for the 3.1 line / G19 decision).
  record  : every process needs meta pinned_fallback_count (any > 0 is listed), status ok; a combination with fewer than 3 ok processes is flagged `incomplete`
            (judged on what exists, flagged; the main session decides at acceptance).
  events  : per process the measured event counts (request phase) vs the predicted event counts, informational (`events_match`).
Output (to --out-dir, default results/S3/real_validation): error_table<tag>.csv, ranking_table<tag>.csv, measured_processes<tag>.csv, verdict<tag>.json
Usage (from code/): python -B -m scripts.compare_S3_4 --pred ../results/S3/real_validation/pred_S3_4.csv [--tag _B1]   |   --selftest
  With split blocks (_B1 / _B2) run once after both blocks without --tag (raw/ is shared; the combination set is taken from the prediction file).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RV = ROOT / "results" / "S3" / "real_validation"
MEM_LINE_MB = 1024.0


def norm(p: str) -> str:
    return str(p).replace("′", "p").replace("'", "p")


def measured(raw: pathlib.Path, empty: bool) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*.meta.json")) if raw.exists() else []:
        m = json.loads(f.read_text())
        rows.append(dict(stream=pathlib.Path(m["trace"]).stem, policy=norm(m["policy"]), cache_mode="empty" if empty else m.get("cache_mode", "warm"),
                         proc=int(f.name.rsplit("_p", 1)[1].split(".")[0]), status=m.get("status"), service_s=m.get("service_s"),
                         service_s_excl_input=m.get("service_s_excl_input"), input_s_total=m.get("input_s_total"),
                         total_incl_load_s=m.get("total_incl_load_s"), load_s=m.get("load_s"), peak_reserved_mb=m.get("peak_reserved_mb"),
                         pinned_fallback_count=m.get("pinned_fallback_count"), n_evicted=m.get("n_evicted_total"), n_denied=m.get("n_denied"),
                         event_counts=json.dumps(m.get("event_counts"), sort_keys=True), file=f.name))
    return pd.DataFrame(rows)


def run(pred_path, raw, raw_empty, out_dir, tag):
    pred = pd.read_csv(pred_path)
    pred["policy"] = pred.policy.map(norm)
    meas = pd.concat([measured(raw, False), measured(raw_empty, True)], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    meas.to_csv(out_dir / f"measured_processes{tag}.csv", index=False)
    rows = []
    for pr in pred.itertuples():
        judged = int(pr.cache_mode == "warm")
        g = meas[(meas.stream == pr.stream) & (meas.policy == pr.policy) & (meas.cache_mode == pr.cache_mode)] if len(meas) else meas
        ok = g[g.status == "ok"] if len(g) else g
        vals = ok.service_s.to_numpy(float) if len(ok) else np.array([])
        med = float(np.median(vals)) if len(vals) else np.nan
        sim = float(pr.pred_total_s) if hasattr(pr, "pred_total_s") and pr.pred_total_s == pr.pred_total_s else np.nan
        pk = ok.peak_reserved_mb.to_numpy(float) if len(ok) else np.array([])
        pred_peak = float(pr.pred_peak_reserved_est_mb) if hasattr(pr, "pred_peak_reserved_est_mb") else np.nan
        ev_match = int(sum(json.loads(e) == json.loads(pr.event_counts) for e in ok.event_counts)) if len(ok) and isinstance(pr.event_counts, str) else None
        rows.append(dict(stream=pr.stream, policy=pr.policy, cache_mode=pr.cache_mode, role=getattr(pr, "role", ""), judged=judged, n_ok=len(vals),
                         n_fail=int((g.status != "ok").sum()) if len(g) else 0, incomplete=int(len(vals) < (3 if judged else 1)),
                         measured_values_s=";".join(f"{v:.3f}" for v in vals), measured_median_s=med,
                         spread_rel=(vals.max() - vals.min()) / med if len(vals) > 1 else np.nan, sim_s=sim,
                         error=abs(sim - med) / med if med == med and sim == sim else np.nan,
                         sim_excl_input_s=getattr(pr, "pred_excl_input_s", np.nan), measured_excl_input_median_s=float(np.median(ok.service_s_excl_input)) if len(ok) else np.nan,
                         measured_incl_load_median_s=float(np.median(ok.total_incl_load_s)) if len(ok) else np.nan,
                         peak_pred_mb=pred_peak, peak_meas_max_mb=float(pk.max()) if len(pk) else np.nan,
                         peak_over_pred_mb=float(pk.max() - pred_peak) if len(pk) and pred_peak == pred_peak else np.nan,
                         mem_line_pass=int(pk.max() <= pred_peak + MEM_LINE_MB) if len(pk) and pred_peak == pred_peak else np.nan,
                         pinned_fallback_total=int(pd.to_numeric(g.pinned_fallback_count, errors="coerce").fillna(-1).sum()) if len(g) else np.nan,
                         events_match=ev_match, pred_error=str(getattr(pr, "error", "")) if str(getattr(pr, "error", "nan")) != "nan" else ""))
    err = pd.DataFrame(rows)
    err["pass_10pct"] = np.where(err.judged == 1, (err.error <= 0.10).astype(int), np.nan)
    err.to_csv(out_dir / f"error_table{tag}.csv", index=False)
    rk = []
    for s, g in err[err.judged == 1].groupby("stream"):
        g = g.set_index("policy")
        for pa, pb in itertools.combinations(sorted(g.index), 2):
            ta, tb = g.loc[pa, "measured_median_s"], g.loc[pb, "measured_median_s"]
            if not (ta == ta and tb == tb):
                continue
            delta = np.nanmax([g.loc[pa, "spread_rel"], g.loc[pb, "spread_rel"]]) if (g.loc[[pa, pb], "spread_rel"].notna().any()) else 0.0
            thr = max(0.02, 2 * delta)
            gap = abs(ta - tb) / min(ta, tb)
            tied = gap < thr
            sa, sb = g.loc[pa, "sim_s"], g.loc[pb, "sim_s"]
            rk.append(dict(stream=s, a=pa, b=pb, measured_a=ta, measured_b=tb, sim_a=sa, sim_b=sb, rel_gap=gap, tie_threshold=thr,
                           tied=int(tied), order_agrees=int(np.sign(ta - tb) == np.sign(sa - sb)), counts=int(not tied)))
    rk = pd.DataFrame(rk, columns=["stream", "a", "b", "measured_a", "measured_b", "sim_a", "sim_b", "rel_gap", "tie_threshold", "tied",
                                   "order_agrees", "counts"])
    rk.to_csv(out_dir / f"ranking_table{tag}.csv", index=False)
    j = err[err.judged == 1]
    n_pred_missing = int((j.pred_error != "").sum())
    bad_rank = rk[(rk.counts == 1) & (rk.order_agrees == 0)]
    verdict = dict(n_judged=int(len(j)), n_with_no_ok_process=int((j.n_ok == 0).sum()), n_incomplete=int(j.incomplete.sum()),
                   n_prediction_errors=n_pred_missing, all_within_10pct=bool(len(j) > 0 and (j.error <= 0.10).all()),
                   max_error=float(j.error.max()) if len(j) else None, worst_combo=(f"{j.loc[j.error.idxmax(), 'stream']}/{j.loc[j.error.idxmax(), 'policy']}" if len(j) and j.error.notna().any() else None),
                   n_nontied_pairs=int(rk.counts.sum()) if len(rk) else 0, n_tied_pairs=int(rk.tied.sum()) if len(rk) else 0,
                   ranking_agrees=bool(len(bad_rank) == 0) if len(rk) else None,
                   reversed_pairs=[f"{r.stream}:{r.a}<>{r.b}" for r in bad_rank.itertuples()],
                   mem_line_all_pass=bool((j.mem_line_pass == 1).all()) if len(j) and j.mem_line_pass.notna().all() else None,
                   mem_line_violations=[f"{r.stream}/{r.policy}:+{r.peak_over_pred_mb:.0f}MB" for r in j.itertuples() if r.mem_line_pass == 0],
                   pinned_fallback_flagged=[f"{r.stream}/{r.policy}:{r.pinned_fallback_total}" for r in err.itertuples() if r.pinned_fallback_total not in (0, np.nan) and r.pinned_fallback_total == r.pinned_fallback_total and r.pinned_fallback_total != 0],
                   n_failed_processes=int(err.n_fail.sum()), pred_file=str(pred_path))
    verdict["pass"] = bool(verdict["all_within_10pct"] and verdict["ranking_agrees"] is not False and verdict["n_with_no_ok_process"] == 0
                           and verdict["n_prediction_errors"] == 0)
    (out_dir / f"verdict{tag}.json").write_text(json.dumps(verdict, indent=2))
    pd.set_option("display.width", 220)
    print(err[["stream", "policy", "cache_mode", "judged", "n_ok", "measured_median_s", "sim_s", "error", "spread_rel", "peak_over_pred_mb", "mem_line_pass"]].round(4).to_string(index=False))
    if len(rk):
        print(rk[["stream", "a", "b", "rel_gap", "tie_threshold", "tied", "order_agrees"]].round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1))
    return verdict


def selftest():
    """Synthetic metas: (1) all within 10 %, order kept -> pass; (2) one combination 15 % off and one non-tied pair reversed -> fail."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cmp34_", dir=os.environ.get("TMPDIR")))
    pred = pd.DataFrame([
        dict(stream="WX_N300_seed100_B", policy=p, cache_mode="warm", role=p, pred_total_s=t, pred_excl_input_s=t - 1, pred_peak_reserved_est_mb=10000.0,
             event_counts=json.dumps({"steady": 299, "eager_first": 1}))
        for p, t in (("B0", 100.0), ("B1", 150.0), ("Ours", 101.0))] + [
        dict(stream="WX_N300_seed100_B", policy="B3′", cache_mode="warm", role="B3prime", pred_total_s=120.0, pred_excl_input_s=119.0,
             pred_peak_reserved_est_mb=9000.0, event_counts=json.dumps({"steady": 299, "eager_first": 1}))])
    pp = tmp / "pred.csv"
    pred.to_csv(pp, index=False)

    def write(d, scale, over=0.0, reverse=False):
        (tmp / d).mkdir(exist_ok=True)
        for r in pred.itertuples():
            for p in range(3):
                t = r.pred_total_s * scale.get(r.policy, 1.0) * (1 + 0.005 * (p - 1))
                if reverse and r.policy == "B1":
                    t = 90.0  # B1 measured faster than B0 although predicted slower: non-tied reversal
                meta = dict(trace=f"x/{r.stream}.jsonl", policy=r.policy, cache_mode="warm", status="ok", service_s=t, service_s_excl_input=t - 1, input_s_total=1.0,
                            total_incl_load_s=t + 0.5, load_s=0.5, peak_reserved_mb=r.pred_peak_reserved_est_mb + 500 + over, pinned_fallback_count=0,
                            event_counts={"steady": 299, "eager_first": 1})
                (tmp / d / f"{r.policy.replace('′', 'p')}_{r.stream}_p{p}.meta.json").write_text(json.dumps(meta))

    write("raw_ok", {"B0": 1.03, "B1": 0.98, "Ours": 1.04})
    v1 = run(pp, tmp / "raw_ok", tmp / "none", tmp / "out_ok", "")
    write("raw_bad", {"B0": 1.03, "Ours": 1.15}, over=800.0, reverse=True)
    v2 = run(pp, tmp / "raw_bad", tmp / "none", tmp / "out_bad", "")
    assert v1["pass"] is True and v1["mem_line_all_pass"] is True, v1
    assert v2["pass"] is False and v2["all_within_10pct"] is False and v2["ranking_agrees"] is False and v2["mem_line_all_pass"] is False, v2
    print("SELFTEST OK", tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--raw", default=str(RV / "raw"))
    ap.add_argument("--raw-empty", default=str(RV / "raw_empty"))
    ap.add_argument("--out-dir", default=str(RV))
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.pred:
        sys.exit("--pred is required")
    run(pathlib.Path(a.pred), pathlib.Path(a.raw), pathlib.Path(a.raw_empty), pathlib.Path(a.out_dir), a.tag)


if __name__ == "__main__":
    main()
