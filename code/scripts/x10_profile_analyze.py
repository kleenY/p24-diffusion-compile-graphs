"""S4-X X10 (CPU): kernel-class breakdown of the Nsight Systems traces of E0 and C1 (SD-Turbo, 512x512 B1 no-CFG; written by window_S4X_X10.sh), with EXACTLY the
S4X rules of scripts/ceiling_S4X.py: classify() / RULES / CATS / BIG (conv, gemm, attention = big ops; layout_convert, norm_reduce, elementwise_copy = small kernels;
triton_fused = Inductor-generated; other) and load_trace() (events inside the NVTX request<i> ranges; GPU-busy = union of kernel + memcpy + memset intervals).
Both are IMPORTED, not copied.  The .nsys-rep -> .sqlite export is p24.nsys_analyze.to_sqlite (`nsys export --type sqlite`; reused if the sqlite exists).
Output: <profile-dir>/profile_summary.json (per arm: per-category ms/request, kernel total, busy union, traced request wall, n_requests, kernel counts per category,
top unclassified kernel names) and <profile-dir>/kernel_breakdown_x10.csv.  Completeness (card sec 3 "trace complete"): >= 3 NVTX request ranges and > 0 kernels.
Usage (from code/): python -B -m scripts.x10_profile_analyze [--profile-dir ../results/S4X/x10/profile] [--shape 512x512_b1]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib

from p24.nsys_analyze import to_sqlite
from scripts.ceiling_S4X import BIG, CATS, classify, load_trace

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))


def analyze_one(sqlite_path: pathlib.Path) -> dict:
    t = load_trace(sqlite_path)
    n = t["n_req"]
    tot_ns = sum(e - s for s, e, _ in t["kern"])
    per = collections.defaultdict(lambda: [0, 0])
    names = collections.defaultdict(lambda: [0, 0])
    for s, e, nm in t["kern"]:
        c = classify(nm)
        per[c][0] += e - s
        per[c][1] += 1
        if c == "other":
            names[nm][0] += e - s
            names[nm][1] += 1
    return dict(n_requests=n, n_kernels=len(t["kern"]), kernel_total_ms=tot_ns / 1e6 / n, busy_ms=t["busy_ms"], traced_wall_ms=t["wall_ms"],
                traced_idle_ms=t["wall_ms"] - t["busy_ms"],
                ms={c: per[c][0] / 1e6 / n for c in CATS}, kernels_per_req={c: per[c][1] / n for c in CATS},
                big_ms=sum(per[c][0] for c in BIG) / 1e6 / n, other_fraction=(per["other"][0] / tot_ns if tot_ns else float("nan")),
                top_other=[dict(name=nm[:140], frac=v[0] / tot_ns, per_req=v[1] / n) for nm, v in sorted(names.items(), key=lambda x: -x[1][0])[:8]],
                complete=bool(n >= 3 and len(t["kern"]) > 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile-dir", default=str(ROOT / "results/S4X/x10/profile"))
    ap.add_argument("--shape", default="512x512_b1")
    ap.add_argument("--arms", default="E0,C1")
    a = ap.parse_args(argv)
    pdir = pathlib.Path(a.profile_dir)
    res, rows = {"shape": a.shape, "cfg": 0, "rules": "scripts/ceiling_S4X.py classify()/RULES (imported)", "arms": {}}, []
    for arm in a.arms.split(","):
        rep = pdir / f"{arm}_{a.shape}_cfg0.nsys-rep"
        sq = pdir / f"{arm}_{a.shape}_cfg0.sqlite"
        if not sq.exists() and rep.exists():
            sq = to_sqlite(rep)
        if not sq.exists():
            res["arms"][arm] = dict(complete=False, error=f"missing {rep.name} / {sq.name}")
            continue
        try:
            r = analyze_one(sq)
        except Exception as e:  # noqa: BLE001
            r = dict(complete=False, error=repr(e)[:400])
        r["source"] = str(sq)
        res["arms"][arm] = r
        if r.get("complete"):
            for c in CATS:
                rows.append(dict(arm=arm, shape=a.shape, category=c, gpu_ms_per_req=r["ms"][c], n_kernels_per_req=r["kernels_per_req"][c],
                                 frac_of_kernel_time=r["ms"][c] / r["kernel_total_ms"] if r["kernel_total_ms"] else float("nan"),
                                 kernel_total_ms=r["kernel_total_ms"], busy_ms=r["busy_ms"], traced_wall_ms=r["traced_wall_ms"]))
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "profile_summary.json").write_text(json.dumps(res, indent=1))
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(pdir / "kernel_breakdown_x10.csv", index=False)
    for arm, r in res["arms"].items():
        if r.get("complete"):
            print(arm, f"kernel_total {r['kernel_total_ms']:.2f} ms busy {r['busy_ms']:.2f} traced_wall {r['traced_wall_ms']:.2f} big {r['big_ms']:.2f} other_frac {r['other_fraction'] * 100:.2f}%",
                  {c: round(v, 2) for c, v in r["ms"].items()})
        else:
            print(arm, "INCOMPLETE", r.get("error", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
