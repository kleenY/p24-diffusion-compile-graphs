"""Launch-overhead decomposition from an Nsight Systems report.

For every NVTX range "request<i>" (one end-to-end request, see bench --nsys-window):
  wall_ms        NVTX range duration (host view, includes the final synchronize)
  gpu_busy_ms    union of kernel + memcpy/memset intervals inside the range
  gpu_idle_frac  1 - busy / wall   (time the GPU waits for the host: launch/dispatch overhead)
  n_kernels      kernels executed (graph kernels counted per node with --cuda-graph-trace=node)
  n_launch_api   host-side launch API calls (cudaLaunchKernel*/cuLaunchKernel*/cudaGraphLaunch*)
Usage: python -m p24.nsys_analyze <report.nsys-rep> [--csv out.csv]
"""
from __future__ import annotations

import argparse
import pathlib
import sqlite3
import subprocess

NSYS = "/usr/local/cuda/bin/nsys"


def to_sqlite(rep: pathlib.Path) -> pathlib.Path:
    db = rep.with_suffix(".sqlite")
    if not db.exists():
        subprocess.run([NSYS, "export", "--type", "sqlite", "-o", str(db), "-f", "true", str(rep)],
                       check=True, capture_output=True)
    return db


def union_len(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def tables(con):
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


def analyze(rep: pathlib.Path):
    con = sqlite3.connect(to_sqlite(rep))
    t = tables(con)
    nv = con.execute("""select start, end, coalesce(text, (select value from StringIds where id=textId))
                        from NVTX_EVENTS where end is not null""").fetchall()
    reqs = [(s, e, n) for s, e, n in nv if n and n.startswith("request")]
    gpu = []
    for tab in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if tab in t:
            gpu += [(s, e, tab) for s, e in con.execute(f"select start, end from {tab}")]
    api = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in t:
        api = con.execute("""select r.start, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
                             join StringIds s on r.nameId = s.id""").fetchall()
    rows = []
    for s, e, n in reqs:
        ins = [(a, b) for a, b, _ in gpu if a >= s and b <= e]
        kern = sum(1 for a, b, tab in gpu if tab.endswith("KERNEL") and a >= s and b <= e)
        launches = sum(1 for ts, name in api if s <= ts <= e and ("LaunchKernel" in name or "GraphLaunch" in name))
        graph_launches = sum(1 for ts, name in api if s <= ts <= e and "GraphLaunch" in name)
        wall = (e - s) / 1e6
        busy = union_len(ins) / 1e6
        rows.append({"report": rep.stem, "range": n, "wall_ms": wall, "gpu_busy_ms": busy,
                     "gpu_idle_frac": 1 - busy / wall if wall else float("nan"), "n_kernels": kern,
                     "n_launch_api": launches, "n_graph_launch": graph_launches})
    return rows


if __name__ == "__main__":
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    allrows = []
    for r in a.reports:
        allrows += analyze(pathlib.Path(r))
    for r in allrows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    if a.csv and allrows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0]))
            w.writeheader()
            w.writerows(allrows)
