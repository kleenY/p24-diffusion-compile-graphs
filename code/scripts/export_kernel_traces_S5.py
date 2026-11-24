"""S5 (2026-11-24): export kernel-level profiling traces from the Nsight Systems sqlite files used in the paper.

Why: the course asks for profiling traces in the public repository. The original .nsys-rep / .sqlite files embed host
metadata (host name, user name, environment, command lines), so the repository ships a text export instead:
one CSV per trace with every GPU kernel and memcpy inside the NVTX `request<i>` ranges (the steady-state requests that the
kernel breakdown in the paper averages), plus the NVTX request ranges themselves.

Columns (kernels):  request, kind, name (short kernel name), start_ns (relative to the first request start), dur_ns,
                    stream, grid, block, graph_node (non-empty when launched from a CUDA Graph)
Columns (requests): request, start_ns, dur_ns

Usage (CPU only, from the project root):  python -B code/scripts/export_kernel_traces_S5.py
Output: results/S5/profiling_traces/<group>/<trace>.kernels.csv, <trace>.requests.csv, and index.csv (source sqlite SHA-256).
Read-only on the sqlite files; refuses to overwrite an existing output file with different content.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
OUT = ROOT / "results/S5/profiling_traces"
# (group, sqlite path relative to ROOT, what it backs in the paper)
TRACES = []
for arm in ("E0", "C1", "C2", "G1", "G2"):
    for cfg in (0, 1):
        TRACES.append(("S1_512_b1", f"results/S1/nsys_q2/{arm}_b1_cfg{cfg}.sqlite", "kernel breakdown at 512x512 B1 (Tab. kernels)"))
for cfg in (0, 1):
    TRACES.append(("S1_512_b1", f"results/S1/nsys_q/C1+G2_b1_cfg{cfg}.sqlite", "kernel breakdown at 512x512 B1"))
for n in ("E0_256x256_b1_cfg0", "G1_256x256_b1_cfg0", "E0_768x768_b4_cfg1", "G1_768x768_b4_cfg1"):
    TRACES.append(("S2_256_768", f"results/S2/cost_table/S-1/nsys/{n}.sqlite", "idle-gap / capture saving at 256x256 and 768x768"))
for n in ("C1+G1_512x512_b1_cfg0",):
    TRACES.append(("S3_accel", f"results/S3/accel/nsys/{n}.sqlite", "C1+G1 at 512x512"))
for n in ("E0_512x512_b1_cfg0", "C1_512x512_b1_cfg0"):
    TRACES.append(("S4X_X10_sdturbo", f"results/S4X/x10/profile/{n}.sqlite", "second model (SD-Turbo) mechanism check"))


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_once(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"exists with different content: {path}"
        return
    path.write_text(text, encoding="utf-8")


def export_one(src: pathlib.Path):
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    reqs = con.execute("select text, start, end from NVTX_EVENTS where end is not null and text like 'request%' order by start").fetchall()
    assert reqs, f"no NVTX request ranges in {src}"
    t0 = reqs[0][1]
    names = dict(con.execute("select id, value from StringIds"))
    rows = []
    for st, en, sid, stream, gx, gy, gz, bx, by, bz, node in con.execute(
            "select start, end, shortName, streamId, gridX, gridY, gridZ, blockX, blockY, blockZ, graphNodeId "
            "from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        rows.append((st, en, "kernel", names.get(sid, str(sid)), stream, f"{gx}x{gy}x{gz}", f"{bx}x{by}x{bz}", node))
    try:
        for st, en, kind, stream, node in con.execute(
                "select m.start, m.end, e.label, m.streamId, m.graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY m "
                "left join ENUM_CUDA_MEMCPY_OPER e on m.copyKind = e.id order by m.start"):
            rows.append((st, en, f"memcpy {kind}", "", stream, "", "", node))
    except sqlite3.Error:
        pass
    rows.sort()
    k = io.StringIO()
    w = csv.writer(k, lineterminator="\n")
    w.writerow(["request", "kind", "name", "start_ns", "dur_ns", "stream", "grid", "block", "graph_node"])
    n = 0
    for st, en, kind, name, stream, grid, block, node in rows:
        req = next((r[0] for r in reqs if r[1] <= st and en <= r[2]), None)
        if req is None:
            continue
        w.writerow([req, kind, name, st - t0, en - st, stream, grid, block, "" if node is None else node])
        n += 1
    r = io.StringIO()
    wr = csv.writer(r, lineterminator="\n")
    wr.writerow(["request", "start_ns", "dur_ns"])
    for text, st, en in reqs:
        wr.writerow([text, st - t0, en - st])
    return k.getvalue(), r.getvalue(), n, len(reqs)


def main() -> int:
    idx = [["group", "trace", "source_sqlite", "source_sha256", "n_records_in_requests", "n_requests", "used_for"]]
    for group, rel, used in TRACES:
        src = ROOT / rel
        if not src.is_file():
            print("missing", rel, file=sys.stderr)
            return 1
        kcsv, rcsv, n, nr = export_one(src)
        stem = src.stem
        write_once(OUT / group / f"{stem}.kernels.csv", kcsv)
        write_once(OUT / group / f"{stem}.requests.csv", rcsv)
        idx.append([group, stem, rel, sha256(src), n, nr, used])
        print(f"{group}/{stem}: {n} records in {nr} requests")
    s = io.StringIO()
    csv.writer(s, lineterminator="\n").writerows(idx)
    write_once(OUT / "index.csv", s.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
