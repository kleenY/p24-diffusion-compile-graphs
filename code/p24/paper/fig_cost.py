"""Fig.2 of the paper (S5 P0-4): cost figure.  Only the S2 part is built now.

  (a) steady-state speed-up r = 1 - latency(arm) / latency(E0) versus the E0 latency of the shape (26 workload shapes, arms G1
      and C1+G1).  Source: results/S2/cost_table/merged/cost_table.csv (steady_ms, steady_ci_lo/hi per arm and shape); the 26-shape set and the
      G1 values are cross-checked against results/S2/cost_table/hs2_r_table.csv (frozen H-S2 table; |delta r| reported by check_against_hs2()).
      Error bars: conservative interval of r from the 95% steady-state CIs of the two arms,
          [1 - arm_hi / E0_lo , 1 - arm_lo / E0_hi]   (the S2 CSVs hold CIs of the arm latencies, not of r itself).
  (b) one-time cost per component x cold level for 3 representative shapes: needs the block-A supplement -> interface only
      (fig2b_onetime_costs raises NotImplementedError until block A is frozen).

    python -m p24.paper.fig_cost [--out paper/paper/figs/fig2a_r.pdf]

CPU only; matplotlib + csv; deterministic PDF (p24.paper.style).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from p24.paper import style as st

CT = "results/S2/cost_table/merged/cost_table.csv"
HS2 = "results/S2/cost_table/hs2_r_table.csv"
ARMS = [("G1", "G1 (graph only)", st.SERIES[0], "o"), ("C1+G1", "C1+G1 (compile + graph)", st.SERIES[1], "s")]
FIG_SIZE = (st.COL_W, 1.6)
KEY = ("H", "W", "B", "cfg")


def _read(root: Path, rel: str) -> List[Dict[str, str]]:
    with open(root / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def r_points(root: Path = st.ROOT) -> List[Dict[str, float]]:
    """One record per HS2 shape and arm: e0_ms, r, r_lo, r_hi (fractions).  Sorted by (e0_ms, arm) -> deterministic."""
    ct = {(r["arm"],) + tuple(r[k] for k in KEY): r for r in _read(root, CT)}
    out = []
    for h in _read(root, HS2):
        k = tuple(h[c] for c in KEY)
        e = ct[("E0",) + k]
        e_ms, e_lo, e_hi = (float(e[c]) for c in ("steady_ms", "steady_ci_lo", "steady_ci_hi"))
        for arm, *_ in ARMS:
            a = ct[(arm,) + k]
            a_ms, a_lo, a_hi = (float(a[c]) for c in ("steady_ms", "steady_ci_lo", "steady_ci_hi"))
            out.append(dict(arm=arm, shape="x".join(k[:2]) + "_B" + k[2] + "_cfg" + k[3], e0_ms=e_ms, r=1 - a_ms / e_ms,
                            r_lo=1 - a_hi / e_lo, r_hi=1 - a_lo / e_hi, source=a["source"] + "/" + e["source"]))
    return sorted(out, key=lambda d: (d["e0_ms"], d["arm"]))


def check_against_hs2(root: Path = st.ROOT) -> Dict[str, float]:
    """Cross-check of the plotted G1 r with the frozen hs2_r_table.csv (same 26 shapes, small difference from the later merge)."""
    pts = {(p["shape"]): p for p in r_points(root) if p["arm"] == "G1"}
    worst, inside = 0.0, 0
    hs = _read(root, HS2)
    for h in hs:
        p = pts["x".join(h[c] for c in KEY[:2]) + "_B" + h["B"] + "_cfg" + h["cfg"]]
        worst = max(worst, abs(p["r"] - float(h["r"])))
        inside += p["r_lo"] - 1e-12 <= float(h["r"]) <= p["r_hi"] + 1e-12
    return {"n_shapes": len(hs), "max_abs_dr": worst, "n_hs2_r_inside_ci": inside}


def fig2a(out: Path, root: Path = st.ROOT) -> Dict[str, float]:
    plt = st.mpl()
    pts = r_points(root)
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")
    st.style_ax(ax)
    for arm, label, col, mk in ARMS:
        p = [d for d in pts if d["arm"] == arm]
        x = [d["e0_ms"] for d in p]
        y = [100 * d["r"] for d in p]
        err = [[100 * (d["r"] - d["r_lo"]) for d in p], [100 * (d["r_hi"] - d["r"]) for d in p]]
        ax.errorbar(x, y, yerr=err, fmt=mk, color=col, ms=2.8, mew=0, elinewidth=0.6, capsize=0, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("E0 steady latency (ms, log)", color=st.INK2, labelpad=1)
    ax.set_ylabel("Steady speed-up r (%)", color=st.INK2, labelpad=1)
    ax.set_ylim(0, 36)
    ax.minorticks_off()
    ax.set_xticks([50, 100, 200, 500, 1000])
    ax.set_xticklabels(["50", "100", "200", "500", "1000"])
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.2, top=0.97)
    st.save_pdf(fig, out)
    plt.close(fig)
    return check_against_hs2(root)


def fig2b_onetime_costs(out: Path, block_a=None, root: Path = st.ROOT):
    """Interface for Fig.2(b): stacked one-time cost (capture / compile components) x cold level for 3 representative shapes.
    Needs the block-A cost supplement (frozen SHA16 + registry_extra entries); not built in P0."""
    raise NotImplementedError("Fig.2(b) waits for block A (cost-table supplement); see goals/S5 item 11 (Fig.2)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(st.FIG_DIR / "fig2a_r.pdf"))
    a = ap.parse_args(argv)
    res = fig2a(Path(a.out))
    print(f"wrote {a.out}; cross-check vs hs2_r_table.csv: {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
