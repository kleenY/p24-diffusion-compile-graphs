"""One-command build of the P0 paper artefacts (S5 P0-4): registry_extra CSVs + tables, numbers.tex, Fig.1, Fig.2(a).

    python -m p24.paper.build_all [--check-tex FILE]

Order: tab_metrics / tab_policy (write registry_extra/*.csv and tables/*.tex) -> p24.paper.build (numbers.tex, manifest) ->
Fig.1 (reproduce_main_figure --paper layer) -> Fig.2(a).  Prints "<sha16>  <relative path>" for every PDF / table / numbers.tex so that two
consecutive runs can be diffed (goals/S5 item 22).  CPU only; no torch.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from p24.paper import build as bld
from p24.paper import fig_cost, tab_metrics, tab_policy
from p24.paper import style as st

OUTPUTS = ["paper/paper/figs/fig1_main.pdf", "paper/paper/figs/fig2a_r.pdf", "paper/paper/tables/tab1_metrics.tex",
           "paper/paper/tables/tab2_policy.tex", "paper/paper/numbers.tex", "paper/paper/figs/regime/fig_regime_v2.pdf",
           "paper/paper/figs/regime/regime_realpoints_add_S5.csv"]
FIG1_RAW = "results/S3/sim_eval/main_raw.csv"


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def fig1(root: Path = st.ROOT) -> None:
    sys.path.insert(0, str(root / "code"))
    from p24 import reproduce_main_figure as R
    R.ANNOT["unvalidated"] = ["B1"]
    t = R.ratio_table(R.load(root / FIG1_RAW, "default"))
    R.draw_paper(t, "W2", R.DEFAULT_POLICIES, root / R.PAPER_OUT)


def regime(root: Path = st.ROOT) -> None:
    """Regime map v2 (S5, 11-23): frozen S4-X simulator grid + all real points incl. X5/X8/X9 (scripts/regime_fig_v2_S5.py, 0444).
    Reads only CSV/JSON files under results/ (no simulation, no torch); written to paper/paper/figs/regime/."""
    sys.path.insert(0, str(root / "code"))
    from scripts import regime_fig_v2_S5 as R
    R.OUT_V2 = root / "paper/paper/figs/regime"
    R.build_points()
    R.cmd_plot(None, R.OUT_V2)


def build_all(root: Path = st.ROOT) -> dict:
    tab_metrics.main([])
    tab_policy.main([])
    res = bld.build(root)
    fig1(root)
    fig_cost.main([])
    regime(root)
    return {p: sha16(root / p) for p in OUTPUTS} | {"manifest": res["manifest_sha16"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    for p, h in build_all().items():
        print(f"{h}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
