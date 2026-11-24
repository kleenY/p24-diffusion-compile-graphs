"""Main figure of S3 3.3 from a raw CSV only (no simulation, no trace, no cost table is read).

One command, three layers (S4 4.4a; goals/S4_detailed_goals_v1.md "Gate-2 revision" 4-5):
  (a) python -m p24.reproduce_main_figure --from-csv results/S3/sim_eval/main_raw.csv [--out DIR] [--unvalidated B1]
        redraw in seconds from a raw CSV (`--raw` is the old spelling of the same option; behaviour unchanged, byte-identical figures).
  (b) python -m p24.reproduce_main_figure [--out results/S4/reproduce] [--procs 4]
        no input CSV: re-run the request-level simulator (scripts/sim_eval_S3.py, split eval, experiment main, `--missing impute`, i.e. exactly
        the S3 3.3 command) on the FROZEN cost table + S3 addendum + frozen evaluation traces, write <out>/main_raw.csv, compare it with the frozen
        results/S3/sim_eval/main_raw.csv (all columns except wall-clock columns must agree; report in <out>/compare_vs_frozen.json), draw <out>/figs.
        About 3-11 min CPU depending on --procs (<= 8).  Across environments the CSV numbers must agree; PNG bytes only in the same environment.
  (c) python -m p24.reproduce_main_figure --quick [--dry-run]
        real-GPU scaled-down check from an EMPTY compile cache (temp dir): 2 shapes cost micro-benchmark + 1 short trace x {E0, B1, Ours},
        GPU memory hard cap 14 GB, target <= 15 min; prints predicted vs measured total service time and draws a reduced main figure.
        Implemented in scripts/reproduce_quick_S4.py.  --dry-run: CPU only (prediction + plan + time estimate), no GPU.
Frozen reference SHA16 of the PNG (same environment): fig_main_W2.png no annotation e6d8122a695c569a ; with --unvalidated B1  7ff201527617b73f.

Layer (a) details.
Main figure  : workload W2; x = N (log); y = total service time / B0 of the SAME (workload, N, seed, cold level); three panels =
               cold level (empty cache / disk cache / in-process warm); line = mean over seeds, band = min-max over seeds.
Appendix     : the same layout for W1, W3, W4, W5 (fig_app_<W>.*).
Also writes  : main_figure_table.csv (the plotted numbers, "table view" of the figure) and a list of (policy, panel) cells that have no
               plotted value because the simulation reported a missing cost.
Style        : dataviz palette (categorical slots in the documented order, light surface), thin lines, hairline solid grid, single y axis,
               legend always present; B0 is the y = 1 reference, B7 (oracle that knows the stream's shape counts) is dashed ink-grey.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd

# ---- dataviz reference palette (light mode): categorical slots in the documented order, chrome tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # blue orange aqua yellow magenta green violet red
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
DISPLAY = {"B3p": "B3′", "B3pp": "B3″", "B6p": "B6′", "Ours-NE": "Ours-NE"}
PANELS = [("empty", "Empty compile cache"), ("disk", "Disk cache (pre-warmed)"), ("hot", "In-process warm")]
DEFAULT_POLICIES = ["Ours", "B1", "B3p", "B3pp", "B4", "B5", "B6p", "B7"]
ORACLE = "B7"
ANNOT = {"unvalidated": []}   # set by --unvalidated (S3 3.4 / C-19): policies whose simulated totals failed the real-replay check
YCAP = 1.35          # y axis is capped here so that differences of a few percent stay readable; larger means are listed in the panel


def disp(p: str) -> str:
    return DISPLAY.get(p, p)


def load(raw: pathlib.Path, budget: str) -> pd.DataFrame:
    d = pd.read_csv(raw)
    need = {"variant", "workload", "N", "seed", "cold", "budget", "status", "service_s", "boot", "prior_kind"}
    assert need <= set(d.columns), f"not a sim_eval_S3 raw CSV, missing {need - set(d.columns)}"
    d = d[(d.budget == budget) & (d.boot == -1) & (d.prior_kind == "A")]
    if "trace_kind" in d:
        d = d[d.trace_kind.isin(["file", "gen_default"])]
    return d


def ratio_table(d: pd.DataFrame) -> pd.DataFrame:
    ok = d[d.status == "ok"]
    key = ["workload", "N", "seed", "cold"]
    b0 = ok[ok.variant == "B0"].set_index(key).service_s
    r = ok.set_index(key).copy()
    r["b0"] = b0.reindex(r.index).to_numpy()
    r = r.reset_index()
    r = r[r.b0.notna()]
    r["ratio"] = r.service_s / r.b0
    g = r.groupby(["workload", "cold", "variant", "N"]).ratio
    t = g.agg(mean="mean", lo="min", hi="max", n_seeds="count").reset_index()
    return t


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)


def fmt_n(n: float) -> str:
    e = math.log10(n)
    return f"$10^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-9 else f"{int(n)}"


def draw(t: pd.DataFrame, d: pd.DataFrame, w: str, policies, out: pathlib.Path, formats, dpi, split: str, nseeds: int, budget: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.edgecolor": AXIS, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True, facecolor=SURFACE)
    tw = t[t.workload == w]
    order = [p for p in policies if p in set(tw.variant)]
    colored = [p for p in policies if p != ORACLE]
    color = {p: SERIES[i % len(SERIES)] for i, p in enumerate(colored)}
    color[ORACLE] = INK2
    mark = {p: MARKERS[i % len(MARKERS)] for i, p in enumerate(colored)}
    mark[ORACLE] = "x"
    handles, labels = {}, {}
    ymin, ymax = 1.0, 1.0
    clipped, notes = {}, []
    for ax, (cold, title) in zip(axes, PANELS):
        style(ax)
        ax.set_xscale("log")
        sub = tw[tw.cold == cold]
        ax.axhline(1.0, color=MUTED, linewidth=1.0, zorder=1)
        for p in order:
            s = sub[sub.variant == p].sort_values("N")
            if s.empty:
                continue
            c = color[p]
            if p != ORACLE and len(s) > 1:
                ax.fill_between(s.N, s.lo, s.hi, color=c, alpha=0.14, linewidth=0, zorder=2)
            (ln,) = ax.plot(s.N, s["mean"], color=c, linewidth=1.8 if p == "Ours" else 1.4, marker=mark[p], markersize=4.5,
                            linestyle="--" if p == ORACLE else "-", zorder=4 if p == "Ours" else 3, markeredgewidth=1.0 if p == ORACLE else 0)
            handles[p], labels[p] = ln, disp(p) + ("†" if p in ANNOT["unvalidated"] else "")
            ymin = min(ymin, float(s.lo.min()))
            ymax = max(ymax, float(s.hi.max()))
            clipped.setdefault(cold, []).extend((disp(p), int(r.N), float(r.mean)) for r in s.itertuples() if r.mean > YCAP)
        # policies that were requested but have no plotted value in this panel (missing cost in the cost table)
        dd = d[(d.workload == w) & (d.cold == cold)]
        missing = [disp(p) for p in order if (dd.variant == p).any() and (dd[(dd.variant == p)].status != "ok").all()]
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("N (requests per process)", color=INK2, fontsize=10)
        ns = sorted(tw.N.unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([fmt_n(n) for n in ns])
        ax.minorticks_off()
        if clipped.get(cold):
            notes.append(f"[{title}] above the axis (mean): " + "; ".join(f"{p} {v:.1f}x at N={fmt_n(n)}" for p, n, v in clipped[cold]))
        if missing:
            notes.append(f"[{title}] no value (cost not measured in the S2 cost table): " + ", ".join(missing))
    top = min(ymax, YCAP)
    pad = 0.04 * (top - ymin + 1e-9)
    axes[0].set_ylim(ymin - pad, top + pad)                            # larger values are clipped and listed in the notes below
    axes[0].set_ylabel("Total service time / B0   (lower is better)", color=INK2, fontsize=10)
    axes[-1].text(1.01, 1.0, "B0 = 1", transform=axes[-1].get_yaxis_transform(), ha="left", va="center", fontsize=8, color=INK2)
    leg = fig.legend([handles[p] for p in order if p in handles], [labels[p] for p in order if p in handles], loc="lower center",
                     ncol=len(handles), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.15), handlelength=2.4)
    for tx in leg.get_texts():
        tx.set_color(INK)
    note = (f"workload {w}; budget {budget}; line = mean, band = min-max over {nseeds} seeds; B7 = oracle with known shape counts (dashed). "
            f"Source: {split} split.")
    if ANNOT["unvalidated"]:
        note = (f"workload {w}; budget {budget}; line = mean, band = min-max over {nseeds} seeds; B7 (dashed) = knapsack with known shape counts, "
                f"not an upper bound. Source: {split} split.\n"
                f"† simulation not validated on real replays: total-time error 12.0-12.5% > 10% (one-time costs over-estimated; steady state within 1.5%).")
    if split == "dev":
        note = "DEVELOPMENT STREAMS (A half, seeds 0-4): tool check, not a result.  " + note
    if ANNOT["unvalidated"]:                       # annotated layout: wrap long notes so nothing is clipped or overlaps
        import textwrap
        wrapped = [ln2 for n_ in notes for ln2 in textwrap.wrap(n_, 230)]
        fig.text(0.01, 0.075, "\n".join(wrapped), fontsize=7.5, color=INK2, ha="left", va="bottom", linespacing=1.4)
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom", linespacing=1.4)
        fig.tight_layout(rect=(0, 0.33, 1, 1))
        leg.set_bbox_to_anchor((0.5, 0.25))
    else:
        fig.text(0.01, 0.03, "\n".join(notes), fontsize=7.5, color=INK2, ha="left", va="bottom", linespacing=1.4)
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.25, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(out.with_suffix("." + f), dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# ---- S5 P0 item 4 (added; the functions above are unchanged): paper-sized deterministic Fig.1 ------------------------
PAPER_OUT = "paper/paper/figs/fig1_main.pdf"
PAPER_SIZE = (7.16, 1.9)        # IEEEtran two-column full width x height budget (goals/S5 item 5.2 / task P0-4)


def draw_paper(t: pd.DataFrame, w: str, policies, out: pathlib.Path):
    """Full-width (7.16 x 1.9 in) W2 main figure for the paper: no in-figure notes (they go to the caption), fonts 6-7 pt,
    pdf.fonttype 42, no CreationDate, fixed font and fixed layout => byte-identical PDF on repeated runs (same environment).
    Returns a dict with the numbers a caption writer needs (clipped means)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": AXIS,
                         "text.color": INK, "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.2, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 3, figsize=PAPER_SIZE, sharey=True, facecolor="white")
    tw = t[t.workload == w]
    order = [p for p in policies if p in set(tw.variant)]
    colored = [p for p in policies if p != ORACLE]
    color = {p: SERIES[i % len(SERIES)] for i, p in enumerate(colored)}
    color[ORACLE] = INK2
    mark = {p: MARKERS[i % len(MARKERS)] for i, p in enumerate(colored)}
    mark[ORACLE] = "x"
    handles, labels, clipped = {}, {}, []
    ymin, ymax = 1.0, 1.0
    for ax, (cold, title) in zip(axes, PANELS):
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
            ax.spines[sp].set_linewidth(0.6)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)
        ax.set_xscale("log")
        sub = tw[tw.cold == cold]
        ax.axhline(1.0, color=MUTED, linewidth=0.6, zorder=1)
        for p in order:
            s = sub[sub.variant == p].sort_values("N")
            if s.empty:
                continue
            c = color[p]
            if p != ORACLE and len(s) > 1:
                ax.fill_between(s.N, s.lo, s.hi, color=c, alpha=0.14, linewidth=0, zorder=2)
            (ln,) = ax.plot(s.N, s["mean"], color=c, linewidth=1.1 if p == "Ours" else 0.8, marker=mark[p], markersize=2.6,
                            linestyle="--" if p == ORACLE else "-", zorder=4 if p == "Ours" else 3, markeredgewidth=0.6 if p == ORACLE else 0)
            handles[p], labels[p] = ln, disp(p) + ("\u2020" if p in ANNOT["unvalidated"] else "")
            ymin, ymax = min(ymin, float(s.lo.min())), max(ymax, float(s.hi.max()))
            clipped += [(cold, disp(p), int(r.N), float(r.mean)) for r in s.itertuples() if r.mean > YCAP]
        ax.set_title(title, color=INK, loc="left", pad=2)
        ns = sorted(tw.N.unique())
        ax.set_xticks(ns)
        ax.set_xticklabels([fmt_n(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("N (requests per process)", color=INK2, labelpad=1)
    top = min(ymax, YCAP)
    pad = 0.04 * (top - ymin + 1e-9)
    axes[0].set_ylim(ymin - pad, top + pad)
    axes[0].set_ylabel("Total service time / B0", color=INK2, labelpad=1)
    leg = fig.legend([handles[p] for p in order if p in handles], [labels[p] for p in order if p in handles], loc="upper center",
                     ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.3, handletextpad=0.5)
    for tx in leg.get_texts():
        tx.set_color(INK)
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.2, top=0.8, wspace=0.07)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None}, facecolor="white")
    plt.close(fig)
    return {"clipped_means": clipped, "ylim": (float(ymin - pad), float(top + pad))}


ROOT = pathlib.Path(__file__).resolve().parents[2]          # <project>/code/p24/this file -> <project>
FROZEN_RAW = ROOT / "results/S3/sim_eval/main_raw.csv"
TIME_COLS = {"sim_wall_s"}                                  # wall-clock columns of sim_eval_S3 raw CSVs: excluded from the numeric comparison


def compare_with_frozen(new: pathlib.Path, frozen: pathlib.Path = FROZEN_RAW, rtol: float = 1e-9) -> dict:
    """Row-by-row comparison of a re-run main_raw.csv with the frozen one (same key columns -> same rows); timing columns excluded."""
    a, b = pd.read_csv(new), pd.read_csv(frozen)
    cols = [c for c in b.columns if c not in TIME_COLS]
    res = dict(new=str(new), frozen=str(frozen), n_rows_new=len(a), n_rows_frozen=len(b), excluded_cols=sorted(TIME_COLS),
               same_columns=list(a.columns) == list(b.columns), max_abs_diff={}, n_mismatch_cells=0, mismatch_cols=[])
    if len(a) != len(b) or set(cols) - set(a.columns):
        res["match"] = False
        return res
    key = ["experiment", "split", "variant", "workload", "N", "seed", "budget", "cold", "prior_kind", "trace_kind", "boot"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    bad = 0
    for c in cols:
        if pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
            nan_same = (np.isnan(x) == np.isnan(y)).all()
            m = ~(np.isnan(x) | np.isnan(y))
            d = float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0
            ok = nan_same and bool(np.allclose(x[m], y[m], rtol=rtol, atol=1e-12))
            res["max_abs_diff"][c] = d
        else:
            ok = bool((a[c].fillna("").astype(str) == b[c].fillna("").astype(str)).all())
        if not ok:
            res["mismatch_cols"].append(c)
            bad += 1
    res["n_mismatch_cols"] = bad
    res["match"] = bad == 0
    return res


def run_simulation(out: pathlib.Path, procs: int) -> pathlib.Path:
    """Layer (b): the S3 3.3 `main` experiment on the frozen inputs, via the unchanged scripts/sim_eval_S3.py (subprocess; CPU only)."""
    import os
    import subprocess
    csv = out / "main_raw.csv"
    if csv.exists():
        sys.exit(f"{csv} exists (sim_eval_S3 refuses to overwrite; its outputs are 0444). Use a new --out, or `--from-csv {csv}` to just redraw.")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT / "code") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-B", str(ROOT / "code/scripts/sim_eval_S3.py"), "--split", "eval", "--experiment", "main", "--out", str(out),
           "--missing", "impute", "--procs", str(procs)]
    print("simulator:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return csv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-csv", "--raw", dest="raw", default=None,
                    help="layer (a): main_raw.csv written by scripts/sim_eval_S3.py (experiment main); --raw is the old name. Omit => layer (b) re-simulation")
    ap.add_argument("--out", default=None, help="output directory (default with a CSV: <csv dir>/figs; re-simulation: results/S4/reproduce)")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES), help="variant labels in legend/color order")
    ap.add_argument("--main-workload", default="W2")
    ap.add_argument("--appendix", default="W1,W3,W4,W5")
    ap.add_argument("--budget", default="default")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--unvalidated", default="", help="comma list of policies to mark with a dagger as not validated on real replays (S3: B1)")
    ap.add_argument("--procs", type=int, default=4, help="layer (b): simulator worker processes (1-8)")
    ap.add_argument("--paper", action="store_true",
                    help="S5 paper figure: full-width 7.16x1.9 in deterministic PDF of the W2 main figure (B1 dagger on unless --unvalidated is given); "
                         "writes only --paper-out, no PNG/CSV; needs --from-csv")
    ap.add_argument("--paper-out", default=None, help=f"with --paper: output PDF (default <project>/{PAPER_OUT})")
    ap.add_argument("--quick", action="store_true", help="layer (c): real-GPU scaled-down reproduction (scripts/reproduce_quick_S4.py)")
    ap.add_argument("--dry-run", action="store_true", help="with --quick: CPU-only plan + prediction, no GPU")
    a, rest = ap.parse_known_args()
    if a.quick:
        sys.path.insert(0, str(ROOT / "code"))
        from scripts import reproduce_quick_S4 as Q
        q = (["--dry-run"] if a.dry_run else []) + (["--out", a.out] if a.out else []) + rest
        sys.exit(Q.main(q))
    if rest:
        ap.error("unrecognised arguments: " + " ".join(rest))
    ANNOT["unvalidated"] = [x for x in a.unvalidated.split(",") if x]
    if a.paper:
        if not a.raw:
            ap.error("--paper needs --from-csv (it redraws from the frozen raw CSV only)")
        import os
        os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
        if not ANNOT["unvalidated"]:
            ANNOT["unvalidated"] = ["B1"]
        t = ratio_table(load(pathlib.Path(a.raw), a.budget))
        res = draw_paper(t, a.main_workload, a.policies.split(","), pathlib.Path(a.paper_out) if a.paper_out else ROOT / PAPER_OUT)
        print(f"wrote {a.paper_out or ROOT / PAPER_OUT}; clipped means (above y cap {YCAP}): {res['clipped_means']}; ylim={res['ylim']}")
        return
    if a.raw:
        raw = pathlib.Path(a.raw)
        out = pathlib.Path(a.out) if a.out else raw.parent / "figs"
    else:                                                       # layer (b): re-simulate, compare, draw
        base = pathlib.Path(a.out) if a.out else ROOT / "results/S4/reproduce"
        raw = run_simulation(base, a.procs)
        rep = compare_with_frozen(raw)
        (base / "compare_vs_frozen.json").write_text(__import__("json").dumps(rep, indent=1, ensure_ascii=False))
        print(f"compare with frozen {FROZEN_RAW.name}: match={rep['match']} rows {rep['n_rows_new']}/{rep['n_rows_frozen']} "
              f"mismatch_cols={rep['mismatch_cols']} -> {base / 'compare_vs_frozen.json'}", flush=True)
        out = base / "figs"
    out.mkdir(parents=True, exist_ok=True)
    d = load(raw, a.budget)
    t = ratio_table(d)
    split = str(d.split.iloc[0]) if "split" in d else "?"
    nseeds = int(t.n_seeds.max())
    policies = a.policies.split(",")
    t.to_csv(out / "main_figure_table.csv", index=False)
    formats = a.formats.split(",")
    draw(t, d, a.main_workload, policies, out / f"fig_main_{a.main_workload}", formats, a.dpi, split, nseeds, a.budget)
    for w in [x for x in a.appendix.split(",") if x]:
        if (t.workload == w).any():
            draw(t, d, w, policies, out / f"fig_app_{w}", formats, a.dpi, split, nseeds, a.budget)
        else:
            print(f"appendix {w}: no rows, skipped", file=sys.stderr)
    nm = d[d.status != "ok"].groupby(["variant", "cold"]).size()
    if len(nm):
        print("cells without value (status != ok):", {f"{k[0]}|{k[1]}": int(v) for k, v in nm.items()})
    print(f"wrote {out} (split={split}, seeds<= {nseeds})")


if __name__ == "__main__":
    main()
