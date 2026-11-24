"""S1 figures from raw CSVs (re-generates everything from results/S1).
fig_s1_latency.png   steady-state latency per arm (median across processes, 95% bootstrap CI), one panel per shape key
fig_s1_overhead.png  per-request time split into GPU-busy and GPU-idle (host launch/dispatch overhead), from Nsight
Usage: python -m scripts.plot_s1 --bench results/S1/bench --nsys results/S1/nsys --out results/S1/figs
"""
import argparse, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from p24 import stats

BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated reference palette)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
ORDER = ["E0", "E1", "G1", "G2", "C1", "C2", "C3", "C1+G1", "C1+G2", "C5", "C6"]
LABEL = {"E0": "E0 eager", "E1": "E1 eager+CL+fQKV", "G1": "G1 graph/comp.", "G2": "G2 graph/whole",
         "C1": "C1 compile", "C2": "C2 reduce-overhead", "C3": "C3 max-autotune", "C1+G1": "C1+G1",
         "C1+G2": "C1+G2", "C5": "C5 dynamic", "C6": "C6 regional"}


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def latency(benches, out, include_contended=False):
    """benches: one or more measurement windows; stats (incl. break-even) are computed within each window,
    and an arm already present in an earlier window is not taken again from a later one (bridge arms)."""
    parts = []
    for i, b in enumerate(benches):
        x = stats.summarize(stats.load(b), include_contended).assign(window=pathlib.Path(b).name)
        if parts:
            x = x[~x.arm.isin(set(pd.concat(parts).arm))]
        parts.append(x)
    s = pd.concat(parts, ignore_index=True)
    if s.empty:
        raise SystemExit("no uncontended rows (use --include-contended for dev plots)")
    s.to_csv(out / "s1_summary.csv", index=False)
    keys = s[["H", "W", "B", "cfg"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), sharey=True)
    axes = [axes] if len(keys) == 1 else axes
    all_arms = [a for a in ORDER if a in set(s.arm)]  # common rows so shared y-labels stay correct
    pos = {a: i for i, a in enumerate(all_arms)}
    for ax, (H, W, B, cfg) in zip(axes, keys):
        g = s[(s.H == H) & (s.W == W) & (s.B == B) & (s.cfg == cfg)].set_index("arm")
        arms = [a for a in all_arms if a in g.index]
        y = [pos[a] for a in arms]
        med = g.loc[arms, "steady_wall_ms"]
        ax.hlines(list(y), 0, med, color=BLUE, lw=2)
        ax.errorbar(med, list(y), xerr=[med - g.loc[arms, "ci_lo"], g.loc[arms, "ci_hi"] - med],
                    fmt="o", ms=5, color=BLUE, ecolor=INK, elinewidth=1, capsize=2)
        for yi, a in zip(y, arms):
            ax.text(med[a], yi + 0.3, f"{med[a]:.0f}", fontsize=7, color=INK, ha="center")
        ax.set_yticks(range(len(all_arms)), [LABEL[a] for a in all_arms])
        if ax is axes[0]:
            ax.invert_yaxis()  # shared y: invert once
        ax.set_title(f"{H}x{W}, B={B}, {'CFG' if cfg else 'no CFG'}", fontsize=9, color=INK)
        ax.set_xlabel("steady-state latency per request (ms)", fontsize=8, color=MUTED)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_s1_latency.png", dpi=200)


def overhead(nsys_dirs, out):
    from p24 import nsys_analyze
    rows = []
    for nd in nsys_dirs:
        for rep in sorted(pathlib.Path(nd).glob("*.nsys-rep")):
            rows += nsys_analyze.analyze(rep)
    if not rows:
        return
    d = pd.DataFrame(rows)
    d.to_csv(out / "s1_nsys_requests.csv", index=False)
    g = d.groupby("report").agg(wall=("wall_ms", "median"), busy=("gpu_busy_ms", "median"),
                                kern=("n_kernels", "median"), launch=("n_launch_api", "median")).reset_index()
    g["idle"] = g.wall - g.busy
    g.to_csv(out / "s1_nsys_summary.csv", index=False)
    g["arm"] = g.report.str.rsplit("_", n=2).str[0]
    g["cfg"] = g.report.str.endswith("cfg1")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for ax, cfg in zip(axes, [False, True]):
        h = g[g.cfg == cfg].set_index("arm")
        arms = [a for a in ORDER if a in h.index]
        y = list(range(len(arms)))
        ax.barh(y, h.loc[arms, "busy"], color=BLUE, height=0.6, label="GPU busy", edgecolor="white", lw=2)
        ax.barh(y, h.loc[arms, "idle"], left=h.loc[arms, "busy"], color=ORANGE, height=0.6,
                label="GPU idle (host overhead)", edgecolor="white", lw=2)
        for yi, a in zip(y, arms):
            ax.text(h.loc[a, "wall"] + 2, yi, f"{h.loc[a, 'launch']:.0f} launches", va="center", fontsize=7, color=MUTED)
        ax.set_yticks(y, [LABEL[a] for a in arms])
        ax.invert_yaxis()
        ax.set_title(f"512x512, B=1, {'CFG' if cfg else 'no CFG'}", fontsize=9, color=INK)
        ax.set_xlabel("time per request (ms)", fontsize=8, color=MUTED)
        style(ax)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig_s1_overhead.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, nargs="+")
    ap.add_argument("--nsys", default=None, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-contended", action="store_true", help="dev only; never for reported figures")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    latency([pathlib.Path(b) for b in a.bench], out, a.include_contended)
    if a.nsys:
        overhead(a.nsys, out)
    print("figures ->", out)
