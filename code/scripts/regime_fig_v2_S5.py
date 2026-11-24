"""S5 (2026-11-23): regime figure v2 = the frozen S4-X regime figure (scripts/regime_S4X.py `plot`, unchanged inputs) plus the
pre-registered real points obtained after it was drawn: X5 (W2, N=1e4, disk: Ours/B1), X8 (W2, N=1e3, empty: Ours/B0, B1/B0, Ours/B1)
and X9 (W2, N=1e4, disk: B1/B0, Ours/B0, B1-AOT/B0).  Values are read from the read-only verdict files, predictions from the frozen
prediction files; nothing is typed in.  The old figure (results/S4X/regime/fig_regime.pdf, 0444) and its script are not changed.
Output: results/S5/regime_v2/{regime_realpoints_add_S5.csv, fig_regime_v2.pdf}.
Usage: cd code; "$PROJ_PYTHON" -B -m scripts.regime_fig_v2_S5
"""
from __future__ import annotations
import json
import pandas as pd
from scripts.regime_S4X import OUT, ROOT, NS, WLS

OUT_V2 = ROOT / "results/S5/regime_v2"


def build_points():
    j8 = json.load(open(ROOT / "results/S4X/x8/verdict/h12_verdict.json"))["verdict_all_processes"]["T_mean"]
    j9 = json.load(open(ROOT / "results/S4X/x9/verdict/h123_verdict.json"))["verdict_all_processes"]
    p8 = pd.read_csv(ROOT / "results/S4X/x8/pred_X8.csv").set_index("policy")["pred_total_s"]
    p5 = pd.read_csv(ROOT / "results/S4X/x5/pred_X5.csv").set_index("policy")["pred_total_s"]
    t9 = j9["T_mean"]
    rows = []
    def add(src, cache, N, num, den, m, p):
        rows.append(dict(source=src, workload="W2", cache=cache, N=N, num=num, den=den, ratio_measured=m, ratio_pred=p,
                         same_side_of_1=int((m - 1) * (p - 1) > 0 or (abs(m - 1) < 1e-9 and abs(p - 1) < 1e-9))))
    add("X5 (seed133, N=10000)", "disk", 10000, "Ours", "B1", t9["Ours_X5"] / t9["B1_X5"], p5["Ours"] / p5["B1"])
    add("X8 (seed135, N=1000)", "empty", 1000, "Ours", "B1", j8["Ours"] / j8["B1"], p8["Ours"] / p8["B1"])
    add("X8 (seed135, N=1000)", "empty", 1000, "Ours", "B0", j8["Ours"] / j8["B0"], p8["Ours"] / p8["B0"])
    add("X8 (seed135, N=1000)", "empty", 1000, "B1", "B0", j8["B1"] / j8["B0"], p8["B1"] / p8["B0"])
    add("X9 (seed133, N=10000)", "disk", 10000, "B1", "B0", t9["B1_X5"] / t9["B0"], p5["B1"] / p5["B0"])
    add("X9 (seed133, N=10000)", "disk", 10000, "Ours", "B0", t9["Ours_X5"] / t9["B0"], p5["Ours"] / p5["B0"])
    add("X9 (seed133, N=10000)", "disk", 10000, "B1-AOT", "B0", t9["B1-AOT"] / t9["B0"], p5["B1-AOT"] / p5["B0"])
    df = pd.DataFrame(rows)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_V2 / "regime_realpoints_add_S5.csv", index=False)
    print(df.to_string(index=False))
    return df


def cmd_plot(a, OUT_V2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 7
    g = pd.read_csv(OUT / "regime_grid.csv")
    rp = pd.read_csv(OUT / "regime_realpoints.csv")
    ad = pd.read_csv(OUT_V2 / "regime_realpoints_add_S5.csv")
    COL = {"wide": "#1f77b4", "default": "k", "tight": "#d62728"}
    LAB = {"wide": "wide budget", "default": "default budget", "tight": "tight budget"}

    def series(world, cache, wl, policy, budget, col):
        x = g[(g.world == world) & (g.cache == cache) & (g.workload == wl) & (g.policy == policy) & (g.budget == budget)].sort_values("N")
        return x

    def panels(title):
        fig, axs = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=True, sharey=False)
        fig.suptitle(title, fontsize=9)
        return fig, axs

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    def fin(fig, axs, ylab, ticks, bot=0.07):
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                tk = ticks[cache]
                ax.axhline(1.0, color="0.5", lw=0.8, ls="-", zorder=0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(tk[0] / 1.15, tk[-1] * 1.15)
                ax.yaxis.set_major_locator(FixedLocator(tk))
                ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_title(f"{wl}, {'empty cache' if cache == 'empty' else 'disk cache'}", fontsize=7.5)
                ax.set_xticks(NS)
                ax.set_xticklabels([str(n) if n < 1000 else f"{n//1000}k" for n in NS], fontsize=6)
                ax.minorticks_off()
                if j == 0:
                    ax.set_ylabel(ylab)
                if i == 1:
                    ax.set_xlabel("requests per process N")
                ax.grid(alpha=0.25, lw=0.4)
        h, l = axs[1][0].get_legend_handles_labels()
        seen = {}
        for ax in axs.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(ll, hh)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, fontsize=6.5, frameon=False)
        fig.tight_layout(rect=(0, bot, 1, 0.95))

    with PdfPages(OUT_V2 / "fig_regime_v2.pdf") as pdf:
        # ---------- page 1: Ours / B1
        fig, axs = panels("Ours (reuse-aware, frozen S3 policy) / B1 (compile on first sight): total service time ratio; <1 = Ours faster")
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                for b in ("wide", "default", "tight"):
                    x = series("S3", cache, wl, "Ours", b, "ratio_B1_median")
                    ax.plot(x.N, x.ratio_B1_median, color=COL[b], lw=1.3 if b == "default" else 0.9, ls="-", marker="o", ms=2.5, label=f"sim S3 world, {LAB[b]}")
                    if b == "default":
                        ax.fill_between(x.N, x.ratio_B1_min, x.ratio_B1_max, color=COL[b], alpha=0.15, lw=0)
                if cache == "disk":
                    for b in ("wide", "default", "tight"):
                        x = series("V2", "disk", wl, "Ours", b, "ratio_B1_median")
                        ax.plot(x.N, x.ratio_B1_median, color=COL[b], lw=1.0 if b == "default" else 0.7, ls="--", marker="s", ms=2.0, label=f"sim V2 world (real-calibrated), {LAB[b]}")
                    pts = rp[(rp.workload == wl) & (rp.num == "Ours") & (rp.den == "B1")]
                    for src, mk in (("S3 3.4 real (seed100, N=300)", "o"), ("S4-P block C (seed101, N=300)", "D")):
                        q = pts[pts.source == src]
                        if len(q):
                            ax.scatter(q.N, q.ratio_measured, facecolors="none", edgecolors="m", marker=mk, s=28, lw=1.2, zorder=5, label=f"real 4090: {src.split(' (')[0]}")
                q = ad[(ad.cache == cache) & (ad.workload == wl) & (ad.num == "Ours") & (ad.den == "B1")]
                for _, r in q.iterrows():
                    ax.scatter([r.N], [r.ratio_measured], color="m", marker="*", s=60, lw=0.5, edgecolors="k", zorder=6, label="real 4090, pre-registered (X5 disk / X8 empty)")
        fin(fig, axs, "Ours / B1", {"empty": [0.04, 0.1, 0.2, 0.5, 1.0, 1.2], "disk": [0.2, 0.3, 0.5, 0.7, 1.0, 1.2]})
        pdf.savefig(fig)
        plt.close(fig)
        # ---------- page 2: everything / B0
        fig, axs = panels("Total service time relative to all-eager B0 (<1 = faster than B0).\nEmpty-cache row: B1/Ours from S3 world (imputed); AOT arms from V2 world with offline build time charged (cross-world, order of magnitude only).\nDisk row: B1 above 1.5 is off-scale (up to 5x at N=100)")
        STY = {"B1": ("#8c564b", "^", "-"), "Ours": ("k", "o", "-"), "B1-AOT": ("#2ca02c", "s", "-"), "B1-AOT-K": ("#ff7f0e", "D", "-"),
               "B1-AOT|cold": ("#2ca02c", "s", "-"), "B1-AOT-K|cold": ("#ff7f0e", "D", "-")}
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                if cache == "disk":
                    for p in ("B1", "Ours", "B1-AOT", "B1-AOT-K"):
                        x = series("V2", "disk", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim V2: {p}")
                    for p in ("B1", "Ours"):
                        x = series("S3", "disk", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.0, lw=0.7, ls="--", label=f"sim S3: {p}")
                    pts = rp[(rp.workload == wl) & (rp.den == "B0")]
                    for num, mk in (("B1", "^"), ("Ours", "o"), ("B1-AOT", "s"), ("B1-AOT-K", "D")):
                        q = pts[pts.num == num]
                        if len(q):
                            ax.scatter(q.N, q.ratio_measured, facecolors="none", edgecolors=STY[num][0], marker=mk, s=28, lw=1.2, zorder=5, label=f"real 4090: {num}")
                else:
                    for p in ("B1", "Ours"):
                        x = series("S3", "empty", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim S3 (imputed): {p}")
                    for p in ("B1-AOT|cold", "B1-AOT-K|cold"):
                        x = series("V2", "empty", wl, p, "default", "")
                        c, mk, ls = STY[p]
                        ax.plot(x.N, x.ratio_B0_median, color=c, marker=mk, ms=2.5, lw=1.1, ls=ls, label=f"sim V2: {p.split('|')[0]} + build time")
        for i, cache in enumerate(("empty", "disk")):
            for j, wl in enumerate(WLS):
                ax = axs[i][j]
                q = ad[(ad.cache == cache) & (ad.workload == wl) & (ad.den == "B0")]
                for _, r in q.iterrows():
                    ax.scatter([r.N], [r.ratio_measured], color=STY[r.num][0], marker="*", s=60, lw=0.5, edgecolors="k", zorder=6, label=f"real 4090, pre-registered (X8/X9): {r.num}")
        fin(fig, axs, "policy / B0", {"empty": [0.8, 1, 2, 5, 10, 20, 50, 100, 200], "disk": [0.75, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]}, bot=0.13)
        pdf.savefig(fig)
        plt.close(fig)
    print("wrote", OUT_V2 / "fig_regime_v2.pdf")


if __name__ == "__main__":
    build_points()
    cmd_plot(None, OUT_V2)
