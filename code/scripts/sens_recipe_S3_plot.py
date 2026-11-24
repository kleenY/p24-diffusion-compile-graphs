"""Summary + figure for scripts/sens_recipe_S3.py (reads only; writes summary csv / png / checks into results/S3/accel/sensitivity/).
Total service time (service_s) relative to B0, per seed then mean over seeds, vs N. Ours vs Ours* (star table); B1 reference taken from the formal
evaluation main_raw.csv if present (read-only), else from the control run of the same script.  Also checks control == formal evaluation (Ours/B0/B1, W2)."""
import json, pathlib, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
D = ROOT / ("results/S3/accel/sensitivity" + ("/dev" if (len(sys.argv) > 1 and sys.argv[1] == "dev") else ""))
star = pd.read_csv(D / "ours_star_W2_raw.csv"); starc = pd.read_csv(D / "ours_star_W2_dconst_raw.csv"); ctrl = pd.read_csv(D / "ours_ctrl_W2_raw.csv")
formal_p = ROOT / "results/S3/sim_eval/main_raw.csv"
k = ["variant", "workload", "N", "seed", "cold"]
chk = {}
if formal_p.exists():
    f = pd.read_csv(formal_p); f = f[(f.workload == "W2") & (f.budget == "default") & f.variant.isin(["B0", "B1", "Ours"])]
    m = f.merge(ctrl, on=k, suffixes=("_f", "_c"))
    chk = dict(formal_main_raw=str(formal_p), n_formal_rows=len(f), n_ctrl_rows=len(ctrl), n_matched=len(m),
               max_abs_diff_service_s=float((m.service_s_f - m.service_s_c).abs().max()),
               max_abs_diff_total_incl_load_s=float((m.total_incl_load_s_f - m.total_incl_load_s_c).abs().max()),
               n_status_not_ok_formal=int((f.status != "ok").sum()))
    ref = f
else:
    ref = ctrl
b0 = ctrl[ctrl.variant == "B0"].set_index(["N", "seed", "cold"]).service_s
def rel(df, var):
    x = df[df.variant == var].set_index(["N", "seed", "cold"]).service_s
    return (x / b0).rename("r")
o = rel(ctrl, "Ours"); s = rel(star, "Ours"); b1 = rel(ctrl, "B1")
sc = rel(starc, "Ours")
o_sc = (starc[starc.variant == "Ours"].set_index(["N", "seed", "cold"]).service_s / ctrl[ctrl.variant == "Ours"].set_index(["N", "seed", "cold"]).service_s - 1) * 100
o_s = (star[star.variant == "Ours"].set_index(["N", "seed", "cold"]).service_s / ctrl[ctrl.variant == "Ours"].set_index(["N", "seed", "cold"]).service_s - 1) * 100
rows = []
for (cold, N), g in pd.DataFrame({"ours": o, "star": s, "b1": b1, "d_pct": o_s, "starc": sc, "dc_pct": o_sc}).groupby(level=["cold", "N"]):
    rows.append(dict(cold=cold, N=N, ours_over_B0=g.ours.mean(), star_over_B0=g.star.mean(), B1_over_B0=g.b1.mean(),
                     star_vs_ours_pct_mean=g.d_pct.mean(), star_vs_ours_pct_min=g.d_pct.min(), star_vs_ours_pct_max=g.d_pct.max(),
                     star_const_over_B0=g.starc.mean(), star_const_vs_ours_pct_mean=g.dc_pct.mean(), star_const_vs_ours_pct_min=g.dc_pct.min(), star_const_vs_ours_pct_max=g.dc_pct.max(), n_seeds_const_worse=int((g.dc_pct > 0).sum()), n_seeds_star_better=int((g.d_pct < 0).sum()), n_seeds_star_worse=int((g.d_pct > 0).sum()), n_seeds_equal=int((g.d_pct == 0).sum()), n_seeds=len(g)))
sm = pd.DataFrame(rows); sm.to_csv(D / "summary_star_vs_ours.csv", index=False)
fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
for j, (cold, ttl) in enumerate(zip(["empty", "disk", "hot"], ["cold level: empty compile cache (impute)", "cold level: disk cache pre-warmed (H4 setting)", "cold level: in-process warm (hot)"])):
    g = sm[sm.cold == cold].sort_values("N"); ax = axs[0, j]
    for col, lab, st in (("ours_over_B0", "Ours (frozen v2)", "o-"), ("star_over_B0", "Ours★ prop (d ~ steady)", "s--"), ("star_const_over_B0", "Ours★ const (d = median)", "^-."), ("B1_over_B0", "B1 (reference; off-scale at small N)", ":")):
        ax.plot(g.N, g[col], st, label=lab, lw=1.8)
    ax.set_xscale("log"); ax.set_ylim(0.8, 1.1); ax.set_xlabel("N (requests)"); ax.set_ylabel("total service time / B0"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
    ax = axs[1, j]
    ax.plot(g.N, g.star_vs_ours_pct_mean, "s-", color="C1", label="prop (d ~ steady), mean over seeds")
    ax.fill_between(g.N, g.star_vs_ours_pct_min, g.star_vs_ours_pct_max, color="C1", alpha=.2, label="prop seed min-max")
    ax.plot(g.N, g.star_const_vs_ours_pct_mean, "^-.", color="C3", label="const (d = median), mean over seeds")
    ax.fill_between(g.N, g.star_const_vs_ours_pct_min, g.star_const_vs_ours_pct_max, color="C3", alpha=.2, label="const seed min-max")
    ax.axhline(0, color="k", lw=.8); ax.set_xscale("log"); ax.set_xlabel("N (requests)"); ax.set_ylabel("Ours★ vs Ours, service time (%)"); ax.grid(alpha=.3)
axs[0, 1].legend(fontsize=8); axs[1, 1].legend(fontsize=8)
fig.suptitle("W2 panel, default budget, mean over %d seeds (%s split); positive = Ours★ slower" % (sm.n_seeds.iloc[0], "dev" if D.name == "dev" else "eval"), fontsize=11)
fig.text(0.5, 0.005, "★ = official Inductor recipe, unmeasured shapes extrapolated; sensitivity, not a main result", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig(D / "fig_ours_star_W2.png", dpi=130)
(D / "control_check.json").write_text(json.dumps(chk, indent=1))
pd.set_option("display.width", 250); print(sm.round(4).to_string()); print(chk)
