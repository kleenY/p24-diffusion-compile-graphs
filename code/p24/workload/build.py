"""One-shot generation of the S2 2.4 artifacts: split file, development traces, validation streams, seed table, statistics report.

Usage (project root = PROJ_ROOT):
    export CUDA_VISIBLE_DEVICES=''
    "$PROJ_PYTHON" -B -m p24.workload.build
No evaluation traces are generated (half B is used only for statistics and the split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import generate as G
from .split import (RESOLUTIONS, half_by_prompt, half_by_user, map_resolutions, res_index,
                    user_missing_mask)

ROOT = Path(".")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(r) -> str:
    H, W = r
    return f"H{H}xW{W}"


def res_dist(hw: np.ndarray) -> dict[str, float]:
    c = np.bincount(res_index(hw), minlength=4)
    return {label(r): float(c[i] / c.sum()) for i, r in enumerate(RESOLUTIONS)}


def tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p[k] - q[k]) for k in p)


def half_stats(df, half_arr, hw, code, users_ok):
    m = half_arr == code
    n = int(m.sum())
    u = df["user_name"].to_numpy()[m]
    ok = users_ok[m]
    vc = pd.Series(u[ok]).value_counts()
    return {
        "rows": n,
        "users": int(len(vc)),
        "rows_with_missing_user": int((~ok).sum()),
        "top10_user_row_share": [float(x) for x in (vc.head(10) / n).round(6)],
        "top10_users_total_share": float(vc.head(10).sum() / n),
        "shape_dist": res_dist(hw[m]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = root / "results/S2/workload"
    (out / "dev_traces").mkdir(parents=True, exist_ok=True)
    (out / "validation_streams").mkdir(parents=True, exist_ok=True)

    meta_path = root / "data/diffusiondb/metadata.parquet"
    df = pd.read_parquet(meta_path, columns=["prompt", "width", "height", "user_name"])
    nrows = len(df)
    hw = map_resolutions(df["width"].to_numpy(), df["height"].to_numpy())
    miss = user_missing_mask(df)
    users_ok = ~miss

    hu = half_by_user(df)
    hp = half_by_prompt(df)

    # split file
    split_df = pd.DataFrame({"row": np.arange(nrows, dtype=np.int32),
                             "half": np.where(hu == 0, "A", "B")})
    split_path = out / "split_A_B.parquet"
    split_df.to_parquet(split_path, index=False, compression="zstd")
    (out / "split_A_B.parquet.sha256").write_text(f"{sha256_file(split_path)}  split_A_B.parquet\n")

    # statistics
    w = df["width"].to_numpy().astype(np.int64); h = df["height"].to_numpy().astype(np.int64)
    a = w / h
    stats = {
        "metadata_sha256": sha256_file(meta_path),
        "total_rows": nrows,
        "total_users": int(df.loc[users_ok, "user_name"].nunique()),
        "missing_user_rows": int(miss.sum()),
        "missing_user_share": float(miss.mean()),
        "raw_640x640_share": float(((w == 640) & (h == 640)).mean()),
        "raw_area_le_640sq_share": float(((w * h) <= 640 * 640).mean()),
        "raw_area_le_768sq_share": float(((w * h) <= 768 * 768).mean()),
        "raw_aspect_share": {"a<0.9": float((a < 0.9).mean()), "0.9<=a<=1.1": float(((a >= 0.9) & (a <= 1.1)).mean()),
                              "a>1.1": float((a > 1.1).mean())},
        "mapped_shape_dist_overall": res_dist(hw),
        "split_by_user": {"A": half_stats(df, hu, hw, 0, users_ok), "B": half_stats(df, hu, hw, 1, users_ok)},
        "split_by_prompt": {"A": half_stats(df, hp, hw, 0, users_ok), "B": half_stats(df, hp, hw, 1, users_ok)},
    }
    for key in ("split_by_user", "split_by_prompt"):
        stats[key]["tv_distance_A_vs_B"] = tv(stats[key]["A"]["shape_dist"], stats[key]["B"]["shape_dist"])
    stats["split_agreement_user_vs_prompt"] = float((hu == hp).mean())

    # pools and scenarios
    poolA = G.make_pool(df, hu, "A")
    probsA = G.shape_probs(poolA)
    vstats = {}
    manifest = {}
    for name in ("V1", "V2", "V3"):
        reqs, shapes, p = G.generate_validation(poolA, name)
        fn = out / "validation_streams" / f"{name}_N{len(reqs)}_seed{G.VALIDATION[name]}_A.jsonl"
        fn.write_bytes(G.stream_bytes(reqs))
        manifest[str(fn.relative_to(out))] = sha256_file(fn)
        vstats[name] = {"n": len(reqs), "seed": G.VALIDATION[name],
                        "shapes": [{"H": s[0], "W": s[1], "B": s[2], "cfg": s[3],
                                    "prob_normalized": pp, "prob_raw": probsA[s]} for s, pp in zip(shapes, p)],
                        "has_B3": any(s[2] == 3 for s in shapes),
                        "raw_prob_mass_covered": float(sum(probsA[s] for s in shapes))}
    stats["validation_streams"] = vstats
    stats["A_mapped_shape_dist"] = res_dist(poolA.hw)

    for sc in G.SCENARIOS:
        for seed in G.DEV_SEEDS:
            reqs = G.generate_scenario(poolA, sc, args.n, seed)
            fn = out / "dev_traces" / f"{sc}_N{args.n}_seed{seed}_A.jsonl"
            fn.write_bytes(G.stream_bytes(reqs))
            manifest[str(fn.relative_to(out))] = sha256_file(fn)
    (out / "sha256_manifest.txt").write_text("".join(f"{v}  {k}\n" for k, v in sorted(manifest.items())))

    seeds = {
        "dev_seeds": {"values": G.DEV_SEEDS,
                      "use": "development/prior: policy tuning, validation streams, development traces; drawn from half A only"},
        "eval_seeds": {"values": G.EVAL_SEEDS,
                       "use": "evaluation: drawn from half B only; no evaluation traces are generated during S2, seeds are only registered"},
        "validation_stream_seeds": G.VALIDATION,
        "dev_trace_N": args.n,
    }
    (out / "seeds.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=2) + "\n")
    stats["declared_distribution"] = {"B": dict(zip(map(str, G.B_VALUES), G.B_PROBS)), "cfg_on": G.CFG_PROB,
                                      "note": "declared, not measured"}
    (out / "workload_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    write_md(out / "workload_stats.md", stats)
    print("done; rows", nrows, "A", stats["split_by_user"]["A"]["rows"], "B", stats["split_by_user"]["B"]["rows"])


def write_md(path: Path, s: dict):
    L = ["# Workload statistics (S2 2.4, auto-generated)", ""]
    L.append(f"- total rows {s['total_rows']}, users {s['total_users']}, empty user_name {s['missing_user_rows']} rows ({s['missing_user_share']:.4%})")
    L.append(f"- raw 640x640 share {s['raw_640x640_share']:.4%}; area <=640^2 share {s['raw_area_le_640sq_share']:.2%}; area <=768^2 share {s['raw_area_le_768sq_share']:.2%}")
    L.append(f"- overall share of the 4 mapped resolutions: " + ", ".join(f"{k} {v:.2%}" for k, v in s["mapped_shape_dist_overall"].items()))
    L.append("")
    for key, nm in (("split_by_user", "split by user (primary)"), ("split_by_prompt", "split by prompt (alternative, comparison only)")):
        L.append(f"## {nm}")
        for hf in "AB":
            d = s[key][hf]
            L.append(f"- {hf}: {d['rows']} rows, {d['users']} users, {d['rows_with_missing_user']} rows with empty user; top-10 users' row share " +
                     ", ".join(f"{x:.2%}" for x in d["top10_user_row_share"]) + f" (total {d['top10_users_total_share']:.2%})")
            L.append("  shape distribution: " + ", ".join(f"{k} {v:.2%}" for k, v in d["shape_dist"].items()))
        L.append(f"- A/B shape-distribution total variation distance: {s[key]['tv_distance_A_vs_B']:.5f}")
        L.append("")
    L.append(f"fraction of rows with identical assignment under the two splits: {s['split_agreement_user_vs_prompt']:.2%}")
    L.append("")
    for n in ("V2", "V3"):
        v = s["validation_streams"][n]
        L.append(f"## {n} (N={v['n']}, seed {v['seed']}, contains B=3: {v['has_B3']}, raw probability mass covered {v['raw_prob_mass_covered']:.3f})")
        L.append("| H | W | B | cfg | normalized probability | raw probability |\n|---|---|---|---|---|---|")
        for x in v["shapes"]:
            L.append(f"| {x['H']} | {x['W']} | {x['B']} | {x['cfg']} | {x['prob_normalized']:.4f} | {x['prob_raw']:.4f} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
