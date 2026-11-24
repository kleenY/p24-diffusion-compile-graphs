"""S4 step 4.1: paired GenEval analysis of the three groups (E0 reference, C1G1, PAD).

PRE-REGISTERED RULE (written by the main session before any GenEval number exists; copied verbatim into every output):
  * Unit = prompt.  Per prompt and group: score_p = mean over the 4 samples of the official `correct` flag.
  * Overall score = mean over the 6 GenEval task tags of (mean of score_p over the prompts of that tag)
    (= the official summary_scores.py "Overall score (avg. over tasks)" when every prompt has the same number of samples).
  * Delta = group - E0 (paired by prompt), for the Overall score.
  * Bootstrap: 10000 resamples of PROMPTS WITH REPLACEMENT, stratified by task tag (the six tags keep their prompt counts, the Overall
    metric is always defined), rng = numpy.random.default_rng(SeedSequence([20261202, group_no])), group_no: E0=0, C1G1=1, PAD=2.
    95% interval = percentile [2.5, 97.5] of the bootstrap Delta.
  * A group is "quality unchanged" iff the 95% CI of Delta-Overall lies COMPLETELY inside [-0.02, +0.02].  Otherwise the group is
    "NOT shown unchanged" (an extra descriptive note says whether the CI is wholly below -0.02 = degradation, or just too wide).
  * Sub-task (tag) Deltas are REPORTED ONLY, never judged.  E0 absolute scores are reported as a pipeline sanity check next to the
    published SD1.5 GenEval value, which is NOT FILLED IN here (marked "unverified - look up the paper table and cite").
Extras (not part of the verdict): number of discordant images (E0 correct vs group correct), and for PAD a sensitivity line on samples
0-2 only (the B3->B4 padded requests, equivalence-tested in S2) versus sample 3 (B1->B4, an extension of the S2-tested path).

Input : <results>/<GROUP>.jsonl  (official evaluate_images.py output; the image path ends in <prompt:05d>/samples/<sample:05d>.png)
Output: <analysis>/geneval_S4_summary.csv, geneval_S4_verdict.txt  (file names get `_dryrun` with --synthetic)
--synthetic : writes fake result files (E0 base, C1G1 = ~1% flipped, PAD = degraded) to a scratch dir and runs the whole pipeline on them.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
RES_DIR = ROOT / "results/S4/geneval/results"
ANA_DIR = ROOT / "results/S4/geneval/analysis"
GROUPS = ("E0", "C1G1", "PAD")
GROUP_NO = {"E0": 0, "C1G1": 1, "PAD": 2}
B_BOOT = 10000
MARGIN = 0.02
SEED_TAG = 20261202
RULE_TEXT = ("RULE (pre-registered): quality unchanged iff the 95% percentile CI (10000 prompt-bootstrap resamples, stratified by tag, "
             "SeedSequence([20261202, group_no])) of Delta-Overall (group - E0, paired by prompt) lies completely inside [-0.02, +0.02]. "
             "Sub-task Deltas are reported only, not judged.")
PATH_RE = re.compile(r"(\d+)/samples/(\d+)\.png$")
PUBLIC_SD15 = "unverified (public SD1.5 GenEval value not filled in; look up the GenEval/SD3-era papers and cite, then compare)"


def load_results(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_json(path, orient="records", lines=True)
    m = df["filename"].astype(str).str.extract(PATH_RE)
    if m.isna().any().any():
        raise ValueError(f"{path}: cannot parse prompt/sample index from filename")
    df["prompt_idx"], df["sample_idx"] = m[0].astype(int), m[1].astype(int)
    df["correct"] = df["correct"].astype(bool)
    return df[["prompt_idx", "sample_idx", "tag", "correct"]]


def per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """-> one row per prompt: tag, score (mean correct over its samples), n."""
    g = df.groupby("prompt_idx").agg(tag=("tag", "first"), score=("correct", "mean"), n=("correct", "size"))
    return g


def overall_from_prompt_scores(tags: np.ndarray, scores: np.ndarray, tag_order: List[str]) -> float:
    return float(np.mean([scores[tags == t].mean() for t in tag_order]))


def tag_order_of(tags: np.ndarray) -> List[str]:
    seen: List[str] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen


def bootstrap_overall(tags: np.ndarray, vals: np.ndarray, group_no: int, B: int = B_BOOT, seed_tag: int = SEED_TAG):
    """Stratified (by tag) prompt bootstrap of the Overall mean of `vals` (per-prompt scores or per-prompt Deltas).
    Returns (point, lo, hi, boot_array, per_tag dict tag -> (point, lo, hi))."""
    rng = np.random.default_rng(np.random.SeedSequence([seed_tag, group_no]))
    order = tag_order_of(tags)
    boot_tags = []
    per_tag = {}
    for t in order:
        v = vals[tags == t]
        idx = rng.integers(0, len(v), size=(B, len(v)))
        bt = v[idx].mean(axis=1)
        boot_tags.append(bt)
        per_tag[t] = (float(v.mean()),) + tuple(float(x) for x in np.percentile(bt, [2.5, 97.5]))
    boot = np.mean(boot_tags, axis=0)
    point = float(np.mean([vals[tags == t].mean() for t in order]))
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return point, lo, hi, boot, per_tag


def verdict_from_ci(lo: float, hi: float, margin: float = MARGIN) -> str:
    return "UNCHANGED" if (lo >= -margin and hi <= margin) else "NOT_SHOWN_UNCHANGED"


def describe(lo: float, hi: float, margin: float = MARGIN) -> str:
    if verdict_from_ci(lo, hi, margin) == "UNCHANGED":
        return "CI wholly inside the +-0.02 margin"
    if hi < -margin:
        return "CI wholly below -0.02 (degradation beyond the margin)"
    if lo > margin:
        return "CI wholly above +0.02 (difference beyond the margin, improvement)"
    return "CI crosses a margin edge (not enough evidence for equivalence; not necessarily a degradation)"


def analyze(dfs: Dict[str, pd.DataFrame], B: int = B_BOOT, margin: float = MARGIN) -> dict:
    pp = {g: per_prompt(d) for g, d in dfs.items()}
    base = pp["E0"]
    out = {"groups": {}, "e0": {}}
    for g, p in pp.items():
        if not p.index.equals(base.index):
            raise ValueError(f"{g}: prompt set differs from E0 ({len(p)} vs {len(base)} prompts)")
        if not (p["tag"].values == base["tag"].values).all():
            raise ValueError(f"{g}: tags differ from E0")
    tags = base["tag"].values.astype(object)
    # E0 absolute
    pt, lo, hi, _, per_tag = bootstrap_overall(tags, base["score"].values, GROUP_NO["E0"], B)
    out["e0"] = {"overall": pt, "ci_lo": lo, "ci_hi": hi, "per_tag": per_tag, "n_prompts": len(base),
                 "n_images": int(base["n"].sum()), "pct_images_correct": float(dfs["E0"]["correct"].mean())}
    for g in dfs:
        if g == "E0":
            continue
        delta = pp[g]["score"].values - base["score"].values
        pt, lo, hi, _, per_tag = bootstrap_overall(tags, delta, GROUP_NO[g], B)
        ab_pt = overall_from_prompt_scores(tags, pp[g]["score"].values, tag_order_of(tags))
        # discordant images (paired by prompt+sample)
        m = dfs["E0"].merge(dfs[g], on=["prompt_idx", "sample_idx"], suffixes=("_e0", "_g"))
        e0_only, g_only = int((m.correct_e0 & ~m.correct_g).sum()), int((~m.correct_e0 & m.correct_g).sum())
        res = {"overall": ab_pt, "delta": pt, "ci_lo": lo, "ci_hi": hi, "verdict": verdict_from_ci(lo, hi, margin),
               "note": describe(lo, hi, margin), "per_tag_delta": per_tag, "n_pairs": len(m),
               "discordant_e0_only": e0_only, "discordant_group_only": g_only,
               "per_tag_abs": {t: float(pp[g]["score"].values[tags == t].mean()) for t in tag_order_of(tags)}}
        if g == "PAD":  # sensitivity: samples 0-2 (B3->B4) vs sample 3 (B1->B4); not part of the verdict
            sens = {}
            for name, sel in (("samples_0-2 (B3->B4)", lambda d: d.sample_idx <= 2), ("sample_3 (B1->B4)", lambda d: d.sample_idx == 3)):
                a = {k: per_prompt(d[sel(d)]) for k, d in (("E0", dfs["E0"]), (g, dfs[g]))}
                dl = a[g]["score"].values - a["E0"]["score"].values
                p_, l_, h_, _, _ = bootstrap_overall(a["E0"]["tag"].values.astype(object), dl, GROUP_NO[g], B)
                sens[name] = {"delta": p_, "ci_lo": l_, "ci_hi": h_}
            res["pad_sensitivity"] = sens
        out["groups"][g] = res
    return out


def write_outputs(res: dict, ana_dir: pathlib.Path, suffix: str, B: int, extra_header: str = "") -> None:
    ana_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    e0 = res["e0"]
    rows.append(dict(group="E0", metric="Overall", abs_score=e0["overall"], delta=np.nan, ci_lo=e0["ci_lo"], ci_hi=e0["ci_hi"],
                     verdict="", note="E0 absolute; CI = prompt bootstrap of the score itself"))
    for t, (pt, lo, hi) in e0["per_tag"].items():
        rows.append(dict(group="E0", metric=t, abs_score=pt, delta=np.nan, ci_lo=lo, ci_hi=hi, verdict="", note="E0 absolute"))
    for g, r in res["groups"].items():
        rows.append(dict(group=g, metric="Overall", abs_score=r["overall"], delta=r["delta"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                         verdict=r["verdict"], note=r["note"]))
        for t, (pt, lo, hi) in r["per_tag_delta"].items():
            rows.append(dict(group=g, metric=t, abs_score=r["per_tag_abs"][t], delta=pt, ci_lo=lo, ci_hi=hi, verdict="report-only", note=""))
        for name, s in r.get("pad_sensitivity", {}).items():
            rows.append(dict(group=g, metric=f"sensitivity:{name}", abs_score=np.nan, delta=s["delta"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                             verdict="report-only", note="not part of the verdict"))
    pd.DataFrame(rows).to_csv(ana_dir / f"geneval_S4_summary{suffix}.csv", index=False, float_format="%.5f")
    L = [extra_header, RULE_TEXT, f"bootstrap resamples B={B}; margin = +-{MARGIN}", "",
         f"E0 absolute: Overall = {e0['overall']:.4f} (95% CI {e0['ci_lo']:.4f} .. {e0['ci_hi']:.4f}); prompts={e0['n_prompts']}, "
         f"images={e0['n_images']}, %images correct={e0['pct_images_correct']:.2%}",
         "E0 per tag: " + ", ".join(f"{t}={v[0]:.3f}" for t, v in e0["per_tag"].items()),
         f"published SD1.5 GenEval value for comparison: {PUBLIC_SD15}", ""]
    for g, r in res["groups"].items():
        L.append(f"[{g}] Overall = {r['overall']:.4f}; Delta vs E0 = {r['delta']:+.4f}  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                 f"-> VERDICT: {r['verdict']} ({r['note']})")
        L.append("     per-tag Delta (report only): " + ", ".join(f"{t} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
                                                                for t, v in r["per_tag_delta"].items()))
        L.append(f"     paired images={r['n_pairs']}; discordant: E0-only-correct={r['discordant_e0_only']}, "
                 f"{g}-only-correct={r['discordant_group_only']}")
        for name, s in r.get("pad_sensitivity", {}).items():
            L.append(f"     sensitivity (not judged) {name}: Delta {s['delta']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]")
    (ana_dir / f"geneval_S4_verdict{suffix}.txt").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------ synthetic data
TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
TAG_N = {"single_object": 80, "two_object": 99, "counting": 80, "colors": 94, "position": 100, "color_attr": 100}


def make_synthetic(dirpath: pathlib.Path, seed: int = 7) -> None:
    """E0 ~ per-prompt success probability; C1G1 flips ~1% of images; PAD is degraded (-0.15 on success probability)."""
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    rows, pidx = [], 0
    base = {}
    for t in TAGS:
        for _ in range(TAG_N[t]):
            p = rng.uniform(0.1, 0.95)
            base[pidx] = (t, p, rng.random(4) < p)
            pidx += 1
    def dump(name, fn):
        recs = []
        for i, (t, p, c) in base.items():
            cc = fn(i, p, c)
            for s in range(4):
                recs.append({"filename": f"/x/images/{name}/{i:05d}/samples/{s:05d}.png", "tag": t, "prompt": "p", "correct": bool(cc[s]),
                             "reason": "", "metadata": "{}", "details": "{}"})
        pd.DataFrame(recs).to_json(dirpath / f"{name}.jsonl", orient="records", lines=True)
    dump("E0", lambda i, p, c: c)
    r1 = np.random.default_rng(seed + 1)
    dump("C1G1", lambda i, p, c: np.where(r1.random(4) < 0.01, ~c, c))
    r2 = np.random.default_rng(seed + 2)
    dump("PAD", lambda i, p, c: r2.random(4) < max(p - 0.15, 0.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=RES_DIR)
    ap.add_argument("--out", type=pathlib.Path, default=ANA_DIR)
    ap.add_argument("--boot", type=int, default=B_BOOT)
    ap.add_argument("--synthetic", type=pathlib.Path, default=None,
                    help="scratch dir: write fake <GROUP>.jsonl there, analyse them, outputs get the suffix _dryrun")
    a = ap.parse_args(argv)
    suffix, header = "", ""
    if a.synthetic is not None:
        make_synthetic(a.synthetic)
        a.results, suffix, header = a.synthetic, "_dryrun", "*** DRY RUN ON SYNTHETIC DATA - NOT A RESULT ***"
    dfs = {g: load_results(a.results / f"{g}.jsonl") for g in GROUPS}
    res = analyze(dfs, a.boot)
    write_outputs(res, a.out, suffix, a.boot, header)
    print((a.out / f"geneval_S4_verdict{suffix}.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
