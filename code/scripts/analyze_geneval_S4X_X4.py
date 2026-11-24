"""S4-X X4: paired GenEval analysis of the two supplementary comparisons (new file; analyze_geneval_S4.py stays frozen and its functions are imported unchanged).

Pairs (Delta = group - reference, paired by prompt, Overall = mean of the 6 task means):
  AOT      vs E0     (E0 = the S4 E0 results, results/S4/geneval/results/E0.jsonl, NOT regenerated; guidance 1.0 / no CFG)
  C1G1CFG  vs E0CFG  (both guidance 1.5 / CFG; new)

PRE-REGISTERED RULE = the S4 rule, unchanged (margin +-0.02, 10000 prompt-bootstrap resamples stratified by tag, 95% percentile CI, one-sided-equivalence style:
quality "unchanged" iff the CI of Delta-Overall lies COMPLETELY inside [-0.02, +0.02]; sub-task Deltas report-only).  Only the numbering of the bootstrap
rng streams is new (SeedSequence([20261202, no]); S4 used E0=0, C1G1=1, PAD=2): AOT=3, C1G1CFG=4, E0CFG (absolute CI)=5; the S4 E0 absolute CI keeps 0.
Both pairs are judged separately; no multiplicity adjustment (two pre-declared, non-overlapping questions).

Input : <results>/{AOT,E0CFG,C1G1CFG}.jsonl (official evaluate_images.py output via eval_geneval_S4.sh with IMG_ROOT/RES_ROOT of X4), S4 E0.jsonl.
Output: <analysis>/geneval_S4X_X4_summary.csv, geneval_S4X_X4_verdict.txt (suffix _dryrun with --synthetic).  Refuses when a group has != 2212 images or the prompt set/tags differ.
--synthetic <dir> : fake inputs (AOT ~1% flipped, C1G1CFG degraded) -> whole pipeline, outputs suffixed _dryrun; NOT a result.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import sys
from typing import Dict

import numpy as np
import pandas as pd

from scripts import analyze_geneval_S4 as AN

ROOT = AN.ROOT
S4_RES = ROOT / "results/S4/geneval/results"
X4_RES = ROOT / "results/S4X/x4/results"
X4_ANA = ROOT / "results/S4X/x4/analysis"
PAIRS = (("AOT", "E0", 3, 0), ("C1G1CFG", "E0CFG", 4, 5))     # (group, reference, group_no, reference_no)
N_IMAGES = 2212
RULE_TEXT = AN.RULE_TEXT.replace("RULE (pre-registered)", "RULE (pre-registered in S4, unchanged for S4-X X4)")


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def analyze_pair(ref: pd.DataFrame, grp: pd.DataFrame, group_no: int, ref_no: int, B: int) -> dict:
    for name, d in (("reference", ref), ("group", grp)):
        if len(d) != N_IMAGES:
            raise ValueError(f"{name}: {len(d)} images, expected {N_IMAGES}")
    AN.GROUP_NO["E0"], AN.GROUP_NO["G"] = ref_no, group_no      # used only to seed the bootstrap streams of AN.analyze
    r = AN.analyze({"E0": ref, "G": grp}, B)
    return {"ref": r["e0"], "grp": r["groups"]["G"]}


def write_outputs(res: Dict[str, dict], files: Dict[str, str], ana_dir: pathlib.Path, suffix: str, B: int, header: str) -> None:
    ana_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    L = [header, RULE_TEXT, f"bootstrap resamples B={B}; margin = +-{AN.MARGIN}", ""]
    for g, ref_name, gno, rno in PAIRS:
        r = res[g]
        e, d = r["ref"], r["grp"]
        rows.append(dict(pair=g, row=ref_name, metric="Overall", abs_score=e["overall"], delta=np.nan, ci_lo=e["ci_lo"], ci_hi=e["ci_hi"], verdict="",
                         note=f"{ref_name} absolute (prompt bootstrap of the score itself, stream no {rno})"))
        rows.append(dict(pair=g, row=g, metric="Overall", abs_score=d["overall"], delta=d["delta"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], verdict=d["verdict"], note=d["note"]))
        for t, (pt, lo, hi) in d["per_tag_delta"].items():
            rows.append(dict(pair=g, row=g, metric=t, abs_score=d["per_tag_abs"][t], delta=pt, ci_lo=lo, ci_hi=hi, verdict="report-only", note=""))
        L += [f"=== pair {g} vs {ref_name} (bootstrap stream no {gno}) ===",
              f"{ref_name} absolute: Overall = {e['overall']:.4f} (95% CI {e['ci_lo']:.4f} .. {e['ci_hi']:.4f}); prompts={e['n_prompts']}, images={e['n_images']}, "
              f"%images correct={e['pct_images_correct']:.2%}",
              f"{ref_name} per tag: " + ", ".join(f"{t}={v[0]:.3f}" for t, v in e["per_tag"].items()),
              f"[{g}] Overall = {d['overall']:.4f}; Delta vs {ref_name} = {d['delta']:+.4f}  95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  -> VERDICT: {d['verdict']} ({d['note']})",
              "     per-tag Delta (report only): " + ", ".join(f"{t} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for t, v in d["per_tag_delta"].items()),
              f"     paired images={d['n_pairs']}; discordant: {ref_name}-only-correct={d['discordant_e0_only']}, {g}-only-correct={d['discordant_group_only']}", ""]
    L.append("published SD1.5 GenEval value for comparison: " + AN.PUBLIC_SD15)
    L.append("input files (SHA256-16): " + "; ".join(f"{k}={v}" for k, v in files.items()))
    pd.DataFrame(rows).to_csv(ana_dir / f"geneval_S4X_X4_summary{suffix}.csv", index=False, float_format="%.5f")
    (ana_dir / f"geneval_S4X_X4_verdict{suffix}.txt").write_text("\n".join(L) + "\n")


def make_synthetic(dirpath: pathlib.Path) -> None:
    """S4's synthetic generator: E0 base, 'C1G1' ~1% flipped, 'PAD' degraded -> E0 / AOT / C1G1CFG; E0CFG = a different draw of the same base (E0 copy)."""
    tmp = dirpath / "_s4syn"
    AN.make_synthetic(tmp)
    dirpath.mkdir(parents=True, exist_ok=True)
    for src, dst in (("E0", "E0"), ("C1G1", "AOT"), ("E0", "E0CFG"), ("PAD", "C1G1CFG")):
        shutil.copy(tmp / f"{src}.jsonl", dirpath / f"{dst}.jsonl")
    shutil.rmtree(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s4-results", type=pathlib.Path, default=S4_RES, help="directory with the S4 E0.jsonl (reference of the AOT pair)")
    ap.add_argument("--results", type=pathlib.Path, default=X4_RES, help="directory with AOT/E0CFG/C1G1CFG .jsonl")
    ap.add_argument("--out", type=pathlib.Path, default=X4_ANA)
    ap.add_argument("--boot", type=int, default=AN.B_BOOT)
    ap.add_argument("--synthetic", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)
    suffix, header = "", ""
    if a.synthetic is not None:
        make_synthetic(a.synthetic)
        a.s4_results = a.results = a.synthetic
        suffix, header = "_dryrun", "*** DRY RUN ON SYNTHETIC DATA - NOT A RESULT ***"
    paths = {"E0": a.s4_results / "E0.jsonl", **{g: a.results / f"{g}.jsonl" for g in ("AOT", "E0CFG", "C1G1CFG")}}
    dfs = {k: AN.load_results(p) for k, p in paths.items()}
    res = {g: analyze_pair(dfs[ref], dfs[g], gno, rno, a.boot) for g, ref, gno, rno in PAIRS}
    write_outputs(res, {k: sha16(p) for k, p in paths.items()}, a.out, suffix, a.boot, header)
    print((a.out / f"geneval_S4X_X4_verdict{suffix}.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
