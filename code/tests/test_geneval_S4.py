"""S4 4.1 GenEval scripts: CPU-only tests (seed rule, request planning / padding keeps real samples only, bootstrap verdict)."""
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_geneval_S4 as AN
from scripts import gen_geneval_S4 as GEN

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))


# ------------------------------------------------------------------------------------------------ generation plan
def test_seed_rule_is_group_independent_and_injective():
    seeds = {(p, s): GEN.seed_for(p, s) for p in range(553) for s in range(4)}
    assert len(set(seeds.values())) == 553 * 4
    assert all(v >= GEN.SEED_BASE and v < 2 ** 62 for v in seeds.values())
    assert not (set(seeds.values()) & {GEN.PAD_SEED0 + j for j in range(8)})   # never collides with filler seeds


def test_all_groups_cover_same_images_with_same_seeds():
    cover = {}
    for g in GEN.GROUPS:
        covered = {}
        for r in GEN.plan_requests(g, 7, "a photo of x", 4):
            pr, sd = GEN.pad_batch(r.prompt, [GEN.seed_for(r.prompt_idx, s) for s in r.sample_idxs], r.served_B)
            assert len(pr) == len(sd) == r.served_B
            for row, s in enumerate(r.sample_idxs):
                covered[s] = sd[row]
        cover[g] = covered
    assert cover["E0"] == cover["C1G1"] == cover["PAD"] == {s: GEN.seed_for(7, s) for s in range(4)}


def test_pad_plan_shapes_and_fillers():
    reqs = GEN.plan_requests("PAD", 0, "p", 4, "pad")
    assert [(r.user_B, r.served_B, r.n_filler) for r in reqs] == [(3, 4, 1), (1, 4, 3)]
    reqs_n = GEN.plan_requests("PAD", 0, "p", 4, "native")
    assert [(r.user_B, r.served_B) for r in reqs_n] == [(3, 4), (1, 1)]
    for g in ("E0", "C1G1"):
        assert all(r.served_B == 1 and r.n_filler == 0 for r in GEN.plan_requests(g, 0, "p", 4))
    pr, sd = GEN.pad_batch("real", [11, 12, 13], 4)
    assert pr == ["real"] * 3 + [GEN.PAD_FILLER_PROMPT] and sd == [11, 12, 13, GEN.PAD_SEED0 + 3]
    pr, sd = GEN.pad_batch("real", [11], 4)
    assert sd == [11, GEN.PAD_SEED0 + 1, GEN.PAD_SEED0 + 2, GEN.PAD_SEED0 + 3]


def test_padding_keeps_only_real_rows():
    # fake runner output: row i filled with value i -> padded rows (index >= n_real) must never reach the saved set
    out = np.stack([np.full((3, 2, 2), float(i)) for i in range(4)])
    kept = GEN.take_real(out, 3)
    assert kept.shape[0] == 3 and set(np.unique(kept)) == {0.0, 1.0, 2.0}
    assert GEN.take_real(out, 1).shape[0] == 1


def test_filler_convention_matches_equivalence_scripts():
    pytest.importorskip("torch")
    from p24.bench import PROMPTS_DEFAULT
    assert GEN.PAD_FILLER_PROMPT == PROMPTS_DEFAULT[0]
    src = (ROOT / "code/scripts/run_equivalence.py").read_text()
    assert f"PAD_SEED0 = {GEN.PAD_SEED0}" in src


def test_select_prompts_limit_and_full():
    assert GEN.select_prompts(553, None) == list(range(553))
    sel = GEN.select_prompts(553, 12)
    assert sel == sorted(sel) and sel[0] == 0 and sel[-1] == 552 and len(sel) == 12
    assert GEN.select_prompts(553, 1) == [0]


def test_official_metadata_has_553_prompts():
    meta = GEN.load_metadata()
    assert len(meta) == 553 and all({"tag", "prompt"} <= set(m) for m in meta)


def test_dry_run_runs_without_gpu_and_lists_jobs(capsys, tmp_path):
    rc = GEN.main(["--dry-run", "--limit", "5", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "[PAD]" in out and "served-B counts={4: 10}" in out and "[E0]" in out and "requests=20 images=20" in out
    assert not list(tmp_path.iterdir())     # dry-run writes nothing


def test_resume_detection(tmp_path):
    r = GEN.plan_requests("PAD", 3, "p", 4)[0]
    assert not GEN.request_done(tmp_path, r)
    for s in (0, 1):
        p = GEN.img_path(tmp_path, "PAD", 3, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert not GEN.request_done(tmp_path, r)
    p = GEN.img_path(tmp_path, "PAD", 3, 2)
    p.write_bytes(b"x")
    assert GEN.request_done(tmp_path, r)
    assert GEN.img_path(tmp_path, "PAD", 3, 2).as_posix().endswith("PAD/00003/samples/00002.png")


# ------------------------------------------------------------------------------------------------ analysis
def _dfs(tmp_path):
    AN.make_synthetic(tmp_path)
    return {g: AN.load_results(tmp_path / f"{g}.jsonl") for g in AN.GROUPS}


def test_bootstrap_verdict_on_synthetic(tmp_path):
    res = AN.analyze(_dfs(tmp_path), B=2000)
    c, p = res["groups"]["C1G1"], res["groups"]["PAD"]
    assert c["verdict"] == "UNCHANGED" and -0.02 <= c["ci_lo"] <= c["ci_hi"] <= 0.02
    assert p["verdict"] == "NOT_SHOWN_UNCHANGED" and p["ci_hi"] < -0.02 and "degradation" in p["note"]
    assert set(p["pad_sensitivity"]) == {"samples_0-2 (B3->B4)", "sample_3 (B1->B4)"}
    assert 0.3 < res["e0"]["overall"] < 0.7


def test_verdict_rule_edges():
    assert AN.verdict_from_ci(-0.02, 0.02) == "UNCHANGED"
    assert AN.verdict_from_ci(-0.0201, 0.0) == "NOT_SHOWN_UNCHANGED"
    assert AN.verdict_from_ci(0.0, 0.0201) == "NOT_SHOWN_UNCHANGED"
    assert AN.verdict_from_ci(-0.05, 0.05) == "NOT_SHOWN_UNCHANGED"     # wide CI is NOT "unchanged"


def test_bootstrap_is_seeded_reproducible_and_group_dependent():
    tags = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    v = np.random.default_rng(0).normal(size=60)
    r1 = AN.bootstrap_overall(tags, v, 1, 500)
    r2 = AN.bootstrap_overall(tags, v, 1, 500)
    r3 = AN.bootstrap_overall(tags, v, 2, 500)
    assert r1[1:3] == r2[1:3] and (r1[3] == r2[3]).all()
    assert not np.array_equal(r1[3], r3[3])
    assert abs(r1[0] - np.mean([v[:30].mean(), v[30:].mean()])) < 1e-12


def test_overall_matches_official_summary_definition(tmp_path):
    dfs = _dfs(tmp_path)
    d = dfs["E0"]
    official = np.mean([t_df["correct"].mean() for _, t_df in d.groupby("tag", sort=False)])   # summary_scores.py definition
    pp = AN.per_prompt(d)
    mine = AN.overall_from_prompt_scores(pp["tag"].values.astype(object), pp["score"].values, AN.tag_order_of(pp["tag"].values))
    assert abs(official - mine) < 1e-12


def test_outputs_get_dryrun_suffix(tmp_path):
    rc = AN.main(["--synthetic", str(tmp_path / "syn"), "--out", str(tmp_path / "ana"), "--boot", "300"])
    assert rc == 0
    names = sorted(p.name for p in (tmp_path / "ana").iterdir())
    assert names == ["geneval_S4_summary_dryrun.csv", "geneval_S4_verdict_dryrun.txt"]
    txt = (tmp_path / "ana/geneval_S4_verdict_dryrun.txt").read_text()
    assert "DRY RUN ON SYNTHETIC DATA" in txt and "RULE (pre-registered)" in txt and "unverified" in txt


def test_mismatched_prompt_sets_rejected(tmp_path):
    dfs = _dfs(tmp_path)
    dfs["PAD"] = dfs["PAD"][dfs["PAD"].prompt_idx != 3]
    with pytest.raises(ValueError):
        AN.analyze(dfs, B=50)
