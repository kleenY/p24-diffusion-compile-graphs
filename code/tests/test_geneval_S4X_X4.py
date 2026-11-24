"""S4-X X4 scripts: CPU-only tests (seed/prompt identity with S4, group registry, AOT preflight, analysis reproduces the frozen S4 verdict)."""
import json
import os
import pathlib

import pandas as pd
import pytest

from scripts import analyze_geneval_S4 as AN
from scripts import analyze_geneval_S4X_X4 as AX
from scripts import gen_geneval_S4 as G
from scripts import gen_geneval_S4X_X4 as X

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))


def test_registry_and_modes():
    assert X.GROUPS == ("AOT", "E0CFG", "C1G1CFG")
    assert X.GROUP_CFG == {"AOT": False, "E0CFG": True, "C1G1CFG": True}
    assert X.GROUP_GUIDANCE == {"AOT": 1.0, "E0CFG": 1.5, "C1G1CFG": 1.5}
    assert X.GROUP_ARM == {"AOT": "AOT", "E0CFG": "E0", "C1G1CFG": "C1+G1"}
    assert (G.H, G.W) == (512, 512)


def test_seed_rule_is_the_s4_rule():
    assert X.G.seed_for is G.seed_for and X.planned_seeds([0, 552])[(552, 3)] == 202612020000 + 100 * 552 + 3


@pytest.mark.skipif(not (X.S4_E0_DIR / "manifest.csv").exists(), reason="S4 E0 manifest absent")
def test_prompts_and_seeds_identical_to_s4_generation():
    v = X.verify_against_s4(G.load_metadata())
    assert v["identical"], v
    assert v["n_planned"] == 2212


def test_args_guard():
    with pytest.raises(SystemExit):
        X.parse_args(["--group", "PAD"])
    with pytest.raises(SystemExit):
        X.parse_args(["--group", "AOT", "--verify-s4"])
    with pytest.raises(SystemExit):
        X.parse_args(["--group", "AOT", "--samples", "3"])
    a = X.parse_args(["--group", "E0CFG", "--dry-run"])
    assert a.mem_cap_gb == 14.0 and a.out == X.DEFAULT_OUT


def test_aot_shape_tag_and_pkg_dir():
    from p24.pipeline import ShapeKey  # light import path of the project (torch import only, no CUDA)
    k = ShapeKey(512, 512, 1, False)
    assert k.tag() == "512x512_b1_nocfg" and X.aot_pkg_dir().name == k.tag()
    assert X.AOT_SHAPE_TAG == f"{k.H}x{k.W}x{k.B}x{int(k.cfg)}"


@pytest.mark.skipif(not X.AOT_PKG_ROOT.exists(), reason="AOT packages absent")
def test_aot_preflight_ok_on_real_packages():
    pf = X.aot_preflight()
    assert pf["ok"], pf


def test_aot_preflight_detects_missing(tmp_path):
    pf = X.aot_preflight(tmp_path / "nopkg")
    assert not pf["ok"] and set(pf["missing_components"]) == set(X.AOT_COMPONENTS)


def test_analysis_numbering_is_new_and_distinct():
    assert [p[2] for p in AX.PAIRS] == [3, 4] and AX.PAIRS[1][3] == 5
    assert not ({3, 4, 5} & {0, 1, 2})        # S4 streams: E0=0, C1G1=1, PAD=2


@pytest.mark.skipif(not (AX.S4_RES / "C1G1.jsonl").exists(), reason="S4 results absent")
def test_pair_machinery_reproduces_frozen_s4_verdict_c1g1():
    ref, grp = AN.load_results(AX.S4_RES / "E0.jsonl"), AN.load_results(AX.S4_RES / "C1G1.jsonl")
    r = AX.analyze_pair(ref, grp, group_no=1, ref_no=0, B=AN.B_BOOT)
    assert round(r["grp"]["delta"], 4) == 0.0 and round(r["grp"]["ci_lo"], 4) == -0.0037 and round(r["grp"]["ci_hi"], 4) == 0.0037
    assert round(r["ref"]["overall"], 4) == 0.3665 and r["grp"]["verdict"] == "UNCHANGED"


def test_synthetic_pipeline(tmp_path):
    AX.main(["--synthetic", str(tmp_path / "syn"), "--out", str(tmp_path / "o"), "--boot", "300"])
    t = (tmp_path / "o" / "geneval_S4X_X4_verdict_dryrun.txt").read_text()
    assert "NOT A RESULT" in t and "pair AOT vs E0" in t and "pair C1G1CFG vs E0CFG" in t
    df = pd.read_csv(tmp_path / "o" / "geneval_S4X_X4_summary_dryrun.csv")
    ov = df[df.metric == "Overall"]
    assert set(ov.verdict.dropna()) >= {"UNCHANGED", "NOT_SHOWN_UNCHANGED"}


def test_analysis_refuses_wrong_image_count():
    ref = AN.load_results(AX.S4_RES / "E0.jsonl") if (AX.S4_RES / "E0.jsonl").exists() else None
    if ref is None:
        pytest.skip("S4 results absent")
    with pytest.raises(ValueError):
        AX.analyze_pair(ref, ref.iloc[:-4], 3, 0, 50)
