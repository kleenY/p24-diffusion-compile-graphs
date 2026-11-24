"""Tests for the P0-4 paper figure/table scripts (S5): determinism and SHA-unchanged checks.  CPU only, no torch.

Run (about 40 s):
  export CUDA_VISIBLE_DEVICES=''; cd code; "$PROJ_PYTHON" -B -m pytest -p no:cacheprovider tests/test_paper_figs.py
"""
import difflib
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from p24.paper import build as bld
from p24.paper import build_all, check as chk, fig_cost, tab_metrics, tab_policy
from p24.paper import style as st

ROOT = st.ROOT
RAW = ROOT / "results/S3/sim_eval/main_raw.csv"
MAIN_PKG = ROOT / "code/p24/reproduce_main_figure.py"
PRE = ROOT / "code/p24/reproduce_main_figure_v_preS5.py"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", SOURCE_DATE_EPOCH="1700000000",
           PYTHONPATH=str(ROOT / "code"))


def sha16(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def run_fig(tmp: Path, *args) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-B", "-m", "p24.reproduce_main_figure", "--from-csv", str(RAW), *args]
    return subprocess.run(cmd, check=True, env=ENV, cwd=str(ROOT / "code"), capture_output=True, text=True)


def media_box(pdf: Path):
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", Path(pdf).read_bytes())
    return float(m.group(3)) / 72.0, float(m.group(4)) / 72.0


# ----------------------------------------------------------------------------- SHA unchanged (signed-off code)
def test_frozen_raw_csv_sha():
    assert sha16(RAW) == "076f123353333614"


def test_default_and_b1_png_sha_unchanged(tmp_path):
    run_fig(tmp_path, "--out", str(tmp_path / "d"), "--formats", "png", "--appendix", "")
    run_fig(tmp_path, "--out", str(tmp_path / "b"), "--formats", "png", "--appendix", "", "--unvalidated", "B1")
    assert sha16(tmp_path / "d/fig_main_W2.png") == "e6d8122a695c569a"
    assert sha16(tmp_path / "b/fig_main_W2.png") == "7ff201527617b73f"


def test_preS5_copy_and_only_additions():
    assert sha16(PRE) == "7b1d7744e8256d68"                       # the signed-off version, saved before the --paper branch
    a = PRE.read_text(encoding="utf-8").splitlines()
    b = MAIN_PKG.read_text(encoding="utf-8").splitlines()
    removed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    assert removed == []                                           # incremental rule: the new file only ADDS lines
    assert (PRE.stat().st_mode & 0o222) == 0                       # 0444


# ----------------------------------------------------------------------------- determinism
def test_fig1_paper_deterministic_and_size(tmp_path):
    o1, o2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    run_fig(tmp_path, "--paper", "--paper-out", str(o1))
    run_fig(tmp_path, "--paper", "--paper-out", str(o2))
    assert o1.read_bytes() == o2.read_bytes()
    w, h = media_box(o1)
    assert abs(w - 7.16) < 0.01 and abs(h - 1.9) < 0.01
    raw = o1.read_bytes()
    assert b"CreationDate" not in raw and b"/Type3" not in raw and b"/FontFile2" in raw      # no date, fonttype 42 (embedded TrueType)
    assert sha16(o1) == sha16(ROOT / "paper/paper/figs/fig1_main.pdf")                        # the committed file is what the script makes


def test_fig2a_deterministic_and_size(tmp_path):
    o1, o2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    r1 = fig_cost.fig2a(o1)
    fig_cost.fig2a(o2)
    assert o1.read_bytes() == o2.read_bytes()
    w, h = media_box(o1)
    assert abs(w - 3.5) < 0.01 and abs(h - 1.6) < 0.01
    assert b"CreationDate" not in o1.read_bytes()
    assert r1["n_shapes"] == 26 and r1["n_hs2_r_inside_ci"] == 26 and r1["max_abs_dr"] < 0.002
    assert sha16(o1) == sha16(ROOT / "paper/paper/figs/fig2a_r.pdf")


def test_fig2b_is_interface_only():
    with pytest.raises(NotImplementedError):
        fig_cost.fig2b_onetime_costs(Path("x.pdf"))


def test_tables_deterministic_and_match_disk():
    assert tab_metrics.render() == tab_metrics.render() == (st.TAB_DIR / "tab1_metrics.tex").read_text(encoding="utf-8")
    assert tab_policy.render() == tab_policy.render() == (st.TAB_DIR / "tab2_policy.tex").read_text(encoding="utf-8")
    from p24.paper._tabutil import registry_csv_text
    assert registry_csv_text(tab_metrics.registry()) == (ROOT / "paper/paper/registry_extra/tab1_metrics.csv").read_text(encoding="utf-8")
    assert registry_csv_text(tab_policy.registry()) == (ROOT / "paper/paper/registry_extra/tab2_policy.csv").read_text(encoding="utf-8")


def test_build_all_twice_same_sha():
    first = build_all.build_all()
    second = build_all.build_all()
    assert first == second
    assert set(build_all.OUTPUTS) <= set(first)


# ----------------------------------------------------------------------------- tables use macros only
def test_table_macros_all_defined_or_placeholder():
    manifest = {r["macro"] for r in chk.read_manifest(ROOT / "paper/paper/numbers_manifest.csv")}
    from p24.paper._tabutil import macros_in
    for name in ("tab1_metrics.tex", "tab2_policy.tex"):
        used = macros_in((st.TAB_DIR / name).read_text(encoding="utf-8"))
        missing = [m for m in used if m not in manifest and not m.endswith("TBD")]
        assert missing == []
    # placeholders are not registered
    assert not any(m.endswith("TBD") for m in manifest)


def test_check_passes_on_smoke_tables_and_flags_hardcoded_number(tmp_path):
    smoke = ROOT / "paper/paper/smoke/main_smoke_tables.tex"
    res = chk.run_check(ROOT, None, [smoke])
    assert (len(res.a_mismatches), len(res.b_mismatches), len(res.c_hits), len(res.e_undefined_macros)) == (0, 0, 0, 0)
    assert res.ok and len(res.f_placeholder_macros) == 4
    bad_tab = tmp_path / "bad_tab.tex"
    bad_tab.write_text((st.TAB_DIR / "tab1_metrics.tex").read_text(encoding="utf-8").replace("\\NTabISteadyEZeroFiveBOne", "94.8"), encoding="utf-8")
    bad = tmp_path / "bad.tex"
    bad.write_text("\\documentclass{article}\n\\begin{document}\n\\input{bad_tab.tex}\n\\end{document}\n", encoding="utf-8")
    assert len(chk.run_check(ROOT, None, [bad]).c_hits) >= 1
