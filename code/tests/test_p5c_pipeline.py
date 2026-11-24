"""S4-P P5c script chain on SYNTHETIC data (CPU only; nothing here is a measurement; no output goes under results/).
Covers: aot_cost_table (tier rules, delta_cold, k rule, flags, strictness, 0444, no overwrite), swap_c1 + simulator hand check, sim_h6 (eval path on stand-in traces, guards, bit-for-bit reproduction of the P5 step-2 table),
h6_verdict (pass rules, bootstrap, H6e), predict_S4P_P5c (old rows bit-identical to predict_S4P_p3_v2, B1-AOT row), a123_verdict (A1-A3, V1-V4, NOT_QUIET parsing), gen_traces_S4P_P5c guards (guard function only: the generator is not run).
Run: export CUDA_VISIBLE_DEVICES=''; nice -n 10 "$PROJ_PYTHON" -B -m pytest code/tests/test_p5c_pipeline.py -q
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from p24.policy import make_policy
from p24.policy.base import Shape
from p24.sim import simulator as SIM
from p24.sim.costs import MissingCost
from scripts import a123_verdict_S4P_P5c as A123
from scripts import aot_cost_table_S4P_P5c as CT
from scripts import aot_tables_S4P_P5c as T
from scripts import gen_traces_S4P_P5c as GT
from scripts import h6_verdict_S4P_P5c as H6
from scripts import sim_h6_S4P_P5c as SH
from scripts import synth_S4P_P5c as Y

ROOT = T.ROOT
DRY_TABLE = ROOT / "results/S4P/p3/cost_v2_dryrun/cost_table_v2.csv"
DRY_ADD = ROOT / "results/S4P/p3/cost_v2_dryrun/addendum_rebuild_v2.csv"
V3_FLAT = ROOT / "results/S4P/v3/policy_params_v3_flat.json"
CODE = ROOT / "code"


# ============================================================================================ helpers
def build_table(root: pathlib.Path, name="out", extra=()):
    out = root / name
    out.mkdir(exist_ok=True)
    o = out / "cost_table_v2_aot_SYNTHETIC.csv"
    CT.main(["--synthetic", "--v2-table", str(DRY_TABLE), "--measure-dir", str(root / "measure_A_aot"), "--prebuild-dir", str(root / "prebuild/build"),
             "--fallback-csv", str(root / "aot_fallback_shapes_SYNTHETIC.csv"), "--out", str(o), *extra])
    return o


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("p5c_world")
    truth = Y.make_measure(root, k_ratio=1.0)
    Y.make_eval_like(root, workloads=("W2",), Ns=(1000,))
    Y.make_blockC(root)
    for sub, pre in (("runs_seq", "aot_seq"), ("runs_cold", "aot_cold")):                       # the existing aggregator, for the cross-check
        subprocess.run([sys.executable, "-B", "-m", "scripts.aot_cost_rows_S4P_P5c", "--runs-dir", str(root / "measure_A_aot" / sub), "--out", str(root / "measure_A_aot" / pre)],
                       cwd=CODE, check=True, capture_output=True, env=dict(os.environ, PYTHONPATH=str(CODE), CUDA_VISIBLE_DEVICES=""))
    out = build_table(root)
    return dict(root=root, truth=truth, table=out, df=pd.read_csv(out))


def raw_obs(root, sub):
    import glob
    res = []
    for f in sorted(glob.glob(str(root / "measure_A_aot" / sub / "AOTW_*.json"))):
        j = json.load(open(f))
        for pos, k in enumerate(j["per_key"]):
            res.append((k["key"], pos, k["first_req_s"], k["steady_wall_median_ms"], j["proc_id"]))
    return res


# ============================================================================================ 1. aot_cost_table
def test_tiers_follow_the_preregistered_rules(world):
    df, root = world["df"], world["root"]
    aot = df[df.arm == "AOT"]
    assert len(aot) == 32 and len(df) == len(pd.read_csv(DRY_TABLE)) + 32
    seq, cold = raw_obs(root, "runs_seq"), raw_obs(root, "runs_cold")
    delta = []
    for r in aot.itertuples():
        tag = Shape(r.H, r.W, r.B, r.cfg).tag()
        o = [x for x in seq if x[0] == tag]
        inproc = [x[2] - x[3] / 1e3 for x in o if x[1] > 0]
        assert r.n__inproc_diskcache == len(inproc) and len(inproc) in (2, 3)
        assert r.resid_s__inproc_diskcache == pytest.approx(np.median(inproc), rel=1e-12)
        assert r.steady_ms == pytest.approx(np.median([x[3] for x in o]), rel=1e-12)
        c = [x[2] - x[3] / 1e3 for x in cold if x[0] == tag]
        if c:
            assert r.n__cold_diskcache == 3 and r.aot_cold_source == "measured"
            assert r.resid_s__cold_diskcache == pytest.approx(np.median(c), rel=1e-12)
            delta.append(np.median(c) - r.resid_s__inproc_diskcache)
        else:
            assert r.aot_cold_source == "inproc+delta_cold" and r.n__cold_diskcache == 0
    assert len(delta) == 7
    d = float(np.median(delta))
    for r in aot.itertuples():
        if r.aot_cold_source != "measured":
            assert r.resid_s__cold_diskcache == pytest.approx(r.resid_s__inproc_diskcache + d, rel=1e-12)
        assert r.aot_delta_cold_s == pytest.approx(d, rel=1e-12)
        # empty tiers = build wall-clock + the matching disk tier
        assert r.resid_s__cold_empty == pytest.approx(r.aot_build_wall_s + r.resid_s__cold_diskcache, rel=1e-12)
        assert r.resid_s__inproc_empty == pytest.approx(r.aot_build_wall_s + r.resid_s__inproc_diskcache, rel=1e-12)
        assert np.isnan(r.fc_te_s__cold_diskcache) and r.fc_source == "aot_all_in_resid"
    # the three first shapes of the sequences have n=2; the fallback shape is flagged but its numbers are untouched
    assert sorted(aot[aot.n__inproc_diskcache == 2][["H", "W", "B", "cfg"]].itertuples(index=False, name=None)) == sorted([(512, 512, 2, 0), (512, 768, 1, 0), (512, 512, 3, 0)])
    fb = aot[aot.aot_fallback == 1]
    assert len(fb) == 1 and (fb.H.iloc[0], fb.W.iloc[0], fb.B.iloc[0], fb.cfg.iloc[0]) == (512, 512, 3, 0) and "aot_fallback" in fb.aot_flag.iloc[0]
    meta = json.load(open(str(world["table"]).replace(".csv", "_meta.json")))
    assert meta["warnings"] == [] and meta["synthetic"] is True and meta["n_cold_shapes"] == 7 and meta["build_calibration"]["applied"] is False


def test_non_aot_rows_unchanged_and_files_frozen(world):
    out = world["table"]
    new = pd.read_csv(out)
    old = pd.read_csv(DRY_TABLE)
    pd.testing.assert_frame_equal(new[new.arm != "AOT"][list(old.columns)].reset_index(drop=True), old, check_dtype=False)
    assert T.is_ro(out) and T.is_ro(str(out).replace(".csv", "_meta.json"))
    with pytest.raises(SystemExit):                                                       # never overwritten
        build_table(world["root"])


@pytest.mark.parametrize("ratio,applied", [(1.10, False), (1.14, False), (1.16, True), (0.80, True)])
def test_build_wallclock_calibration_rule(tmp_path, ratio, applied):
    Y.make_measure(tmp_path, k_ratio=ratio)
    out = build_table(tmp_path)
    aot = pd.read_csv(out)
    aot = aot[aot.arm == "AOT"]
    k = json.load(open(str(out).replace(".csv", "_meta.json")))["build_calibration"]
    assert k["k"] == pytest.approx(ratio, rel=1e-9) and k["applied"] is applied
    assert (aot.aot_build_wall_s / aot.aot_build_wall_raw_s).round(9).eq(round(ratio, 9) if applied else 1.0).all()


def test_truncated_aotbuild_uses_prebuild_values_and_flags(tmp_path):
    Y.make_measure(tmp_path, k_ratio=1.5, truncate_build=True)
    aot = pd.read_csv(build_table(tmp_path))
    aot = aot[aot.arm == "AOT"]
    assert (aot.aot_k_applied == 1.0).all() and aot.aot_flag.str.contains("build_nonexclusive").all()


def test_missing_sequence_data_is_not_guessed(tmp_path):
    Y.make_measure(tmp_path)
    for f in (tmp_path / "measure_A_aot/runs_seq").glob("AOTW_seq32_p1_SYNTHETIC.json"):      # drop one process: shapes that were only seen at position>=2 there still have 2 obs ...
        f.unlink()
    for f in (tmp_path / "measure_A_aot/runs_seq").glob("*.json"):                            # ... but with no sequence at all nothing can be built
        f.unlink()
    with pytest.raises(SystemExit):
        build_table(tmp_path)


def test_contended_process_is_flagged_not_dropped(tmp_path):
    Y.make_measure(tmp_path)
    f = tmp_path / "measure_A_aot/runs_seq/AOTW_seq32_p0_SYNTHETIC.json"
    j = json.load(open(f))
    j["gpu0_other_procs_start"] = 1
    f.write_text(json.dumps(j))
    aot = pd.read_csv(build_table(tmp_path))
    aot = aot[aot.arm == "AOT"]
    assert aot.aot_flag.str.contains("other_gpu0_proc").all() and aot.n__inproc_diskcache.min() >= 2


def test_cross_check_against_the_existing_aggregator(world):
    meta = json.load(open(str(world["table"]).replace(".csv", "_meta.json")))
    assert not any("MISMATCH" in w for w in meta["warnings"])


def test_output_location_guards(tmp_path):
    with pytest.raises(AssertionError):
        T.check_not_official_output(ROOT / "results/S4P/p5c/x.csv", True)             # SYNTHETIC never under results/
    with pytest.raises(AssertionError):
        T.check_not_official_output(tmp_path / "x.csv", False)                       # official only under results/S4P/p5c
    T.check_not_official_output(tmp_path / "x.csv", True)
    T.check_not_official_output(T.P5C / "x.csv", False)
    with pytest.raises(AssertionError):
        Y.guard_out(ROOT / "results/S4P/p5c/synthetic")


# ============================================================================================ 2. swap + simulator (hand check)
def mkreq(sh):
    return dict(req_id="r", H=sh.H, W=sh.W, B=sh.B, cfg=sh.cfg)


def test_b1_aot_hand_computed_total_and_fallback_is_e0(world):
    df_imp, add_df, ct0 = SH.load_world("v2", world["table"], DRY_ADD)
    fb = T.read_fallback(world["root"] / "aot_fallback_shapes_SYNTHETIC.csv")
    tabs = SH.make_tables(ct0, fb)
    a, b = Shape(512, 512, 1, 0), Shape(512, 768, 2, 1)
    f = Shape(512, 512, 3, 0)
    reqs = [mkreq(x) for x in (a, a, b, a, f, f, b)]
    aot = ct0.df[ct0.df.arm == "AOT"].set_index(["H", "W", "B", "cfg"])
    e0 = lambda sh, t: ct0.one_time("E0", sh, t).total_s
    for cm, t_first, t_next in (("warm", "cold_diskcache", "inproc_diskcache"), ("empty", "cold_empty", "inproc_empty")):
        tab = tabs["aot"]
        s = lambda x: aot.loc[(x.H, x.W, x.B, x.cfg), "steady_ms"] / 1e3
        o = lambda x, t: aot.loc[(x.H, x.W, x.B, x.cfg), f"resid_s__{t}"]
        want = (s(a) + o(a, t_first)) + s(a) + (s(b) + o(b, t_next)) + s(a)
        want += (ct0.steady_s("E0", f) + e0(f, t_next)) + ct0.steady_s("E0", f) + s(b)                 # fallback shape: E0 service and E0 one-time
        got = SIM.simulate(reqs, make_policy("B1", cost=None), tab, SIM.SimOptions(cache_mode=cm, overhead_aware=True)).summary["service_s"]
        assert got == pytest.approx(want, rel=1e-12), cm
    # swapped table leaves the other arms untouched
    for arm in ("E0", "G1", "C1+G1"):
        pd.testing.assert_frame_equal(tabs["aot"].df[tabs["aot"].df.arm == arm], ct0.df[ct0.df.arm == arm])
    # nofb variant prices the fallback shape as AOT
    assert tabs["aot_nofb"].steady_s("C1", f) == pytest.approx(aot.loc[(512, 512, 3, 0), "steady_ms"] / 1e3)


def test_shape_without_aot_row_raises_missing_cost(world):
    df = pd.read_csv(world["table"])
    drop = (df.arm == "AOT") & (df.H == 768) & (df.W == 768) & (df.B == 4) & (df.cfg == 1)
    from p24.sim import addendum as ADD
    ct = ADD.load_with_addendum(df[~drop], DRY_ADD)
    tab = T.swap_c1(ct, T.aot_provider(ct))
    with pytest.raises(MissingCost):
        SIM.simulate([mkreq(Shape(768, 768, 4, 1))], make_policy("B1", cost=None), tab, SIM.SimOptions(cache_mode="warm"))


# ============================================================================================ 3. sim_h6 on stand-in eval traces / P5 reproduction / guards
def sim_args(world, out, extra=()):
    r = world["root"]
    return ["--split", "eval", "--synthetic", "--table-csv", str(world["table"]), "--addendum-csv", str(DRY_ADD), "--fallback-csv", str(r / "aot_fallback_shapes_SYNTHETIC.csv"),
            "--eval-dir", str(r / "eval_like"), "--eval-manifest", str(r / "eval_like/sha256_manifest_eval_SYNTHETIC.txt"), "--workloads", "W2", "--N", "1000", "--colds", "disk,empty",
            "--out", str(out), *extra]


@pytest.fixture(scope="module")
def sim_eval(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("p5c_sim") / "eval"
    assert SH.main(sim_args(world, out)) == 0
    return out


def test_sim_eval_path_rows_and_pairing(sim_eval, world):
    d = pd.read_csv(sim_eval / "main_raw.csv")
    assert set(d.status) == {"ok"} and set(d.seed) == set(range(120, 130)) and set(d.workload) == {"W2"}
    assert {"B0", "B1", "Ours", "Ours-v3", "B1-AOT", "v2-AOT", "B1-AOT-nofb", "B1-AOT-sens", "B1-excl", "B1-AOT-excl"} <= set(d.variant)
    p = d.pivot_table(index=["cold", "seed"], columns="variant", values="service_s")
    assert (p["B1-AOT"] < p["B1"]).loc["disk"].all()                                          # AOT one-time is far below C1's in the dry-run table
    assert (p["B1-AOT-sens"].loc["disk"] == p["B1-AOT"].loc["disk"]).all()                    # the sensitivity only touches the empty inproc tier
    assert (p["B1-AOT"].loc["empty"] > p["B1"].loc["empty"]).all() and (p["B1-AOT-sens"].loc["empty"] < p["B1-AOT"].loc["empty"]).all()
    assert (p["B1-AOT-nofb"].loc["disk"] < p["B1-AOT"].loc["disk"]).all()                      # disk level: fallback shape priced as AOT is cheaper than E0 service (empty level: the E0 shape saves the build -> opposite)
    assert (p["B1-AOT-nofb"].loc["empty"] > p["B1-AOT"].loc["empty"]).all()
    meta = json.load(open(sim_eval / "main_meta.json"))
    assert meta["synthetic"] is True and meta["offline"]["aot_build_total_s"] == pytest.approx(2120.0) and meta["fallback_shapes"] == ["512x512_b3_nocfg"]
    with pytest.raises(SystemExit):                                                           # never overwritten
        SH.main(sim_args(world, sim_eval))


def test_sim_guards(world, tmp_path):
    base = sim_args(world, tmp_path / "x")
    with pytest.raises(AssertionError):                                                       # official run needs a 0444 fallback list / default dirs (synthetic fallback csv is 0644)
        SH.main([a for a in base if a != "--synthetic"])
    with pytest.raises(AssertionError):                                                       # SYNTHETIC output must not be under results/
        SH.main(sim_args(world, ROOT / "results/S4P/p5c/synthetic_must_fail"))
    with pytest.raises(AssertionError):                                                       # seeds outside 120-129
        SH.main(sim_args(world, tmp_path / "y", ["--seeds", "110,111"]))
    with pytest.raises(AssertionError):
        SH.main(sim_args(world, tmp_path / "z", ["--seeds", "101"]))
    with pytest.raises(AssertionError):                                                       # N outside the evaluation set
        SH.main(sim_args(world, tmp_path / "w", ["--N", "300"]))
    assert not (ROOT / "results/S4P/p5c/synthetic_must_fail").exists()


def test_dry_run_checks_manifest_and_0444_without_reading_traces(world, tmp_path):
    assert SH.main(sim_args(world, tmp_path / "d", ["--dry-run"])) == 0
    tr = world["root"] / "eval_like/eval_traces/W2_N1000_seed120_B.jsonl"
    os.chmod(tr, 0o644)
    try:
        with pytest.raises(AssertionError):
            SH.main(sim_args(world, tmp_path / "d2", ["--dry-run"]))
    finally:
        os.chmod(tr, 0o444)


def test_reproduces_p5_step2_bit_for_bit(tmp_path):
    ns = type("NS", (), dict(out=str(tmp_path / "repro"), synthetic=True))()
    assert SH.cmd_repro_p5(ns) == 0
    r = json.load(open(tmp_path / "repro/repro_p5_step2_check.json"))
    assert r["n_compared"] == 160 and r["n_bit_equal"] == 160 and r["max_abs_diff"] == 0.0 and r["passed"] is True


# ============================================================================================ 4. h6 verdict
def test_pass_rule_8_of_10_and_mean():
    f = lambda xs, line: H6.stat(pd.Series(xs), line)
    assert f([0.79] * 8 + [0.85, 0.9], 0.80)["passed"] is False          # mean 0.8 + ...: 8 seeds under the line but the mean is above the line
    assert f([0.70] * 8 + [0.85, 0.90], 0.80)["passed"] is True          # 8/10 and mean 0.755 <= 0.80
    assert f([0.70] * 7 + [0.9, 0.9, 0.9], 0.80)["k_le_line"] == 7 and f([0.70] * 7 + [0.9, 0.9, 0.9], 0.80)["passed"] is False   # mean ok, only 7/10
    assert f([0.80] * 10, 0.80)["passed"] is True                        # equality counts as <= line
    assert f([0.5] * 4, 0.8)["need"] == 4 and f([0.5] * 5, 0.8)["need"] == 4 and f([0.5] * 10, 0.8)["need"] == 8


def test_bootstrap_is_reproducible_and_matches_the_loop_definition():
    x = np.random.default_rng(3).random(10)
    assert H6.boot_ci(x) == H6.boot_ci(x)
    rng = np.random.default_rng(0)
    m = np.array([x[rng.integers(0, 10, 10)].mean() for _ in range(10000)])
    assert H6.boot_ci(x) == (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def synth_raw(ratios, n_seeds=10, e_over=None, full_grid=True):
    """raw frame: B1 = 100, B0 = 95 + ..., old others 97+, B1-AOT = ratio x B1 (per seed) in the W2/W5 cells, 0.99 elsewhere (e_over overrides one H6e cell)."""
    rows = []
    for w in H6.WL_ALL:
        if not full_grid and w not in ("W2", "W5"):
            continue
        for n in H6.N_ALL if full_grid else (1000, 10000):
            for cold in H6.COLDS:
                for s in range(n_seeds):
                    r = ratios.get((w, n, cold), [0.99] * n_seeds)[s]
                    if e_over and (w, n, cold) == e_over[0]:
                        r = e_over[1]
                    vals = dict(B1=100.0, B0=110.0, B2=105.0, B3=106.0, B3p=107.0, B3pp=108.0, B4=109.0, B5=104.0, B6=111.0, B6p=112.0, Ours=103.0, **{"Ours-v3": 102.0})
                    vals["B1-AOT"] = 100.0 * r
                    vals["B1-AOT-nofb"] = 100.0 * r
                    vals["B1-excl"] = 90.0
                    vals["B1-AOT-excl"] = 90.0 * r
                    vals["B1-AOT-sens"] = 100.0 * r
                    vals["v2-AOT"] = 0.9 * vals["Ours"]
                    for v, t in vals.items():
                        rows.append(dict(split="eval", status="ok", variant=v, workload=w, N=n, cold=cold, seed=100 + s, service_s=t))
    return pd.DataFrame(rows)


def test_h6_verdict_logic_on_hand_built_frame():
    good = {("W2", 1000, "disk"): [0.70] * 10, ("W5", 1000, "disk"): [0.75] * 10, ("W2", 10000, "disk"): [0.95] * 10, ("W5", 10000, "disk"): [0.96] * 10}
    v = H6.verdict(synth_raw(good))
    assert v["confirmatory"] is True
    assert v["H6a"]["passed"] and v["H6b"]["passed"] and v["H6c"]["passed"] and v["H6d"]["passed"] and v["H6e"]["passed"] and v["main_conclusion"]["all"]
    # H6c: best old policy per seed is B1 (100) -> ratio vs best old equals the ratio vs B1 here
    c = {(x["workload"], x["N"]): x for x in v["H6c"]["cells"]}
    assert c[("W2", 1000)]["mean"] == pytest.approx(0.70) and json.loads(c[("W2", 1000)]["best_old_counts"]) == {"B1": 10}
    # one workload fails H6a (mean 0.82 > 0.80) -> H6a false, main false
    bad = {**good, ("W5", 1000, "disk"): [0.82] * 10}
    v2 = H6.verdict(synth_raw(bad))
    assert v2["H6a"]["passed"] is False and v2["main_conclusion"]["all"] is False and v2["H6b"]["passed"] is True
    # H6e: one cell mean 1.006 -> fails; 1.005 -> passes
    assert H6.verdict(synth_raw(good, e_over=(("W3", 100, "disk"), 1.006)))["H6e"]["passed"] is False
    assert H6.verdict(synth_raw(good, e_over=(("W3", 100, "disk"), 1.005)))["H6e"]["passed"] is True
    # H6c uses the BEST old policy: make Ours-v3 cheaper than B1-AOT in one cell only
    v3 = H6.verdict(synth_raw(good).assign(service_s=lambda d: np.where((d.variant == "Ours-v3") & (d.workload == "W2") & (d.N == 1000) & (d.cold == "disk"), 60.0, d.service_s)))
    assert v3["H6c"]["passed"] is False and v3["H6a"]["passed"] is True
    # H6f is reported without a line
    f = {x["workload"] + str(x["N"]): x for x in v["H6f"]["cells"]}
    assert "passed" not in f["W21000"] or f["W21000"].get("line") is None


def test_h6_labels_partial_or_dev_frames_as_not_confirmatory():
    v = H6.verdict(synth_raw({}, full_grid=False))
    assert v["confirmatory"] is False and "NOT confirmatory" in v["label"]
    d = synth_raw({}, n_seeds=5).assign(split="dev")
    assert H6.verdict(d)["confirmatory"] is False
    miss = synth_raw({}).query("variant != 'B3pp'")
    assert H6.verdict(miss)["missing_variants"] == ["B3pp"] and H6.verdict(miss)["confirmatory"] is False


def test_h6_main_writes_files_and_refuses_overwrite(sim_eval, tmp_path):
    out = tmp_path / "ver"
    assert H6.main(["--raw", str(sim_eval / "main_raw.csv"), "--out-dir", str(out), "--synthetic"]) == 0
    assert {p.name for p in out.iterdir()} == {"h6_verdict.json", "h6_cells_judged.csv", "h6_full_table.csv", "h6_sensitivity.csv"}
    j = json.load(open(out / "h6_verdict.json"))
    assert j["confirmatory"] is False and j["synthetic"] is True
    with pytest.raises(SystemExit):
        H6.main(["--raw", str(sim_eval / "main_raw.csv"), "--out-dir", str(out), "--synthetic"])
    full = pd.read_csv(out / "h6_full_table.csv")
    disk = full[full.cold == "disk"].iloc[0]
    assert disk.K_star == pytest.approx(2120.0 / disk.saving_per_process_s)


# ============================================================================================ 5. Pred-1 (bit-identical old rows)
@pytest.fixture(scope="module")
def preds(world, tmp_path_factory):
    d = tmp_path_factory.mktemp("p5c_pred")
    from scripts import s4p_p3_common as C0
    from scripts import predict_S4P_p3_v2 as ORIG
    from scripts import predict_S4P_P5c as NEW
    root = world["root"]
    old_rv4, old_argv = C0.RV4, sys.argv
    try:
        C0.RV4 = root / "streams"
        sys.argv = ["x", "--table-csv", str(DRY_TABLE), "--addendum-csv", str(DRY_ADD), "--v3-params-json", str(V3_FLAT), "--out", str(d / "orig.csv"), "--dryrun"]
        ORIG.main()
    finally:
        C0.RV4, sys.argv = old_rv4, old_argv
    NEW.main(["--table-csv", str(world["table"]), "--addendum-csv", str(DRY_ADD), "--fallback-csv", str(root / "aot_fallback_shapes_SYNTHETIC.csv"), "--v3-params-json", str(V3_FLAT),
              "--stream-dir", str(root / "streams"), "--out", str(d / "new.csv"), "--dryrun"])
    return d


def test_pred1_old_rows_are_bit_identical(preds):
    a, b = pd.read_csv(preds / "orig.csv"), pd.read_csv(preds / "new.csv")
    old = b[b.policy != "B1-AOT"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[list(a.columns)], a, check_exact=True)
    assert {"B0", "B1", "B4", "Ours", "v3", "B3′"} == set(a.policy) and len(b[b.policy == "B1-AOT"]) == 3


def test_pred1_b1_aot_row(preds, world):
    b = pd.read_csv(preds / "new.csv")
    for stem, g in b.groupby("stream"):
        aot, b1 = g[g.policy == "B1-AOT"].iloc[0], g[g.policy == "B1"].iloc[0]
        assert aot.pred_total_s < b1.pred_total_s and aot.event_counts == b1.event_counts and aot.pred_input_s == b1.pred_input_s
        reqs = [json.loads(l) for l in open(world["root"] / f"streams/{stem}.jsonl")]
        assert aot.n_fallback_requests == sum(1 for q in reqs if (q["H"], q["W"], q["B"], q["cfg"]) == (512, 512, 3, 0))
    meta = json.load(open(preds / "new_meta.json"))
    assert meta["dryrun"] is True and meta["aot_fallback"]["shapes"] == ["512x512_b3_nocfg"]


def test_pred1_refuses_official_use_without_freezing_inputs(world, tmp_path):
    from scripts import predict_S4P_P5c as NEW
    base = ["--table-csv", str(world["table"]), "--addendum-csv", str(DRY_ADD), "--fallback-csv", str(world["root"] / "aot_fallback_shapes_SYNTHETIC.csv"), "--v3-params-json", str(V3_FLAT)]
    with pytest.raises(AssertionError):                                                       # not --dryrun: fallback list not 0444 / output not in real_validation
        NEW.main([*base, "--out", str(tmp_path / "p.csv")])
    with pytest.raises(SystemExit):
        NEW.main([*base, "--out", str(tmp_path / "p.csv"), "--dryrun", "--freeze"])
    with pytest.raises(SystemExit):
        NEW.main([*base, "--out", str(tmp_path / "p.csv"), "--stream-dir", str(tmp_path)])
    with pytest.raises(AssertionError):
        NEW.main([*base, "--out", str(ROOT / "results/S4P/p3/real_validation/pred_must_not_exist.csv"), "--dryrun", "--stream-dir", str(world["root"] / "streams")])
    assert not (ROOT / "results/S4P/p3/real_validation/pred_must_not_exist.csv").exists()


# ============================================================================================ 6. a123 verdict
def run_a123(world, tmp_path, **kw):
    root = tmp_path / "root"
    Y.make_blockC(root, **kw)
    out = tmp_path / "v"
    sub = pd.DataFrame([dict(stream=f"{w}_N300_seed101_B", policy=p, cache_mode="warm", role=p, pred_total_s=100.0, pred_excl_input_s=99.0, event_counts="") for w in A123.WLS for p in ("B0", "B1", "B1-AOT")])
    sub.to_csv(tmp_path / "pred1.csv", index=False)
    assert A123.main(["--raw", str(root / "blockC/raw"), "--log-dir", str(root / "blockC/logs"), "--dryrun-dir", str(root / "blockC/dryrun"), "--pred1", str(tmp_path / "pred1.csv"),
                      "--out-dir", str(out), "--synthetic"]) == 0
    return json.load(open(out / "a123_verdict.json")), root


def test_a123_numbers_and_validity_on_clean_data(world, tmp_path):
    res, root = run_a123(world, tmp_path)
    P = A123.load_procs(root / "blockC/raw")
    for w in A123.WLS:
        s = f"{w}_N300_seed101_B"
        va, v1, v0 = (P[(P.stream == s) & (P.policy == p)].service_s.mean() for p in ("B1-AOT", "B1", "B0"))
        got = res["verdict_all_processes"]["streams"][w]
        assert got["A1_ratio"] == pytest.approx(va / v1) and got["A3_ratio"] == pytest.approx(va / v0) and got["n_aot"] == 3
        lo, hi = got["A1_ci95"]
        assert lo <= got["A1_ratio"] <= hi
    assert res["validity"]["all"] is True and res["A123_judged"] is True
    A1 = res["verdict_all_processes"]["A1"]
    assert A1["passed"] is True and A1["combined"] == pytest.approx(np.exp(np.mean(np.log(A1["per_stream"]))))
    assert res["verdict_all_processes"]["A2"]["passed"] is True                                   # 0.4 s first requests vs 6 s
    assert res["verdict_all_processes"]["A3"]["passed"] is False                                   # synthetic: B1-AOT is slower than 0.92 x B0 by construction -> the verdict must say so
    assert res["verdict_all_processes"]["real_machine_holds"] is False
    assert set(res["validity"]["V4"]["fallback_shapes"]) == {"512x512x3x0"}
    assert res["pred1_errors"]["all_within_10pct"] is False and res["pred0_errors"]["aot_rows"] and "excluded_processes" not in res


def test_a123_not_quiet_flag_and_dynamo_failure_block_the_judgement(world, tmp_path):
    res, _ = run_a123(world, tmp_path / "a", flag_b1aot=True)
    assert res["validity"]["V2"]["passed"] is False and res["validity"]["V2"]["not_quiet_flagged"] == ["B1-AOT_W2_warm_p0"]
    assert res["A123_judged"] is False and isinstance(res["conclusion"], str) and "NOT JUDGED" in res["conclusion"]
    assert res["excluded_processes"] == ["B1-AOT_W2_warm_p0"] and res["verdict_excluding_flagged"]["streams"]["W2"]["n_aot"] == 2    # the excluded variant is reported next to it
    res2, _ = run_a123(world, tmp_path / "b", bad_dynamo=True)
    assert res2["validity"]["V3"]["passed"] is False and res2["A123_judged"] is False


def test_not_quiet_parser_handles_both_step_formats(tmp_path):
    (tmp_path / "block_Cv2_ALL.log").write_text("NOT_QUIET load1=18>16\nSTEP [3/66] B1_W2_warm_p0 rc=0 elapsed=1s\nSTEP B0_W2_warm_p0 rc=0\nNOT_QUIET gpu0_busy\nSTEP [5/66] B1-AOT_W3_warm_p1 rc=0\n")
    nq = A123.not_quiet_tags(tmp_path)
    assert set(nq) == {"B1_W2_warm_p0", "B1-AOT_W3_warm_p1"}


def test_a123_bootstrap_reproducible():
    a, b = [np.array([1.0, 2.0, 3.0])] * 3, [np.array([3.0, 4.0, 5.0])] * 3
    assert A123.boot_ratios(a, b) == A123.boot_ratios(a, b)
    ci, g = A123.boot_ratios(a, b)
    assert ci[0][0] <= 2.0 / 4.0 <= ci[0][1] and g[0] <= (2.0 / 4.0) <= g[1]


# ============================================================================================ 7. trace generator guards (guard function only; the generator is NOT run)
def test_trace_generator_guards(world, tmp_path):
    ro = world["table"]
    GT.guard(ro)
    with pytest.raises(SystemExit):
        GT.guard(tmp_path / "missing.csv")
    rw = tmp_path / "rw.csv"
    rw.write_bytes(pathlib.Path(ro).read_bytes())
    os.chmod(rw, 0o644)
    with pytest.raises(SystemExit):
        GT.guard(rw)
    for bad in ([119], [101], [3], [100], [120, 109]):
        with pytest.raises(AssertionError):
            GT.guard(ro, seeds=bad)
    GT.guard(ro, seeds=[120, 129])
    assert GT.SEEDS == list(range(120, 130)) and not (set(GT.SEEDS) & GT.FORBIDDEN) and GT.OUT == T.P5C
    with pytest.raises(SystemExit):                                                           # preregistration SHA mismatch is refused
        fake = tmp_path / "prereg.md"
        fake.write_text("x")
        os.chmod(fake, 0o444)
        GT.guard(ro, prereg=fake)
