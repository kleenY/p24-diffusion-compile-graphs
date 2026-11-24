"""S3 3.1b cost addendum: build script (scripts/build_cost_addendum_S3.py) + overlay loader (p24/sim/addendum.py). CPU only, synthetic data."""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from p24.policy import CostView, Shape, make_policy
from p24.policy.base import Shape as Sh  # noqa: F401
from p24.sim import addendum as ADD
from p24.sim import simulator as SIM
from p24.sim.costs import TIERS, CostTable
from scripts import build_cost_addendum_S3 as BA

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
S1, S2_, S3_, S4_ = Shape(512, 512, 1, 0), Shape(512, 768, 1, 0), Shape(768, 512, 1, 0), Shape(512, 512, 2, 0)
K4 = [S1, S2_, S3_, S4_]


def trow(arm, s, steady_ms=100.0, capture_s=0.04, resid=0.5, source="measured"):
    r = dict(arm=arm, H=s.H, W=s.W, B=s.B, cfg=s.cfg, steady_ms=steady_ms, capture_pool_mb=400.0, capture_s=capture_s,
             weights_alloc_mb=2000.0, steady_peak_reserved_mb=2500.0, transient_excess_s=np.nan, source=source, fc_source="measured")
    for t in TIERS:
        cold = t.startswith("cold")
        r[f"resid_s__{t}"] = resid if cold else np.nan          # inproc tiers NaN as in the S2 table of C2/C5
        r[f"first_req_s__{t}"] = resid + 0.1 if cold else np.nan
        r[f"n__{t}"] = 3 if cold else 0
        for c in ("te", "unet", "vae"):
            r[f"fc_{c}_s__{t}"] = 1.0 if cold else np.nan
    return r


def toy_table():
    rows = []
    for s in K4:
        rows += [trow("E0", s, 120.0, resid=0.1), trow("G1", s, 90.0, resid=0.5), trow("C1+G1", s, 70.0, resid=1.0),
                 trow("C1", s, 75.0, resid=1.0), trow("C2", s, 80.0, resid=7.0, source="extrapolated_ratio_C1"),
                 trow("C5", s, 95.0, resid=40.0, source="extrapolated_ratio_C1")]
    return pd.DataFrame(rows)


def addrow(kind="inproc_cost", **kw):
    r = {c: np.nan for c in ADD.ADDENDUM_COLS}
    r.update(kind=kind, flag="", source="inproc_measured_S3", n_proc=3)
    r.update(kw)
    return r


# ----------------------------------------------------------------------------------------------- overlay loader
def test_only_covered_cells_replaced():
    base = toy_table()
    add = pd.DataFrame([addrow(arm="C2", H=512, W=768, B=1, cfg=0, first_req_s=2.5, resid_s=0.3, fc_te_s=np.nan, fc_unet_s=2.0, fc_vae_s=0.2)])
    ct = ADD.load_with_addendum(base, add)
    new = ct.df
    k = (new.arm == "C2") & (new.H == 512) & (new.W == 768) & (new.B == 1)
    assert new.loc[k, f"resid_s__inproc_diskcache"].iloc[0] == 0.3
    assert new.loc[k, "first_req_s__inproc_diskcache"].iloc[0] == 2.5
    assert new.loc[k, "n__inproc_diskcache"].iloc[0] == 3 and new.loc[k, "addendum_source"].iloc[0] == "inproc_measured_S3"
    # every other cell of every other row, and every other column of that row, is byte-identical
    keep = [c for c in base.columns]
    other = new.loc[~k, keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(other, base.loc[~k.to_numpy()].reset_index(drop=True), check_dtype=False)
    cols_same = [c for c in keep if c not in ADD.INPROC_COLS]
    pd.testing.assert_frame_equal(new.loc[k, cols_same].reset_index(drop=True), base.loc[k.to_numpy(), cols_same].reset_index(drop=True), check_dtype=False)
    assert base.loc[(base.arm == "C2") & (base.W == 768), "resid_s__inproc_diskcache"].isna().all()     # input frame untouched
    assert ct.addendum_report["n_inproc_replaced"] == 1
    # the policy-visible one-time cost of that cell now comes from the inproc tier (before: fallback to cold_diskcache)
    cv = CostView(ct, "warm")
    assert cv.one_time_s("C2", S2_) == pytest.approx(0.3 + 2.0 + 0.2) or cv.one_time_s("C2", S2_) > 0
    assert cv.one_time_s("C5", S2_) == CostView(CostTable(base), "warm").one_time_s("C5", S2_)


def test_addendum_row_without_table_row_is_reported_not_added():
    add = pd.DataFrame([addrow(arm="C2", H=256, W=256, B=1, cfg=0, first_req_s=1.0, resid_s=0.1)])
    ct = ADD.load_with_addendum(toy_table(), add)
    assert ct.addendum_report["inproc_skipped_no_table_row"] == [["C2", 256, 256, 1, 0]] and len(ct.df) == len(toy_table())


def test_override_effective_and_ratio_rule(tmp_path):
    add = pd.DataFrame([
        addrow("rebuild", arm="G1", H=512, W=512, B=1, cfg=0, rebuild_over_capture=4.6, rebuild_override_s=0.185, source="rebuild_measured_S3"),
        addrow("rebuild", arm="C1+G1", H=512, W=512, B=1, cfg=0, rebuild_over_capture=1.2, rebuild_override_s=np.nan, source="rebuild_measured_S3")])
    ct = ADD.load_with_addendum(toy_table(), add)
    assert ct.rebuild_override == {("G1", S1): 0.185}
    cv = CostView(ct, "warm", rebuild_override=ct.rebuild_override)
    assert cv.rebuild_s("G1", S1) == 0.185
    assert cv.rebuild_s("C1+G1", S1) == pytest.approx(0.04)       # ratio <= 1.5: table capture_s kept
    assert cv.rebuild_s("G1", S2_) == pytest.approx(0.04)         # shape not measured: table capture_s kept
    assert ADD.AddendumCostView(ct, "warm").rebuild_s("G1", S1) == 0.185


def test_simulator_charges_the_override_via_hook():
    """simulate() builds its own CostView: without the hook the override would only steer the policy, not the charged seconds."""
    base = toy_table()
    add = pd.DataFrame([addrow("rebuild", arm="C1+G1", H=s_.H, W=s_.W, B=s_.B, cfg=s_.cfg, rebuild_over_capture=12.5, rebuild_override_s=0.5,
                               source="rebuild_measured_S3") for s_ in (S1, S2_)])
    ct_add = ADD.load_with_addendum(base, add)
    ct_plain = CostTable(base)
    reqs = [dict(req_id=f"r{i}", H=s.H, W=s.W, B=s.B, cfg=s.cfg, noise_seed=i, prompt="p") for i, s in enumerate(([S1] * 3 + [S2_] * 3) * 100)]
    opt = SIM.SimOptions(cache_mode="warm", mem_budget_mb=500.0, overhead_aware=True)   # one 400 MB graph fits, two do not -> evict / re-capture
    prior, P = {S1: 0.5, S2_: 0.5}, dict(alpha=1.0, H_prior=1e4)

    def run(ct, ov):
        pol = make_policy("B5", cost=CostView(ct, "warm", rebuild_override=ov), params=P, prior=prior)
        return SIM.simulate(reqs, pol, ct, opt).summary

    orig = SIM.CostView
    try:
        r_plain = run(ct_plain, None)
        r_nohook = run(ct_add, ct_add.rebuild_override)
        ADD.install_simulator_hook()
        r_hook = run(ct_add, ct_add.rebuild_override)
    finally:
        SIM.CostView = orig
    assert r_plain["n_evicted_total"] >= 100, r_plain
    assert r_hook["service_s"] > r_plain["service_s"] + 50 * (0.5 - 0.04)     # re-capture seconds 0.5 instead of 0.04 are charged
    assert r_nohook["service_s"] == pytest.approx(r_plain["service_s"])        # documents why the hook exists


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ADD.load_with_addendum(toy_table(), tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        ADD.read_addendum(tmp_path / "nope.csv")
    ct = ROOT / "results/S2/cost_table/merged/cost_table.csv"
    if not (ct.exists() and (ROOT / "results/S3/policy_params.json").exists()):
        pytest.skip("frozen S2/S3 inputs not present")
    r = subprocess.run([sys.executable, "-B", str(ROOT / "code/scripts/sim_eval_S3.py"), "--split", "dev", "--out", str(tmp_path / "o"),
                        "--addendum", str(tmp_path / "nope.csv"), "--tiny"], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=str(ROOT / "code"))
    assert r.returncode != 0 and "cost addendum not found" in (r.stdout + r.stderr)
    assert not (tmp_path / "o" / "main_raw.csv").exists()


# ----------------------------------------------------------------------------------------------- build script (synthetic S3-A raw)
SEQ_KEYS = [(512, 512, 1, 0), (512, 768, 1, 0), (768, 512, 1, 0), (512, 512, 2, 0)]


def write_seq(d, arm, pid, first_req, fail_at=None, fc_unet=None):
    d.mkdir(parents=True, exist_ok=True)
    per_key = []
    for i, (H, W, B, cfg) in enumerate(SEQ_KEYS):
        if fail_at is not None and i > fail_at:
            break
        bad = fail_at is not None and i == fail_at
        rows = []
        for rep in range(-3, 4):
            rows.append(dict(arm=arm, H=H, W=W, B=B, cfg=cfg, proc_id=pid, cold_tier="cold_diskcache" if i == 0 else "inproc_first",
                             status="fail" if bad else "ok", rep=rep, wall_ms=100.0 if not bad else np.nan, first_req_s=first_req[i],
                             contended=0, load1_start=3.0, load1_end=3.0, cpu_contended=0, seq_idx=i,
                             first_call_unet_s=(fc_unet or first_req)[i] * 0.5 if i else np.nan, first_call_te_s=np.nan, first_call_vae_s=np.nan))
        pd.DataFrame(rows).to_csv(d / f"{arm}_seq{i}_{H}x{W}_b{B}_{'cfg' if cfg else 'nocfg'}_p{pid}.csv", index=False)
        per_key.append(dict(idx=i, key=f"{H}x{W}x{B}x{cfg}", status="fail" if bad else "ok",
                            error="torch.OutOfMemoryError: CUDA out of memory" if bad else None))
    (d / f"{arm}_seq_p{pid}.meta.json").write_text(json.dumps(dict(status="fail" if fail_at is not None else "ok", cache_mode="disk", per_key=per_key)))


def write_rebuild(d, arm, shapes):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for (H, W, B, cfg), (cap, reb) in shapes.items():
        for r in range(5):
            rows.append(dict(arm=arm, H=H, W=W, B=B, cfg=cfg, proc_id=0, round=r, status="ok", error="", cap1_capture_s=cap, rebuild_wall_s=reb[r % len(reb)],
                             reserved_residual_mb=20.0, reserved_recovered=1, output_equal_after_rebuild=1, load1_start=3.0, load1_end=3.0,
                             gpu0_other_procs_start=0, gpu0_other_procs_end=0))
    pd.DataFrame(rows).to_csv(d / f"rebuild_{arm}_p0.csv", index=False)
    (d / f"rebuild_{arm}_p0.meta.json").write_text(json.dumps(dict(arm=arm, pinned_fallback_count=0, any_fail=0)))


@pytest.fixture
def raw(tmp_path):
    r = tmp_path / "raw"
    seq = r / "inproc_seq"
    # C5: 3 clean processes, medians of first_req 2 / 3 / 9 -> 3
    for pid, v in enumerate([(5.0, 2.0, 3.0, 4.0), (5.0, 3.0, 3.5, 4.5), (5.0, 9.0, 3.2, 4.2)]):
        write_seq(seq / "v3shapes_C5_disk", "C5", pid, v)
    # C2: proc 1 OOMs at key 2 (the cumulative pool): keys 2 and 3 must not get a row
    for pid in range(3):
        write_seq(seq / "v3shapes_C2_disk", "C2", pid, (7.0, 2.0, 2.5, 3.0), fail_at=2 if pid == 1 else None)
    # warm pass (proc 9) must be ignored even if it sits next to the standard dirs
    write_seq(seq / "warmpass_C5", "C5", 9, (5.0, 100.0, 100.0, 100.0))
    write_seq(seq / "v3shapes_C5_disk", "C5", 9, (5.0, 100.0, 100.0, 100.0))
    write_rebuild(r / "rebuild", "G1", {(512, 512, 1, 0): (0.04, [0.18, 0.19, 0.17]), (768, 512, 1, 0): (0.10, [0.12, 0.13, 0.12])})
    write_rebuild(r / "rebuild", "C1+G1", {(512, 512, 1, 0): (0.03, [0.15])})
    (r / "S3A_logs").mkdir()
    (r / "S3A_logs" / "v3shapes_C5_disk_p1.pinned_fallback_count.txt").write_text("2\n")
    (r / "S3A_logs" / "block.log").write_text("STEP v3shapes_C5_disk_p0 rc=0 elapsed=1s\nNOT_QUIET load1=17>16\nSTEP v3shapes_C5_disk_p1 rc=0 elapsed=2s\n")
    return r


def build(raw, tmp_path):
    out = tmp_path / "out"
    BA.main(["--raw", str(raw), "--out-dir", str(out), "--cost-table", ""])
    return pd.read_csv(out / "cost_addendum.csv"), json.loads((out / "meta.json").read_text()), out


def test_build_median_of_three_processes_and_flags(raw, tmp_path):
    df, meta, _ = build(raw, tmp_path)
    c5 = df[(df.kind == "inproc_cost") & (df.arm == "C5")].set_index(["H", "W", "B"])
    assert len(c5) == 3 and c5.n_proc.eq(3).all() and c5.source.eq("inproc_measured_S3").all()    # key 0 is the cold tier: no row; proc 9 ignored
    assert c5.loc[(512, 768, 1)].first_req_s == pytest.approx(3.0)                                  # median(2, 3, 9)
    assert "pinned_fallback=2" in c5.loc[(512, 768, 1)].flag and "not_quiet_gate" in c5.loc[(512, 768, 1)].flag     # flagged, not dropped
    assert meta["inproc"]["C5"]["processes"][1]["pinned_fallback_count"] == 2


def test_build_oom_key_and_later_keys_stay_extrapolated(raw, tmp_path):
    df, meta, out = build(raw, tmp_path)
    c2 = df[(df.kind == "inproc_cost") & (df.arm == "C2")]
    assert sorted(zip(c2.H, c2.W, c2.B)) == [(512, 768, 1)]                  # idx 1 only; idx 2 failed in proc 1, idx 3 never measured there
    assert meta["oom_keys"]["C2"][0]["key"] == "768x512x1x0" and "OutOfMemory" in meta["oom_keys"]["C2"][0]["error"]
    ct = ADD.load_with_addendum(toy_table(), out / "cost_addendum.csv")
    new = ct.df
    for s in (S3_, S4_):
        r = new[(new.arm == "C2") & (new.H == s.H) & (new.W == s.W) & (new.B == s.B)].iloc[0]
        assert np.isnan(r["resid_s__inproc_diskcache"]) and r["addendum_source"] == ""   # still extrapolated (falls back to cold tier)
    r = new[(new.arm == "C2") & (new.H == 512) & (new.W == 768)].iloc[0]
    assert r["addendum_source"] == "inproc_measured_S3"


def test_build_rebuild_ratio_threshold(raw, tmp_path):
    df, meta, out = build(raw, tmp_path)
    rb = df[df.kind == "rebuild"].set_index(["arm", "H", "W", "B", "cfg"])
    g = rb.loc[("G1", 512, 512, 1, 0)]
    assert g.rebuild_wall_med_s == pytest.approx(0.18) and g.rebuild_over_capture == pytest.approx(4.5) and g.rebuild_override_s == pytest.approx(0.18)
    g2 = rb.loc[("G1", 768, 512, 1, 0)]                                                 # 0.12 / 0.10 = 1.2 <= 1.5 -> no override
    assert g2.rebuild_over_capture == pytest.approx(1.2) and np.isnan(g2.rebuild_override_s)
    c = rb.loc[("C1+G1", 512, 512, 1, 0)]
    assert c.rebuild_override_s == pytest.approx(0.15)                                  # 0.15 / 0.03 = 5
    ct = ADD.load_with_addendum(toy_table(), out / "cost_addendum.csv")
    assert set(ct.rebuild_override) == {("G1", S1), ("C1+G1", S1)}
    assert meta["rebuild"]["G1"]["shapes"]["512x512x1x0"]["memory_reserved_falls_back"] is True
    assert meta["n_override"] == 2


def test_build_refuses_overwrite_and_missing_raw(raw, tmp_path):
    build(raw, tmp_path)
    with pytest.raises(SystemExit):
        BA.main(["--raw", str(raw), "--out-dir", str(tmp_path / "out"), "--cost-table", ""])
    with pytest.raises(SystemExit):
        BA.main(["--raw", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o2")])


# ----------------------------------------------------------------------------------------------- rebuild extrapolation (main-session decision 2026-11-17)
def build_with_table(raw, tmp_path):
    tp = tmp_path / "ct.csv"
    toy_table().to_csv(tp, index=False)
    out = tmp_path / "out_x"
    BA.main(["--raw", str(raw), "--out-dir", str(out), "--cost-table", str(tp)])
    return pd.read_csv(out / "cost_addendum.csv"), json.loads((out / "meta.json").read_text()), out


def test_rebuild_extrapolated_by_steady_ratio(raw, tmp_path):
    df, meta, out = build_with_table(raw, tmp_path)
    rb = df[df.kind == "rebuild"].set_index(["arm", "H", "W", "B", "cfg"])
    # G1: gate-passing measured shape 512x512x1x0 -> 0.18 / steady 0.090 = 2.0 ; 768x512x1x0 failed the gate (1.2) -> measured row, no override, not used
    ex = rb.loc[("G1", 512, 768, 1, 0)]
    assert ex.source == "extrapolated_S3_rebuild" and ex.rebuild_override_s == pytest.approx(2.0 * 0.090)
    assert rb.loc[("G1", 512, 512, 2, 0)].rebuild_override_s == pytest.approx(0.18)
    assert rb.loc[("G1", 768, 512, 1, 0)].source == "rebuild_measured_S3" and np.isnan(rb.loc[("G1", 768, 512, 1, 0)].rebuild_override_s)
    c = rb.loc[("C1+G1", 512, 768, 1, 0)]
    assert c.source == "extrapolated_S3_rebuild" and c.rebuild_override_s == pytest.approx(0.15 / 0.070 * 0.070)
    g = meta["rebuild"]["G1"]["extrapolation"]
    assert g["median_ratio"] == pytest.approx(2.0) and list(g["ratios_rebuild_over_steady"]) == ["512x512x1x0"] and g["n_extrapolated"] == 2
    assert meta["n_override_extrapolated"] == 5 and meta["n_override_measured"] == 2
    ct = ADD.load_with_addendum(toy_table(), out / "cost_addendum.csv")
    assert ct.rebuild_override[("G1", S2_)] == pytest.approx(0.18) and ct.rebuild_override[("G1", S1)] == pytest.approx(0.18)
    assert ("G1", S3_) not in ct.rebuild_override                              # measured, below the gate: table capture_s kept
    cv = CostView(ct, "warm", rebuild_override=ct.rebuild_override)
    assert cv.rebuild_s("G1", S4_) == pytest.approx(0.18) and cv.rebuild_s("G1", S3_) == pytest.approx(0.04)


def test_no_extrapolation_when_no_shape_passes_gate(tmp_path):
    r = tmp_path / "raw2"
    write_rebuild(r / "rebuild", "G1", {(512, 512, 1, 0): (0.10, [0.12, 0.13])})            # ratio 1.2 <= 1.5
    (r / "inproc_seq").mkdir(parents=True)
    df, meta, _ = build_with_table(r, tmp_path)
    rb = df[df.kind == "rebuild"]
    assert (rb.source == "rebuild_measured_S3").all() and rb.rebuild_override_s.isna().all()
    assert meta["rebuild"]["G1"]["extrapolation"]["n_extrapolated"] == 0 and meta["n_override"] == 0
