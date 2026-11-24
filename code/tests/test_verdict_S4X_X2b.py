"""S4-X X2b: scripts/m1234_verdict_S4X_X2b.py on SYNTHETIC data (CPU only, no GPU, no NVML call, no real numbers).
  * M1-M4 judging function `eval_cell` and `_med` are byte-identical to the X2 verdict (scripts/m1234_verdict_S4X_X2.py): source diff = empty
  * holds + cap touched (attribution arm above C+delta / allocator OOMs) -> sentence A ; holds + not touched -> sentence B with X and Y ; M1 / M2 / M3 / M4 violated -> FAILS ; no power ; incomplete
  * allocator-conf mismatch (cap / uncap / uncapxe) -> INVALID ; wrong seed -> INVALID ; a trig process inside the judged tree -> INVALID
  * the report-only trigger arm (own directory) never changes the verdict ; triggered / not triggered / missing / repair evidence statements
  * --rec is required (no default) ; the frozen file's last line must carry g=0.8
Run from code/ : CUDA_VISIBLE_DEVICES='' python -B -m pytest tests/test_verdict_S4X_X2b.py -q"""
import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
os.environ.setdefault("PROJ_ROOT", str(ROOT))

from scripts import m1234_verdict_S4X_X2b as V  # noqa: E402

C, DELTA, BPOOL = 16384.0, 1000.0, 6565.1
LIM = C + DELTA
HIDX = list(range(0, 5)) + list(range(295, 300)) + [10, 20, 30]


def _w(path, text):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(text)


def write_proc(root, spec):
    if spec.get("omit"):
        return
    arm, ver, w, p = spec["arm"], spec["ver"], spec["w"], spec["p"]
    d = pathlib.Path(root) / ("trig" if ver == "trig" else ver) if spec.get("dir") is None else pathlib.Path(spec["dir"])
    stem = f"{w}_N300_seed{spec.get('seed', 134)}_B"
    tag = f"{arm}_{stem}_p{p}"
    n = 300
    status = spec.get("status", "ok")
    meta = dict(status=status, policy=spec.get("meta_policy", arm), trace=f"../results/S4X/x2b/confirm/workload/{stem}.jsonl", service_s=spec["service_s"], n_requests=n, n_capture_failed=spec.get("n_capture_failed", 0),
                event_counts={"steady": 270, "capture_denied": spec.get("n_denied", 0)}, error=spec.get("error", ""), pinned_fallback_count=0, n_evicted_total=0)
    _w(d / f"{tag}.meta.json", json.dumps(meta))
    _w(d / f"{tag}.x2.json", json.dumps(dict(calls_match_rows=True, alloc_conf_env=spec.get("alloc_conf"), torch_begin=dict(allocated_mb=2056.6),
                                              torch_end=dict(max_allocated_mb=9000.0, max_reserved_mb=(spec["peak"] or 0) - 700, num_ooms=spec.get("num_ooms", 0), num_alloc_retries=spec.get("num_alloc_retries", 0)))))
    var = dict(LABEL=f"{arm}_{ver}_{w}_p{p}", POLICY=arm, PROC=str(p), TRACE=f"x/{stem}.jsonl", CAP_MB=(str(spec["cap_mb"]) if spec.get("cap_mb") is not None else ""),
               ALLOC_CONF=(spec.get("alloc_conf") or ""), BUDGET_MB=(str(spec["budget"]) if spec.get("budget") is not None else ""))
    _w(d / f"{tag}.x2run.json", json.dumps(dict(tag=tag, rc=0 if status == "ok" else 1, t0=0.0, t1=100.0, variant=var, meta_status=status)))
    _w(d / f"{tag}.nvml.window.json", json.dumps(dict(peak_used_mb=spec["peak"], baseline_used_mb=500.0, achieved_hz=20.0, n_samples=2000)))
    tiers = spec["tiers"]
    cols = "idx,req_id,H,W,B,cfg,status,phase,served_by,graph_mem_mb" if not spec.get("no_served_by") else "idx,req_id,H,W,B,cfg,status,phase,graph_mem_mb"
    rows = [cols]
    for i in range(n):
        rows.append(f"{i},r{i},512,512,1,0,ok,req," + ("" if spec.get("no_served_by") else f"{tiers[i]},") + "0.0")
    _w(d / f"{tag}.csv", "\n".join(rows) + "\n")
    hrows = ["idx,req_id,req_shape,served_key,call_kind,dtype,tensor_shape,nbytes,sha256"]
    for i in sorted(spec["hash_idx"]):
        hrows.append(f"{i},r{i},512x512x1x0,k,E0,float16,1x3,6,{spec['hashes'][i]}")
    _w(d / f"{tag}.x2hash.csv", "\n".join(hrows) + "\n")


def synth(raw, rep, mutate=None, with_trig=True, xe_peak=13300.0):
    """A complete 36-process judged design (+ 4 trig processes in `rep`) that HOLDS with the cap NOT touched."""
    for w in V.STREAMS:
        for arm in V.ARMS:
            for ver in ("cap", "uncap", "uncapxe"):
                for p in V.PROCS:
                    jit = 1.0 + 0.002 * p
                    base = 68.0 * jit
                    peak = {"uncap": (20000.0 if arm == "Ours" else 20800.0), "cap": 13500.0, "uncapxe": xe_peak}[ver]
                    spec = dict(arm=arm, ver=ver, w=w, p=p, service_s=base * (1.01 if ver == "cap" else 1.0), peak=peak, status="ok", tiers=["G1"] * 300, hash_idx=HIDX, hashes={i: f"h{i}" for i in HIDX},
                                cap_mb=(C if ver == "cap" else None), budget=(BPOOL if ver == "cap" else None), alloc_conf={"cap": V.ALLOC_CAP, "uncap": None, "uncapxe": V.ALLOC_XE}[ver])
                    if mutate:
                        mutate(spec)
                    write_proc(raw, spec)
            if with_trig:
                spec = dict(arm=arm, ver="trig", w=w, p=0, service_s=68.7, peak=16900.0, status="ok", tiers=["G1"] * 300, hash_idx=HIDX, hashes={i: f"h{i}" for i in HIDX}, cap_mb=C, budget=BPOOL,
                            alloc_conf=V.ALLOC_TRIG, dir=str(pathlib.Path(rep) / "trig"))
                if mutate:
                    mutate(spec)
                write_proc(rep, spec)


def run(tmp_path, mutate=None, **kw):
    raw, rep = tmp_path / "raw", tmp_path / "raw_report"
    synth(raw, rep, mutate, **kw)
    return V.verdict(raw, rep, C, DELTA, BPOOL), raw, rep


def cell(Vd, w, arm):
    return next(c for c in Vd["cells"] if c["stream"] == w and c["arm"] == arm)


def pick(spec, **kv):
    return all(spec.get(k) == v for k, v in kv.items())


# ---------------------------------------------------------------------------------------------------------------- structure
def test_m1_m4_functions_are_verbatim_copies_of_x2():
    def segs(p):
        s = pathlib.Path(p).read_text()
        return {n.name: ast.get_source_segment(s, n) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    a, b = segs(CODE / "scripts/m1234_verdict_S4X_X2.py"), segs(CODE / "scripts/m1234_verdict_S4X_X2b.py")
    for name in ("eval_cell", "_med"):
        assert a[name] == b[name], f"{name} differs from the X2 verdict"
    assert (CODE / "scripts/m1234_verdict_S4X_X2.py").stat().st_mode & 0o777 == 0o444, "the X2 verdict must stay 0444"


def test_constants():
    assert V.SEED == 134 and V.ARMS == ("Ours", "B4") and V.JUDGED == ("Ours", "B4") and V.STREAMS == ("W2", "W5") and V.RATIO_MAX == 1.03
    assert V.ALLOC_CAP == "garbage_collection_threshold:0.8,expandable_segments:True" and V.ALLOC_XE == "expandable_segments:True"
    assert "B1-AOT" not in V.ARMS and "B1-AOT" not in V.JUDGED and "B1-AOT" not in V.LABEL_RE.pattern


def test_rec_is_required_and_g_must_be_08(tmp_path):
    src = (CODE / "scripts/m1234_verdict_S4X_X2b.py").read_text()
    assert 'ap.add_argument("--rec", required=True' in src and "default=DEFAULT_REC" not in src and "--c\"" not in src
    r = subprocess.run([sys.executable, "-B", "-m", "scripts.m1234_verdict_S4X_X2b"], cwd=CODE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert r.returncode == 2 and "--rec" in r.stderr and "required" in r.stderr
    p = tmp_path / "R.txt"
    p.write_text("x\nC=16384 delta=1156 g=0.8 margin=0 B_pool=6565.1\n\n")
    assert V.parse_rec(p) == dict(C=16384.0, delta=1156.0, g="0.8", margin=0.0, B_pool=6565.1)
    for bad in ("C=16384 delta=1156 g=0.6 margin=0 B_pool=6565.1", "C=16384 delta=1156 g=none margin=0 B_pool=6565.1", "NONE_PASSED", "C=16384 delta=1156"):
        p.write_text("x\n" + bad + "\n")
        with pytest.raises(ValueError):
            V.parse_rec(p)
    r = subprocess.run([sys.executable, "-B", "-m", "scripts.m1234_verdict_S4X_X2b", "--rec", str(p), "--out", str(tmp_path / "o")], cwd=CODE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert r.returncode == 2 and "X2B_VERDICT_ERROR" in r.stdout


# ---------------------------------------------------------------------------------------------------------------- holds (two wordings)
def test_holds_cap_not_touched_sentence_b(tmp_path):
    Vd, raw, rep = run(tmp_path)
    assert Vd["verdict"] == "HOLDS", (Vd["verdict"], Vd["integrity"][:3])
    assert all(c["status"] == "PASS" and c["acts"] for c in Vd["cells"])
    assert Vd["bound_touched"]["touched_any"] is False and Vd["sentence_class"] == "holds_cap_not_touched"
    s = Vd["paper_sentence"]
    X = V._med([V._med([20000.0] * 3), V._med([20800.0] * 3), V._med([20000.0] * 3), V._med([20800.0] * 3)]) / 1024
    Y = 13500.0 / 1024
    assert f"from about {X:.1f} GB to {Y:.1f} GB" in s and "the 16 GiB cap was never reached" in s and "removed fragmentation" in s and "outputs can change (X2)" in s, s
    assert "no new" not in s and "the new seed did not trigger the failure; X2b holds but is not repair evidence" == Vd["trigger"]["statement"]


@pytest.mark.parametrize("how", ["attribution_peak", "cap_oom", "cap_retry"])
def test_holds_cap_touched_sentence_a(tmp_path, how):
    def mut(s):
        if how == "attribution_peak" and s["ver"] == "uncapxe" and s["arm"] == "Ours" and s["w"] == "W2":
            s["peak"] = LIM + 500
        if how == "cap_oom" and s["ver"] == "cap" and pick(s, arm="B4", w="W5", p=1):
            s["num_ooms"] = 3
        if how == "cap_retry" and s["ver"] == "cap" and pick(s, arm="Ours", w="W2", p=2):
            s["num_alloc_retries"] = 1
    Vd, *_ = run(tmp_path, mut)
    assert Vd["verdict"] == "HOLDS", (Vd["verdict"], Vd["integrity"][:3])
    assert Vd["bound_touched"]["touched_any"] is True and Vd["sentence_class"] == "holds_cap_touched" and Vd["paper_sentence"].startswith("With the allocator cap, garbage-collection threshold and expandable segments, whole-process device memory stayed within C+δ")
    assert "≤3% time cost" in Vd["paper_sentence"]


def test_attribution_peak_must_exceed_in_all_three_processes(tmp_path):
    def mut(s):
        if s["ver"] == "uncapxe" and pick(s, arm="Ours", w="W2", p=0):
            s["peak"] = LIM + 500        # only 1 of 3 above the line
    Vd, *_ = run(tmp_path, mut)
    e = next(c for c in Vd["bound_touched"]["cells"] if c["stream"] == "W2" and c["arm"] == "Ours")
    assert e["uncapxe_exceeds_all3"] is False and e["touched"] is False and Vd["bound_touched"]["touched_any"] is False


# ---------------------------------------------------------------------------------------------------------------- M1-M4, power, completeness
def test_m1_violated_fails(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(peak=LIM + 100) if (s["ver"] == "cap" and pick(s, arm="Ours", w="W2", p=1)) else None)
    assert Vd["verdict"] == "FAILS" and not cell(Vd, "W2", "Ours")["M1"] and cell(Vd, "W2", "Ours")["M3"] and Vd["paper_sentence"] is None


def test_m1_boundary_passes(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(peak=LIM) if s["ver"] == "cap" else None)
    assert Vd["verdict"] == "HOLDS"


def test_m2_crash_fails(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(status="fail", error="CUDA out of memory. Tried to allocate") if (s["ver"] == "cap" and pick(s, arm="B4", w="W5", p=2)) else None)
    c = cell(Vd, "W5", "B4")
    assert Vd["verdict"] == "FAILS" and not c["M2"] and c["n_crash"] == 1 and c["n_crash_oom"] == 1


def test_m3_same_tier_hash_difference_fails(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s["hashes"].__setitem__(2, "DIFFERENT") if (s["ver"] == "cap" and pick(s, arm="Ours", w="W5", p=0)) else None)
    c = cell(Vd, "W5", "Ours")
    assert Vd["verdict"] == "FAILS" and not c["M3"] and c["m3_totals"]["n_same_tier_diff"] == 1 and Vd["paper_sentence"] is None
    assert "M3" in " ".join(c["reasons"])


def test_m3_tier_change_is_not_a_difference(tmp_path):
    def mut(s):
        if s["ver"] == "cap" and s["arm"] == "Ours":
            for i in (10, 20, 30):
                s["tiers"][i] = "E0"
                s["hashes"][i] = f"hcap{i}"
    Vd, *_ = run(tmp_path, mut)
    assert Vd["verdict"] == "HOLDS" and cell(Vd, "W2", "Ours")["m3_totals"]["n_tier_changed_all"] == 9


@pytest.mark.parametrize("ratio,expect", [(1.029, "HOLDS"), (1.031, "FAILS")])
def test_m4_line(tmp_path, ratio, expect):
    def mut(s):
        if s["ver"] == "cap" and s["arm"] == "Ours" and s["w"] == "W2":
            s["service_s"] = s["service_s"] / 1.01 * ratio
    Vd, *_ = run(tmp_path, mut)
    assert Vd["verdict"] == expect


def test_no_power_when_an_arm_never_acts(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(peak=17000.0) if (s["ver"] == "uncap" and s["arm"] == "Ours") else None)
    assert Vd["verdict"] == "NO_POWER" and Vd["no_power_arms"] == ["Ours"] and cell(Vd, "W2", "Ours")["status"] == "NOT_ACTING" and Vd["paper_sentence"] is None


def test_one_stream_not_acting_is_excluded(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(peak=17000.0) if (s["ver"] == "uncap" and s["arm"] == "Ours" and s["w"] == "W5") else None)
    assert Vd["verdict"] == "HOLDS" and Vd["not_acting_cells"] == ["W5/Ours"]


def test_incomplete_and_definitive_failure(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(omit=True) if (s["ver"] == "cap" and pick(s, arm="B4", w="W2", p=2)) else None)
    assert Vd["verdict"] == "INCOMPLETE" and Vd["incomplete_cells"] == ["W2/B4"]
    def mut(s):
        if s["ver"] == "cap" and pick(s, arm="B4", w="W2", p=2): s["omit"] = True
        if s["ver"] == "cap" and pick(s, arm="B4", w="W2", p=0): s["peak"] = 19000.0
    Vd, *_ = run(tmp_path / "b", mut)
    assert Vd["verdict"] == "FAILS"


def test_missing_attribution_process_does_not_change_the_verdict(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(omit=True) if s["ver"] == "uncapxe" else None)
    assert Vd["verdict"] == "HOLDS" and Vd["bound_touched"]["touched_any"] is False


# ---------------------------------------------------------------------------------------------------------------- invalid processes
@pytest.mark.parametrize("ver,wrong", [("cap", "garbage_collection_threshold:0.8"), ("cap", "expandable_segments:True"), ("cap", None), ("cap", "expandable_segments:True,garbage_collection_threshold:0.8"),
                                        ("uncap", "expandable_segments:True"), ("uncap", "garbage_collection_threshold:0.8"), ("uncapxe", None), ("uncapxe", V.ALLOC_CAP), ("uncapxe", "expandable_segments:True,x:1")])
def test_alloc_conf_mismatch_is_invalid(tmp_path, ver, wrong):
    def mut(s):
        if s["ver"] == ver and pick(s, arm="Ours", w="W2", p=1):
            s["alloc_conf"] = wrong
    Vd, *_ = run(tmp_path, mut)
    assert Vd["verdict"] == "INVALID" and any("PYTORCH_CUDA_ALLOC_CONF" in x for x in Vd["integrity"]), Vd["integrity"]


def test_wrong_seed_is_invalid(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(seed=132) if (s["ver"] == "uncap" and pick(s, arm="B4", w="W5", p=0)) else None)
    assert Vd["verdict"] == "INVALID" and any("seed-134" in x for x in Vd["integrity"])


def test_cap_budget_policy_checks_are_invalid(tmp_path):
    for mut in (lambda s: s.update(cap_mb=18432.0) if (s["ver"] == "cap" and pick(s, arm="Ours", w="W2", p=0)) else None,
                lambda s: s.update(budget=990.0) if (s["ver"] == "cap" and pick(s, arm="Ours", w="W2", p=0)) else None,
                lambda s: s.update(meta_policy="Ours") if (s["ver"] == "uncap" and pick(s, arm="B4", w="W5", p=0)) else None,
                lambda s: s.update(cap_mb=14336.0) if (s["ver"] == "uncapxe" and pick(s, arm="B4", w="W5", p=0)) else None):
        d = tmp_path / str(id(mut))
        Vd, *_ = run(d, mut)
        assert Vd["verdict"] == "INVALID", Vd["integrity"]


def test_trig_process_in_the_judged_tree_is_invalid(tmp_path):
    raw, rep = tmp_path / "raw", tmp_path / "raw_report"
    synth(raw, rep)
    write_proc(raw, dict(arm="Ours", ver="trig", w="W2", p=0, service_s=68.0, peak=16900.0, tiers=["G1"] * 300, hash_idx=HIDX, hashes={i: f"h{i}" for i in HIDX}, cap_mb=C, budget=BPOOL, alloc_conf=V.ALLOC_TRIG,
                         dir=str(raw / "cap")))
    Vd = V.verdict(raw, rep, C, DELTA, BPOOL)
    assert Vd["verdict"] == "INVALID" and any("not a judged" in x for x in Vd["integrity"])


def test_missing_served_by_is_invalid(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s.update(no_served_by=True) if (s["ver"] == "cap" and pick(s, arm="Ours", w="W2", p=0)) else None)
    assert Vd["verdict"] == "INVALID" and any("TIER FIELD" in x for x in Vd["integrity"])


# ---------------------------------------------------------------------------------------------------------------- the report-only trigger arm
def test_trig_garbage_never_changes_the_verdict(tmp_path):
    base, *_ = run(tmp_path / "a")
    def mut(s):
        if s["ver"] == "trig":
            s.update(status="fail", error="out of memory", service_s=999.0, peak=30000.0, num_ooms=29, alloc_conf="something else", cap_mb=1.0)
            s["hashes"][30] = "DIFFERENT"
    Vd, *_ = run(tmp_path / "b", mut)
    assert Vd["verdict"] == base["verdict"] == "HOLDS" and Vd["failing_cells"] == [] and Vd["paper_sentence"] == base["paper_sentence"]
    assert any("configuration differs" in n for n in Vd["report_notes"])


def test_trig_triggered_and_repair_evidence(tmp_path):
    def mut(s):
        if s["ver"] == "trig" and pick(s, arm="B4", w="W5"):
            s.update(num_ooms=29)
            s["hashes"][30] = "DIFFERENT"
    Vd, *_ = run(tmp_path, mut)
    t = Vd["trigger"]
    e = next(c for c in t["cells"] if c["stream"] == "W5" and c["arm"] == "B4")
    assert Vd["verdict"] == "HOLDS" and t["any_triggered"] is True and e["triggered"] and e["x2b_cap_cell_clean"] and e["repair_evidence"] and t["statement"].startswith("repair evidence")
    assert next(c for c in t["cells"] if c["stream"] == "W2" and c["arm"] == "Ours")["triggered"] is False


def test_trig_triggered_by_hash_only(tmp_path):
    Vd, *_ = run(tmp_path, lambda s: s["hashes"].__setitem__(4, "X") if (s["ver"] == "trig" and pick(s, arm="Ours", w="W2")) else None)
    e = next(c for c in Vd["trigger"]["cells"] if c["stream"] == "W2" and c["arm"] == "Ours")
    assert e["triggered"] and e["hash_vs_uncap_p0"]["n_same_tier_diff"] == 1 and e["num_ooms"] == 0


def test_trig_triggered_but_cap_not_clean_is_not_a_repair(tmp_path):
    def mut(s):
        if s["ver"] == "trig" and pick(s, arm="B4", w="W5"): s.update(num_ooms=5)
        if s["ver"] == "cap" and pick(s, arm="B4", w="W5", p=0): s.update(num_ooms=1)
    Vd, *_ = run(tmp_path, mut)
    e = next(c for c in Vd["trigger"]["cells"] if c["stream"] == "W5" and c["arm"] == "B4")
    assert e["triggered"] and not e["x2b_cap_cell_clean"] and not e["repair_evidence"] and "NOT a repair" in Vd["trigger"]["statement"]


def test_trig_missing_is_reported_not_fatal(tmp_path):
    Vd, *_ = run(tmp_path, with_trig=False)
    assert Vd["verdict"] == "HOLDS" and all(not c["present"] for c in Vd["trigger"]["cells"]) and "incomplete" in Vd["trigger"]["statement"]


# ---------------------------------------------------------------------------------------------------------------- the CLI and the output files
def test_cli_writes_outputs_and_last_line(tmp_path):
    raw, rep = tmp_path / "raw", tmp_path / "raw_report"
    synth(raw, rep)
    rec = tmp_path / "REC.txt"
    rec.write_text("draft\nC=16384 delta=1000 g=0.8 margin=0 B_pool=6565.1\n")
    out = tmp_path / "verdict"
    r = subprocess.run([sys.executable, "-B", "-m", "scripts.m1234_verdict_S4X_X2b", "--rec", str(rec), "--raw", str(raw), "--report-raw", str(rep), "--out", str(out)], cwd=CODE,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.rstrip().splitlines()[-1] == "X2B_VERDICT=HOLDS", r.stdout[-1500:]
    for f in ("verdict_x2b.json", "cell_table.csv", "process_table.csv", "VERDICT_X2b.txt"):
        assert (out / f).exists(), f
    txt = (out / "VERDICT_X2b.txt").read_text()
    assert "BOUND TOUCHED" in txt and "TRIGGER / REPAIR" in txt and "sentence (holds_cap_not_touched)" in txt
    assert len((out / "process_table.csv").read_text().splitlines()) == 1 + 36 + 4
    hdr = (out / "process_table.csv").read_text().splitlines()[0].split(",")
    assert "n_rerecord" in hdr and "num_ooms" in hdr and "num_alloc_retries" in hdr and "n_evicted_total" in hdr and "alloc_conf" in hdr and "nvml_peak_mb" in hdr
    # an INVALID verdict ends with exit code 2 and the last line says so
    raw2, rep2 = tmp_path / "raw2", tmp_path / "rep2"
    synth(raw2, rep2, lambda s: s.update(seed=132) if (s["ver"] == "uncap" and pick(s, arm="B4", w="W5", p=0)) else None)
    r2 = subprocess.run([sys.executable, "-B", "-m", "scripts.m1234_verdict_S4X_X2b", "--rec", str(rec), "--raw", str(raw2), "--report-raw", str(rep2), "--out", str(tmp_path / "v2")], cwd=CODE,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    assert r2.returncode == 2 and r2.stdout.rstrip().splitlines()[-1] == "X2B_VERDICT=INVALID"
