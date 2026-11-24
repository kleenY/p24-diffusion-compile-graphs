"""S2 equivalence tests: judge every (arm, shape) json in results/S2/equiv/ against the frozen tolerances
(tests/tolerances.json, unchanged) and check the provenance fields (code/tolerances/model/fused-weights SHA, time, device).
Results are produced by `python -m scripts.run_equivalence --arm <ARM> --keys ... --out results/S2/equiv`.
Skipped when the directory does not exist. A failing shape is a *result* (protocol sec 10: equiv_pass = 0 in the cost
table, compile tier not admitted on that shape) - this test makes it visible; it does not hide it."""
import hashlib, json, os, pathlib
import pytest

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
TOL_F = pathlib.Path(__file__).parent / "tolerances.json"
TOL = json.loads(TOL_F.read_text())
EQ = ROOT / "results/S2/equiv"
FILES = sorted(p for p in EQ.glob("*.json")) if EQ.exists() else []

pytestmark = pytest.mark.skipif(not EQ.exists() or not FILES, reason="results/S2/equiv/ does not exist or is empty")


# Cells judged NOT equivalent once (plan §5 / §9.8: never re-drawn). 2.7 run 2026-11-15 23:44-23:58:
# G1 serving 512x512 B3 no-CFG through the padded B4 bucket: RMSE ratio 2.665 > 2 (LPIPS 0.0054 ok; native B3 = 1.0).
JUDGED_NOT_EQUIVALENT = {("G1_pad", "512x512_b3_nocfg")}


def test_judged_not_equivalent_cells_stay_recorded_as_failures():
    for stem, k in JUDGED_NOT_EQUIVALENT:
        f = EQ / f"{stem}.json"
        m = json.loads(f.read_text())["by_key"][k]
        assert m["equiv_pass"] == 0 and m["rmse_ratio"] > TOL["c_rmse_ratio_max"], (stem, k, m)


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize("f", FILES, ids=[p.stem for p in FILES])
def test_arm_within_tolerance(f):
    r = json.loads(f.read_text())
    assert r["status"] == "ok", r.get("error")
    for k, m in r["by_key"].items():
        if (f.stem, k) in JUDGED_NOT_EQUIVALENT:  # a judged result (plan §5/§9.8), checked separately below
            continue
        assert m["rmse_ratio"] <= TOL["c_rmse_ratio_max"], (k, m)
        assert m["lpips_vs_E0"] <= TOL["lpips_vs_E0_mean_max"], (k, m)
        assert m["equiv_pass"] == 1, (k, m)


@pytest.mark.parametrize("f", FILES, ids=[p.stem for p in FILES])
def test_provenance_fields(f):
    r = json.loads(f.read_text())
    assert r["status"] == "ok", r.get("error")
    for fld in ("code_sha256_16", "tolerances_sha256", "model_revisions_sha256", "fused_weights_sha256", "time", "device"):
        assert r.get(fld), fld
    assert r["tolerances_sha256"] == _sha(TOL_F), "tolerances.json changed after this result was produced"
    # review S4: the fused-weights SHA is measured from the files; it must agree with model_revisions.json
    assert r.get("fused_weights_sha256_matches_revisions") is True, "measured fused-weights SHA != model_revisions.json"
    assert r["model_revisions_sha256"] == _sha(ROOT / "index/model_revisions.json")
    assert len(r["n_images"] if isinstance(r["n_images"], list) else [0] * r["n_images"]) == TOL["n_prompts"]


@pytest.mark.parametrize("f", FILES, ids=[p.stem for p in FILES])
def test_verdict_consistent_with_numbers(f):
    """equiv_pass stored in the json must equal what the frozen tolerances give from the stored metrics."""
    r = json.loads(f.read_text())
    if r["status"] != "ok":
        pytest.skip("failed run")
    for k, m in r["by_key"].items():
        exp = int(m["rmse_ratio"] <= TOL["c_rmse_ratio_max"] and m["lpips_vs_E0"] <= TOL["lpips_vs_E0_mean_max"])
        assert m["equiv_pass"] == exp, (k, m)
