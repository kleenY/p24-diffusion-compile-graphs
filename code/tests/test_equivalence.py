"""Equivalence tests. Arms are judged against the frozen tolerances in tests/tolerances.json.
Arm results are produced by `python -m scripts.run_equivalence --arm <ARM> --out results/S1/equiv`.
The E0-vs-diffusers test runs live on the GPU."""
import json, os, pathlib
import pytest
import torch

ROOT = pathlib.Path(os.environ.get("PROJ_ROOT", pathlib.Path(__file__).resolve().parents[2]))
TOL = json.loads((pathlib.Path(__file__).parent / "tolerances.json").read_text())
EQ = ROOT / "results/S1/equiv"
ARMS = sorted(p.stem for p in EQ.glob("*.json")) if EQ.exists() else []


@pytest.mark.parametrize("arm", ARMS)
def test_arm_within_tolerance(arm):
    r = json.loads((EQ / f"{arm}.json").read_text())
    assert r["status"] == "ok", r.get("error")
    for k, m in r["by_key"].items():
        assert m["rmse_ratio"] <= TOL["c_rmse_ratio_max"], (k, m)
        assert m["lpips_vs_E0"] <= TOL["lpips_vs_E0_mean_max"], (k, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("cfg", [False, True])
def test_E0_matches_diffusers(cfg):
    from p24.pipeline import LCMPipe, ShapeKey
    p = LCMPipe(str(ROOT / "data/models/sd15_lcm_fused"))
    key = ShapeKey(512, 512, 2, cfg)
    prompts, seeds = ["a red bicycle", "a bowl of ramen"], [11, 12]
    ours = p.run(p.make_inputs(prompts, key, seeds), cfg)
    ref = p.diffusers_reference(prompts, key, seeds)
    diff = (ours.float() - ref.float()).abs().max().item()
    assert diff <= 1 / 255, diff
