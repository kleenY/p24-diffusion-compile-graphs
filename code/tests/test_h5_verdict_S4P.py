"""S4-P P1: scripts/h5_verdict_S4P.py on a synthetic raw frame (hand-computed thresholds). CPU only."""
import numpy as np
import pandas as pd
import pytest

from scripts import h5_verdict_S4P as H


def frame(ratios, n_seeds=10, extra_cell=None):
    """W2 N=1e4 disk: B1 = 100 s; v3 = ratios[i] x 100 for seed i; B0 = 120; best online other = B4 = 110. Optional extra cell (w, cold, N, v3, b1, b4)."""
    rows = []
    for i in range(n_seeds):
        for var, s in (("B1", 100.0), ("Ours-v3", 100.0 * ratios[i]), ("B0", 120.0), ("B4", 110.0), ("Ours", 104.0)):
            rows.append(dict(status="ok", variant=var, workload="W2", N=10000, seed=i, cold="disk", service_s=s))
    if extra_cell:
        w, cold, n, v3, b1, b4 = extra_cell
        for var, s in (("B1", b1), ("Ours-v3", v3), ("B0", 120.0), ("B4", b4), ("Ours", v3)):
            rows.append(dict(status="ok", variant=var, workload=w, N=n, seed=0, cold=cold, service_s=s))
    return pd.DataFrame(rows)


def test_h5a_h5c_counts_and_verdicts_with_ten_seeds():
    r = [1.00, 1.005, 1.01, 1.0099, 1.011, 0.99, 0.995, 1.02, 1.0, 0.98]      # <=1.01: 8 seeds ; <1: 3 seeds
    v = H.verdict(frame(r), "Ours-v3")
    assert v["H5a"]["seeds_le_1p01"] == 8 and v["H5a"]["verdict"] == "supported"
    assert v["H5c"]["seeds_lt_1"] == 3 and v["H5c"]["verdict"] == "not supported"
    assert v["H5c"]["mean_ratio"] == pytest.approx(np.mean(r)) and v["H5c"]["boot_ci95_mean_ratio"][0] <= np.mean(r) <= v["H5c"]["boot_ci95_mean_ratio"][1]
    r2 = [0.99] * 8 + [1.1, 1.1]
    v2 = H.verdict(frame(r2), "Ours-v3")
    assert v2["H5a"]["verdict"] == "supported" and v2["H5c"]["verdict"] == "supported"        # 8/10 seeds


def test_h5b_worst_cells_and_dev_style_label():
    base = frame([1.0] * 10, extra_cell=("W3", "empty", 100000, 104.0, 100.0, 101.0))      # v3/B1 = 1.04, v3/best online (B1=100 is best) = 1.04
    v = H.verdict(base, "Ours-v3")
    assert v["H5b"]["max_v3_over_B1"] == pytest.approx(1.04) and not v["H5b"]["pass_B1"] and v["H5b"]["cell_max_B1"] == ["W3", "empty", "100000"]
    assert v["H5b"]["verdict"] == "not supported"
    ok = frame([1.0] * 10, extra_cell=("W3", "hot", 1000, 101.5, 100.0, 105.0))            # 1.015 <= 1.02 and <= 1.03
    assert H.verdict(ok, "Ours-v3")["H5b"]["verdict"] == "supported"
    dev = H.verdict(frame([1.0] * 5, n_seeds=5), "Ours-v3")
    assert dev["H5a"]["verdict"].startswith("dev-style")


def test_best_online_excludes_b0_and_oracle():
    f = frame([1.0] * 10)
    c = H.cell_ratios(f, "Ours-v3")
    assert c.loc[("W2", "disk", 10000), "best_online"] == "B1" and c.loc[("W2", "disk", 10000), "v3/B0"] == pytest.approx(100 / 120)
