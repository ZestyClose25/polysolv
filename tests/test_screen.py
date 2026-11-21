"""Tests for polysolv.screen (T28), polysolv.ensemble.calibration (T26) and polysolv.inference."""

import numpy as np
import pandas as pd
import pytest
import torch

from polysolv.ensemble import calibration
from polysolv.screen import confidence, temperature_grid, wt_from_log


def test_temperature_grid(cfg):
    s = cfg["screen"]
    assert temperature_grid(56.0, s).tolist() == [25.0, 30.0, 35.0, 40.0, 45.0]   # acetone: bp - 10
    assert temperature_grid(300.0, s)[-1] == s["T_max_C"]                          # capped at Zhou's Th range
    assert temperature_grid(float("nan"), s).tolist() == [25.0]


def test_wt_from_log_inverts_label():
    assert wt_from_log(np.log10(10.0 + 1e-3), 1e-3) == pytest.approx(10.0)
    assert wt_from_log(-3.0, 1e-3) == pytest.approx(0.0)


def test_confidence_levels():
    th = [0.15, 0.30]
    assert confidence(0.1, True, th) == "high"
    assert confidence(0.1, False, th) == "medium"
    assert confidence(0.5, True, th) == "low"


def test_calibration_monotone_toy():
    rng = np.random.default_rng(0)
    sig = rng.uniform(0.05, 1.0, 2000)
    y = rng.normal(0, 1, 2000)
    pred = pd.DataFrame({"y": y, "y_pred": y + rng.normal(0, sig), "y_std": sig, "censored": False})
    tab, summ = calibration(pred, 5)
    assert len(tab) == 5 and tab.mae.is_monotonic_increasing
    assert summ["spearman_sigma_abs_err"] > 0.3
    assert 0.6 < summ["coverage_1sigma"] < 0.76  # ~68 % for Gaussian errors


@pytest.fixture(scope="module")
def proc(cfg):
    from polysolv.dataset import load_processed
    d = cfg["paths"]["processed_dir"]
    if not (d / "graphs.pt").exists():
        pytest.skip("run scripts.build_dataset first")
    return load_processed(d)


def test_solvent_table_new_solvent(cfg, proc):
    from polysolv.inference import NEW_ID_START, solvent_table
    extra = pd.DataFrame({"smiles": ["O=C1CCC2OCC1O2"], "dD": [18.8], "dP": [10.6], "dH": [6.9], "bp_C": [227.0]})
    solv, s_desc, m = solvent_table(["CCO", "O=C1CCC2OCC1O2"], proc, cfg, extra)
    assert m["CCO"] < NEW_ID_START <= m["O=C1CCC2OCC1O2"]
    new = solv.set_index("solvent_id").loc[m["O=C1CCC2OCC1O2"]]
    assert bool(new.has_hsp) and 80 < new.V_cm3_mol < 140   # Cyrene ~ 103 cm3/mol
    assert s_desc.loc[m["O=C1CCC2OCC1O2"]].shape == (7,)


def test_screen_runs_with_untrained_model(cfg, proc):
    from polysolv.model.polysolv_net import build_model
    from polysolv.screen import screen
    torch.manual_seed(0)
    model = build_model(cfg, proc["s_desc"].shape[1], proc["p_desc"].shape[1]).eval()
    green = pd.read_csv(cfg["paths"]["green_solvents"]).head(12)
    ranked, grid = screen([model, model], proc, cfg, green, "PS", "PET", 100.0, 0.3)
    assert {"S_PS_lower", "S_PET_upper", "keep", "score"} <= set(grid.columns)
    assert (grid.T_C < grid.bp_used_C).all()
    assert (grid.sigma_PS == 0).all()  # identical members -> no spread
