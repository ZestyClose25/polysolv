"""Tests for polysolv.evaluate and the HSP baseline (T15)."""

import numpy as np
import pandas as pd
import pytest

from polysolv.evaluate import regression_metrics, selectivity_frame, selectivity_metrics, spearman, topk_hit_rate


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 2], [1, 2]))


def test_regression_metrics():
    m = regression_metrics([0, 1, 2], [0, 1, 2])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == pytest.approx(1.0)
    assert regression_metrics([0, 2], [1, 1])["MAE"] == pytest.approx(1.0)


def test_topk():
    score = pd.Series([0.9, 0.8, 0.1, 0.5], index=list("abcd"))
    truth = pd.Series([True, False, True, False], index=list("abcd"))
    assert topk_hit_rate(score, truth, 2) == 0.5


def _toy_labels():
    rows = []
    # solvent 0: PS 20 wt%, PET 0.1 -> selective; 1: PS 20, PET 5 -> not; 2: PS M, PET 0.2 -> selective
    for sid, ps, pet, cens in [(0, 20.0, 0.1, False), (1, 20.0, 5.0, False), (2, np.nan, 0.2, True)]:
        rows.append((sid, "PS", "Th", ps, np.log10((50.0 if cens else ps) + 1e-3), cens))
        rows.append((sid, "PET", "Th", pet, np.log10(pet + 1e-3), False))
    lab = pd.DataFrame(rows, columns=["solvent_id", "polymer", "T_label", "wt_pct", "y", "censored"])
    solv = pd.DataFrame({"solvent_id": [0, 1, 2], "Th_C": [100.0, 100.0, 100.0]})
    return lab, solv


def test_selectivity_frame():
    lab, solv = _toy_labels()
    sel = selectivity_frame(lab, solv)
    assert sel["selective"].tolist() == [True, False, True]
    m = selectivity_metrics(pd.Series([3.0, 2.0, 1.0], index=[0, 1, 2]), sel, ks=(1, 2))
    assert m["top1_hit"] == 1.0 and m["top2_hit"] == 0.5 and m["n_selective"] == 2


def test_hsp_baseline_runs(cfg):
    if not (cfg["paths"]["processed_dir"] / "zhou_long.csv").exists():
        pytest.skip("run scripts.build_dataset first")
    from baselines.hsp_ranking import main
    from polysolv.config import DEFAULT_CONFIG

    r = main(str(DEFAULT_CONFIG))
    t = r["test"]
    assert t["spearman_neg_chi_prior"]["PS/RT"] > 0.5
    assert t["selectivity_chi_diff"]["n_selective"] > 0
