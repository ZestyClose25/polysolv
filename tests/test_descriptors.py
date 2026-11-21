"""Tests for polysolv.features.descriptors (T13)."""

import numpy as np
import pandas as pd
import pytest

from polysolv.data.merge import merge_all
from polysolv.data.splits import scaffold_split
from polysolv.features.descriptors import (
    Scaler, add_molar_volume, build_descriptors, fit_molar_volume, mcgowan_vx,
)


def test_mcgowan_reference_values():
    # Abraham & McGowan: water 0.1673, toluene 0.8573 (units of 100 cm3/mol)
    assert mcgowan_vx("O") == pytest.approx(16.73, abs=0.01)
    assert mcgowan_vx("Cc1ccccc1") == pytest.approx(85.73, abs=0.01)


def test_molar_volume_calibration(cfg):
    anchors = cfg["descriptors"]["molar_volume_anchors"]
    a, b = fit_molar_volume(anchors)
    errors = [abs(a * mcgowan_vx(s) + b - v) / v for s, v in anchors.items()]
    assert np.mean(errors) < 0.06  # ~4 % mean error on the anchor set


def test_scaler_ignores_nan_and_imputes_mean():
    df = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
    sc = Scaler().fit(df, ["x"])
    z = sc.transform(df)["x"].tolist()
    assert z == pytest.approx([-1.0, 1.0, 0.0])
    assert Scaler.from_dict(sc.to_dict()).transform(df)["x"].tolist() == pytest.approx(z)


@pytest.fixture(scope="module")
def built(zhou_clean, cfg):
    d = cfg["descriptors"]
    t = merge_all(zhou_clean[0], pd.read_csv(cfg["paths"]["polymer_props"]))
    solvents = add_molar_volume(t["solvents"], d["molar_volume_anchors"])
    split = scaffold_split(solvents, tuple(cfg["split"]["fractions"]), cfg["split"]["seed"])
    return solvents, t["polymers"], split, build_descriptors(solvents, t["polymers"], split["train"], d)


def test_every_solvent_has_volume(built):
    solvents = built[0]
    assert solvents["V_cm3_mol"].notna().all() and (solvents["V_cm3_mol"] > 10).all()


def test_descriptor_tables(built, cfg):
    solvents, polymers, split, (s_desc, p_desc, scalers) = built
    d = cfg["descriptors"]
    assert list(s_desc.columns) == d["solvent_features"] + d["solvent_flags"]
    assert list(p_desc.columns) == d["polymer_features"] + d["polymer_flags"]
    assert len(s_desc) == len(solvents) and len(p_desc) == 8
    assert not s_desc.isna().any().any() and not p_desc.isna().any().any()
    # scaler fitted on train only: train columns have mean 0 (where values exist)
    train = s_desc.loc[split["train"]]
    assert train["V_cm3_mol"].mean() == pytest.approx(0.0, abs=1e-9)
    # imputed rows sit at 0 and carry flag 0
    no_bp = s_desc[s_desc.has_bp == 0]
    assert len(no_bp) > 0 and (no_bp.bp_C == 0).all()
    assert p_desc.loc["PS", "has_Tm"] == 0 and p_desc.loc["PS", "Tm_K"] == 0
