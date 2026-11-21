"""Sanity checks on the hand-built CSVs in data/external (T4, T6, T11)."""

import pandas as pd
import pytest
from rdkit import Chem

POLYMERS = {"EVOH", "PE", "PET", "PP", "PS", "PVC", "PA6", "PA66"}

# Zhou Table 1: experimental points used to CALIBRATE COSMO-RS -> must not be in the external test set.
CALIBRATION_POINTS = {
    ("EVOH", "67-68-5", 368.15), ("PE", "108-88-3", 383.15), ("PP", "108-88-3", 383.15),
    ("PS", "108-88-3", 383.15), ("PET", "67-68-5", 408.15), ("PVC", "109-99-9", 323.15),
    ("PA6", "64-19-7", 363.15), ("PA66", "67-68-5", 408.15),
}


def _read(cfg, key):
    path = cfg["paths"][key]
    if not path.exists():
        pytest.skip(f"{path} not built yet")
    return pd.read_csv(path)


def test_polymer_props(cfg):
    df = _read(cfg, "polymer_props")
    assert set(df.polymer) == POLYMERS
    for smi in df.repeat_smiles:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        assert sum(a.GetSymbol() == "*" for a in mol.GetAtoms()) == 2, smi
    assert Chem.MolFromSmiles(df.set_index("polymer").loc["PS", "repeat_smiles"]).GetNumAtoms() == 10
    assert df[["dD", "dP", "dH", "Tg_K", "density_g_cm3"]].notna().all().all()
    assert df.hsp_source.notna().all() and df.thermal_source.notna().all()


def test_exp_test(cfg):
    df = _read(cfg, "exp_test")
    assert set(df.polymer) <= POLYMERS
    assert {"PS", "PET"} <= set(df.polymer)
    assert set(df.label) <= {"good", "non"}
    assert df.source.notna().all()
    for smi in df.smiles:
        assert Chem.MolFromSmiles(smi) is not None, smi
    keys = set(zip(df.polymer, df.cas, df.T_K.round(2)))
    assert not keys & CALIBRATION_POINTS


def test_green_solvents(cfg):
    df = _read(cfg, "green_solvents")
    assert len(df) >= 50
    assert not df.cas.duplicated().any()
    assert set(df.chem21) <= {"recommended", "problematic", "hazardous", "highly_hazardous"}
    for smi in df.smiles:
        assert Chem.MolFromSmiles(smi) is not None, smi
