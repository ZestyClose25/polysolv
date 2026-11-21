"""Tests for the XGBoost baseline feature code (T16). Full training is not run in pytest (~4 min)."""

import pandas as pd
import pytest

pytest.importorskip("xgboost")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

from baselines.xgb_baseline import build_features, morgan_bits, polymer_oligomer  # noqa: E402


@pytest.mark.parametrize("smiles,formula", [
    ("*CC(*)c1ccccc1", "C24H26"),                  # PS trimer
    ("*OCCOC(=O)c1ccc(cc1)C(=O)*", "C30H26O12"),   # PET trimer
    ("*CC*", "C6H14"),                             # PE trimer = hexane
])
def test_trimer_formula(smiles, formula):
    tri = polymer_oligomer(smiles, 3)
    assert rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(tri)) == formula
    assert "*" not in tri


def test_pet_trimer_has_ester_links():
    tri = Chem.MolFromSmiles(polymer_oligomer("*OCCOC(=O)c1ccc(cc1)C(=O)*", 3))
    esters = tri.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O[CH2]"))
    assert len(esters) == 5  # 2 per repeat unit, minus the capped chain end


def test_morgan_bits_shape():
    fp = morgan_bits(["CCO", "c1ccccc1"], 2, 512)
    assert fp.shape == (2, 512) and fp.max() == 1


def test_build_features_shape(cfg):
    d = cfg["paths"]["processed_dir"]
    if not (d / "zhou_long.csv").exists():
        pytest.skip("run scripts.build_dataset first")
    labels = pd.read_csv(d / "zhou_long.csv").head(50)
    xcfg = dict(cfg["xgb"], fp_bits=256)
    X, names = build_features(labels, pd.read_csv(d / "solvents.csv"), pd.read_csv(d / "polymers.csv"), xcfg)
    assert X.shape == (50, len(names)) == (50, 2 * 256 + 5 + 7 + 2)
