"""Tests for polysolv.data.clean (T7)."""

import numpy as np
import pandas as pd
import pytest

from polysolv.data.clean import canonical_smiles, clean_zhou


def _toy():
    rows = [
        # name, smiles, bp, dD, polymer, T, wt, M
        ("A", "CCO", 78.0, 15.8, "PS", "RT", 0.5, False),
        ("A", "CCO", 78.0, 15.8, "PS", "Th", 2.0, False),
        ("B", "OCC", 78.0, 15.8, "PS", "RT", 0.7, False),  # same molecule as A -> dropped
        ("C", "CC=O", np.nan, 15.0, "PS", "RT", 1.0, False),
        ("C", "CC=O", np.nan, 15.0, "PS", "Th", 1.0, False),  # Th copy of RT -> dropped
        ("D", "c1ccccc1", 80.0, 18.4, "PS", "RT", np.nan, False),  # missing label -> dropped
        ("D", "c1ccccc1", 80.0, 18.4, "PS", "Th", np.nan, True),   # M -> censored
        ("E", "not_a_smiles", 50.0, 15.0, "PS", "RT", 1.0, False),  # invalid
    ]
    df = pd.DataFrame(rows, columns=["solvent_name", "smiles", "bp_C", "dD", "polymer", "T_label", "wt_pct", "is_miscible"])
    return df


def test_canonical_smiles():
    assert canonical_smiles("OCC") == canonical_smiles("CCO")
    assert canonical_smiles("xyz") is None
    assert canonical_smiles(None) is None


def test_toy_clean():
    clean, dropped = clean_zhou(_toy(), log_floor=1e-3, miscible_cap=50.0)
    assert dropped["drop_reason"].value_counts().to_dict() == {
        "invalid_smiles": 1, "th_copy_of_rt": 1, "missing_label": 1, "duplicate_molecule": 1,
    }
    assert len(clean) == 4
    cens = clean[clean.censored]
    assert len(cens) == 1 and cens.y.item() == pytest.approx(np.log10(50.0 + 1e-3))
    a = clean[(clean.solvent_name == "A") & (clean.T_label == "RT")]
    assert a.y.item() == pytest.approx(np.log10(0.5 + 1e-3))


def test_real_clean(zhou_long, zhou_clean):
    clean, dropped = zhou_clean
    assert len(clean) + len(dropped) == len(zhou_long)
    assert clean["y"].notna().all()
    assert clean["censored"].sum() == 2
    assert not clean.duplicated(["can_smiles", "polymer", "T_label"]).any()
    assert not ((clean.T_label == "Th") & clean.bp_C.isna()).any()
    assert clean.can_smiles.nunique() == clean.solvent_name.nunique()
    assert {"PS", "PET"} <= set(clean.polymer)
