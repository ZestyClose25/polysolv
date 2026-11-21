"""Tests for polysolv.data.merge (T8)."""

import pandas as pd
import pytest

from polysolv.data.merge import merge_all


@pytest.fixture(scope="module")
def tables(zhou_clean, cfg):
    clean, _ = zhou_clean
    props = pd.read_csv(cfg["paths"]["polymer_props"])
    return clean, merge_all(clean, props)


def test_solvents_unique(tables):
    _, t = tables
    s = t["solvents"]
    assert s.solvent_id.is_unique and s.can_smiles.is_unique
    assert (s.mw > 0).all()


def test_labels_complete(tables):
    clean, t = tables
    lab = t["zhou_long"]
    assert len(lab) == len(clean)
    assert lab.solvent_id.isin(t["solvents"].solvent_id).all()
    assert lab.polymer.isin(t["polymers"].polymer).all()
    assert lab[["T_K", "y"]].notna().all().all()


def test_descriptor_gaps_explained(tables):
    """Only the known gaps: missing bp (RT-only solvents) and missing HSP (has_hsp False)."""
    _, t = tables
    s = t["solvents"]
    assert s.loc[s.has_hsp, ["dD", "dP", "dH"]].notna().all().all()
    assert s.loc[s.has_bp, "bp_C"].notna().all()
    lab = t["zhou_long"].merge(s[["solvent_id", "has_bp"]], on="solvent_id")
    assert not ((lab.T_label == "Th") & ~lab.has_bp).any()
    p = t["polymers"]
    assert p[["dD", "dP", "dH", "Tg_K", "crystallinity_pct", "density_g_cm3"]].notna().all().all()
