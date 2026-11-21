"""Tests for polysolv.features.physics (T14)."""

import numpy as np
import pandas as pd
import pytest

from polysolv.data.merge import merge_all
from polysolv.features.descriptors import add_molar_volume
from polysolv.features.physics import add_prior_columns, b_hsp, chi_prior, ra2

PS = (17.6, 6.1, 4.1)
PET = (18.7, 6.3, 6.7)
TOLUENE, V_TOLUENE = (18.0, 1.4, 2.0), 106.8
WATER, V_WATER = (15.5, 16.0, 42.3), 18.0


def test_ra2_by_hand():
    # 4*(17.6-18.0)^2 + (6.1-1.4)^2 + (4.1-2.0)^2 = 0.64 + 22.09 + 4.41
    assert ra2(*PS, *TOLUENE) == pytest.approx(27.14)
    assert ra2(*PS, *PS) == 0


def test_b_hsp_units():
    # 106.8 cm3/mol * 27.14 MPa / (4 * 8.314 J/mol/K) = 87.16 K
    assert b_hsp(V_TOLUENE, 27.14) == pytest.approx(87.16, abs=0.01)
    assert chi_prior(298.15, 87.16, 0.34) == pytest.approx(0.34 + 87.16 / 298.15)


def test_relative_sanity():
    chi = lambda pol, sol, V: chi_prior(298.15, b_hsp(V, ra2(*pol, *sol)))
    assert chi(PS, WATER, V_WATER) > 2.0                         # water is a hard non-solvent
    assert chi(PS, TOLUENE, V_TOLUENE) < chi(PS, WATER, V_WATER)
    assert chi(PS, TOLUENE, V_TOLUENE) < chi(PET, TOLUENE, V_TOLUENE)  # toluene prefers PS
    # chi falls with temperature when B > 0
    assert chi_prior(393.15, 87.16) < chi_prior(298.15, 87.16)


def test_works_on_arrays():
    a = ra2(np.array([17.6, 18.7]), 6.1, 4.1, 18.0, 1.4, 2.0)
    assert a.shape == (2,)


@pytest.fixture(scope="module")
def prior(zhou_clean, cfg):
    t = merge_all(zhou_clean[0], pd.read_csv(cfg["paths"]["polymer_props"]))
    solvents = add_molar_volume(t["solvents"], cfg["descriptors"]["molar_volume_anchors"])
    return solvents, add_prior_columns(t["zhou_long"], solvents, t["polymers"], cfg["physics"]["A0"])


def test_mask_rows(prior, cfg):
    solvents, lab = prior
    no_hsp = set(solvents.loc[~solvents.has_hsp, "solvent_id"])
    m = lab[lab.solvent_id.isin(no_hsp)]
    assert len(m) > 0 and (m.hsp_mask == 0).all() and (m.B_HSP == 0).all()
    assert (m.chi_prior == cfg["physics"]["A0"]).all()
    assert (lab.loc[lab.hsp_mask == 1, "B_HSP"] >= 0).all()


def test_prior_ranks_ps_solvents(prior):
    """Relative check (absolute chi_prior(PS, toluene) = 0.65 > 0.5 with the chosen PS HSP)."""
    solvents, lab = prior
    ps = lab[(lab.polymer == "PS") & (lab.T_label == "RT") & (lab.hsp_mask == 1)].merge(
        solvents[["solvent_id", "solvent_name"]], on="solvent_id")
    pct = ps.set_index("solvent_name").chi_prior.rank(pct=True)
    assert pct["Toluene"] < 0.5 and pct["Dipentene (Dl-Limonene)"] < 0.5
    assert pct["Water"] > 0.8 and pct["Methanol"] > 0.8
    rho = (-ps.chi_prior).rank().corr(ps.y.rank())
    assert rho > 0.5  # Hansen prior vs Zhou PS labels at RT (measured 0.66)
