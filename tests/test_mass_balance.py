"""Tests for polysolv.mass_balance (T27): the worked example from the build plan."""

import numpy as np
import pytest

from polysolv.mass_balance import mass_balance, pet_leaching, ps_purity, required_solvent, working_concentration


def test_worked_example():
    # 100 g waste, 30 % PS, S_PS_lower 38 wt%, S_PET_upper 0.02 wt%, c_max 10 wt%
    r = mass_balance(100.0, 0.30, 38.0, 0.02, c_max=10.0)
    assert r["c_wt"] == pytest.approx(10.0)
    assert r["m_solvent_g"] == pytest.approx(270.0)
    assert r["m_pet_dissolved_g"] == pytest.approx(0.054, abs=1e-3)
    assert r["ps_purity"] == pytest.approx(0.998, abs=1e-3)
    assert r["solvent_to_waste"] == pytest.approx(2.7)


def test_low_solubility_needs_more_solvent():
    c = working_concentration([5.0, 20.0], c_max=10.0)
    assert c.tolist() == pytest.approx([4.0, 10.0])
    m = required_solvent(100.0, 0.3, c)
    assert m[0] > m[1]


def test_edge_cases():
    assert np.isinf(required_solvent(100.0, 0.3, 0.0))
    assert pet_leaching(0.0, 500.0) == 0.0
    assert ps_purity(100.0, 0.3, 0.0) == 1.0
