"""Tests for polysolv.dissolution (T29)."""

import math

import pytest

from polysolv.dissolution import chi_star, dissolution_time_s


def test_chi_star_inverts_head():
    c0, s, c2 = 1.14, 1.19, -2.35
    cs = chi_star(c0, s, c2, 0.0, 10.0)
    assert c0 - s * cs == pytest.approx(math.log10(10.0))


def test_time_scaling():
    t_ref = dissolution_time_s(0.0, 0.5, 2.5e-4, 1e-10)       # drive = dchi_ref -> L^2 / D0
    assert t_ref == pytest.approx(625.0)
    assert dissolution_time_s(0.25, 0.5, 2.5e-4, 1e-10) == pytest.approx(2 * t_ref)   # half the drive
    assert dissolution_time_s(0.0, 0.5, 5e-4, 1e-10) == pytest.approx(4 * t_ref)      # L doubled
    assert dissolution_time_s(1.0, 0.5, 2.5e-4, 1e-10) > 40 * t_ref                    # no drive -> very slow
