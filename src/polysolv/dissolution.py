"""Order-of-magnitude dissolution-time estimate (T29, optional, physics only, not learned).

Fickian solvent penetration into a polymer flake of half-thickness L:
    t ≈ L² / D_eff
with an effective diffusivity that grows with the thermodynamic driving force:
    D_eff = D0 · max(chi_star − chi, eps) / dchi_ref
chi_star = the model's chi at which the head predicts the PS threshold (ps_min wt%), i.e. the
boundary between "dissolves" and "does not". The learned chi is an *effective* chi (its
threshold is not the Flory-Huggins 0.5), so the driving force is measured from chi_star
instead of using (1 − 2chi). Only meaningful for solvents that pass screening; the output
is an order of magnitude (minutes vs hours vs days), not a prediction to compare with
experiments to better than a factor of ~10.
"""

from __future__ import annotations

import math

import numpy as np


def chi_star(c0: float, slope: float, c2: float, cryst: float, s_wt: float, floor: float = 1e-3) -> float:
    """chi at which the head gives s_wt wt% (raw = log10(s_wt)): c0 - slope*chi + c2*x = log10(s_wt)."""
    return (c0 + c2 * cryst - math.log10(s_wt)) / slope


def dissolution_time_s(chi, chi_threshold: float, half_thickness_m: float, D0: float, dchi_ref: float = 0.5,
                       eps: float = 0.01):
    """Seconds to swell/dissolve a flake of the given half-thickness."""
    drive = np.maximum(chi_threshold - np.asarray(chi, dtype=float), eps)
    return half_thickness_m ** 2 / (D0 * drive / dchi_ref)
