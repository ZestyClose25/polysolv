"""Hansen-parameter prior for the Flory–Huggins parameter (T14).

    Ra²   = 4(δD_p − δD_s)² + (δP_p − δP_s)² + (δH_p − δH_s)²        [MPa]
    B_HSP = V_s · Ra² / (4R)                                         [K]
            (V in cm³/mol, δ in MPa^½ → MPa·cm³/mol = J/mol; ÷ R in J/mol/K)
    χ_prior(T) = A0 + B_HSP / T

The model predicts corrections: χ(T) = (A0 + ΔA) + (B_HSP·mask + ΔB) / T.
``mask`` = 1 only when both polymer and solvent have Hansen parameters.

The scalar functions use plain arithmetic, so they work on floats, numpy arrays
and torch tensors alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_GAS = 8.314  # J / (mol K)


def ra2(dD_p, dP_p, dH_p, dD_s, dP_s, dH_s):
    """Squared Hansen distance (MPa), with the factor 4 on the dispersion term."""
    return 4.0 * (dD_p - dD_s) ** 2 + (dP_p - dP_s) ** 2 + (dH_p - dH_s) ** 2


def b_hsp(V_s, ra2_value):
    """Temperature coefficient of χ from Hansen parameters (K)."""
    return V_s * ra2_value / (4.0 * R_GAS)


def chi_prior(T_K, b, A0: float = 0.34):
    """χ_prior(T) = A0 + B/T."""
    return A0 + b / T_K


def add_prior_columns(
    labels: pd.DataFrame, solvents: pd.DataFrame, polymers: pd.DataFrame, A0: float
) -> pd.DataFrame:
    """Return ``labels`` with ``Ra2, B_HSP, hsp_mask, chi_prior`` columns.

    Rows without Hansen data get ``hsp_mask = 0``, ``Ra2 = B_HSP = 0`` and
    ``chi_prior = A0`` (the network then learns B on its own).
    """
    s = solvents.set_index("solvent_id")[["dD", "dP", "dH", "V_cm3_mol"]].add_suffix("_s")
    p = polymers.set_index("polymer")[["dD", "dP", "dH"]].add_suffix("_p")
    df = labels.join(s, on="solvent_id").join(p, on="polymer")

    r2 = ra2(df.dD_p, df.dP_p, df.dH_p, df.dD_s, df.dP_s, df.dH_s)
    mask = r2.notna() & df.V_cm3_mol_s.notna()
    out = labels.copy()
    out["hsp_mask"] = mask.astype(int)
    out["Ra2"] = np.where(mask, r2, 0.0)
    out["B_HSP"] = np.where(mask, b_hsp(df.V_cm3_mol_s, r2), 0.0)
    out["chi_prior"] = chi_prior(out["T_K"], out["B_HSP"], A0)
    return out
