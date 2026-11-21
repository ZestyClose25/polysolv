"""Mass balance for selective PS dissolution (T27).

Inputs: waste mass m (g), PS mass fraction f_PS, conservative saturation solubilities
S_PS_lower and S_PET_upper (wt% = g polymer per 100 g solution), viscosity cap c_max (wt%).

    c       = min(load_factor * S_PS_lower, c_max)          working PS concentration (wt%)
    m_solv  = m * f_PS * (100 - c) / c                     solvent needed to dissolve all PS
    m_PET   = S_PET_upper / (100 - S_PET_upper) * m_solv    worst case: solvent saturated with PET
    purity  = m * f_PS / (m * f_PS + m_PET)                 PS purity of the recovered polymer
"""

from __future__ import annotations

import numpy as np


def working_concentration(s_ps_lower, c_max: float, load_factor: float = 0.8):
    """PS concentration to run at (wt%): a safety margin below saturation, capped for viscosity."""
    return np.minimum(load_factor * np.asarray(s_ps_lower, dtype=float), c_max)


def required_solvent(m_waste, f_ps, c):
    """Solvent mass (g) to dissolve all PS at concentration c (wt%)."""
    c = np.asarray(c, dtype=float)
    return np.where(c > 0, m_waste * f_ps * (100.0 - c) / np.where(c > 0, c, 1.0), np.inf)


def pet_leaching(s_pet_upper, m_solv):
    """Worst-case dissolved PET (g) if the solvent saturates with PET."""
    s = np.asarray(s_pet_upper, dtype=float)
    return s / (100.0 - s) * m_solv


def ps_purity(m_waste, f_ps, m_pet):
    """Mass fraction of PS in the recovered (re-precipitated) polymer."""
    m_ps = m_waste * f_ps
    return m_ps / (m_ps + np.asarray(m_pet, dtype=float))


def mass_balance(m_waste: float, f_ps: float, s_ps_lower, s_pet_upper, c_max: float,
                 load_factor: float = 0.8) -> dict:
    """All mass-balance quantities at once (arrays allowed for S values)."""
    c = working_concentration(s_ps_lower, c_max, load_factor)
    m_solv = required_solvent(m_waste, f_ps, c)
    m_pet = pet_leaching(s_pet_upper, m_solv)
    return {"c_wt": c, "m_solvent_g": m_solv, "solvent_to_waste": m_solv / m_waste, "m_pet_dissolved_g": m_pet,
            "ps_purity": ps_purity(m_waste, f_ps, m_pet)}
