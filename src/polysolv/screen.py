"""Green-solvent screening for selective PS dissolution (T28).

For each green solvent (greenness >= min_greenness) and each T on the grid
T_min .. min(bp - margin, T_max) step T_step:
    PS_lower  = 10**(mean_PS  - k*sigma_PS)  - floor     (wt%, conservative)
    PET_upper = 10**(mean_PET + k*sigma_PET) - floor
    keep if PS_lower >= ps_min and PET_upper <= pet_max  (T < bp by construction)
    score = PS_lower - PET_upper + w_green * greenness
Per solvent: T window = kept temperatures, operating T = best score, mass balance at that T,
and an order-of-magnitude dissolution time (T29, ``polysolv.dissolution``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polysolv.inference import NEW_ID_START, pair_rows, predict_rows, solvent_table
from polysolv.dissolution import chi_star, dissolution_time_s
from polysolv.mass_balance import mass_balance

CONF = ["low", "medium", "high"]


def temperature_grid(bp_C: float, scfg: dict) -> np.ndarray:
    """Screening temperatures (°C); only T_min if the boiling point is unknown."""
    if np.isnan(bp_C):
        return np.array([scfg["T_min_C"]])
    hi = min(bp_C - scfg["bp_margin_C"], scfg["T_max_C"])
    return np.arange(scfg["T_min_C"], hi + 1e-9, scfg["T_step_C"])


def wt_from_log(y, floor: float):
    """Invert y = log10(wt% + floor)."""
    return np.clip(10.0 ** np.asarray(y, dtype=float) - floor, 0.0, None)


def confidence(sigma_max: float, in_zhou: bool, thresholds) -> str:
    """high / medium / low from the ensemble spread; one level lower for solvents not in Zhou."""
    level = 2 if sigma_max < thresholds[0] else 1 if sigma_max < thresholds[1] else 0
    if not in_zhou:
        level = max(level - 1, 0)
    return CONF[level]


def screen(models, proc: dict, cfg: dict, green: pd.DataFrame, target: str, other: str, m_waste: float,
           f_ps: float, device="cpu") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked table of solvents that pass, full grid with predictions and pass flags)."""
    scfg, sel, floor, mb = cfg["screen"], cfg["selectivity"], cfg["clean"]["log_floor_wt"], cfg["mass_balance"]
    g = green[green.greenness >= scfg["min_greenness"]].reset_index(drop=True)
    solvents, s_desc, mapping = solvent_table(g.smiles, proc, cfg, extra=g)
    g["solvent_id"] = g.smiles.map(mapping)
    g["in_zhou"] = g.solvent_id < NEW_ID_START
    zbp = solvents.set_index("solvent_id").bp_C
    g["bp_used_C"] = [zbp.loc[i] if pd.notna(zbp.loc[i]) else b for i, b in zip(g.solvent_id, g.bp_C)]

    rows = [(sid, p, T + 273.15, T) for sid, bp in zip(g.solvent_id, g.bp_used_C)
            for T in temperature_grid(float(bp), scfg) for p in (target, other)]
    pairs = pd.DataFrame(rows, columns=["solvent_id", "polymer", "T_K", "T_C"])
    pred = predict_rows(models, pair_rows(pairs, solvents, proc["polymers"], cfg["physics"]["A0"]),
                        solvents, s_desc, proc, device)
    k = scfg["k_sigma"]
    vals = ["y_pred", "y_std"] + (["chi"] if "chi" in pred else [])
    w = pred.pivot_table(index=["solvent_id", "T_C"], columns="polymer", values=vals)
    grid = pd.DataFrame({
        "S_PS_mean": wt_from_log(w[("y_pred", target)], floor),
        "S_PS_lower": wt_from_log(w[("y_pred", target)] - k * w[("y_std", target)], floor),
        "S_PET_mean": wt_from_log(w[("y_pred", other)], floor),
        "S_PET_upper": wt_from_log(w[("y_pred", other)] + k * w[("y_std", other)], floor),
        "sigma_PS": w[("y_std", target)], "sigma_PET": w[("y_std", other)],
        "chi_PS": w[("chi", target)] if "chi" in vals else np.nan,
        "chi_PET": w[("chi", other)] if "chi" in vals else np.nan,
    }, index=w.index).reset_index()
    grid = grid.merge(g[["solvent_id", "name", "chem21", "greenness", "bio_based", "in_zhou", "bp_used_C"]],
                      on="solvent_id")
    grid["pass_PS"] = grid.S_PS_lower >= sel["ps_min"]
    grid["pass_PET"] = grid.S_PET_upper <= sel["pet_max"]
    grid["keep"] = grid.pass_PS & grid.pass_PET
    grid["score"] = grid.S_PS_lower - grid.S_PET_upper + scfg["w_green"] * grid.greenness

    heads = [m.head for m in models if getattr(m, "physics", False)]
    x_ps = float(proc["polymers"].set_index("polymer").crystallinity_pct.fillna(0).loc[target]) / 100
    c_star = float(np.mean([chi_star(h.c0.item(), h.slope.item(), h.c2.item(), x_ps, sel["ps_min"])
                            for h in heads])) if heads else np.nan
    dcfg = cfg["dissolution"]

    out = []
    for _, d in grid[grid.keep].groupby("solvent_id"):
        best = d.loc[d.score.idxmax()]
        m = mass_balance(m_waste, f_ps, best.S_PS_lower, best.S_PET_upper, mb["c_max_wt"], mb["load_factor"])
        out.append({
            "solvent": best["name"], "T_window_C": f"{d.T_C.min():.0f}-{d.T_C.max():.0f}",
            "T_operate_C": float(best.T_C), "S_PS_sat_wt": round(float(best.S_PS_mean), 2),
            "S_PS_lower_wt": round(float(best.S_PS_lower), 2), "S_PET_upper_wt": round(float(best.S_PET_upper), 4),
            "loading_wt": round(float(m["c_wt"]), 2), "solvent_g": round(float(m["m_solvent_g"]), 1),
            "solvent_to_waste": round(float(m["solvent_to_waste"]), 2),
            "PET_dissolved_g": round(float(m["m_pet_dissolved_g"]), 4),
            "PS_purity_pct": round(100 * float(m["ps_purity"]), 3),
            "confidence": confidence(max(best.sigma_PS, best.sigma_PET), bool(best.in_zhou), scfg["conf_sigma"]),
            "sigma_PS": round(float(best.sigma_PS), 3), "sigma_PET": round(float(best.sigma_PET), 3),
            "chi_PS": round(float(best.chi_PS), 3), "chi_PET": round(float(best.chi_PET), 3),
            "dissolution_time_min": round(float(dissolution_time_s(
                best.chi_PS, c_star, dcfg["half_thickness_mm"] / 1000, dcfg["D0_m2_s"], dcfg["dchi_ref"]) / 60), 1)
            if heads else np.nan,
            "greenness": int(best.greenness), "chem21": best.chem21, "bio_based": int(best.bio_based),
            "in_zhou": bool(best.in_zhou), "bp_C": best.bp_used_C, "score": round(float(best.score), 3),
        })
    ranked = pd.DataFrame(out)
    if len(ranked):
        ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
        ranked.index += 1
    return ranked, grid


def rejection_reasons(grid: pd.DataFrame) -> pd.DataFrame:
    """Solvents with no passing temperature and why (best PS_lower, lowest PET_upper over the grid)."""
    bad = grid.groupby("solvent_id").filter(lambda d: not d.keep.any())
    return bad.groupby(["solvent_id", "name"]).agg(
        max_S_PS_lower=("S_PS_lower", "max"), min_S_PET_upper=("S_PET_upper", "min"),
        PS_ok_any=("pass_PS", "any"), PET_ok_any=("pass_PET", "any"), T_max_C=("T_C", "max")).reset_index()
