"""Numeric descriptors for solvents and polymers (T13).

- Solvent molar volume V_s (needed by the Hansen prior) is estimated from SMILES:
  McGowan characteristic volume Vx (Abraham & McGowan, Chromatographia 1987, 23, 243)
  mapped to liquid molar volume with a linear fit on known values from the config.
- Descriptors are z-scored with statistics from the **training solvents only**.
  Missing values become 0 after scaling (= training mean); a 0/1 flag column
  (``has_bp``, ``has_hsp``, ``has_Tm``) tells the model the value was imputed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# McGowan atomic increments (cm3/mol); one bond of any order subtracts 6.56.
MCGOWAN_INCREMENTS = {
    "C": 16.35, "H": 8.71, "O": 12.43, "N": 14.39, "F": 10.48, "Cl": 20.95, "Br": 26.21,
    "I": 34.53, "S": 22.91, "P": 24.87, "Si": 26.83, "B": 18.32, "Se": 27.81, "Sn": 39.35, "Ge": 31.02,
}
MCGOWAN_BOND = 6.56


def mcgowan_vx(smiles: str) -> float:
    """McGowan characteristic volume in cm3/mol (water 16.73, toluene 85.73)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    try:
        atoms = sum(MCGOWAN_INCREMENTS[a.GetSymbol()] for a in mol.GetAtoms())
    except KeyError as err:
        raise ValueError(f"no McGowan increment for element {err} in {smiles}") from None
    return atoms - MCGOWAN_BOND * mol.GetNumBonds()


def fit_molar_volume(anchors: dict[str, float]) -> tuple[float, float]:
    """Fit V = a*Vx + b on known liquid molar volumes. Returns (a, b)."""
    vx = np.array([mcgowan_vx(s) for s in anchors])
    v = np.array(list(anchors.values()), dtype=float)
    a, b = np.polyfit(vx, v, 1)
    return float(a), float(b)


def add_molar_volume(solvents: pd.DataFrame, anchors: dict[str, float]) -> pd.DataFrame:
    """Return a copy of ``solvents`` with ``V_cm3_mol`` (estimated liquid molar volume)."""
    a, b = fit_molar_volume(anchors)
    out = solvents.copy()
    out["V_cm3_mol"] = [a * mcgowan_vx(s) + b for s in out["can_smiles"]]
    return out


@dataclass
class Scaler:
    """Column-wise z-scoring that ignores NaN when fitting and maps NaN to 0 when transforming."""

    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "Scaler":
        for c in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            sd = float(x.std(ddof=0))
            self.mean[c] = float(x.mean())
            self.std[c] = sd if sd > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.mean:
            out[c] = ((pd.to_numeric(df[c], errors="coerce") - self.mean[c]) / self.std[c]).fillna(0.0)
        return out

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(mean=dict(d["mean"]), std=dict(d["std"]))


def solvent_descriptors(solvents: pd.DataFrame, scaler: Scaler, flags: list[str]) -> pd.DataFrame:
    """z-scored solvent features + 0/1 flags, indexed by solvent_id."""
    z = scaler.transform(solvents)
    for f in flags:
        z[f] = solvents[f].astype(float)
    z.index = solvents["solvent_id"].values
    return z


def polymer_descriptors(polymers: pd.DataFrame, scaler: Scaler, flags: list[str]) -> pd.DataFrame:
    """z-scored polymer features + 0/1 flags, indexed by polymer name."""
    z = scaler.transform(polymers)
    for f in flags:
        z[f] = polymers[f].astype(float)
    z.index = polymers["polymer"].values
    return z


def build_descriptors(
    solvents: pd.DataFrame, polymers: pd.DataFrame, train_ids: list[int], cfg_desc: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit scalers (solvents: train split only; polymers: all 8) and return descriptor tables.

    Returns ``(solvent_desc, polymer_desc, scalers)`` where ``scalers`` is JSON-serialisable.
    """
    train = solvents[solvents["solvent_id"].isin(train_ids)]
    s_scaler = Scaler().fit(train, cfg_desc["solvent_features"])
    p_scaler = Scaler().fit(polymers, cfg_desc["polymer_features"])
    s_desc = solvent_descriptors(solvents, s_scaler, cfg_desc["solvent_flags"])
    p_desc = polymer_descriptors(polymers, p_scaler, cfg_desc["polymer_flags"])
    return s_desc, p_desc, {"solvent": s_scaler.to_dict(), "polymer": p_scaler.to_dict()}


def save_scalers(scalers: dict, path: Path) -> None:
    """Write scaler statistics as JSON (reused at screening time)."""
    path.write_text(json.dumps(scalers, indent=1), encoding="utf-8")
