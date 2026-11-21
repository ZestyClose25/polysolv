"""Clean the long Zhou table (T7).

Steps, in order (each dropped row is kept with a ``drop_reason``):

1. ``invalid_smiles``   — SMILES that RDKit cannot parse.
2. ``th_copy_of_rt``    — Th rows of solvents without a boiling point. For these
   162 solvents Zhou set Th = 25 °C, so the Th value is a copy of RT.
3. ``missing_label``    — no numeric value and not ``M`` (bp < RT, or COSMO-RS gap).
4. ``duplicate_molecule`` — several solvents with the same canonical SMILES
   (Hansen-handbook variants, or stereoisomers whose SMILES carry no stereo).
   One representative per molecule is kept: prefer a solvent with HSP, then
   file order.

Target: ``y = log10(wt% + floor)``. ``M`` rows get ``y = log10(cap + floor)``
and ``censored = True`` (the true value is at least the cap).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if the SMILES does not parse."""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _pick_representatives(df: pd.DataFrame) -> set[str]:
    """Choose one solvent_name per canonical SMILES (prefer HSP present, then file order)."""
    solv = df.drop_duplicates("solvent_name")[["solvent_name", "can_smiles", "dD", "solvent_order"]].copy()
    solv["no_hsp"] = solv["dD"].isna()
    solv = solv.sort_values(["can_smiles", "no_hsp", "solvent_order"])
    return set(solv.drop_duplicates("can_smiles")["solvent_name"])


def clean_zhou(long: pd.DataFrame, log_floor: float, miscible_cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the long Zhou table.

    Args:
        long: output of :func:`polysolv.data.load_zhou.load_zhou`.
        log_floor: floor added before log10 (wt%).
        miscible_cap: censoring value (wt%) used for ``M`` labels.

    Returns:
        ``(clean, dropped)``. ``clean`` has extra columns ``can_smiles, y, censored``;
        ``dropped`` has the removed rows plus ``drop_reason``.
    """
    df = long.copy()
    order = {name: i for i, name in enumerate(pd.unique(df["solvent_name"]))}
    df["solvent_order"] = df["solvent_name"].map(order)
    df["can_smiles"] = df["smiles"].map(canonical_smiles)
    df["drop_reason"] = pd.Series(pd.NA, index=df.index, dtype="object")

    def mark(mask: pd.Series, reason: str) -> None:
        new = mask & df["drop_reason"].isna()
        df.loc[new, "drop_reason"] = reason
        log.info("drop %-20s %6d rows", reason, int(new.sum()))

    mark(df["can_smiles"].isna(), "invalid_smiles")
    mark((df["T_label"] == "Th") & df["bp_C"].isna(), "th_copy_of_rt")
    mark(df["wt_pct"].isna() & ~df["is_miscible"], "missing_label")
    keep_names = _pick_representatives(df[df["drop_reason"].isna()])
    mark(~df["solvent_name"].isin(keep_names), "duplicate_molecule")

    dropped = df[df["drop_reason"].notna()].copy()
    clean = df[df["drop_reason"].isna()].drop(columns="drop_reason").copy()

    clean["censored"] = clean["is_miscible"]
    value = clean["wt_pct"].where(~clean["censored"], miscible_cap)
    clean["y"] = np.log10(value + log_floor)

    dup = clean.duplicated(["can_smiles", "polymer", "T_label"])
    if dup.any():
        raise RuntimeError(f"{int(dup.sum())} duplicate (molecule, polymer, T) rows after cleaning")
    log.info("clean rows: %d (%d molecules)", len(clean), clean["can_smiles"].nunique())
    return clean.reset_index(drop=True), dropped.reset_index(drop=True)
