"""Build the processed tables from the clean Zhou data and polymer_props.csv (T8).

Outputs (all keyed so later stages never re-parse names):
- ``solvents``: one row per molecule, with ``solvent_id`` and descriptors.
- ``polymers``: one row per polymer, from ``polymer_props.csv``.
- ``zhou_long``: labels only (solvent_id, polymer, T, y, censored).

Known, explained gaps in solvent descriptors (logged, not filled here):
``bp_C`` is missing for the solvents without a boiling point in Zhou, and
``dD/dP/dH`` for the solvents without Hansen parameters (``has_hsp = False``).
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from polysolv.data.clean import canonical_smiles

log = logging.getLogger(__name__)

SOLVENT_COLS = ["solvent_name", "cosmo_name", "cas", "smiles", "can_smiles", "class1", "class2",
                "bp_C", "Th_C", "dD", "dP", "dH"]
LABEL_COLS = ["solvent_id", "polymer", "T_label", "T_K", "wt_pct", "y", "censored"]


def build_solvents(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per molecule with descriptors and an integer ``solvent_id``."""
    solv = (clean.sort_values("solvent_order")
            .drop_duplicates("can_smiles")[SOLVENT_COLS]
            .reset_index(drop=True))
    solv.insert(0, "solvent_id", range(len(solv)))
    solv["has_hsp"] = solv[["dD", "dP", "dH"]].notna().all(axis=1)
    solv["has_bp"] = solv["bp_C"].notna()
    solv["mw"] = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in solv["can_smiles"]]
    return solv


def build_polymers(props: pd.DataFrame) -> pd.DataFrame:
    """Polymer table with canonical repeat-unit SMILES and availability flags."""
    pol = props.copy()
    pol["can_smiles"] = pol["repeat_smiles"].map(canonical_smiles)
    if pol["can_smiles"].isna().any():
        raise ValueError(f"unparsable repeat_smiles: {pol.loc[pol.can_smiles.isna(), 'polymer'].tolist()}")
    pol["has_hsp"] = pol[["dD", "dP", "dH"]].notna().all(axis=1)
    pol["has_Tm"] = pol["Tm_K"].notna()
    return pol


def build_labels(clean: pd.DataFrame, solvents: pd.DataFrame, polymers: pd.DataFrame) -> pd.DataFrame:
    """Label table keyed by solvent_id and polymer name."""
    missing = set(clean["polymer"]) - set(polymers["polymer"])
    if missing:
        raise ValueError(f"polymers in Zhou but not in polymer_props.csv: {sorted(missing)}")
    ids = solvents.set_index("can_smiles")["solvent_id"]
    lab = clean.assign(solvent_id=clean["can_smiles"].map(ids))
    if lab["solvent_id"].isna().any():
        raise RuntimeError("label rows without a solvent_id")
    return lab[LABEL_COLS].sort_values(["solvent_id", "polymer", "T_label"]).reset_index(drop=True)


def merge_all(clean: pd.DataFrame, props: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{"solvents", "polymers", "zhou_long"}`` and log descriptor gaps."""
    solvents = build_solvents(clean)
    polymers = build_polymers(props)
    labels = build_labels(clean, solvents, polymers)
    log.info("solvents: %d | without bp: %d | without HSP: %d",
             len(solvents), int((~solvents.has_bp).sum()), int((~solvents.has_hsp).sum()))
    log.info("polymers: %d | without Tm: %s", len(polymers), polymers.loc[~polymers.has_Tm, "polymer"].tolist())
    log.info("label rows: %d", len(labels))
    return {"solvents": solvents, "polymers": polymers, "zhou_long": labels}
