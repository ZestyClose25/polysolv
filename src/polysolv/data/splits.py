"""Solvent-scaffold split for the Zhou data (T9).

Every solvent molecule goes to exactly one of train / val / test, so no solvent
is seen in training and testing. Ring-containing solvents are grouped by their
Bemis–Murcko scaffold (a whole group goes to one split). Acyclic solvents have
an empty scaffold; putting them all in one group would force ~half the data into
one split, so each acyclic molecule is its own group.

Assignment follows the "balanced scaffold split": groups larger than
``big_group_frac × n_test`` go to train first, the rest are shuffled with a fixed
seed and filled greedily into train → val → test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


def scaffold_key(smiles: str) -> str:
    """Murcko scaffold SMILES; acyclic molecules return their own SMILES (singleton group)."""
    mol = Chem.MolFromSmiles(smiles)
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaf if scaf else f"acyclic:{smiles}"


def scaffold_split(
    solvents: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    big_group_frac: float = 0.5,
) -> dict[str, list[int]]:
    """Split ``solvents`` (needs ``solvent_id``, ``can_smiles``) by scaffold.

    Returns ``{"train": [...], "val": [...], "test": [...]}`` of solvent_ids.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    keys = solvents["can_smiles"].map(scaffold_key)
    groups = [g["solvent_id"].tolist() for _, g in solvents.groupby(keys, sort=True)]

    n = len(solvents)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    n_test = n - n_train - n_val
    big = [g for g in groups if len(g) > big_group_frac * n_test]
    small = [g for g in groups if len(g) <= big_group_frac * n_test]
    rng = np.random.default_rng(seed)
    small = [small[i] for i in rng.permutation(len(small))]
    big.sort(key=len, reverse=True)

    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for g in big + small:
        if len(split["train"]) + len(g) <= n_train:
            split["train"] += g
        elif len(split["val"]) + len(g) <= n_val:
            split["val"] += g
        else:
            split["test"] += g
    for k in split:
        split[k] = sorted(int(i) for i in split[k])
    log.info("scaffold split (seed %d): %s | %d groups, %d big",
             seed, {k: len(v) for k, v in split.items()}, len(groups), len(big))
    return split


def save_splits(split: dict[str, list[int]], path: Path, seed: int, fractions) -> None:
    """Write splits.json with method and seed metadata."""
    payload = {
        "method": "balanced Bemis-Murcko scaffold split (acyclic = singleton groups)",
        "key": "solvent_id",
        "seed": seed,
        "fractions": list(fractions),
        **split,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
