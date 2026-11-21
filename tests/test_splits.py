"""Tests for polysolv.data.splits (T9)."""

import pandas as pd
import pytest

from polysolv.data.merge import build_solvents
from polysolv.data.splits import scaffold_key, scaffold_split


@pytest.fixture(scope="module")
def solvents(zhou_clean):
    return build_solvents(zhou_clean[0])


def test_scaffold_key():
    assert scaffold_key("Cc1ccccc1") == scaffold_key("CCc1ccccc1") == "c1ccccc1"
    assert scaffold_key("CCO").startswith("acyclic:")


def test_disjoint_and_complete(solvents, cfg):
    s = cfg["split"]
    sp = scaffold_split(solvents, tuple(s["fractions"]), s["seed"], s["big_group_frac"])
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(solvents.solvent_id)
    # no molecule in two splits
    smi = solvents.set_index("solvent_id").can_smiles
    assert not set(smi[list(tr)]) & set(smi[list(te)])


def test_fractions_and_groups(solvents, cfg):
    s = cfg["split"]
    sp = scaffold_split(solvents, tuple(s["fractions"]), s["seed"], s["big_group_frac"])
    n = len(solvents)
    assert abs(len(sp["train"]) / n - 0.8) < 0.03
    assert abs(len(sp["test"]) / n - 0.1) < 0.03
    where = {i: k for k, ids in sp.items() for i in ids}
    keys = solvents.set_index("solvent_id").can_smiles.map(scaffold_key)
    for _, ids in keys.groupby(keys):
        assert len({where[i] for i in ids.index}) == 1  # a scaffold never spans splits


def test_deterministic(solvents):
    assert scaffold_split(solvents, seed=1) == scaffold_split(solvents, seed=1)
    assert scaffold_split(solvents, seed=1) != scaffold_split(solvents, seed=2)
