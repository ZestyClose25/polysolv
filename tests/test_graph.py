"""Tests for polysolv.features.graph (T12)."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polysolv.data.merge import build_solvents  # noqa: E402
from polysolv.features.graph import ATOM_DIM, BOND_DIM, ELEMENTS, GraphCache, smiles_to_graph  # noqa: E402

PS = "*CC(*)c1ccccc1"
LIMONENE = "CC1=CCC(CC1)C(C)=C"


def test_ps_graph():
    g = smiles_to_graph(PS)
    assert g.num_nodes == 10
    assert g.edge_index.shape == (2, 20)  # 10 bonds, both directions
    assert g.x.shape == (10, ATOM_DIM) and g.edge_attr.shape == (20, BOND_DIM)
    star = ELEMENTS.index("*")
    assert int(g.x[:, star].sum()) == 2  # two dummy connection atoms


def test_limonene_graph():
    assert smiles_to_graph(LIMONENE).num_nodes == 10


def test_edges_bidirectional():
    g = smiles_to_graph("CCO")
    pairs = set(map(tuple, g.edge_index.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)


def test_single_atom_has_no_edges():
    g = smiles_to_graph("O")
    assert g.num_nodes == 1 and g.edge_index.shape == (2, 0) and g.edge_attr.shape == (0, BOND_DIM)


def test_one_hot_rows():
    g = smiles_to_graph("c1ccccc1Cl")
    assert (g.x[:, : len(ELEMENTS) + 1].sum(dim=1) == 1).all()  # exactly one element slot per atom


def test_cache_by_canonical_smiles(tmp_path):
    c = GraphCache()
    assert c.get("OCC") is c.get("CCO")
    assert len(c) == 1
    c.save(tmp_path / "g.pt")
    assert len(GraphCache.load(tmp_path / "g.pt")) == 1


def test_all_dataset_smiles_convert(zhou_clean, cfg):
    solvents = build_solvents(zhou_clean[0])
    polymers = pd.read_csv(cfg["paths"]["polymer_props"])
    cache = GraphCache().build(list(solvents.can_smiles) + list(polymers.repeat_smiles))
    assert len(cache) == len(solvents) + len(polymers)
