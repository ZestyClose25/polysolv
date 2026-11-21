"""Tests for polysolv.model.towers (T18)."""

import torch
from torch_geometric.data import Batch

from polysolv.features.graph import ATOM_DIM, BOND_DIM, smiles_to_graph
from polysolv.model.towers import GINTower, MoleculeEncoder


def _batch(smiles):
    return Batch.from_data_list([smiles_to_graph(s) for s in smiles])


def test_tower_shape_and_layers():
    t = GINTower(ATOM_DIM, BOND_DIM, hidden=128, layers=4)
    assert len(t.convs) == 4
    b = _batch(["*CC(*)c1ccccc1", "CCO", "O"])
    assert t(b.x, b.edge_index, b.edge_attr, b.batch).shape == (3, 128)


def test_sum_pooling_is_permutation_invariant():
    torch.manual_seed(0)
    t = GINTower(ATOM_DIM, BOND_DIM, hidden=32, layers=2).eval()
    a, b = _batch(["OCC"]), _batch(["CCO"])  # same molecule, different atom order
    out_a = t(a.x, a.edge_index, a.edge_attr, a.batch)
    out_b = t(b.x, b.edge_index, b.edge_attr, b.batch)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_encoder_gathers_rows():
    torch.manual_seed(0)
    enc = MoleculeEncoder(ATOM_DIM, BOND_DIM, desc_dim=7, hidden=64).eval()
    b = _batch(["CCO", "c1ccccc1"])
    idx = torch.tensor([0, 1, 1, 0])
    desc = torch.zeros(4, 7)
    h = enc(b, idx, desc)
    assert h.shape == (4, 64)
    assert torch.allclose(h[0], h[3]) and torch.allclose(h[1], h[2])
    h.sum().backward()
