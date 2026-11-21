"""SMILES -> PyTorch Geometric graph (T12).

Nodes = heavy atoms (implicit hydrogens, counted as a node feature).
Edges = bonds, stored in both directions. The polymer connection point ``*``
is a dummy atom with its own element slot.

Atom features (ATOM_DIM = 31):
  element one-hot (12: *, C, N, O, S, F, Cl, Br, I, P, Si, other)
  degree one-hot (6: 0..5), formal charge one-hot (3: -1, 0, +1),
  hybridization one-hot (4: SP, SP2, SP3, other), total H one-hot (4: 0..3+),
  aromatic, in ring
Bond features (BOND_DIM = 6): type one-hot (single, double, triple, aromatic), conjugated, in ring
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

from polysolv.data.clean import canonical_smiles

ELEMENTS = ["*", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si"]  # + "other"
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-1, 0, 1]
HYBRIDS = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3]
NUM_HS = [0, 1, 2, 3]
BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]


def _one_hot(value, choices: list, other: bool = False) -> list[float]:
    """One-hot over ``choices``; values not in the list go to an extra 'other' slot if ``other``."""
    vec = [float(value == c) for c in choices]
    if other:
        vec.append(float(value not in choices))
    elif value not in choices:
        vec[-1] = 1.0  # clip to the last bin (e.g. degree > 5, H count > 3)
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """Feature vector of one atom (length ATOM_DIM)."""
    return (
        _one_hot(atom.GetSymbol(), ELEMENTS, other=True)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(max(-1, min(1, atom.GetFormalCharge())), CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDS, other=True)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    """Feature vector of one bond (length BOND_DIM)."""
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [float(bond.GetIsConjugated()), float(bond.IsInRing())]


_probe = Chem.MolFromSmiles("CC")
ATOM_DIM = len(atom_features(_probe.GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(_probe.GetBondWithIdx(0)))


def smiles_to_graph(smiles: str) -> Data:
    """Build a PyG ``Data`` object (x, edge_index, edge_attr, smiles) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)
        src += [i, j]
        dst += [j, i]
        attr += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(attr, dtype=torch.float).reshape(-1, BOND_DIM)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


class GraphCache:
    """Graphs keyed by canonical SMILES, built once and reused."""

    def __init__(self) -> None:
        self._graphs: dict[str, Data] = {}

    def __len__(self) -> int:
        return len(self._graphs)

    def __contains__(self, smiles: str) -> bool:
        return canonical_smiles(smiles) in self._graphs

    def get(self, smiles: str) -> Data:
        key = canonical_smiles(smiles)
        if key is None:
            raise ValueError(f"invalid SMILES: {smiles}")
        if key not in self._graphs:
            self._graphs[key] = smiles_to_graph(key)
        return self._graphs[key]

    def build(self, smiles_list) -> "GraphCache":
        for s in smiles_list:
            self.get(s)
        return self

    def save(self, path: Path) -> None:
        torch.save(self._graphs, path)

    @classmethod
    def load(cls, path: Path) -> "GraphCache":
        cache = cls()
        cache._graphs = torch.load(path, weights_only=False)
        return cache
