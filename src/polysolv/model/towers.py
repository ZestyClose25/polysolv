"""GIN towers (T18): molecule graph (+ descriptors) -> embedding.

GINTower: atom embedding -> n x [GINEConv -> BatchNorm -> ReLU -> dropout] with a
residual connection -> sum pooling. Polymer and solvent use two separate towers.
MoleculeEncoder = GINTower + descriptor MLP -> h (h_p or h_s).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool


class GINTower(nn.Module):
    """Graph -> pooled vector (``hidden``) with edge-aware GIN layers."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=bond_dim, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(torch.relu(norm(conv(h, edge_index, edge_attr))))
        return global_add_pool(h, batch)


class MoleculeEncoder(nn.Module):
    """GINTower + descriptors -> embedding h (one per graph, gathered per row by the caller)."""

    def __init__(self, atom_dim: int, bond_dim: int, desc_dim: int, hidden: int = 128, layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.tower = GINTower(atom_dim, bond_dim, hidden, layers, dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden + desc_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden))

    def forward(self, graph, index: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        """``graph`` = PyG Batch of unique molecules, ``index`` = row -> graph, ``desc`` = per-row descriptors."""
        g = self.tower(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        return self.mlp(torch.cat([g[index], desc], dim=-1))
