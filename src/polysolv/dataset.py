"""PyG dataset + collate for (polymer, solvent, T) rows (T17).

One item = one row of a label table (``zhou_long.csv`` or an inference table with the
same columns). ``collate`` turns a list of row indices into a :class:`PairBatch`:

- ``s_graph`` / ``p_graph``: PyG ``Batch`` of the **unique** solvents / polymers in the
  mini-batch (a batch of 256 rows has ≤ 8 polymers, so each graph is encoded once);
  ``s_index`` / ``p_index`` map every row to its graph in that batch
- ``s_desc`` / ``p_desc``: z-scored descriptors (+ 0/1 flags) per row
- ``T`` (K), ``ra2``, ``b_hsp`` (K), ``mask`` (Hansen prior available), ``cryst`` (fraction)
- ``y`` (log10 wt%), ``censored`` ('M' rows: true value ≥ y), ``row`` (index into the table)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from polysolv.features.descriptors import Scaler, polymer_descriptors
from polysolv.features.graph import GraphCache


@dataclass
class PairBatch:
    """A collated mini-batch. All per-row tensors have length ``n_rows``."""

    s_graph: Batch
    s_index: torch.Tensor
    p_graph: Batch
    p_index: torch.Tensor
    s_desc: torch.Tensor
    p_desc: torch.Tensor
    T: torch.Tensor
    ra2: torch.Tensor
    b_hsp: torch.Tensor
    mask: torch.Tensor
    cryst: torch.Tensor
    y: torch.Tensor
    censored: torch.Tensor
    row: torch.Tensor

    def to(self, device) -> "PairBatch":
        return PairBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def __len__(self) -> int:
        return int(self.y.shape[0])


def load_processed(processed_dir: Path) -> dict:
    """Read every table written by ``scripts.build_dataset`` (graphs as a GraphCache)."""
    d = Path(processed_dir)
    return {
        "labels": pd.read_csv(d / "zhou_long.csv"),
        "solvents": pd.read_csv(d / "solvents.csv"),
        "polymers": pd.read_csv(d / "polymers.csv"),
        "s_desc": pd.read_csv(d / "solvent_descriptors.csv", index_col=0),
        "p_desc": pd.read_csv(d / "polymer_descriptors.csv", index_col=0),
        "splits": json.loads((d / "splits.json").read_text()),
        "scalers": json.loads((d / "scalers.json").read_text()),
        "graphs": GraphCache.load(d / "graphs.pt"),
    }


def refit_polymer_descriptors(polymers: pd.DataFrame, fit_on: list[str], cfg_desc: dict):
    """Polymer descriptors with the scaler fitted on ``fit_on`` only (leave-one-polymer-out, T24)."""
    scaler = Scaler().fit(polymers[polymers.polymer.isin(fit_on)], cfg_desc["polymer_features"])
    return polymer_descriptors(polymers, scaler, cfg_desc["polymer_flags"]), scaler.to_dict()


class PairDataset(Dataset):
    """Rows of a label table with everything the model needs, stored as tensors.

    ``labels`` needs ``solvent_id, polymer, T_K, Ra2, B_HSP, hsp_mask``; ``y`` and
    ``censored`` are optional (missing -> 0 / False, e.g. for screening).
    """

    def __init__(self, labels: pd.DataFrame, solvents: pd.DataFrame, polymers: pd.DataFrame,
                 s_desc: pd.DataFrame, p_desc: pd.DataFrame, graphs: GraphCache) -> None:
        self.labels = labels.reset_index(drop=True)
        lab = self.labels
        s_smiles = solvents.set_index("solvent_id")["can_smiles"]
        p_smiles = polymers.set_index("polymer")["can_smiles"]
        cryst = polymers.set_index("polymer")["crystallinity_pct"].fillna(0.0) / 100.0

        # graphs: one per unique solvent / polymer, rows point to them by position
        self.s_ids = np.sort(lab.solvent_id.unique())
        self.p_names = sorted(lab.polymer.unique())
        self.s_graphs = [graphs.get(s_smiles.loc[i]) for i in self.s_ids]
        self.p_graphs = [graphs.get(p_smiles.loc[p]) for p in self.p_names]
        s_pos = pd.Series(np.arange(len(self.s_ids)), index=self.s_ids)
        p_pos = pd.Series(np.arange(len(self.p_names)), index=self.p_names)

        f = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
        self.s_gid = torch.tensor(s_pos.loc[lab.solvent_id].to_numpy(), dtype=torch.long)
        self.p_gid = torch.tensor(p_pos.loc[lab.polymer].to_numpy(), dtype=torch.long)
        self.s_desc = f(s_desc.loc[lab.solvent_id].to_numpy())
        self.p_desc = f(p_desc.loc[lab.polymer].to_numpy())
        self.T = f(lab.T_K)
        self.ra2 = f(lab.Ra2)
        self.b_hsp = f(lab.B_HSP)
        self.mask = f(lab.hsp_mask)
        self.cryst = f(cryst.loc[lab.polymer])
        self.y = f(lab["y"]) if "y" in lab else torch.zeros(len(lab))
        cens = lab["censored"].astype(bool) if "censored" in lab else np.zeros(len(lab), bool)
        self.censored = torch.tensor(np.asarray(cens), dtype=torch.bool)

    @property
    def s_desc_dim(self) -> int:
        return int(self.s_desc.shape[1])

    @property
    def p_desc_dim(self) -> int:
        return int(self.p_desc.shape[1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> int:
        return i

    def collate(self, idx) -> PairBatch:
        """List of row indices -> PairBatch (unique graphs batched once)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        s_u, s_inv = torch.unique(self.s_gid[idx], return_inverse=True)
        p_u, p_inv = torch.unique(self.p_gid[idx], return_inverse=True)
        return PairBatch(
            s_graph=Batch.from_data_list([self.s_graphs[i] for i in s_u.tolist()]),
            s_index=s_inv,
            p_graph=Batch.from_data_list([self.p_graphs[i] for i in p_u.tolist()]),
            p_index=p_inv,
            s_desc=self.s_desc[idx], p_desc=self.p_desc[idx], T=self.T[idx], ra2=self.ra2[idx],
            b_hsp=self.b_hsp[idx], mask=self.mask[idx], cryst=self.cryst[idx], y=self.y[idx],
            censored=self.censored[idx], row=idx,
        )

    def loader(self, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
        """DataLoader that yields PairBatch objects."""
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, generator=gen)

    def subset_rows(self, rows) -> PairBatch:
        """Collate an explicit set of rows (e.g. the rank mini-batch)."""
        return self.collate(list(rows))
