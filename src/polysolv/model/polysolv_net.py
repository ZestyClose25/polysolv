"""PolysolvNet (T20): two GIN towers -> interaction -> (dA, dB) -> chi(T) -> log10 wt%.

    h_p = Enc_p(polymer graph, polymer descriptors)
    h_s = Enc_s(solvent graph, solvent descriptors)
    dA, dB = MLP([h_p, h_s, h_p*h_s, |h_p-h_s|, Ra2/scale, mask]);  dB *= dB_scale
    chi = (A0 + dA) + (B_HSP*mask + dB) / T
    log10 wt% = c0 - softplus(c1)*chi + c2*crystallinity   (then the label floor: y = log10(wt% + 1e-3))

Training compares the raw (pre-floor) prediction with y (floor rows one-sided, see losses).

``physics=False`` gives the plain two-tower GIN baseline: the MLP sees 1000/T instead of
(Ra2, mask) and predicts log10 wt% directly (no chi, no Hansen prior).
"""

from __future__ import annotations

import torch
from torch import nn

from polysolv.dataset import PairBatch
from polysolv.features.graph import ATOM_DIM, BOND_DIM
from polysolv.model.flory_huggins import FloryHuggins
from polysolv.model.heads import SolubilityHead, floor_log10
from polysolv.model.interaction import InteractionMLP, interaction_vector
from polysolv.model.towers import MoleculeEncoder


class PolysolvNet(nn.Module):
    def __init__(self, s_desc_dim: int, p_desc_dim: int, mcfg: dict, A0: float = 0.34, floor: float = 1e-3):
        super().__init__()
        h, n, dr = mcfg["hidden"], mcfg["gin_layers"], mcfg["dropout"]
        self.physics = bool(mcfg.get("physics", True))
        self.pair = bool(mcfg.get("pair_features", True))
        self.use_p_desc = bool(mcfg.get("polymer_descriptors", True))
        self.floor = floor
        self.ra2_scale = float(mcfg["ra2_scale"])
        self.dB_scale = float(mcfg["dB_scale"])
        self.enc_p = MoleculeEncoder(ATOM_DIM, BOND_DIM, p_desc_dim, h, n, dr)
        self.enc_s = MoleculeEncoder(ATOM_DIM, BOND_DIM, s_desc_dim, h, n, dr)
        layers = tuple(mcfg["interaction_hidden"])
        if self.physics:
            self.interaction = InteractionMLP(h, n_extra=2, layers=layers, dropout=dr, n_out=2, zero_init=True,
                                              pair=self.pair)
            self.fh = FloryHuggins(A0)
            self.head = SolubilityHead(floor=floor)
            self.slope_grid = tuple(mcfg.get("head_slope_grid", (1.0,)))
        else:
            self.interaction = InteractionMLP(h, n_extra=1, layers=layers, dropout=dr, n_out=1, zero_init=False,
                                              pair=self.pair)

    def forward(self, b: PairBatch) -> dict[str, torch.Tensor]:
        p_desc = b.p_desc if self.use_p_desc else torch.zeros_like(b.p_desc)
        h_p = self.enc_p(b.p_graph, b.p_index, p_desc)
        h_s = self.enc_s(b.s_graph, b.s_index, b.s_desc)
        if not self.physics:
            x = interaction_vector(h_p, h_s, 1000.0 / b.T, pair=self.pair)
            raw = self.interaction(x).squeeze(-1)
            return {"raw": raw, "y_pred": floor_log10(raw, self.floor)}
        x = interaction_vector(h_p, h_s, b.ra2 / self.ra2_scale, b.mask, pair=self.pair)
        out = self.interaction(x)
        dA, dB = out[:, 0], self.dB_scale * out[:, 1]
        chi, A, B = self.fh(dA, dB, b.b_hsp, b.mask, b.T)
        raw = self.head.raw(chi, b.cryst)
        return {"raw": raw, "y_pred": floor_log10(raw, self.floor), "chi": chi, "A": A, "B": B, "dA": dA, "dB": dB}


def build_model(cfg: dict, s_desc_dim: int, p_desc_dim: int) -> PolysolvNet:
    """Model from the config's ``model`` and ``physics`` sections."""
    return PolysolvNet(s_desc_dim, p_desc_dim, cfg["model"], cfg["physics"]["A0"], cfg["clean"]["log_floor_wt"])
