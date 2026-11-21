"""Interaction block (T19): [h_p, h_s, h_p*h_s, |h_p - h_s|, Ra2, mask] -> MLP -> (dA, dB).

The last layer starts at zero, so an untrained model returns dA = dB = 0, i.e. exactly
the Hansen prior. ``extra`` lets the no-physics baseline append 1000/T instead.
"""

from __future__ import annotations

import torch
from torch import nn


def interaction_vector(h_p: torch.Tensor, h_s: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
    """Symmetric pair features plus per-row scalars (each ``extra`` has shape (n,))."""
    parts = [h_p, h_s, h_p * h_s, (h_p - h_s).abs()] + [e.unsqueeze(-1) for e in extra]
    return torch.cat(parts, dim=-1)


class InteractionMLP(nn.Module):
    """MLP on the interaction vector with ``n_out`` outputs (2 = dA, dB)."""

    def __init__(self, hidden: int, n_extra: int, layers=(256, 128), dropout: float = 0.1, n_out: int = 2,
                 zero_init: bool = True):
        super().__init__()
        dims, mods = [4 * hidden + n_extra, *layers], []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*mods)
        self.out = nn.Linear(dims[-1], n_out)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))
