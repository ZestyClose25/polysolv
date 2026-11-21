"""Solubility head (T19): log10 wt% = c0 - softplus(c1) * chi + c2 * crystallinity.

softplus(c1) > 0 makes solubility strictly decrease with chi. ``crystallinity`` is the
polymer's crystalline fraction (0 for amorphous PS). ``init_from_prior`` sets c0, c1, c2
by least squares of y on (chi_prior, crystallinity) so training starts from the Hansen prior.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def inv_softplus(x: float) -> float:
    """Inverse of softplus for x > 0."""
    return x + math.log(-math.expm1(-x))


class SolubilityHead(nn.Module):
    """chi -> log10 wt%, monotone decreasing in chi."""

    def __init__(self, c0: float = 0.0, slope: float = 1.0, c2: float = 0.0):
        super().__init__()
        self.c0 = nn.Parameter(torch.tensor(float(c0)))
        self.c1 = nn.Parameter(torch.tensor(inv_softplus(slope)))
        self.c2 = nn.Parameter(torch.tensor(float(c2)))

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.c1)

    def forward(self, chi: torch.Tensor, cryst: torch.Tensor) -> torch.Tensor:
        return self.c0 - self.slope * chi + self.c2 * cryst

    @torch.no_grad()
    def init_from_prior(self, chi_prior, cryst, y, min_slope: float = 0.1) -> dict[str, float]:
        """Least-squares fit y ~ c0 - s*chi_prior + c2*cryst (s clipped at ``min_slope``)."""
        X = np.column_stack([np.ones(len(y)), -np.asarray(chi_prior), np.asarray(cryst)])
        c0, s, c2 = np.linalg.lstsq(X, np.asarray(y), rcond=None)[0]
        s = max(float(s), min_slope)
        self.c0.fill_(float(c0))
        self.c1.fill_(inv_softplus(s))
        self.c2.fill_(float(c2))
        return {"c0": float(c0), "slope": s, "c2": float(c2)}
