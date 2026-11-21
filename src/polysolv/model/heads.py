"""Solubility head (T19): log10 wt% = c0 - softplus(c1) * chi + c2 * crystallinity.

softplus(c1) > 0 makes solubility strictly decrease with chi. ``crystallinity`` is the
polymer's crystalline fraction (0 for amorphous PS).

The prediction goes through the same floor as the label, y = log10(wt% + floor):
    y_pred = log10(10**raw + floor),  raw = c0 - s*chi + c2*x
so very poor solvents saturate at log10(floor) instead of pulling the line flat.

``init_from_prior`` fixes the slope (the trainer picks the best of a config grid) and
fits c0, c2 on the Hansen prior, so training starts from prior + head. All three stay
learnable afterwards. (A steep physically-chosen slope of 4 was tried: training learned
much more slowly, see results/training_logs/try_slope4.log.)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LN10 = math.log(10.0)


def inv_softplus(x: float) -> float:
    """Inverse of softplus for x > 0."""
    return x + math.log(-math.expm1(-x))


def floor_log10(raw: torch.Tensor, floor: float) -> torch.Tensor:
    """log10(10**raw + floor), computed stably."""
    return torch.logaddexp(raw * LN10, torch.full_like(raw, math.log(floor))) / LN10


class SolubilityHead(nn.Module):
    """chi -> y = log10(wt% + floor), monotone decreasing in chi."""

    def __init__(self, c0: float = 0.0, slope: float = 1.0, c2: float = 0.0, floor: float = 1e-3):
        super().__init__()
        self.c0 = nn.Parameter(torch.tensor(float(c0)))
        self.c1 = nn.Parameter(torch.tensor(inv_softplus(slope)))
        self.c2 = nn.Parameter(torch.tensor(float(c2)))
        self.floor = floor

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.c1)

    def raw(self, chi: torch.Tensor, cryst: torch.Tensor) -> torch.Tensor:
        """log10 wt% before the floor."""
        return self.c0 - self.slope * chi + self.c2 * cryst

    def forward(self, chi: torch.Tensor, cryst: torch.Tensor) -> torch.Tensor:
        return floor_log10(self.raw(chi, cryst), self.floor)

    def init_from_prior(self, chi_prior, cryst, y, slope: float, steps: int = 300) -> dict[str, float]:
        """Fix the slope, fit c0 and c2 by MSE on (chi_prior, crystallinity, y)."""
        dev = self.c0.device
        chi_t, x_t, y_t = (torch.tensor(np.asarray(a), dtype=torch.float32, device=dev) for a in (chi_prior, cryst, y))
        with torch.no_grad():
            self.c1.fill_(inv_softplus(slope))
        opt = torch.optim.Adam([self.c0, self.c2], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((self(chi_t, x_t) - y_t) ** 2).mean()
            loss.backward()
            opt.step()
        self.c1.grad = None
        return {"c0": float(self.c0.detach()), "slope": slope, "c2": float(self.c2.detach()),
                "prior_mse": float(loss.detach())}
