"""Flory-Huggins layer (T19): chi(T) = A + B/T, with A and B as corrections to the Hansen prior.

    A = A0 + dA
    B = B_HSP * mask + dB          (B in K)
Temperature enters the model only here.
"""

from __future__ import annotations

import torch
from torch import nn


def chi(A: torch.Tensor, B: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """chi(T) = A + B/T (T in K)."""
    return A + B / T


class FloryHuggins(nn.Module):
    """(dA, dB, B_HSP, mask, T) -> (chi, A, B)."""

    def __init__(self, A0: float = 0.34):
        super().__init__()
        self.A0 = A0

    def forward(self, dA, dB, b_hsp, mask, T):
        A = self.A0 + dA
        B = b_hsp * mask + dB
        return chi(A, B, T), A, B
