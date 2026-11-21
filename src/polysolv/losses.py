"""Losses (T21).

    L = w_mse * censored MSE  +  w_rank * pairwise rank (PS over PET, stage 2)  +  w_phys * physics_reg

- censored MSE on the *raw* log10 wt% (before the label floor): normal squared error;
  'M' rows (true value >= cap) only penalise predictions below the cap; 'floored' rows
  (wt% below the 1e-3 floor, true value <= floor) only penalise predictions above it.
  This keeps gradients alive for very poor solvents (the floor transform is flat there).
- pairwise rank (RankNet): for solvent pairs (i, j) with true margin_i > margin_j, where
  margin = y(PS) - y(PET) at the same T, penalise softplus(-(m̂_i - m̂_j)).
- physics_reg: mean(dA^2) + mean((dB/dB_scale)^2) keeps the corrections small, so chi
  stays close to the Hansen prior unless the data says otherwise.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def censored_mse(pred: torch.Tensor, y: torch.Tensor, censored: torch.Tensor,
                 floored: torch.Tensor | None = None) -> torch.Tensor:
    """MSE with one-sided rows: censored ('M', pred >= y is fine) and floored (pred <= y is fine)."""
    err = pred - y
    err = torch.where(censored, torch.clamp(err, max=0.0), err)
    if floored is not None:
        err = torch.where(floored, torch.clamp(err, min=0.0), err)
    return (err ** 2).mean()


def pairwise_rank_loss(pred_margin: torch.Tensor, true_margin: torch.Tensor, min_gap: float = 0.1) -> torch.Tensor:
    """RankNet loss over all pairs whose true margins differ by more than ``min_gap`` (log10 units)."""
    dt = true_margin.unsqueeze(1) - true_margin.unsqueeze(0)
    dp = pred_margin.unsqueeze(1) - pred_margin.unsqueeze(0)
    pairs = dt > min_gap
    if not pairs.any():
        return pred_margin.sum() * 0.0
    return F.softplus(-dp[pairs]).mean()


def physics_reg(dA: torch.Tensor, dB: torch.Tensor, dB_scale: float) -> torch.Tensor:
    """L2 penalty on the corrections to the Hansen prior (dimensionless)."""
    return (dA ** 2).mean() + ((dB / dB_scale) ** 2).mean()


def total_loss(out: dict, batch, lcfg: dict, dB_scale: float, rank: torch.Tensor | None = None) -> dict:
    """Weighted sum + parts (as tensors). ``rank`` is the precomputed rank term (or None)."""
    parts = {"mse": censored_mse(out.get("raw", out["y_pred"]), batch.y, batch.censored,
                                 getattr(batch, "floored", None))}
    if "dA" in out:
        parts["phys"] = physics_reg(out["dA"], out["dB"], dB_scale)
    if rank is not None:
        parts["rank"] = rank
    w = {"mse": lcfg["w_mse"], "phys": lcfg["w_phys"], "rank": lcfg["w_rank"]}
    parts["total"] = sum(w[k] * v for k, v in parts.items())
    return parts
