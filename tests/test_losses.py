"""Tests for polysolv.losses (T21)."""

import pytest
import torch

from polysolv.losses import censored_mse, pairwise_rank_loss, physics_reg, total_loss


def test_censored_mse():
    y = torch.tensor([1.0, 1.7, 1.7])
    cens = torch.tensor([False, True, True])
    pred = torch.tensor([2.0, 2.5, 1.2])  # censored: above cap -> 0, below cap by 0.5 -> 0.25
    assert censored_mse(pred, y, cens).item() == pytest.approx((1.0 + 0.0 + 0.25) / 3)


def test_rank_loss_prefers_correct_order():
    true = torch.tensor([2.0, 1.0, 0.0])
    good = pairwise_rank_loss(torch.tensor([3.0, 2.0, 1.0]), true)
    bad = pairwise_rank_loss(torch.tensor([1.0, 2.0, 3.0]), true)
    assert good < bad


def test_rank_loss_no_pairs_is_zero():
    assert pairwise_rank_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.5])).item() == 0.0


def test_physics_reg():
    assert physics_reg(torch.tensor([0.1, -0.1]), torch.tensor([300.0, 0.0]), 300.0).item() == pytest.approx(0.01 + 0.5)


def test_total_loss_weights():
    class B:
        y = torch.zeros(2)
        censored = torch.zeros(2, dtype=torch.bool)
    out = {"y_pred": torch.ones(2), "dA": torch.ones(2), "dB": torch.zeros(2)}
    parts = total_loss(out, B, {"w_mse": 1.0, "w_phys": 0.5, "w_rank": 2.0}, 300.0, rank=torch.tensor(0.25))
    assert parts["total"].item() == pytest.approx(1.0 + 0.5 + 0.5)
