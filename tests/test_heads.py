"""Tests for interaction, flory_huggins and heads (T19)."""

import numpy as np
import pytest
import torch

from polysolv.model.flory_huggins import FloryHuggins, chi
from polysolv.model.heads import SolubilityHead, inv_softplus
from polysolv.model.interaction import InteractionMLP, interaction_vector


def test_chi_hand_values():
    A, B, T = torch.tensor([0.34]), torch.tensor([300.0]), torch.tensor([300.0])
    assert chi(A, B, T).item() == pytest.approx(1.34)


def test_flory_huggins_mask_and_corrections():
    fh = FloryHuggins(A0=0.34)
    T = torch.tensor([300.0, 300.0])
    c, A, B = fh(torch.tensor([0.1, 0.0]), torch.tensor([-30.0, 60.0]),
                 torch.tensor([150.0, 150.0]), torch.tensor([1.0, 0.0]), T)
    assert A.tolist() == pytest.approx([0.44, 0.34])
    assert B.tolist() == pytest.approx([120.0, 60.0])  # mask 0 drops the prior
    assert c.tolist() == pytest.approx([0.84, 0.54])


def test_chi_decreases_with_T_when_B_positive():
    T = torch.linspace(280, 400, 10)
    c = chi(torch.tensor(0.34), torch.tensor(200.0), T)
    assert torch.all(c[1:] < c[:-1])


def test_head_is_monotone_in_chi():
    head = SolubilityHead(c0=1.0, slope=2.0, c2=-0.5)
    c = torch.linspace(0, 2, 20)
    y = head(c, torch.zeros(20))
    assert torch.all(y[1:] < y[:-1])
    assert head.slope.item() == pytest.approx(2.0)
    assert head(torch.tensor([0.0]), torch.tensor([1.0])).item() == pytest.approx(0.5)


def test_inv_softplus():
    for x in (0.1, 1.0, 5.0):
        assert torch.nn.functional.softplus(torch.tensor(inv_softplus(x))).item() == pytest.approx(x, rel=1e-5)


def test_head_init_from_prior_recovers_line():
    rng = np.random.default_rng(0)
    c, x = rng.uniform(0, 2, 200), rng.uniform(0, 0.5, 200)
    y = 1.5 - 2.5 * c - 0.8 * x
    fit = SolubilityHead().init_from_prior(c, x, y)
    assert fit["c0"] == pytest.approx(1.5) and fit["slope"] == pytest.approx(2.5) and fit["c2"] == pytest.approx(-0.8)


def test_interaction_starts_at_zero():
    mlp = InteractionMLP(hidden=16, n_extra=2)
    h_p, h_s = torch.randn(5, 16), torch.randn(5, 16)
    x = interaction_vector(h_p, h_s, torch.ones(5), torch.ones(5))
    assert x.shape == (5, 66)
    assert torch.all(mlp(x) == 0)
