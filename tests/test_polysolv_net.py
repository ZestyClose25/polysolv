"""Tests for PolysolvNet (T20): one batch forward/backward, shapes, prior at init."""

import pytest
import torch

from polysolv.dataset import PairDataset, load_processed
from polysolv.model.polysolv_net import build_model


@pytest.fixture(scope="module")
def batch_and_ds(cfg):
    d = cfg["paths"]["processed_dir"]
    if not (d / "graphs.pt").exists():
        pytest.skip("run scripts.build_dataset first")
    p = load_processed(d)
    lab = p["labels"].head(300)
    ds = PairDataset(lab, p["solvents"], p["polymers"], p["s_desc"], p["p_desc"], p["graphs"])
    return ds.collate(list(range(128))), ds


def test_forward_backward_shapes(cfg, batch_and_ds):
    b, ds = batch_and_ds
    torch.manual_seed(0)
    model = build_model(cfg, ds.s_desc_dim, ds.p_desc_dim)
    out = model(b)
    for k in ("y_pred", "chi", "A", "B", "dA", "dB"):
        assert out[k].shape == (128,)
    out["y_pred"].mean().backward()
    assert model.enc_s.tower.embed.weight.grad is not None
    assert model.enc_p.tower.embed.weight.grad is not None


def test_untrained_model_equals_hansen_prior(cfg, batch_and_ds):
    b, ds = batch_and_ds
    model = build_model(cfg, ds.s_desc_dim, ds.p_desc_dim).eval()
    with torch.no_grad():
        out = model(b)
    prior = torch.tensor(ds.labels.chi_prior.to_numpy()[:128], dtype=torch.float32)
    assert torch.allclose(out["chi"], prior, atol=1e-4)
    assert torch.all(out["dA"] == 0) and torch.all(out["dB"] == 0)


def test_plain_gin_baseline(cfg, batch_and_ds):
    b, ds = batch_and_ds
    mcfg = dict(cfg["model"], physics=False)
    model = build_model(dict(cfg, model=mcfg), ds.s_desc_dim, ds.p_desc_dim)
    out = model(b)
    assert set(out) == {"y_pred"} and out["y_pred"].shape == (128,)
