"""Tests for polysolv.train (T22): a tiny run writes logs, history, curves and a checkpoint."""

import copy

import pytest

from polysolv.dataset import PairDataset, load_processed
from polysolv.train import RankSampler, load_checkpoint, predict, train_model


@pytest.fixture(scope="module")
def tiny(cfg):
    d = cfg["paths"]["processed_dir"]
    if not (d / "graphs.pt").exists():
        pytest.skip("run scripts.build_dataset first")
    p = load_processed(d)
    lab = p["labels"]
    mk = lambda ids: PairDataset(lab[lab.solvent_id.isin(ids)], p["solvents"], p["polymers"],  # noqa: E731
                                 p["s_desc"], p["p_desc"], p["graphs"])
    return mk(p["splits"]["train"][:30]), mk(p["splits"]["val"][:10])


def test_tiny_run_writes_logs(cfg, tiny, tmp_path):
    c = copy.deepcopy(cfg)
    c["train"].update(log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ck"), device="cpu")
    c["loss"]["w_rank"] = 0.5
    tr, va = tiny
    model, hist, info = train_model(c, tr, va, "tiny", seed=0, epochs=3)
    assert len(hist) == 3 and {"train_mse", "val_mse", "val_R2", "train_rank"} <= set(hist.columns)
    for f in ("tiny.log", "tiny_history.csv", "tiny_curves.png"):
        assert (tmp_path / "logs" / f).exists()
    m2, meta = load_checkpoint(tmp_path / "ck" / "tiny.pt")
    p1, p2 = predict(model, va, "cpu"), predict(m2, va, "cpu")
    assert (p1.y_pred - p2.y_pred).abs().max() < 1e-5
    assert meta["seed"] == 0


def test_rank_sampler_pairs(tiny):
    tr, _ = tiny
    rs = RankSampler(tr)
    assert len(rs) > 0
    assert all(tr.labels.polymer.iloc[rs.rows_t] == "PS") and all(tr.labels.polymer.iloc[rs.rows_o] == "PET")
    assert (tr.labels.solvent_id.iloc[rs.rows_t].to_numpy() == tr.labels.solvent_id.iloc[rs.rows_o].to_numpy()).all()
