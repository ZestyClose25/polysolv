"""Tests for polysolv.dataset (T17)."""

import pytest
import torch

from polysolv.dataset import PairDataset, load_processed
from polysolv.features.graph import ATOM_DIM, BOND_DIM


@pytest.fixture(scope="module")
def proc(cfg):
    d = cfg["paths"]["processed_dir"]
    if not (d / "graphs.pt").exists():
        pytest.skip("run scripts.build_dataset first")
    return load_processed(d)


@pytest.fixture(scope="module")
def ds(proc):
    lab = proc["labels"]
    lab = lab[lab.solvent_id.isin(proc["splits"]["train"][:40])]
    return PairDataset(lab, proc["solvents"], proc["polymers"], proc["s_desc"], proc["p_desc"], proc["graphs"])


def test_lengths_and_dims(ds):
    assert len(ds) > 100
    assert ds.s_desc_dim == 7 and ds.p_desc_dim == 5


def test_collate_shapes(ds):
    b = ds.collate(list(range(64)))
    assert len(b) == 64
    for t in (b.s_desc, b.p_desc, b.T, b.ra2, b.b_hsp, b.mask, b.cryst, b.y, b.censored):
        assert t.shape[0] == 64
    assert b.s_graph.x.shape[1] == ATOM_DIM and b.s_graph.edge_attr.shape[1] == BOND_DIM
    assert int(b.s_index.max()) == b.s_graph.num_graphs - 1
    assert int(b.p_index.max()) == b.p_graph.num_graphs - 1 <= 7


def test_rows_point_to_the_right_graphs(ds, proc):
    b = ds.collate([0, 5, 17])
    smiles = proc["solvents"].set_index("solvent_id")["can_smiles"]
    for k, r in enumerate(b.row.tolist()):
        sid = ds.labels.solvent_id.iloc[r]
        assert b.s_graph.smiles[int(b.s_index[k])] == smiles.loc[sid]
        assert b.T[k] == pytest.approx(ds.labels.T_K.iloc[r])


def test_loader_covers_all_rows(ds):
    seen = torch.cat([b.row for b in ds.loader(100, shuffle=True, seed=0)])
    assert sorted(seen.tolist()) == list(range(len(ds)))


def test_ps_is_amorphous(ds):
    ps = ds.labels.index[ds.labels.polymer == "PS"][0]
    assert ds.cryst[ps] == 0.0
