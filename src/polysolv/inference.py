"""Predictions for arbitrary (polymer, solvent SMILES, T) rows (used by T23 external test, T26, T28).

Solvents found in Zhou (by canonical SMILES) reuse their Zhou descriptors. New solvents get:
McGowan-based V_s, Hansen parameters / bp only if supplied in ``extra`` (else NaN -> imputed,
has_hsp / has_bp = 0, no Hansen prior), descriptors scaled with the saved train scaler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from polysolv.data.clean import canonical_smiles
from polysolv.dataset import PairDataset
from polysolv.features.descriptors import Scaler, add_molar_volume, solvent_descriptors
from polysolv.features.physics import add_prior_columns

NEW_ID_START = 100_000


def solvent_table(smiles, proc: dict, cfg: dict, extra: pd.DataFrame | None = None):
    """Return (solvents incl. new rows, solvent descriptors incl. new rows, {input smiles: solvent_id}).

    ``extra``: optional table with ``smiles`` and any of ``dD, dP, dH, bp_C`` for solvents not in Zhou.
    """
    zhou = proc["solvents"]
    known = dict(zip(zhou.can_smiles, zhou.solvent_id))
    ext = {}
    if extra is not None:
        for _, r in extra.iterrows():
            ext[canonical_smiles(r["smiles"])] = r
    mapping, new = {}, {}
    for s in dict.fromkeys(smiles):
        c = canonical_smiles(s)
        if c is None:
            raise ValueError(f"invalid SMILES: {s}")
        if c in known:
            mapping[s] = int(known[c])
            continue
        if c not in new:
            r = ext.get(c, {})
            get = lambda k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan  # noqa: E731
            new[c] = {"solvent_id": NEW_ID_START + len(new), "solvent_name": r.get("name", c) if len(r) else c,
                      "smiles": s, "can_smiles": c, "dD": get("dD"), "dP": get("dP"), "dH": get("dH"),
                      "bp_C": get("bp_C"), "Th_C": np.nan}
        mapping[s] = new[c]["solvent_id"]
    solvents, s_desc = zhou, proc["s_desc"]
    if new:
        nd = pd.DataFrame(new.values())
        nd["has_hsp"] = nd[["dD", "dP", "dH"]].notna().all(axis=1)
        nd["has_bp"] = nd["bp_C"].notna()
        nd = add_molar_volume(nd, cfg["descriptors"]["molar_volume_anchors"])
        scaler = Scaler.from_dict(proc["scalers"]["solvent"])
        nd_desc = solvent_descriptors(nd, scaler, cfg["descriptors"]["solvent_flags"])
        solvents = pd.concat([zhou, nd], ignore_index=True)
        s_desc = pd.concat([s_desc, nd_desc[s_desc.columns]])
    return solvents, s_desc, mapping


def pair_rows(pairs: pd.DataFrame, solvents: pd.DataFrame, polymers: pd.DataFrame, A0: float) -> pd.DataFrame:
    """(solvent_id, polymer, T_K) rows + Hansen prior columns, ready for PairDataset."""
    lab = pairs.copy()
    if "T_label" not in lab:
        lab["T_label"] = "ext"
    return add_prior_columns(lab, solvents, polymers, A0)


@torch.no_grad()
def predict_rows(models, rows: pd.DataFrame, solvents, s_desc, proc: dict, device, p_desc=None,
                 batch_size: int = 1024) -> pd.DataFrame:
    """Mean and std over ``models`` of y_pred and chi (std = 0 for one model)."""
    ds = PairDataset(rows, solvents, proc["polymers"], s_desc, proc["p_desc"] if p_desc is None else p_desc,
                     proc["graphs"])
    ys, chis = [], []
    for m in models:
        m.eval()
        y, c = [], []
        for b in ds.loader(batch_size, shuffle=False):
            out = m(b.to(device))
            y.append(out["y_pred"].cpu().numpy())
            if "chi" in out:
                c.append(out["chi"].cpu().numpy())
        ys.append(np.concatenate(y))
        if c:
            chis.append(np.concatenate(c))
    out = ds.labels.copy()
    ys = np.stack(ys)
    out["y_pred"], out["y_std"] = ys.mean(0), ys.std(0)
    if chis:
        chis = np.stack(chis)
        out["chi"], out["chi_std"] = chis.mean(0), chis.std(0)
    out["n_models"] = len(models)
    return out


def external_rows(exp: pd.DataFrame, proc: dict, cfg: dict):
    """exp_test.csv -> (rows for prediction, solvents, s_desc)."""
    solvents, s_desc, mapping = solvent_table(exp.smiles, proc, cfg)
    pairs = exp.assign(solvent_id=exp.smiles.map(mapping))[["solvent_id", "polymer", "T_K"]]
    rows = pair_rows(pairs, solvents, proc["polymers"], cfg["physics"]["A0"])
    keep = ["solvent", "smiles", "wt_pct_exp", "label", "source"]
    rows[keep] = exp[keep].to_numpy()
    rows["in_zhou"] = rows.solvent_id < NEW_ID_START
    return rows, solvents, s_desc
