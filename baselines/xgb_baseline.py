"""XGBoost baseline (T16): Morgan fingerprints + descriptors + T -> log10 wt%.

Features per (polymer, solvent, T) row:
- solvent Morgan fingerprint (radius 2, 2048 bits) from the canonical SMILES
- polymer Morgan fingerprint of a **trimer** (repeat unit joined 3 times, ends capped
  with H) — the "trimer trick" gives the fingerprint realistic backbone environments
- raw solvent descriptors (δD, δP, δH, bp, V_s; NaN allowed) and polymer descriptors
  (Tg, Tm, crystallinity, density, δD, δP, δH)
- T_K and 1/T

Trained on the train-split solvents, early stopping on val, evaluated on test with
the same metrics as the Hansen baseline (T15).

Usage:  python -m baselines.xgb_baseline [--config configs/default.yaml]
Writes: results/xgb_baseline.json, results/xgb_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.evaluate import per_group_spearman, regression_metrics, selectivity_frame, selectivity_metrics

log = logging.getLogger("xgb_baseline")

SOLVENT_COLS = ["dD", "dP", "dH", "bp_C", "V_cm3_mol"]
POLYMER_COLS = ["Tg_K", "Tm_K", "crystallinity_pct", "density_g_cm3", "dD", "dP", "dH"]


def polymer_oligomer(repeat_smiles: str, n: int = 3) -> str:
    """Join a ``*``-terminated repeat unit n times head-to-tail; cap both ends with H."""
    unit = Chem.MolFromSmiles(repeat_smiles)
    stars = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        raise ValueError(f"repeat unit needs exactly two '*': {repeat_smiles}")
    head, tail = stars
    combo = None
    for i in range(n):
        m = Chem.RWMol(unit)
        m.GetAtomWithIdx(head).SetAtomMapNum(i if i > 0 else 0)          # 0 = no link (chain end)
        m.GetAtomWithIdx(tail).SetAtomMapNum(i + 1 if i < n - 1 else 0)
        combo = m if combo is None else Chem.CombineMols(combo, m)
    zipped = Chem.RWMol(Chem.molzip(combo))
    for atom in zipped.GetAtoms():
        if atom.GetAtomicNum() == 0:  # remaining chain-end dummies -> H
            atom.SetAtomicNum(1)
    mol = Chem.RemoveHs(zipped)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def morgan_bits(smiles_list, radius: int, n_bits: int) -> np.ndarray:
    """Morgan fingerprints as a uint8 matrix (n_molecules, n_bits)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.uint8) for s in smiles_list])


def build_features(labels: pd.DataFrame, solvents: pd.DataFrame, polymers: pd.DataFrame, xcfg: dict):
    """Return (X, feature_names) aligned with ``labels`` rows."""
    r, nb = xcfg["fp_radius"], xcfg["fp_bits"]
    s_fp = pd.DataFrame(morgan_bits(solvents.can_smiles, r, nb), index=solvents.solvent_id)
    oligo = [polymer_oligomer(s, xcfg["trimer_n"]) for s in polymers.repeat_smiles]
    p_fp = pd.DataFrame(morgan_bits(oligo, r, nb), index=polymers.polymer)
    s_desc = solvents.set_index("solvent_id")[SOLVENT_COLS].add_prefix("s_")
    p_desc = polymers.set_index("polymer")[POLYMER_COLS].add_prefix("p_")

    X = np.hstack([
        s_fp.loc[labels.solvent_id].to_numpy(np.float32),
        p_fp.loc[labels.polymer].to_numpy(np.float32),
        s_desc.loc[labels.solvent_id].to_numpy(np.float32),
        p_desc.loc[labels.polymer].to_numpy(np.float32),
        labels[["T_K"]].to_numpy(np.float32),
        (1.0 / labels[["T_K"]]).to_numpy(np.float32),
    ])
    names = ([f"sfp{i}" for i in range(nb)] + [f"pfp{i}" for i in range(nb)]
             + list(s_desc.columns) + list(p_desc.columns) + ["T_K", "inv_T"])
    return X, names


def selectivity_from_predictions(pred: pd.DataFrame, target: str = "PS", other: str = "PET") -> pd.Series:
    """Score = predicted y(target) − y(other) at Th, per solvent."""
    th = pred[(pred.T_label == "Th") & pred.polymer.isin([target, other])]
    w = th.pivot_table(index="solvent_id", columns="polymer", values="y_pred", aggfunc="first")
    return (w[target] - w[other]).dropna()


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    d, xcfg, seed = cfg["paths"]["processed_dir"], cfg["xgb"], cfg["seed"]
    labels = pd.read_csv(d / "zhou_long.csv")
    solvents = pd.read_csv(d / "solvents.csv")
    polymers = pd.read_csv(d / "polymers.csv")
    split = json.loads((d / "splits.json").read_text())

    X, names = build_features(labels, solvents, polymers, xcfg)
    y = labels["y"].to_numpy()
    part = {k: labels.solvent_id.isin(split[k]).to_numpy() for k in ("train", "val", "test")}
    log.info("features: %d | rows train %d, val %d, test %d", X.shape[1], *(int(m.sum()) for m in part.values()))

    params = dict(xcfg["params"])
    n_rounds = params.pop("n_estimators")
    params.update(objective="reg:squarederror", eval_metric="rmse", seed=seed)
    dtrain = xgb.DMatrix(X[part["train"]], label=y[part["train"]], feature_names=names)
    dval = xgb.DMatrix(X[part["val"]], label=y[part["val"]], feature_names=names)
    model = xgb.train(params, dtrain, num_boost_round=n_rounds, evals=[(dval, "val")],
                      early_stopping_rounds=xcfg["early_stopping_rounds"], verbose_eval=False)
    log.info("best iteration: %d", model.best_iteration)

    def predict(mask):
        return model.predict(xgb.DMatrix(X[mask], feature_names=names), iteration_range=(0, model.best_iteration + 1))

    test = labels[part["test"]].copy()
    test["y_pred"] = predict(part["test"])
    hsp_ids = set(solvents.loc[solvents.has_hsp, "solvent_id"])

    sel_cfg = cfg["selectivity"]
    sel_all = selectivity_frame(test, solvents, **sel_cfg)
    score = selectivity_from_predictions(test)
    sel_hsp = sel_all[sel_all.index.isin(hsp_ids)]  # same solvents as the Hansen baseline
    results = {
        "seed": seed,
        "best_iteration": int(model.best_iteration),
        "n_features": int(X.shape[1]),
        "test_regression": regression_metrics(test.y, test.y_pred),
        "test_regression_by_polymer": {p: regression_metrics(g.y, g.y_pred) for p, g in test.groupby("polymer")},
        "test_spearman": per_group_spearman(test, "y_pred"),
        "test_selectivity_hsp_solvents": selectivity_metrics(score, sel_hsp),
        "test_selectivity_all": selectivity_metrics(score, sel_all),
        "val_regression": regression_metrics(y[part["val"]], predict(part["val"])),
    }
    res_dir = REPO_ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "xgb_baseline.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    test[["solvent_id", "polymer", "T_label", "T_K", "y", "y_pred"]].to_csv(res_dir / "xgb_test_predictions.csv", index=False)
    r = results["test_regression"]
    log.info("TEST  MAE %.3f RMSE %.3f R2 %.3f (log10 wt%%)", r["MAE"], r["RMSE"], r["R2"])
    log.info("TEST  Spearman PS/RT %.3f PS/Th %.3f PET/RT %.3f PET/Th %.3f",
             *(results["test_spearman"][k] for k in ("PS/RT", "PS/Th", "PET/RT", "PET/Th")))
    log.info("TEST  selectivity (HSP solvents): %s", results["test_selectivity_hsp_solvents"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    main(ap.parse_args().config)
