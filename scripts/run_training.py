"""Train Polysolv (T22) and evaluate it on the Zhou test split.

Usage:
    python -m scripts.run_training                         # stage 1, run name 'polysolv_stage1'
    python -m scripts.run_training --no-physics            # plain two-tower GIN baseline ('gin_plain')
    python -m scripts.run_training --w-rank 0.5            # stage 2 (T25) with the PS>PET rank loss
    python -m scripts.run_training --overfit 100           # sanity: memorise 100 training rows

Writes results/<run>.json, results/<run>_{test,external}_predictions.csv,
results/training_logs/<run>.{log,_history.csv,_curves.png}, checkpoints/<run>.pt
"""

from __future__ import annotations

import argparse
import copy
import json

import pandas as pd

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.dataset import PairDataset, load_processed, split_datasets
from polysolv.evaluate import (external_metrics, per_group_spearman, regression_metrics, selectivity_frame,
                               selectivity_metrics)
from polysolv.inference import external_rows, predict_rows
from polysolv.train import get_device, predict, train_model


def selectivity_score(pred: pd.DataFrame, target: str, other: str, col: str = "y_pred") -> pd.Series:
    """Predicted y(target) - y(other) at Th per solvent."""
    th = pred[(pred.T_label == "Th") & pred.polymer.isin([target, other])]
    w = th.pivot_table(index="solvent_id", columns="polymer", values=col, aggfunc="first")
    return (w[target] - w[other]).dropna()


def test_report(pred: pd.DataFrame, proc: dict, cfg: dict) -> dict:
    """Same metrics as the HSP and XGBoost baselines."""
    solvents, sel_cfg = proc["solvents"], cfg["selectivity"]
    sel_all = selectivity_frame(pred, solvents, **sel_cfg)
    hsp_ids = set(solvents.loc[solvents.has_hsp, "solvent_id"])
    score = selectivity_score(pred, sel_cfg["target"], sel_cfg["other"])
    unc = ~pred.censored.astype(bool)
    return {
        "test_regression": regression_metrics(pred.y[unc], pred.y_pred[unc]),
        "test_regression_by_polymer": {p: regression_metrics(g.y, g.y_pred) for p, g in pred[unc].groupby("polymer")},
        "test_spearman": per_group_spearman(pred, "y_pred"),
        "test_selectivity_hsp_solvents": selectivity_metrics(score, sel_all[sel_all.index.isin(hsp_ids)]),
        "test_selectivity_all": selectivity_metrics(score, sel_all),
    }


def external_report(models, proc: dict, cfg: dict, device, p_desc=None) -> tuple[dict, pd.DataFrame]:
    """Predict the experimental test set (exp_test.csv) and score it."""
    rows, solvents, s_desc = external_rows(pd.read_csv(cfg["paths"]["exp_test"]), proc, cfg)
    pred = predict_rows(models, rows, solvents, s_desc, proc, device, p_desc)
    return external_metrics(pred, "y_pred", cfg["external"]["good_min_wt"], cfg["clean"]["log_floor_wt"]), pred


def overfit_run(cfg: dict, proc: dict, n: int, seed: int) -> dict:
    """Sanity check: train and 'validate' on the same n rows, no regularisation."""
    c = copy.deepcopy(cfg)
    c["model"]["dropout"] = 0.0
    c["loss"]["w_phys"] = 0.0
    c["train"].update(weight_decay=0.0, patience=10 ** 6, lr_patience=10 ** 6, batch_size=n)
    lab = proc["labels"]
    rows = lab[lab.solvent_id.isin(proc["splits"]["train"])].sample(n, random_state=seed)
    ds = PairDataset(rows, proc["solvents"], proc["polymers"], proc["s_desc"], proc["p_desc"], proc["graphs"])
    _, hist, info = train_model(c, ds, ds, f"overfit_{n}", seed, epochs=300)
    return {**info, "final_train_mse": float(hist.train_mse.iloc[-1]), "final_R2": float(hist.val_R2.iloc[-1])}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--w-rank", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    seed = cfg["seed"] if a.seed is None else a.seed
    proc = load_processed(cfg["paths"]["processed_dir"])
    if a.overfit:
        info = overfit_run(cfg, proc, a.overfit, seed)
        print(json.dumps(info, indent=1))
        return info
    if a.no_physics:
        cfg["model"]["physics"] = False
    if a.w_rank is not None:
        cfg["loss"]["w_rank"] = a.w_rank
    run = a.run_name or ("gin_plain" if a.no_physics else "polysolv_stage2" if cfg["loss"]["w_rank"] > 0
                         else "polysolv_stage1")

    ds = split_datasets(proc)
    model, hist, info = train_model(cfg, ds["train"], ds["val"], run, seed, a.epochs)
    device = get_device(cfg["train"]["device"])
    pred = predict(model, ds["test"], device)
    val = predict(model, ds["val"], device)
    results = {**info, **test_report(pred, proc, cfg), "val_regression": regression_metrics(val.y, val.y_pred)}
    results["external"], ext = external_report([model], proc, cfg, device)

    res_dir = REPO_ROOT / "results"
    (res_dir / f"{run}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    keep = [c for c in ["solvent_id", "polymer", "T_label", "T_K", "y", "censored", "y_pred", "chi", "A", "B",
                        "dA", "dB", "chi_prior"] if c in pred]
    pred[keep].to_csv(res_dir / f"{run}_test_predictions.csv", index=False)
    ext.drop(columns=["n_models", "y_std"]).to_csv(res_dir / f"{run}_external_predictions.csv", index=False)
    r, s = results["test_regression"], results["test_spearman"]
    print(f"{run} TEST MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} R2 {r['R2']:.3f} | Spearman PS RT {s['PS/RT']:.2f} "
          f"Th {s['PS/Th']:.2f} PET RT {s['PET/RT']:.2f} Th {s['PET/Th']:.2f}")
    print("selectivity (HSP solvents):", results["test_selectivity_hsp_solvents"])
    print("external:", results["external"]["all"])
    return results


if __name__ == "__main__":
    main()
