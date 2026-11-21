"""Train the 5-seed deep ensemble (T26) and evaluate it on the Zhou test + external test.

Usage:  python -m scripts.run_ensemble [--eval-only]
Writes: checkpoints/ensemble/seed_<s>.pt, results/training_logs/ensemble_seed_<s>.*,
        results/polysolv_ensemble.json, results/polysolv_ensemble_test_predictions.csv,
        results/ensemble_calibration.{csv,png}
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.dataset import load_processed, split_datasets
from polysolv.ensemble import calibration, load_ensemble, member_paths, plot_calibration
from polysolv.inference import predict_rows
from polysolv.train import get_device, train_model
from scripts.run_training import external_report, test_report


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    proc = load_processed(cfg["paths"]["processed_dir"])
    ds = split_datasets(proc)
    device = get_device(cfg["train"]["device"])
    ck_dir = REPO_ROOT / cfg["train"]["checkpoint_dir"]

    if not a.eval_only:
        c = copy.deepcopy(cfg)
        c["loss"]["w_rank"] = cfg["ensemble"]["w_rank"]
        for seed, path in zip(cfg["ensemble"]["seeds"], member_paths(cfg)):
            if path.exists():
                print(f"seed {seed}: exists, skipped")
                continue
            run = f"ensemble_seed_{seed}"
            train_model(c, ds["train"], ds["val"], run, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(ck_dir / f"{run}.pt", path)

    models = load_ensemble(cfg, device)
    test = ds["test"]
    pred = predict_rows(models, test.labels, proc["solvents"], proc["s_desc"], proc, device)
    results = {"seeds": cfg["ensemble"]["seeds"], "w_rank": cfg["ensemble"]["w_rank"], **test_report(pred, proc, cfg)}
    results["external"], ext = external_report(models, proc, cfg, device)
    tab, summ = calibration(pred, cfg["ensemble"]["calib_bins"])
    results["calibration"] = summ
    # single members for reference: spread of test R2 across seeds
    per_seed = []
    for m in models:
        p1 = predict_rows([m], test.labels, proc["solvents"], proc["s_desc"], proc, device)
        per_seed.append(test_report(p1, proc, cfg)["test_regression"]["R2"])
    results["member_test_R2"] = per_seed

    res = REPO_ROOT / "results"
    (res / "polysolv_ensemble.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    pred[["solvent_id", "polymer", "T_label", "T_K", "y", "censored", "y_pred", "y_std", "chi", "chi_std"]].to_csv(
        res / "polysolv_ensemble_test_predictions.csv", index=False)
    ext.to_csv(res / "polysolv_ensemble_external_predictions.csv", index=False)
    tab.to_csv(res / "ensemble_calibration.csv", index=False)
    plot_calibration(tab, res / "ensemble_calibration.png")
    r = results["test_regression"]
    print(f"ENSEMBLE TEST R2 {r['R2']:.3f} MAE {r['MAE']:.3f} | members R2 {per_seed} | calib {summ}")
    print("selectivity:", results["test_selectivity_hsp_solvents"], "| external:", results["external"]["all"])
    return results


if __name__ == "__main__":
    main()
