"""Leave-one-polymer-out (T24): train on 7 polymers, predict the 8th.

For each held-out polymer P: polymer descriptor scaler refitted on the other 7, training
on train-split solvents of the 7 polymers (early stopping on val), evaluation on ALL rows
of P (every solvent; P was never seen) and on the test-split rows of P. Reference: the
Hansen prior (-chi_prior), which needs no training, and the in-distribution stage-1 model.

Usage:  python -m scripts.run_lopo [--polymers PS PET] [--no-physics]
Writes: results/lopo.json (lopo_plain.json), results/lopo.md, results/training_logs/lopo[_plain]_<P>.*
"""

from __future__ import annotations

import argparse
import json

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.dataset import PairDataset, load_processed, refit_polymer_descriptors, split_datasets
from polysolv.evaluate import per_group_spearman, regression_metrics, spearman
from polysolv.train import get_device, predict, train_model


def metrics(pred) -> dict:
    unc = ~pred.censored.astype(bool)
    out = regression_metrics(pred.y[unc], pred.y_pred[unc])
    out["spearman"] = {k.split("/")[1]: v for k, v in per_group_spearman(pred, "y_pred").items()}
    out["spearman_prior"] = {t: spearman(-g.chi_prior[g.hsp_mask == 1], g.y[g.hsp_mask == 1])
                             for t, g in pred.groupby("T_label")}
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--polymers", nargs="*", default=None)
    ap.add_argument("--no-physics", action="store_true", help="plain two-tower GIN (results/lopo_plain.*)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    tag = "lopo_plain" if a.no_physics else "lopo"
    if a.no_physics:
        cfg["model"]["physics"] = False
    proc = load_processed(cfg["paths"]["processed_dir"])
    device = get_device(cfg["train"]["device"])
    res_path = REPO_ROOT / "results" / f"{tag}.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    all_p = sorted(proc["polymers"].polymer)

    for held in a.polymers or cfg["lopo"]["polymers"]:
        keep = [p for p in all_p if p != held]
        p_desc, _ = refit_polymer_descriptors(proc["polymers"], keep, cfg["descriptors"])
        ds = split_datasets(proc, keep, p_desc, parts=("train", "val"))
        model, _, info = train_model(cfg, ds["train"], ds["val"], f"{tag}_{held}", cfg["seed"])
        lab = proc["labels"][proc["labels"].polymer == held]
        held_ds = PairDataset(lab, proc["solvents"], proc["polymers"], proc["s_desc"], p_desc, proc["graphs"])
        pred = predict(model, held_ds, device)
        test = pred[pred.solvent_id.isin(proc["splits"]["test"])]
        results[held] = {"seed": cfg["seed"], "best_epoch": info["best_epoch"], "all_solvents": metrics(pred),
                         "test_solvents": metrics(test)}
        res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        r = results[held]["all_solvents"]
        print(f"LOPO {held}: R2 {r['R2']:.3f} MAE {r['MAE']:.3f} | rho RT {r['spearman'].get('RT', float('nan')):.2f} "
              f"Th {r['spearman'].get('Th', float('nan')):.2f} | prior rho RT {r['spearman_prior'].get('RT'):.2f}")
    write_tables()
    return results


def write_tables() -> None:
    """results/lopo.md: Polysolv and (if run) plain GIN side by side."""
    res = REPO_ROOT / "results"
    load = lambda n: json.loads((res / n).read_text()) if (res / n).exists() else {}  # noqa: E731
    write_table(load("lopo.json"), load("lopo_plain.json"))


def write_table(results: dict, plain: dict | None = None) -> None:
    s1 = REPO_ROOT / "results" / "polysolv_stage1.json"
    ind = json.loads(s1.read_text())["test_regression_by_polymer"] if s1.exists() else {}
    lines = ["# Leave-one-polymer-out (T24)", "",
             "Held-out polymer never seen in training (graph, descriptors or labels). Metrics on all its Zhou rows",
             "(log10 wt%). 'In-dist. R²' = stage-1 model on the test split, polymer seen in training.", "",
             "Plain GIN = same protocol without the physics layer (no Hansen prior, no chi).", "",
             "| Held out | R² | MAE | ρ RT | ρ Th | Plain GIN R² | Plain GIN ρ RT | Plain GIN ρ Th "
             "| Hansen prior ρ RT | Hansen prior ρ Th | In-dist. R² |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    plain = plain or {}
    f = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    for p, r in results.items():
        a = r["all_solvents"]
        q = plain.get(p, {}).get("all_solvents", {})
        lines.append(f"| {p} | {a['R2']:.2f} | {a['MAE']:.2f} | {f(a['spearman'].get('RT'))} | "
                     f"{f(a['spearman'].get('Th'))} | {f(q.get('R2'))} | {f(q.get('spearman', {}).get('RT'))} | "
                     f"{f(q.get('spearman', {}).get('Th'))} | {f(a['spearman_prior'].get('RT'))} | "
                     f"{f(a['spearman_prior'].get('Th'))} | {f(ind.get(p, {}).get('R2'))} |")
    (REPO_ROOT / "results" / "lopo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
