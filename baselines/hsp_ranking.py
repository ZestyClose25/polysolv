"""Hansen baseline (T15): rank solvents by Hansen distance, no training.

Scores (higher = more soluble / more PS-selective):
- solubility:   −Ra²(polymer, solvent)  and  −χ_prior(T) = −(A0 + V_s·Ra²/(4RT))
- selectivity:  Ra²(PET) − Ra²(PS)      and  χ_prior(PET, Th) − χ_prior(PS, Th)

For regression metrics and the external test, the prior is mapped to log10 wt% with the
same head as the model, y = log10(10**(c0 - s*chi_prior + c2*crystallinity) + floor),
fitted on the train split (slope chosen by grid search). Only 4 numbers are fitted.

RED = Ra/R0 is not used: R0 for PS is not available in our sources, and within one
polymer RED gives the same ranking as Ra².

Usage:  python -m baselines.hsp_ranking [--config configs/default.yaml]
Writes: results/hsp_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
import numpy as np

from polysolv.dataset import load_processed
from polysolv.evaluate import (external_metrics, per_group_spearman, regression_metrics, selectivity_frame,
                               selectivity_metrics)
from polysolv.inference import external_rows
from polysolv.model.heads import SolubilityHead

log = logging.getLogger("hsp_baseline")


def hsp_selectivity_scores(labels: pd.DataFrame, target: str = "PS", other: str = "PET") -> pd.DataFrame:
    """Per solvent at Th: Ra² and χ_prior differences (other − target)."""
    th = labels[(labels.T_label == "Th") & labels.polymer.isin([target, other]) & (labels.hsp_mask == 1)]
    w = th.pivot_table(index="solvent_id", columns="polymer", values=["Ra2", "chi_prior"], aggfunc="first")
    return pd.DataFrame({
        "ra2_diff": w[("Ra2", other)] - w[("Ra2", target)],
        "chi_diff": w[("chi_prior", other)] - w[("chi_prior", target)],
    }).dropna()


def evaluate(labels: pd.DataFrame, solvents: pd.DataFrame, ids: list[int] | None, sel_cfg: dict) -> dict:
    """Metrics on the given solvent ids (None = all solvents)."""
    lab = labels if ids is None else labels[labels.solvent_id.isin(ids)]
    lab = lab[lab.hsp_mask == 1].assign(neg_ra2=lambda d: -d.Ra2, neg_chi=lambda d: -d.chi_prior)
    sel = selectivity_frame(lab, solvents, **sel_cfg)
    scores = hsp_selectivity_scores(lab)
    sel = sel.loc[sel.index.intersection(scores.index)]
    return {
        "spearman_neg_ra2": per_group_spearman(lab, "neg_ra2"),
        "spearman_neg_chi_prior": per_group_spearman(lab, "neg_chi"),
        "selectivity_ra2_diff": selectivity_metrics(scores["ra2_diff"], sel),
        "selectivity_chi_diff": selectivity_metrics(scores["chi_diff"], sel),
    }


def fit_prior_head(labels: pd.DataFrame, cryst: pd.Series, floor: float, slopes=(0.25, 0.5, 0.8, 1, 1.5, 2, 3, 4)):
    """Best (by train MSE) floor-log head on chi_prior; returns the fitted SolubilityHead and its fit."""
    ok = (labels.hsp_mask == 1) & ~labels.censored.astype(bool)
    d = labels[ok]
    best = None
    for s in slopes:
        head = SolubilityHead(floor=floor)
        fit = head.init_from_prior(d.chi_prior, cryst.loc[d.polymer], d.y, slope=s)
        if best is None or fit["prior_mse"] < best[1]["prior_mse"]:
            best = (head, fit)
    return best


def head_predict(head: SolubilityHead, chi_prior, cryst) -> np.ndarray:
    import torch
    with torch.no_grad():
        return head(torch.tensor(np.asarray(chi_prior), dtype=torch.float32),
                    torch.tensor(np.asarray(cryst), dtype=torch.float32)).numpy()


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    out_dir = cfg["paths"]["processed_dir"]
    labels = pd.read_csv(out_dir / "zhou_long.csv")
    solvents = pd.read_csv(out_dir / "solvents.csv")
    split = json.loads((out_dir / "splits.json").read_text())
    sel_cfg = cfg["selectivity"]

    proc = load_processed(out_dir)
    cryst = proc["polymers"].set_index("polymer").crystallinity_pct.fillna(0) / 100
    floor = cfg["clean"]["log_floor_wt"]
    head, fit = fit_prior_head(labels[labels.solvent_id.isin(split["train"])], cryst, floor)
    test = labels[labels.solvent_id.isin(split["test"]) & (labels.hsp_mask == 1) & ~labels.censored.astype(bool)]
    test_pred = head_predict(head, test.chi_prior, cryst.loc[test.polymer])

    ext, _, _ = external_rows(pd.read_csv(cfg["paths"]["exp_test"]), proc, cfg)
    ext["y_pred"] = head_predict(head, ext.chi_prior, cryst.loc[ext.polymer])
    ext["neg_chi"] = -ext.chi_prior
    ext_m = external_metrics(ext, "y_pred", cfg["external"]["good_min_wt"], floor)
    ext_m["AUC_neg_chi_prior"] = external_metrics(ext, "neg_chi", wt_score=False)["all"]["AUC"]

    results = {
        "seed": cfg["seed"],
        "note": "no training; HSP solvents only (hsp_mask = 1); head = 3 numbers fitted on train",
        "head": fit,
        "test_regression_prior_head": regression_metrics(test.y, test_pred),
        "test": evaluate(labels, solvents, split["test"], sel_cfg),
        "all": evaluate(labels, solvents, None, sel_cfg),
        "external": ext_m,
    }
    res_dir = REPO_ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "hsp_baseline.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    t = results["test"]
    log.info("TEST  Spearman -chi_prior: PS/RT %.3f PS/Th %.3f PET/RT %.3f PET/Th %.3f",
             *(t["spearman_neg_chi_prior"][k] for k in ("PS/RT", "PS/Th", "PET/RT", "PET/Th")))
    log.info("TEST  selectivity (chi diff): %s", t["selectivity_chi_diff"])
    log.info("ALL   selectivity (chi diff): %s", results["all"]["selectivity_chi_diff"])
    log.info("TEST  prior + head: %s (head %s)", results["test_regression_prior_head"], fit)
    log.info("EXTERNAL: %s", ext_m)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    main(ap.parse_args().config)
