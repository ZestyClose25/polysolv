"""Comparison table (T23): Hansen prior vs XGBoost vs plain GIN vs Polysolv, on the Zhou test
split and the external experimental test set. Rows appear for every results/*.json that exists.

Usage:  python -m scripts.compare_models
Writes: results/comparison.md, results/comparison.json
"""

from __future__ import annotations

import json
import sys

from polysolv.config import REPO_ROOT

RES = REPO_ROOT / "results"
MODELS = [  # (file stem, label)
    ("hsp_baseline", "Hansen prior (+3-number head)"),
    ("xgb_baseline", "XGBoost (Morgan FP + descriptors)"),
    ("gin_plain", "Plain two-tower GIN (no physics)"),
    ("polysolv_stage1", "Polysolv stage 1"),
    ("polysolv_stage2", "Polysolv stage 2 (+ rank loss)"),
    ("polysolv_ensemble", "Polysolv 5-seed ensemble"),
]


def row(stem: str, r: dict) -> dict:
    """Pull the common numbers out of one results file."""
    if stem == "hsp_baseline":
        reg, sp = r["test_regression_prior_head"], r["test"]["spearman_neg_chi_prior"]
        sel = r["test"]["selectivity_chi_diff"]
    else:
        reg, sp, sel = r["test_regression"], r["test_spearman"], r["test_selectivity_hsp_solvents"]
    ext = r.get("external", {})
    e_all, e_z = ext.get("all", {}), ext.get("zhou_table3", {})
    return {
        "MAE": reg["MAE"], "RMSE": reg["RMSE"], "R2": reg["R2"],
        "rho_PS_RT": sp["PS/RT"], "rho_PS_Th": sp["PS/Th"], "rho_PET_RT": sp["PET/RT"], "rho_PET_Th": sp["PET/Th"],
        "sel_top10": sel["top10_hit"], "sel_top20": sel["top20_hit"], "sel_rho_margin": sel["spearman_margin"],
        "ext_AUC": e_all.get("AUC"), "ext_acc": e_all.get("accuracy"), "ext_T3_MAE": e_z.get("measured_MAE"),
        "ext_T3_rho": e_z.get("measured_spearman"),
    }


def fmt(v, nd=2) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def main() -> dict:
    table = {}
    for stem, label in MODELS:
        p = RES / f"{stem}.json"
        if p.exists():
            table[label] = row(stem, json.loads(p.read_text(encoding="utf-8")))
    cols = [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²"), ("rho_PS_RT", "ρ PS RT"), ("rho_PS_Th", "ρ PS Th"),
            ("rho_PET_RT", "ρ PET RT"), ("rho_PET_Th", "ρ PET Th"), ("sel_top10", "sel top-10"),
            ("sel_top20", "sel top-20"), ("sel_rho_margin", "ρ margin"), ("ext_AUC", "ext AUC"),
            ("ext_acc", "ext acc"), ("ext_T3_MAE", "Table 3 MAE"), ("ext_T3_rho", "Table 3 ρ")]
    lines = [
        "# Model comparison (T23)",
        "",
        "Zhou test split = 99 scaffold-split solvents never seen in training (log10 wt%; MAE/RMSE/R² on",
        "uncensored rows). ρ = Spearman per polymer/temperature. Selectivity = PS-over-PET at Th on the 69",
        "test solvents with Hansen parameters (9 selective, base rate 0.13): top-k hit rate and Spearman of",
        "the predicted vs true margin. External = 41 experimental rows (16 Zhou Table 3 + 25 handbook",
        "good/non pairs): AUC good-vs-non, accuracy of 'predicted ≥ 1 wt%', MAE/ρ on the 13 measured Table 3",
        "values (log10 wt%). External solvents are Zhou solvents (34 of 41 in the train split): this tests",
        "COSMO-RS → experiment transfer, not new solvents. Hansen-prior regression uses 3 fitted numbers.",
        "Reference: COSMO-RS itself (Zhou's predictions, the training labels) vs the same 13 measured Table 3",
        "values: MAE 1.27, ρ 0.78 — the ceiling a model trained on COSMO-RS labels can be expected to reach.",
        "",
        "| Model | " + " | ".join(h for _, h in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for label, r in table.items():
        lines.append(f"| {label} | " + " | ".join(fmt(r[k], 3 if k in ('MAE', 'RMSE', 'R2') else 2)
                                                for k, _ in cols) + " |")
    (RES / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / "comparison.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(x for x in lines if x.startswith("|")))
    return table


if __name__ == "__main__":
    main()
