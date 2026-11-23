"""Report figures from the saved results (T30 plotting, T32).

Reads only files in ``results/`` (no training, no model code) and writes PNGs to
``results/figures/``. Each ``fig_*`` function returns the matplotlib Figure so the results
notebook can show it inline; missing inputs are skipped with a message, so the script can be
re-run as more experiments (LOPO, ablations) finish.

Ablation runs are picked up automatically: any ``results/ablation_<name>.json`` written by
``python -m scripts.run_training --run-name ablation_<name> ...`` appears in the ablation figure.

Usage:  python -m scripts.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from polysolv.config import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "results"
FIG_DIR = RES / "figures"

# Reference palette (dataviz skill): categorical slots 1-3, neutral for context marks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2 = "#b5b3ad", "#0b0b0b", "#52514e"
POLYSOLV_NAMES = ("Polysolv stage 1", "Polysolv 5-seed ensemble", "Polysolv (full)")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return out


def _hbar_panels(table: pd.DataFrame, metrics: dict[str, str], title: str, highlight) -> plt.Figure:
    """Small multiples: one horizontal-bar panel per metric (all metrics on a 0-1 scale)."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 0.42 * len(table) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(table))[::-1]
    for ax, (col, label) in zip(axes, metrics.items()):
        vals = table[col].to_numpy(dtype=float)
        colors = [BLUE if highlight(m) else GREY for m in table.index]
        ax.barh(y, np.nan_to_num(vals), height=0.62, color=colors, edgecolor="white", linewidth=1.5)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(max(v, 0) + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, table.index)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def fig_model_comparison() -> plt.Figure | None:
    """HSP vs XGB vs plain GIN vs Polysolv on Zhou test, selectivity and external test."""
    comp = _load("comparison.json")
    if comp is None:
        print("skip model comparison: results/comparison.json missing")
        return None
    t = pd.DataFrame(comp).T
    fig = _hbar_panels(
        t, {"R2": "Zhou test R² (log10 wt%)", "sel_rho_margin": "PS–PET margin ρ (test, Th)",
            "ext_AUC": "External AUC good vs non"},
        "Model comparison (seed 42; ensemble = seeds 42–46)", lambda m: m.startswith("Polysolv"))
    _save(fig, "fig_model_comparison.png")
    return fig


def _run_row(d: dict) -> dict:
    ext = d.get("external", {}).get("all", {})
    return {"R2": d["test_regression"]["R2"],
            "sel_rho_margin": d["test_selectivity_hsp_solvents"]["spearman_margin"],
            "ext_AUC": ext.get("AUC", np.nan)}


def fig_ablations() -> plt.Figure | None:
    """Full Polysolv vs each available ablation run (one change each)."""
    runs = {"Polysolv (full)": "polysolv_stage1.json",
            "− physics layer (plain GIN)": "gin_plain.json",
            "− floored head (linear head, v0)": "v0_linear_head.json"}
    rows = {}
    for label, f in runs.items():
        d = _load(f)
        if d is not None:
            rows[label] = _run_row(d)
    for p in sorted(RES.glob("ablation_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows["− " + p.stem.removeprefix("ablation_").replace("_", " ")] = _run_row(d)
    if len(rows) < 2:
        print("skip ablations: fewer than two runs found")
        return None
    t = pd.DataFrame(rows).T
    fig = _hbar_panels(
        t, {"R2": "Zhou test R²", "sel_rho_margin": "PS–PET margin ρ", "ext_AUC": "External AUC"},
        "Ablations (seed 42; blank = metric not recorded for that run)", lambda m: m == "Polysolv (full)")
    _save(fig, "fig_ablations.png")
    return fig


def fig_parity() -> plt.Figure | None:
    """Predicted vs true log10 wt% on the Zhou test split (5-seed ensemble), PS and PET highlighted."""
    p = RES / "polysolv_ensemble_test_predictions.csv"
    if not p.exists():
        print("skip parity: ensemble test predictions missing")
        return None
    df = pd.read_csv(p)
    df = df[~df["censored"].astype(bool)]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    other = df[~df["polymer"].isin(["PS", "PET"])]
    ax.scatter(other["y"], other["y_pred"], s=8, color=GREY, alpha=0.5, linewidths=0, label="other 6 polymers")
    for pol, c in (("PET", ORANGE), ("PS", BLUE)):
        s = df[df["polymer"] == pol]
        ax.scatter(s["y"], s["y_pred"], s=12, color=c, alpha=0.8, edgecolors="white", linewidths=0.4, label=pol)
    lim = (-3.2, 2.0)
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Zhou COSMO-RS label, log10 wt%")
    ax.set_ylabel("Polysolv ensemble prediction, log10 wt%")
    ax.set_title("Zhou test split (99 unseen solvents)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_parity_ensemble.png")
    return fig


def fig_lopo() -> plt.Figure | None:
    """Leave-one-polymer-out: Spearman at Th (Polysolv vs plain GIN vs Hansen prior) and R²."""
    lopo = _load("lopo.json")
    if not lopo:
        print("skip LOPO: results/lopo.json missing")
        return None
    plain = _load("lopo_plain.json") or {}
    pols = list(lopo)
    x = np.arange(len(pols))
    series = [("Polysolv", BLUE, [lopo[p]["all_solvents"]["spearman"]["Th"] for p in pols])]
    if plain:
        series.append(("Plain GIN", AQUA,
                       [plain.get(p, {}).get("all_solvents", {}).get("spearman", {}).get("Th", np.nan) for p in pols]))
    series.append(("Hansen prior", ORANGE, [lopo[p]["all_solvents"]["spearman_prior"]["Th"] for p in pols]))
    w = 0.8 / len(series)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for i, (lab, c, v) in enumerate(series):
        a1.bar(x + (i - (len(series) - 1) / 2) * w, np.nan_to_num(np.array(v, float)), w * 0.92,
               color=c, label=lab, edgecolor="white", linewidth=1)
    a1.set_xticks(x, pols)
    a1.set_ylim(-0.1, 1)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_title("Solvent ranking of the held-out polymer (ρ at Th)", loc="left", fontsize=9)
    a1.legend(frameon=False, fontsize=8, ncol=len(series), loc="upper left")
    s1 = _load("polysolv_stage1.json") or {}
    ind = s1.get("test_regression_by_polymer", {})
    r2_lopo = [lopo[p]["all_solvents"]["R2"] for p in pols]
    r2_ind = [ind.get(p, {}).get("R2", np.nan) for p in pols]
    for i, (lab, c, v) in enumerate((("held out (LOPO)", BLUE, r2_lopo), ("seen in training (test split)", GREY, r2_ind))):
        a2.bar(x + (i - 0.5) * 0.4, np.nan_to_num(np.array(v, float)), 0.37, color=c, label=lab,
               edgecolor="white", linewidth=1)
    a2.set_xticks(x, pols)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_title("Absolute level: R² (log10 wt%)", loc="left", fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_lopo.png")
    return fig


def fig_screening(top: int = 15) -> plt.Figure | None:
    """Top screened green solvents: conservative PS solubility and worst-case PET leaching at T_operate."""
    p = RES / "screening_PS_PET.csv"
    if not p.exists():
        print("skip screening: results/screening_PS_PET.csv missing")
        return None
    df = pd.read_csv(p).sort_values("rank").head(top)
    y = np.arange(len(df))[::-1]
    labels = [f"{n}  ({w} °C)" for n, w in zip(df["solvent"], df["T_window_C"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 0.33 * len(df) + 1.3), sharey=True,
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    a1.barh(y, df["S_PS_lower_wt"], height=0.62, color=BLUE, edgecolor="white", linewidth=1.5)
    a1.axvline(10, color=INK2, lw=1, ls="--")
    a1.text(10.3, y.max() + 0.6, "keep ≥ 10", fontsize=7.5, color=INK2)
    a1.set_xlabel("PS saturation, lower bound (wt%)")
    a1.set_yticks(y, labels)
    a1.grid(axis="y", visible=False)
    a2.barh(y, df["S_PET_upper_wt"], height=0.62, color=ORANGE, edgecolor="white", linewidth=1.5)
    a2.axvline(0.5, color=INK2, lw=1, ls="--")
    a2.text(0.51, y.max() + 0.6, "keep ≤ 0.5", fontsize=7.5, color=INK2)
    a2.set_xlabel("PET solubility, upper bound (wt%)")
    a2.grid(axis="y", visible=False)
    fig.suptitle(f"Top {len(df)} green solvents for PS/PET (5-seed ensemble, at T_operate)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_screening_top.png")
    return fig


def fig_calibration() -> plt.Figure | None:
    """Ensemble σ vs observed error per σ bin (copy of T26 data, report styling)."""
    p = RES / "ensemble_calibration.csv"
    if not p.exists():
        print("skip calibration: results/ensemble_calibration.csv missing")
        return None
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    m = max(df["rmse"].max(), df["sigma"].max()) * 1.1
    ax.plot([0, m], [0, m], color=INK2, lw=1, ls="--", label="ideal (RMSE = σ)")
    ax.plot(df["sigma"], df["rmse"], color=BLUE, lw=2, marker="o", ms=5, label="observed RMSE per σ bin")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("ensemble σ (log10 wt%)")
    ax.set_ylabel("RMSE (log10 wt%)")
    ax.set_title("Ensemble uncertainty is over-confident", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_calibration.png")
    return fig


ALL_FIGURES = (fig_model_comparison, fig_ablations, fig_parity, fig_lopo, fig_screening, fig_calibration)


def main() -> None:
    for f in ALL_FIGURES:
        fig = f()
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
