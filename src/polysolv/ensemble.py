"""Deep ensemble (T26): N seeds of the same model; prediction = mean ± σ over members.

σ is the spread between members (epistemic uncertainty only; COSMO-RS label noise is not
modelled). ``calibration`` bins rows by σ and reports the mean absolute error per bin; a
useful σ has error growing with σ (Spearman(σ, |err|) > 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polysolv.config import REPO_ROOT
from polysolv.evaluate import spearman
from polysolv.train import load_checkpoint

ENSEMBLE_DIR = "ensemble"


def member_paths(cfg: dict) -> list[Path]:
    d = REPO_ROOT / cfg["train"]["checkpoint_dir"] / ENSEMBLE_DIR
    return [d / f"seed_{s}.pt" for s in cfg["ensemble"]["seeds"]]


def load_ensemble(cfg: dict, device="cpu") -> list:
    """Load every trained member (raises if one is missing)."""
    paths = member_paths(cfg)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ensemble members missing: {missing} (run python -m scripts.run_ensemble)")
    return [load_checkpoint(p, device)[0] for p in paths]


def calibration(pred: pd.DataFrame, n_bins: int = 10, sigma_col: str = "y_std") -> tuple[pd.DataFrame, dict]:
    """Per σ-quantile bin: mean σ, mean |error|, RMSE, n. Plus Spearman(σ, |err|) and coverage."""
    d = pred[~pred.censored.astype(bool)].copy()
    d["abs_err"] = (d.y_pred - d.y).abs()
    d["bin"] = pd.qcut(d[sigma_col].rank(method="first"), n_bins, labels=False)
    tab = d.groupby("bin").agg(sigma=(sigma_col, "mean"), mae=("abs_err", "mean"),
                               rmse=("abs_err", lambda e: float(np.sqrt((e ** 2).mean()))), n=("abs_err", "size"))
    z = d.abs_err / d[sigma_col].clip(lower=1e-6)
    summary = {"spearman_sigma_abs_err": spearman(d[sigma_col], d.abs_err),
               "coverage_1sigma": float((z <= 1).mean()), "coverage_2sigma": float((z <= 2).mean()),
               "mean_sigma": float(d[sigma_col].mean())}
    return tab.reset_index(), summary


def plot_calibration(tab: pd.DataFrame, path: Path) -> None:
    """Mean |error| vs mean σ per bin, with the y = x line for reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=surf)
    ax.set_facecolor(surf)
    ax.grid(color=grid, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hi = float(max(tab.sigma.max(), tab.mae.max())) * 1.1
    ax.plot([0, hi], [0, hi], color=ink2, lw=1, ls="--", label="|error| = σ")
    ax.plot(tab.sigma, tab.mae, color="#2a78d6", lw=2, marker="o", ms=6, mec=surf, mew=2, label="ensemble")
    ax.set_xlabel("mean ensemble σ per bin (log10 wt%)", color=ink2)
    ax.set_ylabel("mean |error| (log10 wt%)", color=ink2)
    ax.tick_params(colors=ink2, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)
    ax.set_title("Calibration on the Zhou test split", color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surf)
    plt.close(fig)
