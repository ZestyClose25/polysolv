"""Evaluation metrics shared by baselines and the model (used from T15; extended in T23).

Selectivity ground truth (Zhou, at Th, Th >= 25 °C):
    PS-selective  <=>  wt%(PS) >= ps_min (or 'M')  AND  wt%(PET) <= pet_max (and not 'M')
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def spearman(a, b) -> float:
    """Spearman rank correlation (NaN-safe, no scipy needed)."""
    s = pd.DataFrame({"a": np.asarray(a, dtype=float), "b": np.asarray(b, dtype=float)}).dropna()
    if len(s) < 3:
        return float("nan")
    return float(s["a"].rank().corr(s["b"].rank()))


def regression_metrics(y, y_hat) -> dict[str, float]:
    """MAE, RMSE and R² on the log10 wt% scale."""
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    err = y_hat - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def per_group_spearman(df: pd.DataFrame, score_col: str, y_col: str = "y",
                       by: tuple[str, ...] = ("polymer", "T_label")) -> dict[str, float]:
    """Spearman(score, y) per group, e.g. {'PS/RT': 0.66, ...}."""
    return {"/".join(map(str, k)): spearman(g[score_col], g[y_col]) for k, g in df.groupby(list(by))}


def selectivity_frame(labels: pd.DataFrame, solvents: pd.DataFrame, target: str = "PS", other: str = "PET",
                      ps_min: float = 10.0, pet_max: float = 0.5, min_T_C: float = 25.0) -> pd.DataFrame:
    """One row per solvent with Th labels for both polymers: y_target, y_other, true margin and ``selective``."""
    th = labels[(labels["T_label"] == "Th") & labels["polymer"].isin([target, other])]
    wide = th.pivot_table(index="solvent_id", columns="polymer", values=["y", "wt_pct", "censored"], aggfunc="first")
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.dropna(subset=[f"y_{target}", f"y_{other}"]).join(solvents.set_index("solvent_id")[["Th_C"]])
    wide = wide[wide["Th_C"] >= min_T_C]
    cens_t = wide[f"censored_{target}"].astype(bool)
    cens_o = wide[f"censored_{other}"].astype(bool)
    wide["selective"] = ((wide[f"wt_pct_{target}"] >= ps_min) | cens_t) & (wide[f"wt_pct_{other}"] <= pet_max) & ~cens_o
    wide["true_margin"] = wide[f"y_{target}"] - wide[f"y_{other}"]
    return wide


def topk_hit_rate(score: pd.Series, truth: pd.Series, k: int) -> float:
    """Fraction of the k highest-scored items that are truly selective."""
    top = score.sort_values(ascending=False).index[:k]
    return float(truth.loc[top].mean())


def selectivity_metrics(score: pd.Series, sel: pd.DataFrame, ks=(10, 20)) -> dict[str, float]:
    """Top-k hit rates and Spearman(score, true margin) for a PS-over-PET selectivity score."""
    score = score.reindex(sel.index)
    out = {f"top{k}_hit": topk_hit_rate(score, sel["selective"], k) for k in ks}
    out["spearman_margin"] = spearman(score, sel["true_margin"])
    out["n_solvents"] = int(len(sel))
    out["n_selective"] = int(sel["selective"].sum())
    out["base_rate"] = float(sel["selective"].mean())
    return out


def auc(score, positive) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count 1/2). NaN if one class is empty."""
    s = pd.Series(np.asarray(score, dtype=float))
    pos = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = s.rank().to_numpy()
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def external_metrics(df: pd.DataFrame, score_col: str = "y_pred", good_min_wt: float = 1.0,
                     floor: float = 1e-3, wt_score: bool = True) -> dict:
    """Metrics on the experimental test set (``exp_test.csv`` rows with a prediction column).

    - measured wt% rows: MAE / Spearman of predicted vs log10(wt_exp + floor)
    - good / non rows:  AUC (score ranks good above non) and, if ``wt_score``, accuracy of
      "predicted wt% >= good_min_wt"  <=>  good
    Reported for all rows and per source (Zhou Table 3 vs handbook pairs).
    """
    def block(d: pd.DataFrame) -> dict:
        out = {"n": int(len(d))}
        m = d.wt_pct_exp.notna()
        if m.sum() >= 3:
            y_exp = np.log10(d.wt_pct_exp[m].astype(float) + floor)
            out["measured_n"] = int(m.sum())
            out["measured_MAE"] = float(np.abs(d[score_col][m] - y_exp).mean()) if wt_score else float("nan")
            out["measured_spearman"] = spearman(d[score_col][m], y_exp)
        good = d.label == "good"
        out["n_good"], out["n_non"] = int(good.sum()), int((~good).sum())
        out["AUC"] = auc(d[score_col], good)
        if wt_score:
            pred_good = d[score_col] >= np.log10(good_min_wt + floor)
            out["accuracy"] = float((pred_good == good).mean())
        return out

    res = {"all": block(df)}
    for src, g in df.groupby(df.source.str.startswith("Zhou").map({True: "zhou_table3", False: "handbook"})):
        res[src] = block(g)
    return res
