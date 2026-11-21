"""Training loop (T22): Stage 1 on Zhou (+ optional stage 2 rank loss, T25).

Every run writes (``train.log_dir``, default ``results/training_logs``):
- ``<run>.log``          human-readable log: config, head init, one line per epoch, best epoch
- ``<run>_history.csv``  per-epoch numbers (train/val losses, val MAE/RMSE/R², lr, time)
- ``<run>_curves.png``   loss and val R² curves
and the best checkpoint (lowest val MSE) to ``train.checkpoint_dir/<run>.pt`` (git-ignored).
"""

from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from polysolv.config import REPO_ROOT
from polysolv.dataset import PairDataset
from polysolv.evaluate import regression_metrics
from polysolv.losses import pairwise_rank_loss, total_loss
from polysolv.model.polysolv_net import PolysolvNet, build_model


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


def run_logger(name: str, log_file: Path) -> logging.Logger:
    """Logger that writes to the console and to ``log_file`` (overwritten)."""
    log = logging.getLogger(f"train.{name}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


class RankSampler:
    """Solvents with both target and other polymer at Th in a dataset -> rank mini-batches."""

    def __init__(self, ds: PairDataset, target: str = "PS", other: str = "PET", seed: int = 0):
        lab = ds.labels.reset_index(drop=True)
        th = lab[(lab.T_label == "Th") & lab.polymer.isin([target, other])]
        w = th.reset_index().pivot_table(index="solvent_id", columns="polymer", values="index", aggfunc="first")
        w = w.dropna().astype(int)
        self.rows_t = w[target].to_numpy()
        self.rows_o = w[other].to_numpy()
        self.true_margin = torch.tensor(lab.y.to_numpy()[self.rows_t] - lab.y.to_numpy()[self.rows_o],
                                        dtype=torch.float32)
        self.ds = ds
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows_t)

    def loss(self, model: PolysolvNet, k: int, device) -> torch.Tensor:
        pick = self.rng.choice(len(self), size=min(k, len(self)), replace=False)
        b = self.ds.subset_rows(np.concatenate([self.rows_t[pick], self.rows_o[pick]])).to(device)
        y = model(b)["y_pred"]
        n = len(pick)
        return pairwise_rank_loss(y[:n] - y[n:], self.true_margin[pick].to(device))


@torch.no_grad()
def predict(model: PolysolvNet, ds: PairDataset, device, batch_size: int = 1024) -> pd.DataFrame:
    """``ds.labels`` + y_pred (and chi, A, B, dA, dB for the physics model)."""
    model.eval()
    chunks: dict[str, list] = {}
    for b in ds.loader(batch_size, shuffle=False):
        out = model(b.to(device))
        for k, v in out.items():
            chunks.setdefault(k, []).append(v.cpu().numpy())
    df = ds.labels.copy()
    for k, v in chunks.items():
        df[k] = np.concatenate(v)
    return df


@torch.no_grad()
def evaluate_loss(model, ds: PairDataset, lcfg: dict, device, batch_size: int = 1024) -> dict:
    """Mean val loss parts (no rank term) + regression metrics on uncensored rows."""
    model.eval()
    sums, n = {}, 0
    preds = []
    for b in ds.loader(batch_size, shuffle=False):
        b = b.to(device)
        out = model(b)
        parts = total_loss(out, b, dict(lcfg, w_rank=0.0), model.dB_scale)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(b)
        n += len(b)
        preds.append(out["y_pred"].cpu().numpy())
    res = {k: v / n for k, v in sums.items()}
    lab = ds.labels
    unc = ~lab.censored.astype(bool).to_numpy()
    res.update(regression_metrics(lab.y.to_numpy()[unc], np.concatenate(preds)[unc]))
    return res


def init_head(model: PolysolvNet, ds: PairDataset) -> dict | None:
    """Fit c0, slope, c2 on train rows with a Hansen prior (Stage-1 start = prior + linear head)."""
    if not model.physics:
        return None
    lab = ds.labels
    ok = (lab.hsp_mask == 1) & ~lab.censored.astype(bool)
    args = (lab.chi_prior[ok].to_numpy(), ds.cryst.numpy()[ok.to_numpy()], lab.y[ok].to_numpy())
    fits = {s: model.head.init_from_prior(*args, slope=s) for s in model.slope_grid}
    best = min(fits, key=lambda s: fits[s]["prior_mse"])
    return model.head.init_from_prior(*args, slope=best)


def save_checkpoint(path: Path, model: PolysolvNet, cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], "A0": cfg["physics"]["A0"],
                "floor": cfg["clean"]["log_floor_wt"],
                "s_desc_dim": model.enc_s.mlp[0].in_features - cfg["model"]["hidden"],
                "p_desc_dim": model.enc_p.mlp[0].in_features - cfg["model"]["hidden"], "meta": meta}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[PolysolvNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PolysolvNet(ck["s_desc_dim"], ck["p_desc_dim"], ck["model_cfg"], ck["A0"], ck["floor"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["meta"]


def plot_curves(hist: pd.DataFrame, path: Path, title: str) -> None:
    """Two panels: train/val MSE (log scale) and val R² per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=surf)
    for ax in axes:
        ax.set_facecolor(surf)
        ax.grid(color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.set_xlabel("epoch", color=ink2)
    a = axes[0]
    a.plot(hist.epoch, hist.train_mse, color="#2a78d6", lw=2, label="train MSE")
    a.plot(hist.epoch, hist.val_mse, color="#eb6834", lw=2, label="val MSE")
    a.set_yscale("log")
    a.set_ylabel("MSE (log10 wt%)²", color=ink2)
    a.legend(frameon=False, fontsize=9, labelcolor=ink)
    b = axes[1]
    b.plot(hist.epoch, hist.val_R2, color="#2a78d6", lw=2)
    b.set_ylabel("val R²", color=ink2)
    best = hist.loc[hist.val_mse.idxmin()]
    b.scatter([best.epoch], [best.val_R2], s=40, color="#2a78d6", edgecolor=surf, linewidth=2, zorder=3)
    b.annotate(f"best epoch {int(best.epoch)}: R² {best.val_R2:.3f}", (best.epoch, best.val_R2),
               textcoords="offset points", xytext=(-10, -16), ha="right", fontsize=9, color=ink)
    fig.suptitle(title, color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surf)
    plt.close(fig)


def train_model(cfg: dict, train_ds: PairDataset, val_ds: PairDataset, run_name: str, seed: int,
                epochs: int | None = None) -> tuple[PolysolvNet, pd.DataFrame, dict]:
    """Train one model; returns (best model, history, info). Logs and checkpoint as in the module doc."""
    tcfg, lcfg = cfg["train"], cfg["loss"]
    log_dir, ckpt_dir = REPO_ROOT / tcfg["log_dir"], REPO_ROOT / tcfg["checkpoint_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log = run_logger(run_name, log_dir / f"{run_name}.log")
    device = get_device(tcfg["device"])
    epochs = epochs or tcfg["epochs"]
    set_seed(seed)

    model = build_model(cfg, train_ds.s_desc_dim, train_ds.p_desc_dim).to(device)
    head = init_head(model, train_ds)
    n_par = sum(p.numel() for p in model.parameters())
    log.info("run %s | seed %d | device %s | physics %s | params %d", run_name, seed, device, model.physics, n_par)
    log.info("rows: train %d, val %d | polymers %s", len(train_ds), len(val_ds), ",".join(train_ds.p_names))
    log.info("model %s", cfg["model"])
    log.info("loss %s | train %s", lcfg, {k: v for k, v in tcfg.items() if k not in ("checkpoint_dir", "log_dir")})
    if head:
        log.info("head init on Hansen prior: c0 %.3f, slope %.3f (best of grid), c2 %.3f, prior MSE %.3f",
                 head["c0"], head["slope"], head["c2"], head["prior_mse"])

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tcfg["lr_factor"], patience=tcfg["lr_patience"])
    ranker = RankSampler(train_ds, cfg["selectivity"]["target"], cfg["selectivity"]["other"], seed) \
        if lcfg["w_rank"] > 0 else None
    if ranker:
        log.info("rank loss on %d train solvents (PS and PET at Th), %d per step", len(ranker), lcfg["rank_solvents"])

    v0 = evaluate_loss(model, val_ds, lcfg, device)
    log.info("epoch   0 | val mse %.4f R2 %.3f (start = Hansen prior + fitted head)", v0["mse"], v0["R2"])
    hist, best, best_state, wait = [], float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        sums, n = {}, 0
        for b in train_ds.loader(tcfg["batch_size"], shuffle=True, seed=seed + ep):
            b = b.to(device)
            rank = ranker.loss(model, lcfg["rank_solvents"], device) if ranker else None
            parts = total_loss(model(b), b, lcfg, model.dB_scale, rank)
            opt.zero_grad()
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(b)
            n += len(b)
        tr = {k: v / n for k, v in sums.items()}
        va = evaluate_loss(model, val_ds, lcfg, device)
        sched.step(va["mse"])
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": ep, "train_loss": tr["total"], "train_mse": tr["mse"], "train_phys": tr.get("phys", np.nan),
               "train_rank": tr.get("rank", np.nan), "val_mse": va["mse"], "val_MAE": va["MAE"],
               "val_RMSE": va["RMSE"], "val_R2": va["R2"], "lr": lr, "sec": time.time() - t0}
        hist.append(row)
        improved = va["mse"] < best
        if improved:
            best, best_state, wait = va["mse"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        extra = f" phys {row['train_phys']:.4f}" if "phys" in tr else ""
        extra += f" rank {row['train_rank']:.4f}" if "rank" in tr else ""
        log.info("epoch %3d | train %.4f (mse %.4f%s) | val mse %.4f MAE %.3f R2 %.3f | lr %.1e | %.1fs%s",
                 ep, tr["total"], tr["mse"], extra, va["mse"], va["MAE"], va["R2"], lr, row["sec"],
                 " *" if improved else "")
        if wait >= tcfg["patience"]:
            log.info("early stop: no val improvement for %d epochs", tcfg["patience"])
            break

    model.load_state_dict(best_state)
    hist = pd.DataFrame(hist)
    b_ep = int(hist.loc[hist.val_mse.idxmin(), "epoch"])
    info = {"run": run_name, "seed": seed, "best_epoch": b_ep, "best_val_mse": best, "epochs_run": int(len(hist)),
            "train_minutes": float(hist.sec.sum() / 60), "n_params": n_par, "head_init": head}
    if model.physics:
        info["head_final"] = {k: float(v.detach()) for k, v in [("c0", model.head.c0), ("slope", model.head.slope), ("c2", model.head.c2)]}
    log.info("best epoch %d | val mse %.4f | %.1f min", b_ep, best, info["train_minutes"])
    hist.to_csv(log_dir / f"{run_name}_history.csv", index=False)
    plot_curves(hist, log_dir / f"{run_name}_curves.png", f"{run_name} (seed {seed})")
    save_checkpoint(ckpt_dir / f"{run_name}.pt", model, cfg, info)
    return model, hist, info
