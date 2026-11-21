"""Screen green solvents for selective PS dissolution from PS/PET waste (T28).

Usage:
    python -m scripts.run_screening --target PS --other PET --waste-mass 100 --ps-fraction 0.3
Uses the deep ensemble (checkpoints/ensemble/, from scripts.run_ensemble); falls back to the
stage-1 checkpoint (sigma = 0) if the ensemble is missing.
Writes: results/screening_<target>_<other>.csv (ranked), ..._grid.csv (all T), ..._rejected.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.dataset import load_processed
from polysolv.ensemble import load_ensemble
from polysolv.screen import rejection_reasons, screen
from polysolv.train import get_device, load_checkpoint


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--target", default="PS")
    ap.add_argument("--other", default="PET")
    ap.add_argument("--waste-mass", type=float, default=100.0, help="g of mixed waste")
    ap.add_argument("--ps-fraction", type=float, default=0.3, help="mass fraction of the target polymer")
    ap.add_argument("--min-greenness", type=int, default=None,
                    help="override screen.min_greenness (0 = all solvents, e.g. to check that DMF is rejected)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.min_greenness is not None:
        cfg["screen"]["min_greenness"] = a.min_greenness
    device = get_device(cfg["train"]["device"])
    proc = load_processed(cfg["paths"]["processed_dir"])
    try:
        models = load_ensemble(cfg, device)
    except FileNotFoundError as err:
        print(f"{err}\n-> using the single stage-1 model (no uncertainty)")
        models = [load_checkpoint(REPO_ROOT / cfg["train"]["checkpoint_dir"] / "polysolv_stage1.pt", device)[0]]
    green = pd.read_csv(cfg["paths"]["green_solvents"])
    ranked, grid = screen(models, proc, cfg, green, a.target, a.other, a.waste_mass, a.ps_fraction, device)
    stem = REPO_ROOT / "results" / f"screening_{a.target}_{a.other}{a.tag}"
    ranked.to_csv(f"{stem}.csv", index_label="rank")
    grid.to_csv(f"{stem}_grid.csv", index=False)
    rejection_reasons(grid).to_csv(f"{stem}_rejected.csv", index=False)
    cols = ["solvent", "T_window_C", "T_operate_C", "S_PS_sat_wt", "loading_wt", "solvent_to_waste",
            "PS_purity_pct", "dissolution_time_min", "confidence", "greenness"]
    print(f"{len(ranked)} of {grid.solvent_id.nunique()} green solvents pass "
          f"(waste {a.waste_mass:g} g, {100 * a.ps_fraction:g} % {a.target}, {len(models)} model(s))")
    if len(ranked):
        print(ranked[cols].to_string())
    return ranked


if __name__ == "__main__":
    main()
