"""Build data/processed from data/raw + data/external (T10).

Usage (from repo root):
    python -m scripts.build_dataset [--config configs/default.yaml]

Writes: solvents.csv, polymers.csv, zhou_long.csv, zhou_dropped.csv, splits.json,
        solvent_descriptors.csv, polymer_descriptors.csv, scalers.json, build.log
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from polysolv.config import DEFAULT_CONFIG, load_config
from polysolv.data.clean import clean_zhou
from polysolv.data.load_zhou import load_zhou
from polysolv.data.merge import merge_all
from polysolv.data.splits import save_splits, scaffold_split
from polysolv.features.descriptors import add_molar_volume, build_descriptors, save_scalers
from polysolv.features.physics import add_prior_columns

log = logging.getLogger("build_dataset")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths, z, c, s = cfg["paths"], cfg["zhou"], cfg["clean"], cfg["split"]
    out = paths["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "build.log", mode="w", encoding="utf-8")],
    )
    log.info("config: %s | seed: %d", config_path, cfg["seed"])

    long = load_zhou(paths["zhou_xlsx"], z["sheet_main"], z["sheet_info"], z["rt_K"], z["polymer_rename"])
    log.info("loaded Zhou: %d long rows", len(long))

    clean, dropped = clean_zhou(long, c["log_floor_wt"], c["miscible_cap_wt"])
    dropped.to_csv(out / "zhou_dropped.csv", index=False)

    d = cfg["descriptors"]
    tables = merge_all(clean, pd.read_csv(paths["polymer_props"]))
    tables["solvents"] = add_molar_volume(tables["solvents"], d["molar_volume_anchors"])
    tables["zhou_long"] = add_prior_columns(tables["zhou_long"], tables["solvents"], tables["polymers"],
                                            cfg["physics"]["A0"])
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    fractions = tuple(s["fractions"])
    split = scaffold_split(tables["solvents"], fractions, s["seed"], s["big_group_frac"])
    save_splits(split, out / "splits.json", s["seed"], fractions)

    s_desc, p_desc, scalers = build_descriptors(tables["solvents"], tables["polymers"], split["train"], d)
    s_desc.to_csv(out / "solvent_descriptors.csv", index_label="solvent_id")
    p_desc.to_csv(out / "polymer_descriptors.csv", index_label="polymer")
    save_scalers(scalers, out / "scalers.json")
    log.info("descriptors: solvent %s, polymer %s (scaler fitted on %d train solvents)",
             s_desc.shape, p_desc.shape, len(split["train"]))
    log.info("wrote %s", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    main(ap.parse_args().config)
