"""Load the Zhou et al. (2023) ESI spreadsheet into a long table.

One output row = one (solvent, polymer, temperature) combination. No rows are
dropped here; cleaning happens in :mod:`polysolv.data.clean`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INFO_COLS = ["solvent_name", "cosmo_name", "cas", "bp_C", "Th_C", "dD", "dP", "dH"]
NUMERIC_INFO = ["bp_C", "Th_C", "dD", "dP", "dH"]
T_LABELS = ("RT", "Th")


def read_main_sheet(path: Path, sheet: str, polymer_rename: dict[str, str]) -> pd.DataFrame:
    """Read the wide solubility sheet and flatten its two-row header.

    Columns become ``INFO_COLS`` followed by ``<polymer>_<RT|Th>``.
    Polymer columns keep their raw values (floats, ``"M"`` or NaN).
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    top = raw.iloc[0].ffill()
    sub = raw.iloc[1]
    n_info = len(INFO_COLS)
    value_cols = [f"{polymer_rename.get(t, t)}_{s}" for t, s in zip(top[n_info:], sub[n_info:])]

    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = INFO_COLS + value_cols
    for c in NUMERIC_INFO:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["solvent_name"] = df["solvent_name"].astype(str).str.strip()
    return df


def read_info_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read SMILES and chemical classes (first two class labels) per solvent."""
    info = pd.read_excel(path, sheet_name=sheet)
    info = info.rename(columns={
        "Solvent name": "solvent_name",
        "SMILES": "smiles",
        "Classes": "class1",
        "Unnamed: 5": "class2",
    })
    info["solvent_name"] = info["solvent_name"].astype(str).str.strip()
    return info[["solvent_name", "smiles", "class1", "class2"]]


def load_zhou(
    path: Path,
    sheet_main: str = "All solubility predictions",
    sheet_info: str = "Solvent Info",
    rt_K: float = 298.15,
    polymer_rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return the Zhou database as a long table.

    Columns: solvent info + ``smiles, class1, class2, polymer, T_label, T_K,
    raw_value, wt_pct, is_miscible``. ``wt_pct`` is NaN for ``"M"`` and for
    missing values.
    """
    main = read_main_sheet(path, sheet_main, polymer_rename or {})
    info = read_info_sheet(path, sheet_info)
    if main["solvent_name"].duplicated().any():
        raise ValueError("solvent_name is not unique in the main sheet")
    wide = main.merge(info, on="solvent_name", how="left", validate="one_to_one")

    value_cols = [c for c in main.columns if c not in INFO_COLS]
    id_cols = INFO_COLS + ["smiles", "class1", "class2"]
    long = wide.melt(id_vars=id_cols, value_vars=value_cols, var_name="col", value_name="raw_value")
    long[["polymer", "T_label"]] = long["col"].str.rsplit("_", n=1, expand=True)
    long = long.drop(columns="col")

    long["is_miscible"] = long["raw_value"].astype(str).str.strip().eq("M")
    long["wt_pct"] = pd.to_numeric(long["raw_value"], errors="coerce")
    long["T_K"] = np.where(long["T_label"] == "RT", rt_K, long["Th_C"] + 273.15)
    return long
