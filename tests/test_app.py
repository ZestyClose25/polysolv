"""Console UI helpers (scripts/app.py) — no models loaded."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.app import find_solvents, fmt_wt, reason

CFG = {"selectivity": {"ps_min": 10.0, "pet_max": 0.5}}


def test_fmt_wt_precision():
    assert fmt_wt(26.39) == "26.4"
    assert fmt_wt(0.4852) == "0.49"
    assert fmt_wt(0.0123) == "0.012"
    assert fmt_wt(1e-5) == "<0.001"
    assert fmt_wt(np.nan) == "–"


def test_reason_covers_every_failure():
    r = lambda ps, pet: pd.Series({"PS_ok_any": ps, "PET_ok_any": pet})  # noqa: E731
    assert "too little PS" in reason(r(False, True), "PS", "PET", CFG)
    assert "PET dissolves too" in reason(r(True, False), "PS", "PET", CFG)
    assert "never at the same T" in reason(r(True, True), "PS", "PET", CFG)
    assert "at every T" in reason(r(False, False), "PS", "PET", CFG)


def _state():
    green = pd.DataFrame({"name": ["d-limonene", "ethanol"], "cas": ["5989-27-5", "64-17-5"],
                          "smiles": ["CC1=CCC(CC1)C(C)=C", "CCO"]})
    solvents = pd.DataFrame({"solvent_name": ["Dipentene (Dl-Limonene)", "Toluene"],
                             "cosmo_name": ["dipentene", "toluene"], "cas": ["138-86-3", "108-88-3"],
                             "can_smiles": ["C=C(C)C1CC=C(C)CC1", "Cc1ccccc1"]})
    return SimpleNamespace(green=green, solvents=solvents)


def test_find_solvents_prefers_green_list_and_dedupes_by_molecule():
    hits = find_solvents(_state(), "limonene")
    assert len(hits) == 1 and hits[0]["source"] == "green list"  # same molecule as Zhou's dipentene


def test_find_solvents_by_cas_and_training_data():
    assert find_solvents(_state(), "108-88-3")[0]["name"] == "Toluene"
    assert find_solvents(_state(), "no-such-solvent") == []
