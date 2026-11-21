"""Tests for polysolv.data.load_zhou (T5)."""

POLYMERS = {"EVOH", "PE", "PET", "PP", "PS", "PVC", "PA6", "PA66"}


def test_row_count(zhou_long):
    assert len(zhou_long) == 1007 * 8 * 2


def test_polymers_and_temps(zhou_long):
    assert set(zhou_long["polymer"]) == POLYMERS
    assert set(zhou_long["T_label"]) == {"RT", "Th"}


def test_every_solvent_has_smiles(zhou_long):
    assert zhou_long["smiles"].notna().all()


def test_wt_nonnegative(zhou_long):
    wt = zhou_long["wt_pct"].dropna()
    assert (wt >= 0).all()
    assert wt.max() < 100


def test_miscible_flag(zhou_long):
    m = zhou_long[zhou_long["is_miscible"]]
    assert len(m) == 2
    assert m["wt_pct"].isna().all()
    assert set(m["polymer"]) == {"PS", "EVOH"}


def test_temperatures_sane(zhou_long, cfg):
    rt = zhou_long[zhou_long["T_label"] == "RT"]
    assert (rt["T_K"] == cfg["zhou"]["rt_K"]).all()
    th = zhou_long[zhou_long["T_label"] == "Th"]["T_K"]  # sub-ambient Th exists: methane Th = 110.65 K
    assert th.notna().all()
    assert th.min() > 100 and th.max() <= 120 + 273.15 + 1e-9


def test_known_value(zhou_long):
    """PS in toluene at Th, from Zhou Table 2 (41.0 wt%)."""
    row = zhou_long[(zhou_long.solvent_name == "Toluene") & (zhou_long.polymer == "PS") & (zhou_long.T_label == "Th")]
    assert abs(row["wt_pct"].item() - 41.0) < 0.1
