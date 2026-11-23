"""Interactive console UI for the green-solvent solubility predictor (rich).

Usage:  python -m scripts.app

Modes
  1  Screen green solvents for a waste stream  -> ranked solvents with T window, loading, solvent amount,
                                                  purity, dissolution time, confidence (+ why others fail)
  2  Evaluate one solvent (name, CAS or SMILES) -> solubility of both polymers vs T, chi(T), verdict,
                                                  mass balance at the best temperature
  3  Model inputs, outputs and test performance
Every model input is asked (press Enter to keep the default shown in brackets).

All predictions go through ``polysolv.screen.screen`` (the same code as ``scripts.run_screening``), so the
UI adds no chemistry or ML logic of its own: it only collects inputs, overrides the config and renders.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from polysolv.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from polysolv.data.clean import canonical_smiles
from polysolv.dataset import load_processed
from polysolv.ensemble import load_ensemble
from polysolv.screen import rejection_reasons, screen
from polysolv.train import get_device, load_checkpoint

RDLogger.DisableLog("rdApp.*")  # invalid SMILES typed by the user are handled, not logged
console = Console()

APP_TITLE = "Green-Solvent Solubility Predictor"
CONF_STYLE = {"high": "bold green", "medium": "bold yellow", "low": "bold red"}
GREEN_LEVELS = {3: "recommended", 2: "problematic", 1: "hazardous", 0: "highly hazardous"}
POLYMER_NAMES = {"PS": "polystyrene", "PET": "poly(ethylene terephthalate)", "PE": "polyethylene",
                 "PP": "polypropylene", "PVC": "poly(vinyl chloride)", "EVOH": "ethylene-vinyl alcohol",
                 "PA6": "nylon 6", "PA66": "nylon 66"}
ZHOU_T_MAX_C = 120.0  # hottest training temperature


# ----------------------------------------------------------------------------------------------- state
class State:
    """Config, processed data and the trained models, loaded once."""

    def __init__(self) -> None:
        self.cfg = load_config(DEFAULT_CONFIG)
        self.device = get_device(self.cfg["train"]["device"])
        self.proc = load_processed(self.cfg["paths"]["processed_dir"])
        try:
            self.models = load_ensemble(self.cfg, self.device)
            self.model_note = f"{len(self.models)}-model ensemble (mean ± σ)"
        except FileNotFoundError:
            ck = REPO_ROOT / self.cfg["train"]["checkpoint_dir"] / "polysolv_stage1.pt"
            self.models = [load_checkpoint(ck, self.device)[0]]
            self.model_note = "single model (ensemble missing: no uncertainty, σ = 0)"
        self.green = pd.read_csv(self.cfg["paths"]["green_solvents"])
        self.solvents = self.proc["solvents"]
        self.polymers = list(self.proc["polymers"].polymer)


# --------------------------------------------------------------------------------------------- prompts
def ask_float(label: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Float prompt with a range check."""
    while True:
        v = FloatPrompt.ask(label, default=default, console=console)
        if (lo is None or v >= lo) and (hi is None or v <= hi):
            return float(v)
        rng = f"{'' if lo is None else lo} … {'' if hi is None else hi}"
        console.print(f"[red]  must be in the range {rng}[/red]")


def ask_int(label: str, default: int, lo: int, hi: int) -> int:
    while True:
        v = IntPrompt.ask(label, default=default, console=console)
        if lo <= v <= hi:
            return int(v)
        console.print(f"[red]  must be between {lo} and {hi}[/red]")


def ask_optional_float(label: str) -> float:
    """Blank -> NaN (unknown)."""
    while True:
        s = Prompt.ask(f"{label} [dim](blank = unknown)[/dim]", default="", show_default=False, console=console)
        if not s.strip():
            return np.nan
        try:
            return float(s)
        except ValueError:
            console.print("[red]  enter a number or leave blank[/red]")


def ask_polymers(st: State) -> tuple[str, str]:
    tab = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    for p in st.polymers:
        tab.add_row(f"[cyan]{p}[/cyan]", POLYMER_NAMES.get(p, ""))
    console.print(Panel(tab, title="Polymers the model was trained on", expand=False))
    target = Prompt.ask("Polymer to [bold]dissolve[/bold]", choices=st.polymers, default="PS", console=console)
    others = [p for p in st.polymers if p != target]
    other = Prompt.ask("Polymer to [bold]keep solid[/bold]", choices=others,
                       default="PET" if target != "PET" else others[0], console=console)
    return target, other


def ask_waste(target: str) -> tuple[float, float]:
    console.print(Rule("[bold]Waste stream", style="dim"))
    m = ask_float("Waste mass (g)", 100.0, lo=1e-6)
    f = ask_float(f"Mass fraction of {target} in the waste (0–1)", 0.3, lo=1e-6, hi=1.0)
    return m, f


def ask_settings(st: State, target: str, other: str, with_greenness: bool) -> dict:
    """Every screening / process input, defaults from configs/default.yaml. Returns a config copy."""
    cfg = copy.deepcopy(st.cfg)
    sc, sel, mb, dc = cfg["screen"], cfg["selectivity"], cfg["mass_balance"], cfg["dissolution"]

    console.print(Rule("[bold]Selectivity rule", style="dim"))
    sel["ps_min"] = ask_float(f"Minimum {target} solubility to accept (wt%)", sel["ps_min"], lo=0)
    sel["pet_max"] = ask_float(f"Maximum {other} solubility to accept (wt%)", sel["pet_max"], lo=0)
    sc["k_sigma"] = ask_float("Safety margin in ensemble σ (0 = use the mean)", sc["k_sigma"], lo=0, hi=3)
    if with_greenness:
        console.print("  greenness: 3 recommended · 2 problematic · 1 hazardous · 0 highly hazardous (CHEM21)")
        sc["min_greenness"] = ask_int("Minimum greenness to screen", sc["min_greenness"], 0, 3)

    console.print(Rule("[bold]Temperature grid", style="dim"))
    sc["T_min_C"] = ask_float("Lowest temperature (°C)", sc["T_min_C"], lo=-50, hi=250)
    sc["T_max_C"] = ask_float("Highest temperature (°C)", sc["T_max_C"], lo=sc["T_min_C"], hi=250)
    if sc["T_max_C"] > ZHOU_T_MAX_C:
        console.print(f"[yellow]  note: above {ZHOU_T_MAX_C:g} °C there is no training data (extrapolation)[/yellow]")
    sc["T_step_C"] = ask_float("Temperature step (°C)", sc["T_step_C"], lo=0.5, hi=50)
    sc["bp_margin_C"] = ask_float("Stay this far below the boiling point (°C)", sc["bp_margin_C"], lo=0, hi=100)

    console.print(Rule("[bold]Process", style="dim"))
    mb["c_max_wt"] = ask_float(f"Viscosity cap on the {target} concentration (wt%)", mb["c_max_wt"], lo=0.1, hi=60)
    mb["load_factor"] = ask_float("Run at this fraction of the conservative saturation", mb["load_factor"],
                                  lo=0.05, hi=1.0)
    thick = ask_float("Flake thickness for the dissolution-time estimate (mm)", 2 * dc["half_thickness_mm"],
                      lo=1e-3, hi=50)
    dc["half_thickness_mm"] = thick / 2
    dc["D0_m2_s"] = ask_float("Solvent diffusivity in the swollen polymer (m²/s)", dc["D0_m2_s"], lo=1e-15, hi=1e-6)
    return cfg


# ------------------------------------------------------------------------------------------- rendering
def fmt_wt(x: float) -> str:
    """wt% with sensible precision for 1e-3 … 50."""
    if not np.isfinite(x):
        return "–"
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}" if x >= 0.1 else f"{x:.3f}" if x >= 0.001 else "<0.001"


def conf_text(c: str) -> Text:
    return Text(c, style=CONF_STYLE.get(c, ""))


def inputs_panel(title: str, rows: list[tuple[str, str]]) -> Panel:
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="dim")
    t.add_column()
    for k, v in rows:
        t.add_row(k, v)
    return Panel(t, title=title, border_style="cyan", expand=False)


def setting_rows(cfg: dict, target: str, other: str, m: float, f: float) -> list[tuple[str, str]]:
    sc, sel, mb, dc = cfg["screen"], cfg["selectivity"], cfg["mass_balance"], cfg["dissolution"]
    return [
        ("dissolve / keep solid", f"{target} / {other}"),
        ("waste", f"{m:g} g, {100 * f:g} % {target}"),
        ("accept if", f"{target} ≥ {sel['ps_min']:g} wt%  and  {other} ≤ {sel['pet_max']:g} wt%  "
                      f"(bounds = mean ∓ {sc['k_sigma']:g}σ)"),
        ("temperatures", f"{sc['T_min_C']:g} → min(bp − {sc['bp_margin_C']:g}, {sc['T_max_C']:g}) °C, "
                         f"step {sc['T_step_C']:g}"),
        ("process", f"cap {mb['c_max_wt']:g} wt%, load factor {mb['load_factor']:g}, "
                    f"flakes {2 * dc['half_thickness_mm']:g} mm, D0 {dc['D0_m2_s']:.0e} m²/s"),
    ]


def ranked_table(ranked: pd.DataFrame, target: str, other: str) -> Table:
    t = Table(title=f"Solvents that dissolve {target} and leave {other} solid", box=box.ROUNDED,
              header_style="bold", title_style="bold", show_lines=False)
    for col, kw in [("#", {"justify": "right"}), ("Solvent", {"style": "bold"}), ("T window\n°C", {}),
                    ("T run\n°C", {"justify": "right"}), (f"{target} sat.\nwt%", {"justify": "right"}),
                    (f"{other} max\nwt%", {"justify": "right"}), ("Loading\nwt%", {"justify": "right"}),
                    ("Solvent\ng (g/g)", {"justify": "right"}), (f"{target}\npurity %", {"justify": "right"}),
                    ("Time\nmin", {"justify": "right"}), ("Conf.", {}), ("CHEM21", {}), ("Notes", {"style": "dim"})]:
        t.add_column(col, **kw)
    for rank, r in ranked.iterrows():
        notes = ", ".join(n for n, ok in (("bio-based", r.bio_based), ("not in training data", not r.in_zhou)) if ok)
        t.add_row(str(rank), r.solvent, r.T_window_C, f"{r.T_operate_C:.0f}", fmt_wt(r.S_PS_sat_wt),
                  fmt_wt(r.S_PET_upper_wt), f"{r.loading_wt:.1f}", f"{r.solvent_g:.0f} ({r.solvent_to_waste:.2f})",
                  f"{r.PS_purity_pct:.1f}", "–" if pd.isna(r.dissolution_time_min) else f"{r.dissolution_time_min:.0f}",
                  conf_text(r.confidence), r.chem21, notes)
    return t


def reason(r, target: str, other: str, cfg: dict) -> str:
    ps, pet = cfg["selectivity"]["ps_min"], cfg["selectivity"]["pet_max"]
    if not r.PS_ok_any and not r.PET_ok_any:
        return f"{target} < {ps:g} and {other} > {pet:g} wt% at every T"
    if not r.PS_ok_any:
        return f"too little {target} dissolves (< {ps:g} wt%)"
    if not r.PET_ok_any:
        return f"{other} dissolves too (> {pet:g} wt%)"
    return "both limits met, but never at the same T"


def rejected_table(grid: pd.DataFrame, target: str, other: str, cfg: dict) -> Table | None:
    rej = rejection_reasons(grid)
    if rej.empty:
        return None
    rej = rej.sort_values("max_S_PS_lower", ascending=False)
    t = Table(title="Rejected solvents", box=box.SIMPLE_HEAD, header_style="bold", title_style="bold")
    t.add_column("Solvent")
    t.add_column(f"best {target}\n(lower) wt%", justify="right")
    t.add_column(f"lowest {other}\n(upper) wt%", justify="right")
    t.add_column("Reason", style="dim")
    for _, r in rej.iterrows():
        t.add_row(r["name"], fmt_wt(r.max_S_PS_lower), fmt_wt(r.min_S_PET_upper), reason(r, target, other, cfg))
    return t


def profile_table(g: pd.DataFrame, target: str, other: str, k: float) -> Table:
    """Per-temperature predictions of one solvent."""
    t = Table(box=box.SIMPLE_HEAD, header_style="bold")
    for col in ["T °C", f"{target} mean", f"{target} lower", f"χ {target}", f"{other} mean", f"{other} upper",
                f"χ {other}", "σ (log)", "Verdict"]:
        t.add_column(col, justify="right" if col != "Verdict" else "left")
    for _, r in g.sort_values("T_C").iterrows():
        verdict = (Text("✓ pass", style="bold green") if r.keep else
                   Text(f"✗ {target} too low" if not r.pass_PS else f"✗ {other} too high", style="red"))
        t.add_row(f"{r.T_C:.0f}", fmt_wt(r.S_PS_mean), fmt_wt(r.S_PS_lower), f"{r.chi_PS:.2f}", fmt_wt(r.S_PET_mean),
                  fmt_wt(r.S_PET_upper), f"{r.chi_PET:.2f}", f"{max(r.sigma_PS, r.sigma_PET):.2f}", verdict)
    t.caption = f"wt% at saturation; lower/upper = mean ∓ {k:g}σ; χ = effective Flory–Huggins parameter"
    return t


def summary_panel(r: pd.Series, target: str, other: str, m: float, f: float) -> Panel:
    rows = [("temperature window", f"{r.T_window_C} °C (run at {r.T_operate_C:.0f} °C)"),
            (f"{target} saturation", f"{fmt_wt(r.S_PS_sat_wt)} wt% (conservative {fmt_wt(r.S_PS_lower_wt)})"),
            (f"{other} dissolved (worst case)", f"{fmt_wt(r.S_PET_upper_wt)} wt% → {r.PET_dissolved_g:.3f} g"),
            ("recommended loading", f"{r.loading_wt:.1f} wt% {target}"),
            ("solvent needed", f"{r.solvent_g:.0f} g for {m:g} g waste ({r.solvent_to_waste:.2f} g/g)"),
            (f"{target} recovered", f"{m * f:.1f} g at {r.PS_purity_pct:.2f} % purity"),
            ("dissolution time", "–" if pd.isna(r.dissolution_time_min) else f"≈ {r.dissolution_time_min:.0f} min "
                                                                              "(order of magnitude)"),
            ("χ at run temperature", f"{target} {r.chi_PS:.2f} · {other} {r.chi_PET:.2f}"),
            ("confidence", f"{r.confidence} (σ {target} {r.sigma_PS:.2f}, {other} {r.sigma_PET:.2f} log units)")]
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="dim")
    t.add_column(style="bold")
    for k, v in rows:
        t.add_row(k, v)
    return Panel(t, title=f"[bold green]✓ {r.solvent} is selective[/bold green]", border_style="green", expand=False)


# --------------------------------------------------------------------------------------------- mode 1
def mode_screen(st: State) -> None:
    console.print(Rule("[bold cyan]1 · Screen green solvents for a waste stream"))
    target, other = ask_polymers(st)
    m, f = ask_waste(target)
    cfg = ask_settings(st, target, other, with_greenness=True)
    with console.status("[cyan]Predicting solubility for every solvent and temperature…"):
        ranked, grid = screen(st.models, st.proc, cfg, st.green, target, other, m, f, st.device)
    n_all = grid.solvent_id.nunique()
    console.print()
    console.print(inputs_panel("Inputs", setting_rows(cfg, target, other, m, f)
                               + [("solvents screened", f"{n_all} with greenness ≥ {cfg['screen']['min_greenness']}"),
                                  ("model", st.model_note)]))
    if ranked.empty:
        console.print(Panel(f"No solvent passes for {target}/{other} with these settings.", border_style="red"))
    else:
        console.print(ranked_table(ranked, target, other))
        console.print(f"[bold]{len(ranked)} of {n_all}[/bold] solvents pass. Solvent g = solvent for the whole "
                      f"waste batch; purity = {target} in the recovered polymer (worst-case {other} leaching).")
    rej = rejected_table(grid, target, other, cfg)
    if rej is not None and Confirm.ask("Show the rejected solvents and why?", default=False, console=console):
        console.print(rej)

    names = sorted(grid.name.unique())
    while True:
        act = Prompt.ask("\n\\[d] details of one solvent · \\[s] save results · \\[b] back",
                         choices=["d", "s", "b"],
                         default="b", console=console)
        if act == "b":
            return
        if act == "s":
            stem = REPO_ROOT / "results" / f"ui_screening_{target}_{other}"
            ranked.to_csv(f"{stem}.csv", index_label="rank")
            grid.to_csv(f"{stem}_grid.csv", index=False)
            console.print(f"[green]saved {stem.relative_to(REPO_ROOT)}.csv and _grid.csv[/green]")
        if act == "d":
            q = Prompt.ask("Solvent rank or (part of) its name", console=console).strip().lower()
            hit = (ranked.loc[int(q), "solvent"] if q.isdigit() and int(q) in ranked.index
                   else next((n for n in names if q and q in n.lower()), None))
            if hit is None:
                console.print("[red]  no such solvent[/red]")
                continue
            console.print(profile_table(grid[grid.name == hit], target, other, cfg["screen"]["k_sigma"]))


# --------------------------------------------------------------------------------------------- mode 2
def find_solvents(st: State, q: str) -> list[dict]:
    """Green list + training solvents matching a name / CAS fragment (green list first, exact names first)."""
    q = q.strip().lower()
    out, seen = [], set()
    for _, r in st.green.iterrows():
        if q in str(r["name"]).lower() or q == str(r.cas).lower():
            c = canonical_smiles(r.smiles)
            seen.add(c)
            out.append({"name": r["name"], "smiles": r.smiles, "cas": r.cas, "source": "green list", "row": r})
    z = st.solvents
    hit = z[z.solvent_name.str.lower().str.contains(q, regex=False) | z.cosmo_name.str.lower().str.contains(q, regex=False)
            | (z.cas.astype(str).str.lower() == q)]
    for _, r in hit.iterrows():
        if r.can_smiles not in seen:
            seen.add(r.can_smiles)
            out.append({"name": r.solvent_name, "smiles": r.can_smiles, "cas": r.cas, "source": "training data",
                        "row": None})
    out.sort(key=lambda d: (str(d["name"]).lower() != q, d["source"] != "green list", len(str(d["name"]))))
    return out


def ask_solvent(st: State) -> pd.DataFrame | None:
    """One-row solvent table in the green-list format (name, smiles, chem21, greenness, bio_based, bp_C, δ)."""
    console.print(Rule("[bold]Solvent", style="dim"))
    while True:
        q = Prompt.ask("Solvent name, CAS number or SMILES", console=console).strip()
        if not q:
            continue
        hits = find_solvents(st, q)
        if hits:
            if len(hits) > 1:
                t = Table(box=box.SIMPLE, header_style="bold")
                for c in ["#", "Name", "CAS", "Found in"]:
                    t.add_column(c)
                for i, h in enumerate(hits[:15], 1):
                    t.add_row(str(i), str(h["name"]), str(h["cas"]), h["source"])
                console.print(t)
                i = ask_int("Pick a number", 1, 1, min(len(hits), 15))
            else:
                i = 1
            h = hits[i - 1]
            if h["row"] is not None:
                return h["row"].to_frame().T.reset_index(drop=True)
            z = st.solvents.set_index("can_smiles").loc[h["smiles"]]
            return pd.DataFrame([{"name": h["name"], "smiles": h["smiles"], "chem21": "not rated", "greenness": 0,
                                  "bio_based": 0, "bp_C": z.bp_C, "dD": z.dD, "dP": z.dP, "dH": z.dH}])
        c = canonical_smiles(q)
        if c is None:
            console.print("[red]  not found by name/CAS and not a valid SMILES — try again[/red]")
            continue
        gmatch = [i for i, s in enumerate(st.green.smiles) if canonical_smiles(s) == c]
        if gmatch:
            r = st.green.iloc[gmatch[0]]
            console.print(f"[green]  this molecule is on the green-solvent list: {r['name']}[/green]")
            return r.to_frame().T.reset_index(drop=True)
        known = st.solvents[st.solvents.can_smiles == c]
        if len(known):
            r = known.iloc[0]
            console.print(f"[green]  this molecule is in the training data: {r.solvent_name}[/green]")
            return pd.DataFrame([{"name": r.solvent_name, "smiles": c, "chem21": "not rated", "greenness": 0,
                                  "bio_based": 0, "bp_C": r.bp_C, "dD": r.dD, "dP": r.dP, "dH": r.dH}])
        console.print("[yellow]  new molecule (not in the training data): the prediction is an extrapolation.[/yellow]")
        name = Prompt.ask("Name for this solvent", default=c, console=console)
        bp = ask_optional_float("Boiling point (°C)")
        if np.isnan(bp):
            console.print("[yellow]  without a boiling point only the lowest temperature is evaluated[/yellow]")
        console.print("  Hansen parameters (MPa½) enable the physics prior; leave blank if unknown.")
        dD, dP, dH = (ask_optional_float(f"  {k}") for k in ("δD", "δP", "δH"))
        if any(np.isnan(v) for v in (dD, dP, dH)):
            dD = dP = dH = np.nan
            console.print("[yellow]  no complete Hansen set → no physics prior (lower confidence)[/yellow]")
        return pd.DataFrame([{"name": name, "smiles": c, "chem21": "not rated", "greenness": 0, "bio_based": 0,
                              "bp_C": bp, "dD": dD, "dP": dP, "dH": dH}])


def mode_single(st: State) -> None:
    console.print(Rule("[bold cyan]2 · Evaluate one solvent"))
    target, other = ask_polymers(st)
    solv = ask_solvent(st)
    m, f = ask_waste(target)
    cfg = ask_settings(st, target, other, with_greenness=False)
    cfg["screen"]["min_greenness"] = -1  # evaluate the chosen solvent whatever its greenness
    solv["greenness"] = pd.to_numeric(solv.greenness, errors="coerce").fillna(0).astype(int)
    with console.status("[cyan]Predicting…"):
        ranked, grid = screen(st.models, st.proc, cfg, solv, target, other, m, f, st.device)
    s = solv.iloc[0].copy()
    g0 = grid.iloc[0]
    z = st.solvents[st.solvents.can_smiles == canonical_smiles(s.smiles)]
    if len(z):  # training solvents: Hansen values from the training data (the green list only has them for others)
        s[["dD", "dP", "dH"]] = z.iloc[0][["dD", "dP", "dH"]].to_numpy()
    has_hsp = all(pd.notna(s.get(k)) for k in ("dD", "dP", "dH"))
    bp = g0.bp_used_C
    console.print()
    console.print(inputs_panel("Inputs", [
        ("solvent", f"{s['name']}   [dim]{s.smiles}[/dim]"),
        ("in training data", "yes" if g0.in_zhou else "[yellow]no (extrapolation)[/yellow]"),
        ("boiling point", "unknown" if pd.isna(bp) else f"{bp:g} °C"),
        ("Hansen δD/δP/δH", f"{s.dD:g} / {s.dP:g} / {s.dH:g} MPa½" if has_hsp else "unknown (no physics prior)"),
        ("CHEM21", f"{s.chem21}" + (f" (greenness {int(s.greenness)})" if s.chem21 != "not rated" else "")),
    ] + setting_rows(cfg, target, other, m, f) + [("model", st.model_note)]))
    console.print(profile_table(grid, target, other, cfg["screen"]["k_sigma"]))
    if len(ranked):
        console.print(summary_panel(ranked.iloc[0], target, other, m, f))
    else:
        r = rejection_reasons(grid).iloc[0]
        console.print(Panel(f"[bold]{s['name']}[/bold] is not selective for {target}/{other}: "
                            f"{reason(r, target, other, cfg)}.\n"
                            f"Best {target} (lower bound) {fmt_wt(r.max_S_PS_lower)} wt%, lowest {other} "
                            f"(upper bound) {fmt_wt(r.min_S_PET_upper)} wt%.",
                            title="[bold red]✗ rejected[/bold red]", border_style="red", expand=False))


# --------------------------------------------------------------------------------------------- mode 3
def mode_info(st: State) -> None:
    ens = json.loads((REPO_ROOT / "results" / "polysolv_ensemble.json").read_text(encoding="utf-8")) \
        if (REPO_ROOT / "results" / "polysolv_ensemble.json").exists() else {}
    inp = Table(title="What the model takes (one prediction = polymer × solvent × temperature)", box=box.ROUNDED,
                header_style="bold", title_style="bold")
    inp.add_column("Input")
    inp.add_column("Details")
    inp.add_column("Asked as", style="dim")
    for row in [
        ("Polymer graph", "repeat unit as a molecular graph, '*' = chain ends", "polymer choice (8 trained)"),
        ("Polymer descriptors", "Tg, Tm, crystallinity, density", "looked up"),
        ("Solvent graph", "molecule as a graph (atoms = nodes, bonds = edges)", "name / CAS / SMILES"),
        ("Solvent descriptors", "δD, δP, δH, boiling point, molar volume", "looked up or typed (new solvents)"),
        ("Temperature", "kelvin; enters only through χ(T) = A + B/T", "temperature grid"),
        ("Hansen prior", "Ra², B_HSP = V·Ra²/(4R), mask", "computed from the δ values"),
    ]:
        inp.add_row(*row)
    out = Table(title="What it returns", box=box.ROUNDED, header_style="bold", title_style="bold")
    out.add_column("Output")
    out.add_column("Details")
    out.add_row("χ(T)", "effective Flory–Huggins parameter (learned correction to the Hansen prior)")
    out.add_row("Saturation solubility", "wt%, mean ± σ over the ensemble")
    out.add_row("Screening (not learned)", "T window, loading, solvent amount, purity, dissolution time, confidence")
    perf = Table(title="Test performance", box=box.ROUNDED, header_style="bold", title_style="bold")
    perf.add_column("Test")
    perf.add_column("Result", justify="right")
    if ens:
        r, sel, ext = ens["test_regression"], ens["test_selectivity_hsp_solvents"], ens["external"]["all"]
        perf.add_row("Unseen solvents (scaffold split), R² log10 wt%", f"{r['R2']:.2f}")
        perf.add_row("Unseen solvents, MAE (log10 units)", f"{r['MAE']:.2f}")
        perf.add_row("PS-vs-PET selectivity ranking, Spearman", f"{sel['spearman_margin']:.2f}")
        perf.add_row("Experimental data, AUC good vs non-solvent", f"{ext['AUC']:.2f}")
    limits = ("Trained on COSMO-RS (simulated) solubilities for 8 polymers at 2 temperatures (RT and ≤ 120 °C). "
              "Reliable for new solvents, not for new polymers; chain length, additives and crystallinity variance "
              "of real waste are not modelled; σ ranks uncertainty but is over-confident.")
    console.print(Group(inp, out, perf, Panel(limits, title="Limits", border_style="yellow")))


# ------------------------------------------------------------------------------------------------ main
def banner(st: State | None = None) -> Panel:
    body = Text.assemble(
        ("Which green solvent dissolves one plastic and leaves the other solid — and at what temperature?\n",
         "italic"),
        ("Physics-informed graph neural network: predicts the Flory–Huggins χ(T), then solubility.", "dim"))
    if st is not None:
        body.append(f"\nModel: {st.model_note} · device: {st.device}", style="dim")
    return Panel(body, title=f"[bold]{APP_TITLE}[/bold]", border_style="cyan", padding=(1, 2))


def main() -> None:
    console.print(banner())
    with console.status("[cyan]Loading data and trained models…"):
        st = State()
    console.print(f"[green]ready[/green] · {st.model_note} · {len(st.solvents)} training solvents · "
                  f"{len(st.green)} green solvents · device {st.device}")
    while True:
        console.print()
        menu = Table(box=None, show_header=False, pad_edge=False)
        for k, v in [("1", "Screen green solvents for a waste stream"),
                     ("2", "Evaluate one solvent (name, CAS or SMILES)"),
                     ("3", "Model inputs, outputs and test performance"), ("q", "Quit")]:
            menu.add_row(f"[bold cyan]{k}[/bold cyan]", v)
        console.print(Panel(menu, title="Menu", border_style="cyan", expand=False))
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "q"], default="1", console=console)
        try:
            if choice == "1":
                mode_screen(st)
            elif choice == "2":
                mode_single(st)
            elif choice == "3":
                mode_info(st)
            else:
                console.print("[dim]bye[/dim]")
                return
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]cancelled — back to the menu[/dim]")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye[/dim]")
