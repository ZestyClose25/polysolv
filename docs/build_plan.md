# Polysolv — Master Build Plan

> **For a new chat:** read section A (context) first, then continue from the first unchecked task in section C.
> Say "continue Polysolv from T__" and paste the progress log (section D).

---

## A. Project context (decisions already made)

**Goal.** ML model that recommends a **green solvent** to selectively dissolve **PS** from a **PS/PET** mixture (PET stays solid). Output per solvent: temperature window, saturation solubility, recommended loading / solvent-to-waste ratio, predicted PS purity, confidence. Dissolution time = optional, physics estimate only.

**Novelty.**
1. Physics-informed output: network predicts Flory–Huggins χ(T) = A + B/T (as a correction to the Hansen prior), not raw solubility.
2. Multi-fidelity training: simulated quantitative data (Zhou) + experimental binary data (Kern).
3. Selectivity-aware screening with uncertainty (deep ensemble).
4. Optional lab validation on real PS/PET waste.

**Data sources.**
| ID | Source | Content | Use |
|---|---|---|---|
| A | Zhou et al., *Green Chem.* 2023, DOI 10.1039/D3GC00404J (ESI spreadsheet) | 8 polymers (EVOH, PE, PP, PS, PET, PVC, nylon 6, nylon 66) × 1007 solvents × 2 temps (RT, Th = min(bp−1, 120) °C); wt%; solvent name, CAS, SMILES, bp, δD/δP/δH, class; `M` = miscible, `NaN` = bp < RT; COSMOtherm-generated | Regression labels, solvent features |
| B | Kern et al., *PCCP* 2022, DOI 10.1039/D2CP03735A (ESI) + GitHub `Ramprasad-Group/polymer_in_solvent_solubility_modeling` | 3,373 polymers × 51 solvents, ~20.7k soluble/insoluble pairs, room temp | Classification labels, polymer diversity |
| C | Hand-built `polymer_props.csv` (Polymer Handbook, Hansen handbook, papers) | Tg, Tm, crystallinity, density, HSP, repeat-unit SMILES for the 8 polymers | Polymer descriptors, physics prior |
| D | Hand-built `green_solvents.csv` (CHEM21, GSK guides, bio-based solvent reviews) | name, CAS, greenness category | Screening filter only |
| E (opt.) | CHAOS DB / OpenSPGen σ-profiles | 51-pt σ-profile per solvent | Ablation feature |

No AMS/COSMOtherm license needed — we only use Zhou's published numbers.

**Architecture.**
```
polymer SMILES → RDKit graph → GIN tower (4×GINEConv, 128, sum-pool) ⊕ polymer descriptors → MLP → h_p
solvent SMILES → RDKit graph → GIN tower (separate weights)           ⊕ solvent descriptors → MLP → h_s
x = [h_p, h_s, h_p⊙h_s, |h_p−h_s|, Ra², mask] → MLP → (ΔA, ΔB)
A = 0.34 + ΔA ;  B = B_HSP·mask + ΔB ;  B_HSP = V_s·Ra²/(4R) ;  Ra² = 4ΔδD² + ΔδP² + ΔδH²
χ(T) = A + B/T
Head 1: log10 wt% = c0 − softplus(c1)·χ + c2·crystallinity_term      (Zhou)
Head 2: p(soluble) = sigmoid(softplus(k)·(0.5 − χ))                  (Kern)
Uncertainty: 5-seed deep ensemble (mean ± σ)
```

**Loss.** `L = λ1·MSE(log wt%, censored for M) + λ2·BCE + λ3·PairwiseRank(PS>PET, optional) + λ4·physics_reg`
**Training.** Stage 1 pretrain Zhou → Stage 2 fine-tune Kern + 20% Zhou replay → Stage 3 (opt.) selectivity rank loss.
**Splits.** Zhou: Bemis–Murcko solvent-scaffold split 80/10/10. Kern: polymer split.
**Baselines.** Hansen RED ranking; XGBoost on Morgan FP (trimer trick for polymers) + descriptors; plain two-tower GIN without physics layer.

**Concentration handling.** Model predicts saturation S_PS, S_PET (± σ). PS:PET ratio of waste is a *screening input*, not a model input. Mass balance:
```
c       = min(0.8·S_PS_lower, c_max)              # c_max = viscosity cap from config
m_solv  = m·f_PS·(100 − c)/c
m_PET   = S_PET_upper/(100 − S_PET_upper)·m_solv  # worst-case leaching
purity  = m·f_PS/(m·f_PS + m_PET)
```

**Screening rule.** For each green solvent, T grid 25 °C → bp−10 °C step 5: keep if PS_lower ≥ 10 wt%, PET_upper ≤ 0.5 wt%, T < bp, greenness OK. `score = PS_lower − PET_upper + w·greenness`.

**Working style.** User writes core code by hand; Claude (incl. Claude Code) explains, hints, reviews — does not write core files unless asked. OK to generate: tests, plots, configs, logging, XGBoost baseline.

**Known limitations.** Zhou labels are simulated (PET-in-DMF was a known miss); only 2 temps per pair; only 8 polymers quantitative; real-waste crystallinity variance; additives not modelled.

---

## B. Repository layout

```
polysolv/
├── CLAUDE.md  README.md  pyproject.toml  .gitignore
├── configs/default.yaml
├── data/raw/  data/external/{polymer_props.csv, green_solvents.csv}  data/processed/
├── src/polysolv/
│   ├── data/{load_zhou.py, load_kern.py, clean.py, merge.py, splits.py}
│   ├── features/{graph.py, descriptors.py, physics.py}
│   ├── dataset.py
│   ├── model/{towers.py, interaction.py, flory_huggins.py, heads.py, polysolv_net.py}
│   ├── losses.py  train.py  ensemble.py  evaluate.py  mass_balance.py  screen.py
├── baselines/{hsp_ranking.py, xgb_baseline.py}
├── scripts/{build_dataset.py, run_training.py, run_screening.py}
├── notebooks/{01_explore_zhou.ipynb, 02_results.ipynb}
└── tests/
```
Legend below: **[U]** = user writes, **[C]** = fine to let Claude generate.

---

## C. Task list (do in order; each has a "done when")

### Phase 1 — Setup & data
- [ ] **T1 Setup** [U] — repo, venv, `pip install rdkit pandas numpy openpyxl jupyter`, folders, `CLAUDE.md`, `.gitignore`, first commit.
  *Done when:* `Chem.MolFromSmiles('*CC(*)c1ccccc1').GetNumAtoms() == 10`.
- [ ] **T2 Download raw data** [U] — Zhou ESI spreadsheet + PDF; Kern ESI + GitHub repo data → `data/raw/`. Note sources in README.
  *Done when:* all files present, untouched.
- [ ] **T3 Explore Zhou** [U] — `01_explore_zhou.ipynb`: sheets, columns, wide/long, solvent count, `M`/`NaN` counts per polymer, PS & PET histograms (RT, Th), count of PS-selective solvents, SMILES parse rate.
  *Done when:* you can describe the file in 3 sentences + have the PS-selective count.
- [ ] **T4 polymer_props.csv** [U] — 8 rows: polymer, repeat_smiles, Tg_K, Tm_K, crystallinity_pct, density, dD, dP, dH, source.
  *Done when:* all SMILES parse and look right when drawn; every value has a source.
- [ ] **T5 load_zhou.py** [U] + tests [C] — wide → long; RT & Th in K per solvent; `is_miscible` flag.
  *Done when:* tests pass (row count, PS/PET present, wt% ≥ 0, sane T).
- [ ] **T6 load_kern.py** [U] — ESI/GitHub tables → `polymer_smiles, solvent, label, T_K=298`.
  *Done when:* ~20.7k rows, ~57% soluble.
- [ ] **T7 clean.py** [U] — drop NaN; `M` → censored at cap; `log10(wt% + 1e-3)`; canonical SMILES; drop invalid; dedupe.
  *Done when:* clean tables saved, dropped rows logged.
- [ ] **T8 merge.py** [U] — map Kern solvents to Zhou solvents by CAS/canonical SMILES; attach solvent & polymer descriptors; write `solvents.csv`, `polymers.csv`, `zhou_long.csv`, `kern_binary.csv`.
  *Done when:* no unexplained missing descriptors for Zhou rows.
- [ ] **T9 splits.py** [U] — scaffold split (Zhou), polymer split (Kern) → `splits.json`; fixed seed.
  *Done when:* zero solvent overlap between Zhou train/test.
- [ ] **T10 build_dataset.py** [C] — one command runs T5→T9.
- [ ] **T11 green_solvents.csv** [U] (parallel, any time before T27) — 50–200 green solvents with CAS + CHEM21/GSK category; flag which are already in Zhou.

### Phase 2 — Features & physics baseline
- [ ] **T12 graph.py** [U] + tests [C] — `atom_features`, `bond_features`, `smiles_to_graph` → PyG `Data`; `*` as dummy atom; bidirectional edges; SMILES-keyed cache. Install `torch`, `torch_geometric`.
  *Done when:* PS graph 10 nodes, limonene 10 nodes, all dataset SMILES convert.
- [ ] **T13 descriptors.py** [U] — z-scored solvent (δD, δP, δH, bp, V) & polymer (Tg, Tm, cryst, density + mask) vectors; save scaler.
- [ ] **T14 physics.py** [U] + tests [C] — `Ra2()`, `B_hsp()` (units: V cm³/mol, δ MPa^½, R = 8.314).
  *Done when:* χ_prior(PS, toluene, 298 K) < 0.5 and χ_prior(PS, water) ≫ 0.5.
- [ ] **T15 HSP baseline** `hsp_ranking.py` [U] — rank solvents for PS-vs-PET by RED; evaluate vs Zhou labels (Spearman, top-k selective hit rate).
  *Done when:* first baseline numbers recorded in D.
- [ ] **T16 XGBoost baseline** [C] — Morgan FP (2048, polymer trimer trick) + descriptors + T → log wt%; same metrics & splits.
  *Done when:* baseline numbers recorded.

### Phase 3 — Core model
- [ ] **T17 dataset.py** [U] — PyG Dataset/`collate` returning (polymer graph, solvent graph, descriptors, T, Ra², mask, label, task_id, censored).
- [ ] **T18 towers.py** [U] — GIN tower (4×GINEConv, 128, BN, ReLU, dropout 0.1, sum-pool).
- [ ] **T19 interaction.py + flory_huggins.py + heads.py** [U] — interaction vector, MLP → ΔA, ΔB; χ = A + B/T; regression & classification heads.
  *Done when:* unit test: hand-set A, B give expected χ and monotone outputs.
- [ ] **T20 polysolv_net.py** [U] — wire everything; forward returns χ, log wt%, p(soluble).
  *Done when:* one batch forward/backward works, shapes correct.
- [ ] **T21 losses.py** [U] — censored MSE, BCE, pairwise rank, physics_reg.
- [ ] **T22 train.py + configs/default.yaml** [U core, C for logging/ckpt] — Stage 1 on Zhou only.
  *Done when:* val loss decreases; overfits 100 rows when asked (sanity).
- [ ] **T23 evaluate.py** [U metrics, C plots] — MAE/RMSE/R²/per-polymer Spearman on Zhou test; compare with T15–T16.
  *Done when:* table: HSP vs XGB vs Polysolv-stage1.

### Phase 4 — Multi-fidelity, ensemble, screening
- [ ] **T24 Stage 2** [U] — add Kern + 20% Zhou replay; report F1/AUC on Kern polymer-split test.
- [ ] **T25 Stage 3 (optional)** [U] — PS>PET pairwise rank loss; check top-k selectivity changes.
- [ ] **T26 ensemble.py** [U] — 5 seeds, mean ± σ; calibration plot (error vs σ).
- [ ] **T27 mass_balance.py** [U] + tests [C] — `required_solvent`, `pet_leaching`, `ps_purity`.
  *Done when:* worked example (100 g, 30% PS, S_PS_lower 38, S_PET_upper 0.02, c_max 10) → 270 g solvent, ~0.054 g PET, ~99.8% purity.
- [ ] **T28 screen.py + run_screening.py** [U] — T grid, rules, score, mass balance → ranked CSV:
  `solvent | T window | S_PS sat | loading | solvent:waste | PS purity | confidence | greenness`.
  *Done when:* sanity — known PS solvents (e.g. limonene, p-cymene class) rank high; PET-dissolving solvents (e.g. GVL at high T) are rejected.
- [ ] **T29 Dissolution time (optional)** [U] — simple diffusion/swelling estimate from χ + particle size.

### Phase 5 — Experiments for the report
- [ ] **T30 Ablations** [U run, C plotting] — −physics layer, −pretraining, −polymer descriptors, −interaction features, +σ-profiles (opt.), `*` dummy vs ring-closure.
- [ ] **T31 Interpretability** [U] — plot predicted χ(T) for PS & PET in top solvents; compare χ with HSP prior; list functional-group trends.
- [ ] **T32 Results notebook** [C] — `02_results.ipynb`: all tables & figures.

### Phase 6 — Validation & write-up
- [ ] **T33 Lab validation (if access)** [U] — top-3 solvents on PS cups + PET bottle flakes; gravimetric dissolution; PS purity (FTIR/DSC); PET loss; compare with predictions.
- [ ] **T34 Report** [U] — intro (recycling problem, STRAP, Zhou, SolNet), methods (data, model, physics, training), results (baselines, ablations, screening, validation), limitations, future work (solvent mixtures, more polymers, active learning).
- [ ] **T35 Presentation + viva prep** [U] — slides; be ready to explain χ, Flory–Huggins, Hansen, GIN, scaffold split, why physics layer.
- [ ] **T36 Clean repo** [U] — README with run commands, pinned requirements, final commit/tag.

---

## D. Progress log (update and paste into new chats)

```
Current task: T1
Done: —
Key numbers:
  - Zhou rows after cleaning: 
  - PS-selective solvents (Zhou, Th): 
  - HSP baseline (Spearman / top-10 hit): 
  - XGBoost baseline: 
  - Polysolv stage 1: 
  - Polysolv stage 2 (Kern F1/AUC): 
Decisions changed since plan: 
Open issues: 
```
