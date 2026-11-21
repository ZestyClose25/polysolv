# Polysolv — Master Build Plan

> **For a new chat:** read section A (context) first, then continue from the first unchecked task in section C.
> Say "continue Polysolv from T__" and paste the progress log (section D).

---

## A. Project context (decisions already made)

**Goal.** ML model that recommends a **green solvent** to selectively dissolve **PS** from a **PS/PET** mixture (PET stays solid). Output per solvent: temperature window, saturation solubility, recommended loading / solvent-to-waste ratio, predicted PS purity, confidence. Dissolution time = optional, physics estimate only.

**Novelty.**
1. Physics-informed output: network predicts Flory–Huggins χ(T) = A + B/T (as a correction to the Hansen prior), not raw solubility.
2. Trained on simulated data (Zhou COSMO-RS), **tested externally against experimental data** (Zhou Table 3 + literature good/non-solvent pairs).
3. Selectivity-aware screening with uncertainty (deep ensemble).
4. Optional lab validation on real PS/PET waste.

**Data sources.**
| ID | Source | Content | Use |
|---|---|---|---|
| A | Zhou et al., *Green Chem.* 2023, DOI 10.1039/D3GC00404J (ESI spreadsheet) | 8 polymers (EVOH, PE, PP, PS, PET, PVC, nylon 6, nylon 66) × 1007 solvents × 2 temps (RT, Th = min(bp−1, 120) °C); wt%; solvent name, CAS, SMILES, bp, δD/δP/δH, class; `M` = miscible, `NaN` = bp < RT; COSMOtherm-generated | Training labels, solvent features |
| B | Kern et al., *PCCP* 2022, DOI 10.1039/D2CP03735A | **NOT USED** — labelled pairs are not public (ESI = figure captions, GitHub CSV = UMAP coordinates). May return as optional classification stage if authors share data. | — |
| C | Hand-built `polymer_props.csv` (Polymer Handbook, Hansen handbook, papers) | Tg, Tm, crystallinity, density, HSP, repeat-unit SMILES for the 8 polymers | Polymer descriptors, physics prior |
| D | Hand-built `green_solvents.csv` (CHEM21, GSK guides, bio-based solvent reviews) | name, CAS, greenness category | Screening filter only |
| E | Hand-built `exp_test.csv` (Zhou Table 3; Polymer Handbook / Hansen handbook good/non-solvent pairs for PS and PET) | experimental solubility or good/non-solvent label | **External test only, never trained on** |
| F (opt.) | CHAOS DB / OpenSPGen σ-profiles | 51-pt σ-profile per solvent | Ablation feature |

No AMS/COSMOtherm license needed — we only use Zhou's published numbers.

**Architecture.**
```
polymer SMILES → RDKit graph → GIN tower (4×GINEConv, 128, sum-pool) ⊕ polymer descriptors → MLP → h_p
solvent SMILES → RDKit graph → GIN tower (separate weights)           ⊕ solvent descriptors → MLP → h_s
x = [h_p, h_s, h_p⊙h_s, |h_p−h_s|, Ra², mask] → MLP → (ΔA, ΔB)
A = 0.34 + ΔA ;  B = B_HSP·mask + ΔB ;  B_HSP = V_s·Ra²/(4R) ;  Ra² = 4ΔδD² + ΔδP² + ΔδH²
χ(T) = A + B/T
Head: log10 wt% = c0 − softplus(c1)·χ + c2·crystallinity_term
Uncertainty: 5-seed deep ensemble (mean ± σ)
```

**Loss.** `L = λ1·MSE(log wt%, censored for M) + λ2·PairwiseRank(PS>PET, optional) + λ3·physics_reg`
**Training.** Stage 1 train on Zhou → Stage 2 (opt.) add selectivity rank loss.
**Splits.** Zhou: Bemis–Murcko solvent-scaffold split 80/10/10 (grouped by canonical SMILES). Leave-one-polymer-out experiment. External experimental test set.
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

**Known limitations.** Zhou labels are simulated (PET-in-DMF was a known miss); only 2 temps per pair (only 1 for the 162 solvents without bp); only 8 polymers; high-MW assumption; real-waste crystallinity variance; additives not modelled.

---

## B. Repository layout

```
polysolv/
├── CLAUDE.md  README.md  pyproject.toml  .gitignore
├── configs/default.yaml
├── data/raw/  data/external/{polymer_props.csv, green_solvents.csv, exp_test.csv}  data/processed/
├── src/polysolv/
│   ├── data/{load_zhou.py, clean.py, merge.py, splits.py}
│   ├── features/{graph.py, descriptors.py, physics.py}
│   ├── dataset.py
│   ├── model/{towers.py, interaction.py, flory_huggins.py, heads.py, polysolv_net.py}
│   ├── losses.py  train.py  ensemble.py  evaluate.py  mass_balance.py  screen.py
├── baselines/{hsp_ranking.py, xgb_baseline.py}
├── scripts/{build_dataset.py, run_training.py, run_screening.py}
├── notebooks/{01_explore_zhou.ipynb, 02_results.ipynb}
└── tests/
```
⚠️ Layout conflict to resolve before T5: CLAUDE.md says `src/` is the package root (`from src.data ...`), but the repo and this plan use `src/polysolv/`.

Legend below: **[U]** = user writes, **[C]** = fine to let Claude generate.

---

## C. Task list (do in order; each has a "done when")

### Phase 1 — Setup & data
- [x] **T1 Setup** [U] — repo, venv, `pip install rdkit pandas numpy openpyxl jupyter`, folders, `CLAUDE.md`, `.gitignore`, first commit.
  *Done when:* `Chem.MolFromSmiles('*CC(*)c1ccccc1').GetNumAtoms() == 10`.
- [x] **T2 Download raw data** [U] — Zhou ESI spreadsheet + PDF → `data/raw/`. Note sources in README. (Kern files were downloaded too but are not used.)
  *Done when:* all files present, untouched.
- [x] **T3 Explore Zhou** [U] — `01_explore_zhou.ipynb`: sheets, columns, wide/long, solvent count, `M`/`NaN` counts per polymer, PS & PET histograms (RT, Th), count of PS-selective solvents, SMILES parse rate.
  *Done when:* you can describe the file in 3 sentences + have the PS-selective count.
- [x] **T4 polymer_props.csv** [U] — 8 rows: polymer, repeat_smiles, Tg_K, Tm_K, crystallinity_pct, density, dD, dP, dH, R0, sources.
  *Done when:* all SMILES parse and look right when drawn; every value has a source. (Values still flagged `verified = no` — see open issues.)
- [x] **T5 load_zhou.py** [U] + tests [C] — wide → long; RT & Th in K per solvent; `is_miscible` flag.
  *Done when:* tests pass (row count, PS/PET present, wt% ≥ 0, sane T).
- [x] **T6 exp_test.csv** [U] — external experimental test set: Zhou Table 3 (read from the PDF) + Polymer Handbook / Hansen handbook good/non-solvent pairs for PS and PET. Columns: polymer, solvent, CAS, SMILES, T_K, wt% (if measured), label (good/non), source.
  *Done when:* every row has a source; no row is a Zhou Table 1 calibration point.
- [x] **T7 clean.py** [U] — drop NaN per row; drop duplicated Th rows of solvents without bp; `M` → censored at cap; `log10(wt% + 1e-3)`; canonical SMILES; drop invalid; dedupe (same `cosmo_name` → keep one; decide stereoisomers).
  *Done when:* clean tables saved, dropped rows logged.
- [x] **T8 merge.py** [U] — attach solvent & polymer descriptors; write `solvents.csv`, `polymers.csv`, `zhou_long.csv`.
  *Done when:* no unexplained missing descriptors for Zhou rows.
- [x] **T9 splits.py** [U] — scaffold split (Zhou), grouped by canonical SMILES → `splits.json`; fixed seed.
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
- [ ] **T17 dataset.py** [U] — PyG Dataset/`collate` returning (polymer graph, solvent graph, descriptors, T, Ra², mask, label, censored).
- [ ] **T18 towers.py** [U] — GIN tower (4×GINEConv, 128, BN, ReLU, dropout 0.1, sum-pool).
- [ ] **T19 interaction.py + flory_huggins.py + heads.py** [U] — interaction vector, MLP → ΔA, ΔB; χ = A + B/T; regression head.
  *Done when:* unit test: hand-set A, B give expected χ and monotone outputs.
- [ ] **T20 polysolv_net.py** [U] — wire everything; forward returns χ, log wt%.
  *Done when:* one batch forward/backward works, shapes correct.
- [ ] **T21 losses.py** [U] — censored MSE, pairwise rank, physics_reg.
- [ ] **T22 train.py + configs/default.yaml** [U core, C for logging/ckpt] — Stage 1 on Zhou.
  *Done when:* val loss decreases; overfits 100 rows when asked (sanity).
- [ ] **T23 evaluate.py** [U metrics, C plots] — MAE/RMSE/R²/per-polymer Spearman on Zhou test; external experimental test set; compare with T15–T16.
  *Done when:* table: HSP vs XGB vs Polysolv-stage1 (Zhou test + external test).

### Phase 4 — Generalization, ensemble, screening
- [ ] **T24 Leave-one-polymer-out** [U] — train on 7 polymers, predict the 8th (at least PS and PET held out); report honestly.
- [ ] **T25 Stage 2 (optional)** [U] — PS>PET pairwise rank loss; check top-k selectivity changes.
- [ ] **T26 ensemble.py** [U] — 5 seeds, mean ± σ; calibration plot (error vs σ).
- [ ] **T27 mass_balance.py** [U] + tests [C] — `required_solvent`, `pet_leaching`, `ps_purity`.
  *Done when:* worked example (100 g, 30% PS, S_PS_lower 38, S_PET_upper 0.02, c_max 10) → 270 g solvent, ~0.054 g PET, ~99.8% purity.
- [ ] **T28 screen.py + run_screening.py** [U] — T grid, rules, score, mass balance → ranked CSV:
  `solvent | T window | S_PS sat | loading | solvent:waste | PS purity | confidence | greenness`.
  *Done when:* sanity — known PS solvents (e.g. limonene/dipentene) rank high; PET-dissolving solvents (e.g. DMF at high T) are rejected. (p-cymene is not in Zhou; GVL has no Th in Zhou — check them as model predictions only.)
- [ ] **T29 Dissolution time (optional)** [U] — simple diffusion/swelling estimate from χ + particle size.

### Phase 5 — Experiments for the report
- [ ] **T30 Ablations** [U run, C plotting] — −physics layer, −polymer descriptors, −interaction features, +σ-profiles (opt.), `*` dummy vs ring-closure.
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
Current task: T5
Done: T1, T2, T3, T4
Key numbers:
  - Zhou raw: 1007 solvents (983 unique molecules), 16 112 long rows before cleaning, 2 `M` labels, 100 % SMILES parse
  - Zhou rows after cleaning: 
  - PS-selective solvents (PS >= 10, PET <= 0.5 wt%): RT 14, Th 104 (Th >= 25 °C)
  - HSP baseline (Spearman / top-10 hit): 
  - XGBoost baseline: 
  - Polysolv stage 1: 
  - Leave-one-polymer-out: 
  - External experimental test: 
Decisions changed since plan:
  - Kern et al. dropped (data not public). Novelty 2 = external experimental test instead of multi-fidelity. No classification head, no BCE.
  - Polymer short names everywhere: EVOH, PE, PET, PP, PS, PVC, PA6, PA66.
  - polymer_props.csv also has R0 (needed for RED in T15) and separate thermal/HSP source columns + `verified` flag.
Open issues:
1. 162 solvents have no bp, and their "Th" is just 25 °C, an exact copy of the RT values. They have only one real temperature. GVL is one of them, so the "GVL rejected at high T" check in T28 can't be tested against Zhou. In T7, drop those duplicate Th rows.
2. There are duplicate molecules of two kinds:
   - Hansen-handbook variants of the same compound. They have identical labels, so keep one.
   - cis/trans isomers. The SMILES carry no stereo, so the graphs are identical, but the labels can differ a lot (crotonaldehyde 38 vs 3.4 wt%).
     For T7 you need to decide how to handle the isomers. For T9, all rows of one molecule must go into the same split.
3. External test set: use only Table 3 of the paper (14 measured values). The Table 1 values were used to calibrate COSMO-RS, so they aren't independent. Read Table 3 in the PDF itself, because text extraction garbles it.
4. polymer_props.csv: all rows `verified = no`. Thermal values are typical handbook values (check against Polymer Handbook). HSP for PE/PET/PP/PVC/PA6/PA66 = Hansen handbook averages (accudynetest polytable_02). PS from Materials 2018, 11, 1529 (no R0). EVOH HSP is an estimate (PE/PVOH volume-fraction mix).
5. With the current PS HSP, χ_prior(PS, toluene, 298 K) = 0.63, so the T14 check (< 0.5) fails. Decide in T14: different PS HSP set, different A0, or relax the check.
6. B_HSP needs solvent molar volume V_s, which Zhou does not provide (no density). Source it for T13/T14 (MW/density, Hansen handbook, or RDKit estimate).
7. 29 Zhou solvents have no HSP → no physics prior. Decide: drop, or mask = 0.
8. RT taken as 298.15 K (assumption, state in report).
9. Package layout conflict: CLAUDE.md `src/` vs repo `src/polysolv/`. Resolve before T5.
```
