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
├── configs/default.yaml           all paths, seeds, cleaning + split settings
├── data/raw/  data/external/{polymer_props.csv, green_solvents.csv, exp_test.csv}  data/processed/
├── src/polysolv/                  package (`pip install -e .`; imports `from polysolv.data.clean import ...`)
│   ├── config.py                  load_config(), REPO_ROOT
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
Processed outputs (`python -m scripts.build_dataset`): `solvents.csv` (+ `V_cm3_mol`), `polymers.csv`, `zhou_long.csv` (+ `hsp_mask, Ra2, B_HSP, chi_prior`), `zhou_dropped.csv`, `splits.json`, `solvent_descriptors.csv`, `polymer_descriptors.csv`, `scalers.json`, `graphs.pt`, `build.log`.

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
- [x] **T10 build_dataset.py** [C] — one command runs T5→T9.
- [x] **T11 green_solvents.csv** [U] (parallel, any time before T27) — 50–200 green solvents with CAS + CHEM21/GSK category; flag which are already in Zhou.

### Phase 2 — Features & physics baseline
- [x] **T12 graph.py** [U] + tests [C] — `atom_features`, `bond_features`, `smiles_to_graph` → PyG `Data`; `*` as dummy atom; bidirectional edges; SMILES-keyed cache. Install `torch`, `torch_geometric`.
  *Done when:* PS graph 10 nodes, limonene 10 nodes, all dataset SMILES convert.
- [x] **T13 descriptors.py** [U] — z-scored solvent (δD, δP, δH, bp, V) & polymer (Tg, Tm, cryst, density + mask) vectors; save scaler.
- [x] **T14 physics.py** [U] + tests [C] — `ra2()`, `b_hsp()`, `chi_prior()`, `add_prior_columns()` (units: V cm³/mol, δ MPa^½, R = 8.314).
  *Done when:* χ_prior(PS, water) ≫ 0.5; toluene & limonene in the better half and water & methanol in the worst 20 % of PS solvents; Spearman(−χ_prior, Zhou PS RT) > 0.5. (Original absolute check χ_prior(PS, toluene) < 0.5 dropped: 0.65 with the chosen PS HSP.)
- [x] **T15 HSP baseline** `hsp_ranking.py` [U] — rank solvents for PS-vs-PET by Hansen distance (Ra², χ_prior; RED not usable: no R0 for PS); evaluate vs Zhou labels (Spearman, top-k selective hit rate). Metrics in `src/polysolv/evaluate.py`, results in `results/hsp_baseline.json`.
  *Done when:* first baseline numbers recorded in D.
- [x] **T16 XGBoost baseline** [C] — Morgan FP (2048, polymer trimer trick) + descriptors + T → log wt%; same metrics & splits. Results in `results/xgb_baseline.json`.
  *Done when:* baseline numbers recorded.

### Phase 3 — Core model
- [x] **T17 dataset.py** [U] — PyG Dataset/`collate` returning (polymer graph, solvent graph, descriptors, T, Ra², mask, label, censored).
- [x] **T18 towers.py** [U] — GIN tower (4×GINEConv, 128, BN, ReLU, dropout 0.1, sum-pool).
- [x] **T19 interaction.py + flory_huggins.py + heads.py** [U] — interaction vector, MLP → ΔA, ΔB; χ = A + B/T; regression head.
  *Done when:* unit test: hand-set A, B give expected χ and monotone outputs.
- [x] **T20 polysolv_net.py** [U] — wire everything; forward returns χ, log wt%.
  *Done when:* one batch forward/backward works, shapes correct.
- [x] **T21 losses.py** [U] — censored MSE, pairwise rank, physics_reg.
- [x] **T22 train.py + configs/default.yaml** [U core, C for logging/ckpt] — Stage 1 on Zhou.
  *Done when:* val loss decreases; overfits 100 rows when asked (sanity).
- [x] **T23 evaluate.py** [U metrics, C plots] — MAE/RMSE/R²/per-polymer Spearman on Zhou test; external experimental test set; compare with T15–T16.
  *Done when:* table: HSP vs XGB vs Polysolv-stage1 (Zhou test + external test).

### Phase 4 — Generalization, ensemble, screening
- [ ] **T24 Leave-one-polymer-out** [U] — train on 7 polymers, predict the 8th (at least PS and PET held out); report honestly.
- [x] **T25 Stage 2 (optional)** [U] — PS>PET pairwise rank loss; check top-k selectivity changes.
- [ ] **T26 ensemble.py** [U] — 5 seeds, mean ± σ; calibration plot (error vs σ).
- [x] **T27 mass_balance.py** [U] + tests [C] — `required_solvent`, `pet_leaching`, `ps_purity`.
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
Current task: T24
Done: T1–T16 (Phase 1 and Phase 2 complete)
Key numbers:
  - Zhou raw: 1007 solvents (983 unique molecules), 16 112 long rows before cleaning, 2 `M` labels, 100 % SMILES parse
  - Zhou rows after cleaning: 13 732 (981 molecules). Dropped: 1296 Th-copy-of-RT, 748 missing label, 336 duplicate molecule, 0 invalid SMILES
  - Solvent descriptors: 157 solvents without bp (RT rows only), 29 without HSP
  - Split (seed 42): train 784 / val 98 / test 99 solvents; 744 scaffold groups (benzene scaffold = 162 solvents, always in train)
  - PS-selective solvents (PS >= 10, PET <= 0.5 wt%): RT 14, Th 104 (Th >= 25 °C)
  - External test set: 16 Zhou Table 3 rows (all 8 polymers) + 25 handbook good/non pairs (PS, PET)
  - Green solvents: 57 (19 recommended, 22 problematic, 10 hazardous, 6 highly hazardous); 54 in Zhou
  - Graphs: 989 cached (981 solvents + 8 polymers); ATOM_DIM 31, BOND_DIM 6
  - Molar volume V_s: McGowan Vx -> V = 1.350·Vx − 3.98, mean error ~4 % on 17 anchor solvents
  - Hansen prior alone vs Zhou PS (RT): Spearman(−χ_prior, y) = 0.66 (Th: 0.41)
  - HSP baseline, test split (99 solvents): Spearman(−χ_prior, y) PS RT 0.68 / Th 0.39, PET RT 0.66 / Th 0.37;
    PS-vs-PET selectivity (χ_prior diff, 69 test solvents, 9 selective, base rate 13 %): top-10 hit 0.30, top-20 0.25, Spearman vs true margin 0.58.
    All solvents (735, 100 selective): top-10 0.10, top-20 0.05 (below base rate 0.14), Spearman margin 0.62
  - XGBoost baseline (seed 42, 4110 features, best iter 2172), test split: MAE 0.32, RMSE 0.51, R² 0.89 (log10 wt%); val R² 0.91.
    Spearman PS RT 0.79 / Th 0.86, PET RT 0.71 / Th 0.87. R² by polymer: PS 0.79, PET 0.85 (PE 0.96 best, PA66 0.78 worst).
    Selectivity (same 69 HSP test solvents): top-10 hit 0.40, top-20 0.35, Spearman vs true margin 0.87  (HSP: 0.30 / 0.25 / 0.58)
  - Polysolv stage 1 (seed 42, 516k params, best epoch 71 of 101, 7.5 min on RTX 2050): test MAE 0.316, RMSE 0.499, R² 0.894 (val R² ~0.93).
    Spearman PS RT 0.83 / Th 0.89, PET RT 0.73 / Th 0.89. Selectivity (69 HSP test solvents): top-10 0.50, top-20 0.40, margin ρ 0.92.
    Sanity: overfits 100 rows (train MSE 4e-6, R² 1.000). Logs: results/training_logs/ (polysolv_stage1.log/_history.csv/_curves.png)
  - Leave-one-polymer-out: 
  - Stage 2 (T25, w_rank 0.2, RankNet on PS−PET margins of 649 train solvents at Th, 64 per step; 16 min):
    test R² 0.886 (stage 1 0.894); selectivity top-10 0.50 / top-20 0.40 (unchanged), margin ρ 0.93 (0.92); external AUC 0.93 (same),
    Table 3 MAE 1.12 (1.17), ρ 0.84 (0.80). -> changes are within single-seed noise; ensemble members use stage 1 (ensemble.w_rank = 0).
  - Plain two-tower GIN (no physics, 1000/T input): test MAE 0.236, RMSE 0.425, R² 0.923; Spearman PS 0.91/0.92, PET 0.83/0.91;
    selectivity 0.50 / 0.45 / margin ρ 0.90; external AUC 0.89. -> better than Polysolv on Zhou-test regression (honest result).
  - External experimental test (41 rows; 34 of the solvents are in the train split, so this is COSMO->experiment transfer):
    AUC good-vs-non / accuracy (pred >= 1 wt%) / Table 3 MAE (13 measured, log10) / Table 3 ρ:
    Hansen 0.74 / 0.78 / 1.59 / 0.18 · XGB 0.88 / 0.80 / 1.22 / 0.80 · plain GIN 0.89 / 0.83 / 1.21 / 0.77 · Polysolv 0.93 / 0.85 / 1.17 / 0.80.
    COSMO-RS itself vs the same 13 measurements: MAE 1.27, ρ 0.78 (ceiling for COSMO-trained models).
    Hansen prior + 3-number head on Zhou test: R² 0.21. Full table: results/comparison.md
  - Mass balance (T27): worked example 100 g, 30 % PS, S_PS_lower 38, S_PET_upper 0.02, c_max 10 -> c 10 wt%, 270 g solvent (2.7 g/g), 0.054 g PET, 99.82 % purity (tests pass).
Decisions changed since plan:
  - Kern et al. dropped (data not public). Novelty 2 = external experimental test instead of multi-fidelity. No classification head, no BCE.
  - Package layout: src/polysolv (installed with `pip install -e .`), imports `from polysolv...`.
  - Polymer short names everywhere: EVOH, PE, PET, PP, PS, PVC, PA6, PA66.
  - polymer_props.csv also has R0 (needed for RED in T15) and separate thermal/HSP source columns + `verified` flag.
  - Cleaning: Th rows of solvents without bp dropped (Th = 25 °C copy of RT); sub-ambient Th rows kept (down to 110.65 K, methane).
  - Dedupe: one solvent per canonical SMILES (prefer HSP present, then file order). Applies to Hansen variants AND stereoisomers (the second isomer is dropped).
  - `M` -> censored at 50 wt% (config clean.miscible_cap_wt).
  - Split: acyclic solvents are singleton groups (otherwise 628 acyclic molecules would form one group); balanced scaffold split.
  - V_s from McGowan volume (calibrated, config descriptors.molar_volume_anchors) because Zhou has no densities.
  - Solvents without HSP kept with hsp_mask = 0 (B_HSP = 0, chi_prior = A0).
  - T14 done-check changed to a relative one: χ_prior(PS, toluene) = 0.65 > 0.5 with the PS HSP 17.6/6.1/4.1, but this set ranks Zhou PS solvents best (ρ 0.66 vs 0.60 for 18.5/4.5/2.9, 0.28 for the handbook average); the network learns ΔA/ΔB anyway.
  - Descriptor scaler fitted on train solvents only; polymer scaler on all 8 polymers (refit without the held-out polymer for leave-one-polymer-out, T24).
  - torch 2.14.1+cu130 (RTX 2050, 4 GB) + torch_geometric 2.8.1; xgboost 3.4.1 (native API, no scikit-learn).
  - Head: y_pred = log10(10**raw + 1e-3) with raw = c0 − softplus(c1)·χ + c2·x_cryst (same floor as the label).
    Rows with wt% < 1e-3 (12 %) are LEFT-censored in the loss (one-sided, like `M`), on the raw prediction -> no vanishing gradient.
    Head init: c0, c2 fitted on the Hansen prior, slope = best of a grid (0.5). A fixed steep slope (4) learned much slower (try_slope4.log).
    v0 (plain linear head, no floor): test R² 0.863 (results/v0_linear_head.json).
  - Collate batches only the unique graphs per mini-batch; GIN layers have residual connections; interaction last layer zero-init (model starts at prior + head).
  - Shared metrics live in src/polysolv/evaluate.py (Spearman, MAE/RMSE/R², selectivity top-k); T23 extends it.
  - XGB features include raw HSP descriptors but NOT chi_prior/Ra² (keeps it a non-physics baseline).
Open issues:
1. polymer_props.csv: all rows `verified = no`. Thermal values are typical handbook values (check against Polymer Handbook). HSP for PE/PET/PP/PVC/PA6/PA66 = Hansen handbook averages (accudynetest polytable_02). PS from Materials 2018, 11, 1529 (no R0). EVOH HSP is an estimate (PE/PVOH volume-fraction mix).
2. Polysolv (R² 0.894) ties XGBoost (0.888) but loses to the plain GIN (0.923) on Zhou-test regression; it wins on external AUC (0.93) and selectivity margin ρ (0.92). The physics layer has to earn its place in leave-one-polymer-out (T24) and temperature behaviour (T31); report this trade-off honestly.
3. The Hansen prior's absolute χ is too high for good PS solvents (toluene 0.65, limonene 0.68; THF 0.48). Its ranking is reasonable at RT (ρ 0.66) but weaker at Th (ρ 0.41) — the learned ΔA/ΔB must fix this; check in T31.
4. McGowan-based V_s overestimates very large molecules (triolein ~ +14 %); fine for a prior.
5. RT taken as 298.15 K (assumption, state in report).
6. exp_test.csv handbook pairs and green_solvents.csv CHEM21 categories are `verified = no` (from memory of the Polymer Handbook / CHEM21 guide). Check before the report. Eucalyptol and alpha-pinene are not in CHEM21 (category by analogy).
7. GVL is in Zhou only at 25 °C, and p-cymene / Cyrene / alpha-pinene are not in Zhou at all -> screening for them relies purely on model extrapolation.
8. `zhou_solvent_id` in green_solvents.csv depends on the build order; re-run the flag if cleaning rules change.
```
