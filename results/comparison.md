# Model comparison (T23)

Zhou test split = 99 scaffold-split solvents never seen in training (log10 wt%; MAE/RMSE/R² on
uncensored rows). ρ = Spearman per polymer/temperature. Selectivity = PS-over-PET at Th on the 69
test solvents with Hansen parameters (9 selective, base rate 0.13): top-k hit rate and Spearman of
the predicted vs true margin. External = 41 experimental rows (16 Zhou Table 3 + 25 handbook
good/non pairs): AUC good-vs-non, accuracy of 'predicted ≥ 1 wt%', MAE/ρ on the 13 measured Table 3
values (log10 wt%). External solvents are Zhou solvents (34 of 41 in the train split): this tests
COSMO-RS → experiment transfer, not new solvents. Hansen-prior regression uses 3 fitted numbers.
Reference: COSMO-RS itself (Zhou's predictions, the training labels) vs the same 13 measured Table 3
values: MAE 1.27, ρ 0.78 — the ceiling a model trained on COSMO-RS labels can be expected to reach.

| Model | MAE | RMSE | R² | ρ PS RT | ρ PS Th | ρ PET RT | ρ PET Th | sel top-10 | sel top-20 | ρ margin | ext AUC | ext acc | Table 3 MAE | Table 3 ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hansen prior (+3-number head) | 1.176 | 1.364 | 0.205 | 0.68 | 0.39 | 0.66 | 0.37 | 0.30 | 0.25 | 0.58 | 0.74 | 0.78 | 1.59 | 0.18 |
| XGBoost (Morgan FP + descriptors) | 0.318 | 0.512 | 0.888 | 0.79 | 0.85 | 0.71 | 0.87 | 0.40 | 0.35 | 0.87 | 0.88 | 0.80 | 1.22 | 0.80 |
| Plain two-tower GIN (no physics) | 0.236 | 0.425 | 0.923 | 0.91 | 0.92 | 0.83 | 0.91 | 0.50 | 0.45 | 0.90 | 0.89 | 0.83 | 1.21 | 0.77 |
| Polysolv stage 1 | 0.316 | 0.499 | 0.894 | 0.83 | 0.89 | 0.73 | 0.89 | 0.50 | 0.40 | 0.92 | 0.93 | 0.85 | 1.17 | 0.80 |
| Polysolv stage 2 (+ rank loss) | 0.327 | 0.516 | 0.886 | 0.84 | 0.88 | 0.70 | 0.87 | 0.50 | 0.40 | 0.93 | 0.93 | 0.85 | 1.12 | 0.84 |
| Polysolv 5-seed ensemble | 0.300 | 0.476 | 0.904 | 0.86 | 0.89 | 0.77 | 0.90 | 0.50 | 0.40 | 0.94 | 0.91 | 0.83 | 1.18 | 0.79 |
