# Leave-one-polymer-out (T24)

Held-out polymer never seen in training (graph, descriptors or labels). Metrics on all its Zhou rows
(log10 wt%). 'In-dist. R²' = stage-1 model on the test split, polymer seen in training.

Plain GIN = same protocol without the physics layer (no Hansen prior, no chi).

| Held out | R² | MAE | ρ RT | ρ Th | Plain GIN R² | Plain GIN ρ RT | Plain GIN ρ Th | Hansen prior ρ RT | Hansen prior ρ Th | In-dist. R² |
|---|---|---|---|---|---|---|---|---|---|---|
| PS | 0.01 | 1.05 | 0.64 | 0.78 | -1.01 | 0.65 | 0.74 | 0.66 | 0.41 | 0.86 |
| PET | 0.12 | 0.95 | 0.34 | 0.59 | -0.94 | 0.62 | 0.68 | 0.58 | 0.47 | 0.83 |
| PE | 0.40 | 0.93 | 0.89 | 0.85 | – | – | – | 0.62 | 0.01 | 0.93 |
| PP | -0.29 | 1.23 | 0.69 | 0.72 | – | – | – | 0.60 | 0.29 | 0.85 |
| PVC | 0.40 | 0.62 | 0.51 | 0.66 | – | – | – | 0.68 | 0.40 | 0.88 |
| EVOH | 0.16 | 1.16 | 0.49 | 0.71 | – | – | – | 0.48 | 0.32 | 0.93 |
| PA6 | 0.43 | 0.81 | 0.40 | 0.65 | – | – | – | 0.50 | 0.26 | 0.80 |
| PA66 | 0.74 | 0.45 | 0.67 | 0.86 | – | – | – | 0.53 | 0.44 | 0.82 |
