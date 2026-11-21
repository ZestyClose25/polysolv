# Polysolv — Model Architecture

Physics-informed two-tower GNN for **selective green-solvent recommendation** (PS / PET separation).
Trained on Zhou et al. COSMO-RS data only; experimental data is used for external testing only. (Kern et al. is not used: its labelled data is not public.)

---

## 1. End-to-end pipeline

```mermaid
flowchart LR
    subgraph DATA["Data sources"]
        D1["Zhou et al. 2023<br/>COSMO-RS solubility<br/>8 polymers x 1007 solvents x T<br/>(quantitative, wt%)"]
        D2["Experimental test set<br/>Zhou Table 3 + handbook<br/>good / non-solvent pairs<br/>(external test only)"]
        D3["HSP tables<br/>(dD, dP, dH)"]
        D4["Green solvent guides<br/>CHEM21 / GSK<br/>+ boiling points"]
    end

    subgraph MODEL["Polysolv model"]
        M["Two-tower GNN<br/>+ interaction head<br/>+ Flory-Huggins layer"]
    end

    subgraph SCREEN["Selectivity screening"]
        S1["Score every candidate<br/>green solvent s and T"]
        S2["Selectivity =<br/>Sol(PS) - Sol(PET)"]
        S3["Constraints:<br/>T &lt; bp(s), greenness pass,<br/>uncertainty below threshold"]
    end

    OUT["Ranked output:<br/>solvent, T window,<br/>max PS loading wt%,<br/>selectivity margin, confidence"]

    D1 -->|train| M
    D2 -.->|evaluate only| OUT
    D3 -->|features| M
    M --> S1 --> S2 --> S3 --> OUT
    D4 --> S3
```

---

## 2. Model internals

```mermaid
flowchart TB
    subgraph PIN["Polymer input"]
        P1["Repeat-unit graph<br/>with * end atoms<br/>(e.g. PS, PET)"]
        P2["Polymer descriptors<br/>Tg, Tm, crystallinity, density<br/>(no Mw: high-MW limit)"]
    end

    subgraph SIN["Solvent input"]
        S1["Solvent molecular graph<br/>(RDKit from SMILES)"]
        S2["Solvent descriptors<br/>HSP, bp, molar volume"]
    end

    PT["Polymer tower<br/>GIN x 4 layers (GINEConv, 128)<br/>+ sum readout"]
    ST["Solvent tower<br/>GIN x 4 layers (separate weights)<br/>+ sum readout"]

    P1 --> PT
    S1 --> ST

    HP["h_p (polymer embedding)<br/>concat descriptors"]
    HS["h_s (solvent embedding)<br/>concat descriptors"]

    PT --> HP
    P2 --> HP
    ST --> HS
    S2 --> HS

    INT["Interaction features<br/>[h_p, h_s, h_p*h_s, |h_p - h_s|, Ra2, mask]"]
    HP --> INT
    HS --> INT

    MLP["Interaction MLP"]
    INT --> MLP

    FH["Physics layer: Flory-Huggins<br/>A = 0.34 + dA, B = B_HSP*mask + dB<br/>chi(T) = A + B / T"]
    MLP -->|"predicts dA, dB"| FH
    T["Temperature T"] --> FH

    O1["Regression head<br/>log10 wt% = c0 - softplus(c1)*chi + c2*cryst<br/>(trained on COSMO-RS data)"]
    UQ["Uncertainty<br/>5-seed deep ensemble"]

    FH --> O1
    O1 --> UQ
```

---

## 3. Training and evaluation

```mermaid
flowchart LR
    A["Stage 1: Train<br/>Zhou COSMO-RS data<br/>loss = censored MSE on log wt%"] --> C["Stage 2: Selectivity tuning (optional)<br/>pairwise ranking loss:<br/>Sol(PS, s, T) &gt; Sol(PET, s, T)"]
    C --> D["Evaluate<br/>solvent-scaffold split (unseen solvents)<br/>leave-one-polymer-out<br/>external experimental test set"]
```

**Total loss:**
`L = λ1·censored MSE(log wt%) + λ2·RankLoss(selectivity, optional) + λ3·physics_reg`

`physics_reg` keeps χ(T) monotonic/smooth and penalises non-physical A, B values.

---

## 4. Component summary

| Block | Input | Output | Why |
|---|---|---|---|
| Polymer tower | Repeat-unit graph + Tg/Tm/crystallinity/density | `h_p` | Graph alone misses PET crystallinity |
| Solvent tower | Solvent graph + HSP/bp | `h_s` | ~1000 solvents justifies a GNN; generalises to unseen green solvents |
| Interaction head | `[h_p, h_s, h_p*h_s, \|h_p-h_s\|, Ra², mask]` | ΔA, ΔB | Captures polymer–solvent interactions (dot product is too weak) |
| Flory–Huggins layer | A, B, T | χ(T) → solubility | Interpretable, extrapolates in temperature |
| Regression head | χ, crystallinity | log10 wt% | Solubility falls monotonically with χ |
| Uncertainty | Ensemble | σ per prediction | Flags unreliable recommendations |
| Screening layer | Predictions + green guide + bp | Ranked solvent list | Turns solubility into a **selectivity** decision |

---

## 5. Baselines and ablations

- **Baselines:** HSP RED ranking · XGBoost on Morgan fingerprints · plain two-tower (no physics layer)
- **Ablations:** − physics layer · − polymer descriptors · − interaction features · +σ-profiles (opt.)
- **Generalization:** leave-one-polymer-out (8 polymers is too few to claim polymer generalization)
- **External test:** experimental data (Zhou Table 3, handbook good/non-solvent pairs), never trained on
- **Validation:** gravimetric dissolution test of top-3 predicted solvents on real PS + PET waste (if lab access)

---

## 6. Optional: dissolution time

Not learned (little data). Can be estimated post-hoc with a simple swelling/diffusion model using predicted χ and particle size.
