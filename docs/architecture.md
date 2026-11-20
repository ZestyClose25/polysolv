# Polysolv — Model Architecture

Physics-informed, multi-fidelity two-tower GNN for **selective green-solvent recommendation** (PS / PET separation).

---

## 1. End-to-end pipeline

```mermaid
flowchart LR
    subgraph DATA["Data sources"]
        D1["Zhou et al. 2023<br/>COSMO-RS solubility<br/>8 polymers x 1007 solvents x T<br/>(quantitative, wt%)"]
        D2["Ramprasad / SolNet<br/>experimental solvent vs non-solvent<br/>(binary labels)"]
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

    D1 -->|pretrain| M
    D2 -->|fine-tune| M
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
        P2["Polymer descriptors<br/>Tg, Tm, crystallinity, Mw"]
    end

    subgraph SIN["Solvent input"]
        S1["Solvent molecular graph<br/>(RDKit from SMILES)"]
        S2["Solvent descriptors<br/>HSP, bp, molar volume"]
    end

    PT["Polymer tower<br/>GIN x L layers<br/>+ sum/attention readout"]
    ST["Solvent tower<br/>GIN or D-MPNN x L layers<br/>+ readout"]

    P1 --> PT
    S1 --> ST

    HP["h_p (polymer embedding)<br/>concat descriptors"]
    HS["h_s (solvent embedding)<br/>concat descriptors"]

    PT --> HP
    P2 --> HP
    ST --> HS
    S2 --> HS

    INT["Interaction features<br/>[h_p, h_s, h_p*h_s, |h_p - h_s|, Ra_HSP]"]
    HP --> INT
    HS --> INT

    MLP["Interaction MLP"]
    INT --> MLP

    FH["Physics layer: Flory-Huggins<br/>chi(T) = A + B / T"]
    MLP -->|"predicts A, B"| FH
    T["Temperature T"] --> FH

    O1["Head 1 - regression<br/>solubility wt% at T<br/>(trained on COSMO-RS data)"]
    O2["Head 2 - classification<br/>good / non-solvent<br/>(trained on experimental data)"]
    UQ["Uncertainty<br/>deep ensemble / MC dropout"]

    FH --> O1
    FH --> O2
    O1 --> UQ
    O2 --> UQ
```

---

## 3. Training strategy (multi-fidelity)

```mermaid
flowchart LR
    A["Stage 1: Pretrain<br/>COSMO-RS data<br/>loss = MSE on solubility(T)"] --> B["Stage 2: Fine-tune<br/>experimental labels<br/>loss = BCE (+ small MSE replay)"]
    B --> C["Stage 3: Selectivity tuning (optional)<br/>pairwise ranking loss:<br/>Sol(PS, s, T) &gt; Sol(PET, s, T)"]
    C --> D["Evaluate<br/>solvent-scaffold split<br/>(unseen solvents)"]
```

**Total loss:**
`L = λ1·MSE(solubility) + λ2·BCE(good/non-solvent) + λ3·RankLoss(selectivity) + λ4·physics_reg`

`physics_reg` keeps χ(T) monotonic/smooth and penalises non-physical A, B values.

---

## 4. Component summary

| Block | Input | Output | Why |
|---|---|---|---|
| Polymer tower | Repeat-unit graph + Tg/Tm/crystallinity | `h_p` | Graph alone misses PET crystallinity |
| Solvent tower | Solvent graph + HSP/bp | `h_s` | ~1000 solvents justifies a GNN; generalises to unseen green solvents |
| Interaction head | `[h_p, h_s, h_p*h_s, \|h_p-h_s\|, Ra]` | A, B | Captures polymer–solvent interactions (dot product is too weak) |
| Flory–Huggins layer | A, B, T | χ(T) → solubility | Interpretable, extrapolates in temperature |
| Dual heads | χ | wt% + good/non-solvent | Uses both simulated and experimental data |
| Uncertainty | Ensemble | σ per prediction | Flags unreliable recommendations |
| Screening layer | Predictions + green guide + bp | Ranked solvent list | Turns solubility into a **selectivity** decision |

---

## 5. Baselines and ablations

- **Baselines:** HSP RED ranking · XGBoost on Morgan fingerprints · plain two-tower (no physics layer)
- **Ablations:** − physics layer · − pretraining · − polymer descriptors · − interaction features
- **Validation:** gravimetric dissolution test of top-3 predicted solvents on real PS + PET waste (if lab access)

---

## 6. Optional: dissolution time

Not learned (little data). Can be estimated post-hoc with a simple swelling/diffusion model using predicted χ and particle size.
