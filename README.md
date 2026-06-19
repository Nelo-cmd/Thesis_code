# Autonomous Carbonate-Scale Prediction & Remediation System
### Thesis codebase — physics + machine learning hybrid

This repository implements the downhole carbonate (CaCO3) scaling prediction
system described in the methodology. It pairs a first-principles wellbore
geochemistry simulator with supervised machine-learning classifiers, and was
built to test — honestly — whether thermodynamic saturation indices add
predictive value over raw water chemistry for field-inspection scale labels.

-------------------------------------------------------------------------------
## Directory layout

    thesis/
    ├── data/
    │   └── prep.py            Phase 1: load, validity checks, charge-balance
    │                          Cl- inference, molality + ionic strength,
    │                          engineered ionic ratios, naive + grouped splits
    ├── phys/                  Phase 2: the physics core
    │   ├── constants.py       Plummer-Busenberg K1, K2, Ksp (calcite, aragonite)
    │   ├── pitzer.py          Pitzer ion-interaction activity model (HMW84)
    │   ├── pressure.py        Molar-volume pressure correction for Ksp
    │   └── wellbore.py        Depth-resolved P/T/pH + SI profile simulator
    ├── ml/                    Phase 3: classifiers + explainability
    │   ├── train.py           3 feature sets x 3 models x 4 eval regimes
    │   └── shap_analysis.py   SHAP global feature importance
    ├── tests/
    │   └── validate_pitzer.py Pitzer validation vs published gamma_pm values
    └── outputs/               generated artifacts (parquet/npz/npy)

-------------------------------------------------------------------------------
## How to run (order matters — each step writes artifacts the next one reads)

    pip install pandas openpyxl pyarrow numpy scikit-learn xgboost shap scipy

    # Phase 1 — prepare data, build splits  -> outputs/prepared.parquet, splits.npz
    python3 data/prep.py

    # Phase 2 — sanity-check the physics
    python3 phys/constants.py          # equilibrium constants vs literature
    python3 tests/validate_pitzer.py   # activity coefficients vs literature
    python3 phys/pressure.py           # pressure-correction magnitude
    python3 phys/wellbore.py           # single-sample SI profile demo

    # run the full physics simulation over all 505 samples
    # (writes outputs/with_physics.parquet) — see snippet in NOTES below
    python3 -c "import sys; sys.path.insert(0,'.'); import pandas as pd; \
        from phys.wellbore import simulate_dataframe; \
        simulate_dataframe(pd.read_parquet('outputs/prepared.parquet'))\
        .to_parquet('outputs/with_physics.parquet')"

    # Phase 3 — train + evaluate, then explain
    python3 ml/train.py                # -> outputs/ml_results.parquet
    python3 ml/shap_analysis.py        # -> outputs/shap_rank.parquet

-------------------------------------------------------------------------------
## Key results so far (Phases 1-3 complete)

Physics validation
  * K1, K2, Ksp match Plummer-Busenberg (1982) to 4 decimal places.
  * Pitzer gamma_pm within 0.3-1% of published NaCl / CaCl2 / MgCl2 values
    across the full ionic-strength range (to 5-6 mol/kg).

The physics-value experiment (Random Forest F1)
                chemistry   +ratios   +physics
    naive          0.782      0.836     0.809
    grouped        0.809      0.848     0.837
  -> Engineered ionic ratios help (+0.04 F1). Thermodynamic SI features do NOT
     (5.5% of total SHAP importance). This is the thesis's central empirical
     finding: equilibrium saturation indices are necessary but not sufficient
     predictors of field scaling.

Generalisation
  * Well-grouped split ~ naive split  -> no material leakage.
  * Cross-field (train A test B, and vice-versa): AUC collapses to 0.35-0.66.
    The models learn field-specific brine signatures, not a transferable
    scaling law. Most published field-scale ML papers never run this test.

-------------------------------------------------------------------------------
## NOTES / design decisions (for the write-up)

  * Cl- is not measured; inferred by charge balance. Pre-inference charge
    imbalance ~96% confirms Cl- was the missing dominant anion (expected brine).
  * One reference well configuration (TVD/BHP/BHT/gradients) is applied to all
    samples because the dataset has no per-well downhole conditions. This makes
    SI a deterministic transform of measured chemistry, which is WHY it adds no
    orthogonal ML signal. Documented as a limitation, addressed by sensitivity
    analysis (Phase 5, pending).
  * class imbalance (~70% scale) handled with class_weight / scale_pos_weight;
    F1 and AUC reported, not just accuracy.

-------------------------------------------------------------------------------
## Status

  [x] Phase 1  data prep + splits
  [x] Phase 2  physics simulator (validated)
  [x] Phase 3  ML classifiers + SHAP
  [ ] Phase 4  alarm logic + integrated decision system
  [ ] Phase 5  sensitivity analysis on assumed gradients
  [ ] Phase 6  dynamic validation scenarios
  [ ] Phase 7  figures + Chapter 4/5 write-up + front matter
