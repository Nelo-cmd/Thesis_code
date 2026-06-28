"""
Phase 7 — sensitivity of the physics-value result to the assumed wellbore config.

The negative result is conditional on one assumed geometry (wellbore.REF). Here we
vary the key configuration parameters one-at-a-time, recompute the SI and kinetic
features, and re-test the marginal AUC of +SI and +kinetics WITHIN FIELD B (the
confound-free regime). If physics stays ~worthless (and the small SI whisper does
not grow into real signal) across every plausible geometry, the conclusion is robust.

Field B only (balanced, no field confound) keeps this both fast and clean.
Run:  python ml/sensitivity.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physics import wellbore as WB

CHEM = ["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm", "TDS_ppm", "pH",
        "Cl_ppm", "ionic_strength", "Ca_HCO3_ratio", "Mg_Ca_ratio", "SO4_Ca_ratio", "Ca_TDS_norm"]
SI = CHEM + ["SI_max_cal", "SI_mean_cal"]
KIN = SI + ["krate_max", "krate_mean", "krate_integral"]

# one-at-a-time variations around REF (label, overrides)
VARIATIONS = [
    ("REF (baseline)", {}),
    ("BHT 70 C", {"BHT_C": 70.0}),
    ("BHT 100 C", {"BHT_C": 100.0}),
    ("BHT 120 C", {"BHT_C": 120.0}),
    ("WHT 25 C", {"WHT_C": 25.0}),
    ("TVD 6000 ft", {"TVD_ft": 6000.0}),
    ("TVD 12000 ft", {"TVD_ft": 12000.0, "BHP_psi": 6000.0}),
    ("no degassing (a=0)", {"a_degas": 0.0}),
    ("strong degassing (a=0.8)", {"a_degas": 0.8}),
    ("low BHP 3000 psi", {"BHP_psi": 3000.0}),
]


def folds(frame, y, groups, n_rep=4):
    f = []
    for r in range(n_rep):
        f += list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                       random_state=100 + r).split(frame, y, groups))
    return [(tr, te) for tr, te in f if len(np.unique(y[te])) == 2]


def auc_set(frame, feats, y, fl):
    X = frame[feats].values; a = []
    for tr, te in fl:
        spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="logloss", random_state=0,
                          scale_pos_weight=spw)
        m.fit(X[tr], y[tr]); a.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(a)


def main():
    df = pd.read_parquet(os.path.join(ROOT, "outputs", "prepared.parquet"))
    dfB = df[df.Field == "B"].reset_index(drop=True)
    yB = dfB["Inspection_Result"].values.astype(int)
    gB = dfB["Well_Number"].values
    fl = folds(dfB, yB, gB)

    print("=" * 82)
    print("SENSITIVITY of physics value to assumed wellbore config (XGBoost, within Field B)")
    print(f"  n={len(dfB)}, folds={len(fl)}; reports AUC and marginal dAUC vs the chemistry baseline")
    print("=" * 82)
    print(f"{'config':26}{'SI_max(med)':>12}{'chem':>8}{'+SI':>8}{'dSI':>8}{'+kin':>9}{'dkin':>8}")

    for label, override in VARIATIONS:
        cfg = {**WB.REF, **override}
        sim = WB.simulate_dataframe(dfB, cfg)
        a_c = auc_set(sim, CHEM, yB, fl)
        a_s = auc_set(sim, SI, yB, fl)
        a_k = auc_set(sim, KIN, yB, fl)
        print(f"{label:26}{sim['SI_max_cal'].median():>12.2f}"
              f"{a_c.mean():>8.3f}{a_s.mean():>8.3f}{a_s.mean()-a_c.mean():>+8.3f}"
              f"{a_k.mean():>9.3f}{a_k.mean()-a_s.mean():>+8.3f}")

    print("\n  dSI  = AUC(+SI) - AUC(chemistry)      [marginal value of thermodynamics]")
    print("  dkin = AUC(+kinetics) - AUC(+SI)       [marginal value of kinetics over SI]")
    print("  Robust negative result <=> dSI and dkin stay ~0 across all configs.")


if __name__ == "__main__":
    main()
