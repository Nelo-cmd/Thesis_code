"""
Phase 6 — the physics-value experiment, done rigorously and confound-aware.

Central question: does adding thermodynamic SI, then mechanistic KINETICS, predict
real field scaling better than brine chemistry alone?

Verified design (after an independent adversarial review):
  * Nested feature sets chemistry -> +SI -> +kinetics; deltas are marginal & paired.
  * Dead/degenerate features dropped (onset_depth_ft, supersat_fraction,
    highrisk_zone_ft, krate_peak_depth_ft, and SI_max_arag which duplicates SI_max_cal).
  * Primary metric AUC (threshold-free) + average precision; F1/accuracy are
    misleading (70% positive -> trivial all-positive F1 ~ 0.83).
  * THREE regimes:
      - pooled (repeated stratified)          <- CONFOUNDED by field identity
      - grouped by well (leakage-free)        <- still field-confounded
      - within Field B, grouped               <- DECISIVE: balanced, no field confound
  * Field-confound diagnostic: pooled chemistry vs chemistry+Field shows how much of
    the pooled signal is just "guess the field" (Simpson's paradox; Ca_TDS_norm flips
    sign between fields). So absolute pooled AUC and cross-field claims are reported
    as confounded, not at face value.
  * Per-fold class-imbalance handling; paired deltas across identical folds.

CAVEAT: all SI/kinetic features use one assumed wellbore configuration (wellbore.REF),
so they are deterministic transforms of surface chemistry. The result is conditional
on that; per-well measured P/T would be needed to fully test physics value.

Run:  python ml/experiment.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- feature sets (nested, degenerate features pruned) ----------------------
CHEM = ["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm", "TDS_ppm", "pH",
        "Cl_ppm", "ionic_strength",
        "Ca_HCO3_ratio", "Mg_Ca_ratio", "SO4_Ca_ratio", "Ca_TDS_norm"]
SI = CHEM + ["SI_max_cal", "SI_mean_cal"]
KIN = SI + ["krate_max", "krate_mean", "krate_integral"]
SETS = {"chemistry": CHEM, "+SI": SI, "+kinetics": KIN}
MODELS = ["LogReg", "RandomForest", "XGBoost"]


def make_model(name, y_tr):
    if name == "LogReg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                      class_weight="balanced", random_state=0, n_jobs=-1)
    spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         random_state=0, scale_pos_weight=spw)


def valid_folds(folds, y):
    return [(tr, te) for tr, te in folds if len(np.unique(y[te])) == 2]


def eval_on_folds(frame, feats, y, folds, name):
    X = frame[feats].values
    aucs, aps = [], []
    for tr, te in folds:
        m = make_model(name, y[tr]); m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
    return np.array(aucs), np.array(aps)


def delta(a, b):
    d = a - b
    p = stats.ttest_rel(a, b).pvalue if len(d) > 1 and d.std() > 0 else np.nan
    return d.mean(), d.std(), 100 * np.mean(d > 0), p


def pooled_folds(frame, y, n_rep=10):
    return valid_folds(list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=n_rep, random_state=42).split(frame, y)), y)


def grouped_folds(frame, y, groups, n_rep=10):
    f = []
    for r in range(n_rep):
        f += list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                       random_state=100 + r).split(frame, y, groups))
    return valid_folds(f, y)


def regime_table(frame, y, folds, title):
    print(f"\n--- regime: {title}  (n={len(frame)}, folds={len(folds)}, base rate={y.mean():.3f}) ---")
    print(f"{'model':14}{'chemistry':>16}{'+SI':>16}{'+kinetics':>16}")
    store = {}
    for mdl in MODELS:
        auc = {s: eval_on_folds(frame, f, y, folds, mdl)[0] for s, f in SETS.items()}
        store[mdl] = auc
        print(f"{mdl:14}" + "".join(f"{auc[s].mean():>10.3f}+/-{auc[s].std():.3f}"[:16].rjust(16)
                                    for s in SETS))
    print("  marginal AUC (paired): mean dAUC [%folds improved] p")
    for mdl in MODELS:
        m1, s1, f1, p1 = delta(store[mdl]["+SI"], store[mdl]["chemistry"])
        m2, s2, f2, p2 = delta(store[mdl]["+kinetics"], store[mdl]["+SI"])
        print(f"    {mdl:13} SI-vs-chem {m1:+.4f}[{f1:3.0f}%]p={p1:.2f}"
              f"   kin-vs-SI {m2:+.4f}[{f2:3.0f}%]p={p2:.2f}")
    return store


def main():
    df = pd.read_parquet(os.path.join(ROOT, "outputs", "with_physics.parquet")).reset_index(drop=True)
    y = df["Inspection_Result"].values.astype(int)
    # Well_Number is not globally unique (some numbers span both fields) -> group on
    # the composite well identity so grouped folds are truly leakage-free.
    groups = (df["Field"].astype(str) + "_" + df["Well_Number"].astype(str)).values

    print("=" * 78)
    print("THE PHYSICS-VALUE EXPERIMENT (rigorous, confound-aware) — AUC mean +/- std")
    print("  trivial baseline AUC = 0.500. Pooled/cross-field are FIELD-CONFOUNDED;")
    print("  the decisive test is WITHIN FIELD B (balanced, no field-identity signal).")
    print("=" * 78)

    regime_table(df, y, pooled_folds(df, y), "POOLED (CONFOUNDED by field)")
    regime_table(df, y, grouped_folds(df, y, groups), "GROUPED by well (leakage-free, still field-confounded)")

    dfB = df[df.Field == "B"].reset_index(drop=True)
    yB = dfB["Inspection_Result"].values.astype(int)
    gB = dfB["Well_Number"].values
    regime_table(dfB, yB, grouped_folds(dfB, yB, gB), "WITHIN FIELD B, grouped  <-- DECISIVE (no field confound)")

    # ---- field-confound diagnostic (Simpson's paradox) ----
    print("\n" + "=" * 78)
    print("FIELD-CONFOUND DIAGNOSTIC — how much of 'chemistry' is just guessing the field?")
    print("=" * 78)
    dff = df.copy(); dff["Field_is_A"] = (df.Field == "A").astype(int)
    folds = pooled_folds(df, y)
    a_chem = eval_on_folds(df, CHEM, y, folds, "XGBoost")[0]
    a_chemF = eval_on_folds(dff, CHEM + ["Field_is_A"], y, folds, "XGBoost")[0]
    print(f"  pooled XGB AUC  chemistry            = {a_chem.mean():.3f}")
    print(f"  pooled XGB AUC  chemistry + Field    = {a_chemF.mean():.3f}   (jump = +{a_chemF.mean()-a_chem.mean():.3f})")
    for fld in ["A", "B"]:
        sub = df[df.Field == fld]
        r = np.corrcoef(sub["Ca_TDS_norm"], sub["Inspection_Result"])[0, 1]
        print(f"  corr(Ca_TDS_norm, label) in Field {fld} = {r:+.3f}   "
              f"(field positive rate {sub['Inspection_Result'].mean():.2f})")
    print("  -> sign flip across fields = Simpson's paradox; pooled AUC rewards field identity.")


if __name__ == "__main__":
    main()
