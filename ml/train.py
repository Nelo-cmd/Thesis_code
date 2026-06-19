"""
Phase 3 - Supervised classifier development and the physics-value experiment.

Trains Logistic Regression, Random Forest and XGBoost under three feature sets
(chemistry only -> +engineered ratios -> +physics) and three evaluation regimes
(naive stratified split, grouped-by-well split, and cross-field generalisation).

The chemistry-only vs +physics comparison is the empirical test of the thesis's
central claim. The grouped and cross-field regimes expose the leakage and field
confounding identified in the data analysis.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---- portable paths ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

DATA = OUTPUTS / "with_physics.parquet"

FS_CHEM = ["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm", "TDS_ppm", "pH"]
FS_RATIOS = FS_CHEM + ["Cl_ppm", "ionic_strength",
                       "Ca_HCO3_ratio", "Mg_Ca_ratio", "SO4_Ca_ratio", "Ca_TDS_norm"]
FS_PHYS = FS_RATIOS + ["SI_max_cal", "SI_mean_cal", "SI_max_arag",
                       "onset_depth_ft", "supersat_fraction", "highrisk_zone_ft"]
FEATURE_SETS = {"chemistry": FS_CHEM, "chem+ratios": FS_RATIOS, "full(+physics)": FS_PHYS}


def make_models():
    return {
        "LogReg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000,
                                                       class_weight="balanced"))]),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=8,
                                               min_samples_leaf=3,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=42,
                                 scale_pos_weight=150.0 / 355.0),
    }


def metrics(y, yp, yprob):
    out = {"acc": accuracy_score(y, yp),
           "prec": precision_score(y, yp, zero_division=0),
           "rec": recall_score(y, yp, zero_division=0),
           "f1": f1_score(y, yp, zero_division=0)}
    out["auc"] = roc_auc_score(y, yprob) if len(np.unique(y)) > 1 else np.nan
    return out


def evaluate(df, tr, te, feats):
    X = df[feats].values
    y = df["Inspection_Result"].values
    res = {}
    for name, model in make_models().items():
        model.fit(X[tr], y[tr])
        yp = model.predict(X[te])
        yprob = model.predict_proba(X[te])[:, 1]
        res[name] = metrics(y[te], yp, yprob)
    return res


def main():
    df = pd.read_parquet(DATA).reset_index(drop=True)
    sp = np.load(OUTPUTS / "splits.npz")
    y = df["Inspection_Result"].values
    groups = df["Well_Number"].values

    regimes = {
        "naive": (sp["naive_tr"], sp["naive_te"]),
        "grouped": (sp["grouped_tr"], sp["grouped_te"]),
        "field_B->A": (np.where(df.Field == "B")[0], np.where(df.Field == "A")[0]),
        "field_A->B": (np.where(df.Field == "A")[0], np.where(df.Field == "B")[0]),
    }

    rows = []
    for reg, (tr, te) in regimes.items():
        for fsname, feats in FEATURE_SETS.items():
            res = evaluate(df, tr, te, feats)
            for model, m in res.items():
                rows.append({"regime": reg, "features": fsname, "model": model, **m})
    R = pd.DataFrame(rows)
    R.to_parquet(OUTPUTS / "ml_results.parquet")

    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("=" * 78)
    print("THE PHYSICS-VALUE EXPERIMENT  (F1 / AUC by feature set)")
    print("=" * 78)
    for reg in regimes:
        print(f"\n--- regime: {reg} ---")
        sub = R[R.regime == reg].pivot_table(index="model", columns="features",
                                             values="f1")
        sub = sub[["chemistry", "chem+ratios", "full(+physics)"]]
        print("F1 score:")
        print(sub.round(3).to_string())
    print("\n" + "=" * 78)
    print("BEST MODEL FULL METRICS PER REGIME (full feature set)")
    print("=" * 78)
    best = R[R.features == "full(+physics)"]
    print(best[["regime", "model", "acc", "prec", "rec", "f1", "auc"]]
          .round(3).to_string(index=False))

    # cross-validated F1 on the grouped training fold (honest internal estimate)
    print("\n" + "=" * 78)
    print("GROUPED 5-FOLD CV (full features) - internal generalisation estimate")
    print("=" * 78)
    X = df[FS_PHYS].values
    gkf = GroupKFold(n_splits=5)
    for name, model in make_models().items():
        sc = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring="f1")
        print(f"  {name:14} F1 = {sc.mean():.3f} +/- {sc.std():.3f}")


if __name__ == "__main__":
    main()