"""
Phase 1 - Data preparation for the scale-prediction system.

Loads the Al-Hajri (2020) inspection-labelled dataset, runs validity and
charge-balance checks, infers chloride by electroneutrality, engineers the
ionic ratios from methodology section 3.5.1, and produces both the naive
stratified split and the grouped-by-well split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

# ---- portable paths ---------------------------------------------------------
# This file lives at thesis_code/data/prep.py.  parents[1] is thesis_code/.
ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data_in"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RAW = "al_hajri_2020_extracted_rows.xlsx"
# -----------------------------------------------------------------------------

# Molar masses (g/mol) and charges for the measured ions
MW = {"Na": 22.990, "Ca": 40.078, "Mg": 24.305,
      "SO4": 96.06, "HCO3": 61.017, "Cl": 35.453}
Z = {"Na": 1, "Ca": 2, "Mg": 2, "SO4": 2, "HCO3": 1, "Cl": 1}


def load_raw(path=RAW):
    df = pd.read_excel(path)
    df["Sample_Date"] = pd.to_datetime(df["Sample_Date"], errors="coerce")
    df["Inspection_Date"] = pd.to_datetime(df["Inspection_Date"], errors="coerce")
    return df


def validity_report(df):
    """Physical-plausibility checks. Returns a dict of flag counts."""
    chem = ["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm", "TDS_ppm"]
    rep = {}
    rep["n_rows"] = len(df)
    rep["negative_or_zero_chem"] = int((df[chem] <= 0).any(axis=1).sum())
    rep["pH_out_of_range"] = int(((df["pH"] < 4) | (df["pH"] > 10)).sum())
    ion_sum = df[["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm"]].sum(axis=1)
    rep["ion_sum_gt_TDS"] = int((ion_sum > df["TDS_ppm"] * 1.05).sum())
    rep["target_pos"] = int((df["Inspection_Result"] == 1).sum())
    rep["target_neg"] = int((df["Inspection_Result"] == 0).sum())
    rep["unique_wells"] = int(df["Well_Number"].nunique())
    return rep


def charge_balance(df):
    """Compute cation/anion equivalents and infer Cl- by electroneutrality."""
    out = df.copy()
    cat_eq = sum(out[f"{i}_ppm"] / MW[i] * Z[i] for i in ["Na", "Ca", "Mg"])
    an_eq_meas = sum(out[f"{i}_ppm"] / MW[i] * Z[i] for i in ["SO4", "HCO3"])
    out["cation_eq"] = cat_eq
    out["anion_eq_measured"] = an_eq_meas
    out["CBE_pre_pct"] = 100 * (cat_eq - an_eq_meas) / (cat_eq + an_eq_meas)
    cl_eq = (cat_eq - an_eq_meas).clip(lower=0.0)
    out["Cl_meq"] = cl_eq
    out["Cl_ppm"] = cl_eq * MW["Cl"] / Z["Cl"]
    return out


def to_molality(df):
    """ppm (mg/kg solution) -> molality (mol/kg water) using TDS for water mass."""
    out = df.copy()
    kg_water = (1.0 - out["TDS_ppm"] / 1e6).clip(lower=0.5)
    for i in ["Na", "Ca", "Mg", "SO4", "HCO3", "Cl"]:
        out[f"m_{i}"] = (out[f"{i}_ppm"] / 1000.0 / MW[i]) / kg_water
    out["ionic_strength"] = 0.5 * sum(out[f"m_{i}"] * Z[i] ** 2
                                      for i in ["Na", "Ca", "Mg", "SO4", "HCO3", "Cl"])
    return out


def engineer_features(df):
    """Ionic ratios from methodology 3.5.1 (molar basis)."""
    out = df.copy()
    ca = out["Ca_ppm"] / MW["Ca"]
    hco3 = out["HCO3_ppm"] / MW["HCO3"]
    mg = out["Mg_ppm"] / MW["Mg"]
    so4 = out["SO4_ppm"] / MW["SO4"]
    eps = 1e-9
    out["Ca_HCO3_ratio"] = ca / (hco3 + eps)
    out["Mg_Ca_ratio"] = mg / (ca + eps)
    out["SO4_Ca_ratio"] = so4 / (ca + eps)
    out["Ca_TDS_norm"] = out["Ca_ppm"] / (out["TDS_ppm"] + eps)
    return out


def make_splits(df, test_size=0.20, seed=42):
    """Return index arrays for naive stratified and grouped-by-well splits."""
    y = df["Inspection_Result"].values
    groups = df["Well_Number"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    naive_tr, naive_te = next(sss.split(df, y))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    grp_tr, grp_te = next(gss.split(df, y, groups))

    return {"naive": (naive_tr, naive_te), "grouped": (grp_tr, grp_te)}


def build():
    df = load_raw()
    rep = validity_report(df)
    df = charge_balance(df)
    df = to_molality(df)
    df = engineer_features(df)
    splits = make_splits(df)
    return df, rep, splits


if __name__ == "__main__":
    df, rep, splits = build()
    print("=== VALIDITY REPORT ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("\n=== CHARGE BALANCE (pre-Cl inference) ===")
    print(df["CBE_pre_pct"].describe().to_string())
    print("\n=== INFERRED Cl- (ppm) ===")
    print(df["Cl_ppm"].describe().to_string())
    print("\n=== IONIC STRENGTH (mol/kg) ===")
    print(df["ionic_strength"].describe().to_string())
    print("\n=== ENGINEERED RATIOS ===")
    print(df[["Ca_HCO3_ratio", "Mg_Ca_ratio", "SO4_Ca_ratio", "Ca_TDS_norm"]].describe().to_string())
    for name, (tr, te) in splits.items():
        ytr = df["Inspection_Result"].values[tr]
        yte = df["Inspection_Result"].values[te]
        wtr = set(df["Well_Number"].values[tr]); wte = set(df["Well_Number"].values[te])
        print(f"\n=== SPLIT: {name} ===")
        print(f"  train n={len(tr)} pos%={ytr.mean():.3f} | test n={len(te)} pos%={yte.mean():.3f}")
        print(f"  well overlap train and test = {len(wtr & wte)} wells")
    df.to_parquet("outputs/prepared.parquet")
    np.savez("outputs/splits.npz",
             naive_tr=splits["naive"][0], naive_te=splits["naive"][1],
             grouped_tr=splits["grouped"][0], grouped_te=splits["grouped"][1])
    print(f"\nsaved {OUTPUTS / 'prepared.parquet'}")
    print(f"saved {OUTPUTS / 'splits.npz'}")
