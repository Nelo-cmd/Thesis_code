"""
Phase 4 validation: our saturation-index engine vs PHREEQC (the field-standard oracle).

We compare calcite SI computed by our code (Plummer-Busenberg Ksp + Pitzer/HMW84
activities) against PHREEQC with the Pitzer database (pitzer.dat), sample-by-sample,
at the SAME thermodynamic condition: each sample's measured pH, 25 C, 1 atm.

This isolates and tests the thermodynamic ENGINE (activities + Ksp + carbonate
speciation). It deliberately does NOT test the wellbore P/T extrapolation — PHREEQC
cannot validate our assumed downhole gradients; those are addressed by the sensitivity
analysis. A consistent bias here is harmless for ML features (it preserves ranking);
random scatter is what would undermine trust.

Run from anywhere:  python tests/validate_phreeqc.py
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from physics import constants as C
from physics.pitzer import activity_coefficients
from phreeqpython import PhreeqPython

T_REF = 298.15  # 25 C
IONS = ["Na", "Ca", "Mg", "Cl", "SO4", "HCO3"]


def si_ours(sample, T_K=T_REF):
    """Calcite SI from our engine at (measured pH, T_K, 1 atm)."""
    m = {k: float(sample[f"m_{k}"]) for k in IONS}
    gam = activity_coefficients({**m, "CO3": 1e-6}, T=T_K)
    gCa, gHCO3 = gam.get("Ca", 1.0), gam.get("HCO3", 0.6)
    aH = 10.0 ** (-float(sample["pH"]))
    a_CO3 = C.K2(T_K) * gHCO3 * m["HCO3"] / aH
    IAP = gCa * m["Ca"] * a_CO3
    return float(np.log10(IAP) - C.logKsp_calcite(T_K))


def si_phreeqc(pp, sample):
    """Calcite SI from PHREEQC (pitzer.dat) at (measured pH, 25 C, 1 atm)."""
    sol = pp.add_solution({
        "units": "mg/l", "temp": 25.0, "pH": float(sample["pH"]),
        "Na": float(sample["Na_ppm"]), "Ca": float(sample["Ca_ppm"]),
        "Mg": float(sample["Mg_ppm"]), "Cl": float(sample["Cl_ppm"]),
        "S(6)": f"{float(sample['SO4_ppm'])} as SO4",
        "Alkalinity": f"{float(sample['HCO3_ppm'])} as HCO3",
    })
    si = float(sol.si("Calcite"))
    sol.forget()  # free the C-side solution so 505 runs stay light
    return si


def main():
    df = pd.read_parquet(os.path.join(ROOT, "outputs", "prepared.parquet"))
    pp = PhreeqPython(database="pitzer.dat")

    ours, phq, ok = [], [], []
    for _, s in df.iterrows():
        try:
            o, p = si_ours(s), si_phreeqc(pp, s)
            ours.append(o); phq.append(p); ok.append(True)
        except Exception:
            ours.append(np.nan); phq.append(np.nan); ok.append(False)

    ours, phq = np.array(ours), np.array(phq)
    mask = np.array(ok) & np.isfinite(ours) & np.isfinite(phq)
    o, p = ours[mask], phq[mask]
    diff = o - p

    # rank correlation without scipy dependency
    def spearman(a, b):
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        return float(np.corrcoef(ra, rb)[0, 1])

    slope, intercept = np.polyfit(p, o, 1)

    print("=" * 70)
    print("PHREEQC VALIDATION  —  calcite SI: ours vs PHREEQC (pitzer.dat)")
    print(f"  condition: measured pH, 25 C, 1 atm   |   n = {mask.sum()} / {len(df)}")
    print("=" * 70)
    print(f"  bias  (mean ours - phreeqc) = {diff.mean():+.3f}   <- systematic offset")
    print(f"  MAE   (mean |diff|)         = {np.abs(diff).mean():.3f}")
    print(f"  RMSE                        = {np.sqrt((diff**2).mean()):.3f}")
    print(f"  std of diff                 = {diff.std():.3f}   <- random scatter")
    print(f"  Pearson r                   = {np.corrcoef(o, p)[0,1]:.4f}")
    print(f"  Spearman rho (ranking)      = {spearman(o, p):.4f}")
    print(f"  bias-corrected MAE          = {np.abs(diff - diff.mean()).mean():.3f}")
    print(f"  linear fit  ours = {slope:.3f}*phreeqc + {intercept:+.3f}")
    for thr in (0.1, 0.2, 0.3):
        print(f"  within {thr:.1f} SI              = {100*np.mean(np.abs(diff) <= thr):5.1f} %")
        print(f"  within {thr:.1f} after debias     = {100*np.mean(np.abs(diff - diff.mean()) <= thr):5.1f} %")

    out = pd.DataFrame({"SI_ours": ours, "SI_phreeqc": phq, "ok": ok})
    out.to_parquet(os.path.join(ROOT, "outputs", "phreeqc_validation.parquet"))
    print(f"\nsaved outputs/phreeqc_validation.parquet")


if __name__ == "__main__":
    main()
