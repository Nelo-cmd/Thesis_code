"""
Calcite precipitation / dissolution kinetics — the Plummer-Wigley-Parkhurst (PWP) rate law.

This is the "amber box" of the model (thesis Module 3): it answers *how fast*
calcite reacts — the piece the equilibrium saturation index (SI) cannot supply.

Net rate per unit reactive surface area, PWP (1978) as implemented in the
PHREEQC v3 manual (Parkhurst & Appelo, 2013):

    r = ( k1*a(H+) + k2*a(CO2) + k3*a(H2O) ) * ( 1 - (IAP/Ksp)^(2/3) )
      = [ forward dissolution rate ]         * [ thermodynamic affinity brake ]

Sign convention follows PHREEQC:  r > 0 -> dissolution,  r < 0 -> precipitation.

  * The bracket is the FORWARD rate: three parallel mechanisms — attack by H+,
    by dissolved CO2, and by water. Each k is temperature dependent.
  * The (1 - 10^(2/3*SI)) factor is the AFFINITY BRAKE: exactly 0 at equilibrium
    (SI=0), positive when undersaturated (SI<0 -> dissolves), negative when
    supersaturated (SI>0 -> precipitates).

RIGOR / HONESTY (for the write-up):
  * The three forward k-terms are mechanistic (each tied to a real reactant).
  * The 2/3 exponent in the brake is SEMI-EMPIRICAL (calibrated, not first
    principles). We test sensitivity to it later.
  * Original PWP calibration was ~5-48 C; we apply it to 40-85 C wellbore brine
    and validate the result against PHREEQC (phreeqpython, pitzer.dat).
  * Absolute magnitude needs the reactive surface area per well, which we do not
    have, so we use the per-area rate as a RELATIVE deposition-rate index.

Rate constants verified verbatim from the official PHREEQC v3 source. T in Kelvin.
"""
import os
import sys
import numpy as np

# make `from physics import ...` work no matter the current directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics import constants as C


def pwp_constants(T_K):
    """Temperature-dependent PWP forward rate constants (k1, k2, k3).

    k1 — attack by H+         (dominant at low pH)
    k2 — attack by CO2/H2CO3  (the term that makes PWP right for CO2-driven wellbores)
    k3 — attack by water      (dominant at high pH; note the T<=25 / T>25 branch)

    Accepts scalars or numpy arrays (so a depth profile of T works directly).
    """
    TC = T_K - 273.15
    k1 = 10.0 ** (0.198 - 444.0 / T_K)
    k2 = 10.0 ** (2.84 - 2177.0 / T_K)
    k3 = np.where(TC <= 25.0,
                  10.0 ** (-5.86 - 317.0 / T_K),
                  10.0 ** (-1.10 - 1737.0 / T_K))
    return k1, k2, k3


def co2_activity(a_H, a_HCO3, T_K):
    """Activity of dissolved CO2 from the first carbonic-acid dissociation.

        CO2(aq) + H2O <-> H+ + HCO3- ,   K1 = a(H+) a(HCO3-) / a(CO2)
        =>  a(CO2) = a(H+) * a(HCO3-) / K1(T)

    Ties the kinetic CO2 term back to measured bicarbonate and pH (Module 5).
    """
    return a_H * a_HCO3 / C.K1(T_K)


def pwp_net_rate(SI_calcite, a_H, a_CO2, a_H2O=1.0, T_K=298.15, area=1.0):
    """PWP net reaction rate per unit area.

    PHREEQC sign convention: > 0 dissolution, < 0 precipitation.
    `area` is a relative surface-area factor (default 1 -> specific rate).
    """
    k1, k2, k3 = pwp_constants(T_K)
    r_forward = k1 * a_H + k2 * a_CO2 + k3 * a_H2O          # always > 0
    affinity_brake = 1.0 - 10.0 ** ((2.0 / 3.0) * SI_calcite)
    return area * r_forward * affinity_brake


def deposition_rate(SI_calcite, a_H, a_CO2, a_H2O=1.0, T_K=298.15, area=1.0):
    """Calcite precipitation (scaling) rate, >= 0.

    This is the quantity of interest for scaling: 0 unless supersaturated (SI>0),
    growing with both the thermodynamic push (SI) and the kinetic speed (forward rate).
    """
    net = pwp_net_rate(SI_calcite, a_H, a_CO2, a_H2O, T_K, area)
    return np.maximum(0.0, -net)


if __name__ == "__main__":
    print("=== PWP forward rate constants (per unit area) ===")
    for tc in [25, 40, 60, 85]:
        k1, k2, k3 = pwp_constants(tc + 273.15)
        print(f"  {tc:>3} C: k1={float(k1):.3e}  k2={float(k2):.3e}  k3={float(k3):.3e}")

    print("\n=== sign behaviour across saturation (T=85 C, a_H=1e-6, a_CO2=1e-3) ===")
    T, aH, aCO2 = 358.15, 1e-6, 1e-3
    for si in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        net = float(pwp_net_rate(si, aH, aCO2, 1.0, T))
        dep = float(deposition_rate(si, aH, aCO2, 1.0, T))
        tag = "dissolve" if net > 0 else ("equilibrium" if net == 0 else "PRECIPITATE")
        print(f"  SI={si:+.1f}: net={net:+.3e}  deposition={dep:.3e}  [{tag}]")

    print("\n=== worked example: supersaturated brine at 85 C ===")
    aH = 10.0 ** (-6.0)
    aHCO3 = 0.5 * 5e-3                       # gamma_HCO3 ~ 0.5, m_HCO3 ~ 5 mmol/kg
    aCO2 = float(co2_activity(aH, aHCO3, T))
    print(f"  a(CO2) from pH=6, a(HCO3)={aHCO3:.2e}  ->  {aCO2:.3e}")
    for si in [1.0, 2.0, 3.0]:
        dep = float(deposition_rate(si, aH, aCO2, 1.0, T))
        print(f"  SI={si:+.1f}: deposition-rate index = {dep:.3e}")
