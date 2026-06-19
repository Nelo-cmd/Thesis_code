"""
Validation of the Pitzer implementation against published mean activity
coefficients (gamma_pm) at 25 C. If these reproduce the literature within a
few percent, the activity model is trustworthy for the brine SI calculation.

Reference gamma_pm values (Robinson & Stokes / Pitzer 1991):
  NaCl   at m = 0.1, 0.5, 1.0, 2.0, 3.0, 5.0
  CaCl2  at m = 0.1, 0.5, 1.0, 2.0
  MgCl2  at m = 0.1, 0.5, 1.0
"""
import sys
sys.path.insert(0, ".")
from physics.pitzer import activity_coefficients, gamma_pm, A_phi

LIT = {
    "NaCl": {0.1: 0.778, 0.5: 0.681, 1.0: 0.657, 2.0: 0.668, 3.0: 0.714, 5.0: 0.874},
    "CaCl2": {0.1: 0.518, 0.5: 0.448, 1.0: 0.500, 2.0: 0.792},
    "MgCl2": {0.1: 0.529, 0.5: 0.481, 1.0: 0.569},
}


def test_nacl():
    print("NaCl  m      gamma_calc   gamma_lit   err%")
    for m, lit in LIT["NaCl"].items():
        g = activity_coefficients({"Na": m, "Cl": m})
        gpm = gamma_pm(g["Na"], g["Cl"], 1, 1)
        print(f"     {m:>4}    {gpm:.4f}      {lit:.4f}    {100*(gpm-lit)/lit:+.1f}")


def test_cacl2():
    print("\nCaCl2 m      gamma_calc   gamma_lit   err%")
    for m, lit in LIT["CaCl2"].items():
        g = activity_coefficients({"Ca": m, "Cl": 2 * m})
        gpm = gamma_pm(g["Ca"], g["Cl"], 1, 2)
        print(f"     {m:>4}    {gpm:.4f}      {lit:.4f}    {100*(gpm-lit)/lit:+.1f}")


def test_mgcl2():
    print("\nMgCl2 m      gamma_calc   gamma_lit   err%")
    for m, lit in LIT["MgCl2"].items():
        g = activity_coefficients({"Mg": m, "Cl": 2 * m})
        gpm = gamma_pm(g["Mg"], g["Cl"], 1, 2)
        print(f"     {m:>4}    {gpm:.4f}      {lit:.4f}    {100*(gpm-lit)/lit:+.1f}")


def test_seawater():
    # Standard seawater molalities (mol/kg). Expected Ca2+ activity coeff ~0.20-0.26,
    # CO3^2- ~0.04-0.07 (strongly depressed by ionic strength).
    sw = {"Na": 0.486, "Mg": 0.0547, "Ca": 0.0107,
          "Cl": 0.566, "SO4": 0.0293, "HCO3": 0.00183, "CO3": 0.00027}
    g = activity_coefficients(sw)
    print("\nSeawater individual ion activity coefficients:")
    for ion in ["Na", "Ca", "Mg", "Cl", "SO4", "HCO3", "CO3"]:
        print(f"   gamma({ion:>4}) = {g[ion]:.4f}")
    print("   [lit: gamma_Ca ~0.20-0.26, gamma_CO3 ~0.04-0.07]")


if __name__ == "__main__":
    print(f"A_phi(25C) = {A_phi(298.15):.4f}  [0.3915]\n")
    test_nacl()
    test_cacl2()
    test_mgcl2()
    test_seawater()
