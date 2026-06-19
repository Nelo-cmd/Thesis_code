"""
Pressure correction of the solubility product via the molar-volume method.

  ln(K_P / K_1bar) = -(dVr/RT)(P - Pref) + (dKr/2RT)(P - Pref)^2

dVr is the molar volume change of the dissolution reaction
CaCO3(s) -> Ca2+ + CO3^2-, dKr the corresponding compressibility change.
Values from Millero (1982/2001) partial molar volumes; the correction is
deliberately first-order because the underlying wellbore pressure profile is
itself an assumed gradient (see thesis sensitivity analysis).

R = 83.14 cm^3 bar / (mol K). Pressure in bar.
"""
import numpy as np

R = 83.14  # cm^3 bar / mol / K

# Molar volume / compressibility change of dissolution (cm^3/mol, cm^3/mol/bar)
DVR = {"calcite": -58.0, "aragonite": -53.0}
DKR = {"calcite": -8.0e-3, "aragonite": -7.0e-3}


def pressure_factor(P_bar, T, phase="calcite", P_ref=1.0):
    """Return ln(K_P / K_1bar) for the given phase."""
    dV = DVR[phase]
    dK = DKR[phase]
    dP = P_bar - P_ref
    return -(dV / (R * T)) * dP + (dK / (2 * R * T)) * dP ** 2


def apply(logKsp_1bar, P_bar, T, phase="calcite"):
    """Apply pressure correction to a base-10 log Ksp."""
    return logKsp_1bar + pressure_factor(P_bar, T, phase) / np.log(10.0)


if __name__ == "__main__":
    T = 350.0  # K (~77 C)
    for P_psi in [500, 2000, 4500]:
        P_bar = P_psi * 0.0689476
        f = pressure_factor(P_bar, T, "calcite")
        print(f"P={P_psi:>4} psi ({P_bar:6.1f} bar): "
              f"ln(KP/K1)={f:+.3f}  KP/K1={np.exp(f):.3f}  "
              f"dlogKsp={f/np.log(10):+.3f}")
