"""
Temperature-dependent equilibrium constants for the carbonate system.

All correlations are from Plummer & Busenberg (1982), "The solubilities of
calcite, aragonite and vaterite in CO2-H2O solutions between 0 and 90 C..."
Geochimica et Cosmochimica Acta 46, 1011-1040. T in Kelvin throughout.
Valid 0-90 C at 1 atm; pressure is applied separately (see pressure.py).
"""
import numpy as np


def _log10(x):
    return np.log10(x)


def logK1(T):
    """First dissociation of carbonic acid: H2CO3* <-> H+ + HCO3-."""
    return (-356.3094 - 0.06091964 * T + 21834.37 / T
            + 126.8339 * _log10(T) - 1684915.0 / T ** 2)


def logK2(T):
    """Second dissociation: HCO3- <-> H+ + CO3^2-."""
    return (-107.8871 - 0.03252849 * T + 5151.79 / T
            + 38.92561 * _log10(T) - 563713.9 / T ** 2)


def logKsp_calcite(T):
    """CaCO3(calcite) <-> Ca2+ + CO3^2-."""
    return (-171.9065 - 0.077993 * T + 2839.319 / T + 71.595 * _log10(T))


def logKsp_aragonite(T):
    """CaCO3(aragonite) <-> Ca2+ + CO3^2-."""
    return (-171.9773 - 0.077993 * T + 2903.293 / T + 71.595 * _log10(T))


def K1(T):
    return 10.0 ** logK1(T)


def K2(T):
    return 10.0 ** logK2(T)


def Ksp_calcite(T):
    return 10.0 ** logKsp_calcite(T)


def Ksp_aragonite(T):
    return 10.0 ** logKsp_aragonite(T)


if __name__ == "__main__":
    T = 298.15
    print("Self-check at 25 C (expected literature values in brackets):")
    print(f"  logK1       = {logK1(T):.4f}  [-6.35]")
    print(f"  logK2       = {logK2(T):.4f}  [-10.33]")
    print(f"  logKsp_cal  = {logKsp_calcite(T):.4f}  [-8.48]")
    print(f"  logKsp_arag = {logKsp_aragonite(T):.4f}  [-8.34]")
    print("\nTemperature dependence (retrograde solubility check):")
    for tc in [25, 50, 75, 90]:
        tk = tc + 273.15
        print(f"  {tc:>3} C: logKsp_calcite = {logKsp_calcite(tk):.4f}")
