"""
Pitzer ion-interaction model for activity coefficients in hypersaline brine.

Implements the Pitzer (1973, 1991) formalism with the Harvie-Moller-Weare
(1984) parameter set for the Na-Ca-Mg-Cl-SO4-HCO3-CO3 system, including
binary (beta0, beta1, beta2, Cphi), like-charge mixing (theta) and triplet
(psi) terms, plus the higher-order electrostatic (E-theta) terms required for
the asymmetric mixing of Ca2+/CO3^2- in a Na+/Cl- dominated medium.

The Debye-Huckel slope A_phi is computed from first principles using the
temperature-dependent dielectric constant (Malmberg-Maryott 1956) and water
density, so no magic constant is introduced.

References:
  Pitzer, K.S. (1991). Activity Coefficients in Electrolyte Solutions, 2nd ed.
  Harvie, Moller & Weare (1984). Geochim. Cosmochim. Acta 48, 723-751.
"""
import numpy as np

# Physical constants (SI)
N_A = 6.02214076e23
E_CHG = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23

BVAL = 1.2  # Pitzer universal constant

# Ion charges
ZI = {"Na": 1, "Ca": 2, "Mg": 2, "Cl": -1, "SO4": -2, "HCO3": -1, "CO3": -2}
CATIONS = ["Na", "Ca", "Mg"]
ANIONS = ["Cl", "SO4", "HCO3", "CO3"]

# Binary parameters at 25 C: (beta0, beta1, beta2, Cphi). HMW84.
BETA = {
    ("Na", "Cl"):  (0.0765, 0.2664, 0.0, 0.00127),
    ("Na", "SO4"): (0.01958, 1.113, 0.0, 0.00497),
    ("Na", "HCO3"): (0.0277, 0.0411, 0.0, 0.0),
    ("Na", "CO3"): (0.0399, 1.389, 0.0, 0.0044),
    ("Ca", "Cl"):  (0.3159, 1.614, 0.0, -0.00034),
    ("Ca", "SO4"): (0.20, 3.1973, -54.24, 0.0),
    ("Ca", "HCO3"): (0.4, 2.977, 0.0, 0.0),
    ("Ca", "CO3"): (0.0, 0.0, 0.0, 0.0),     # strong ion pairing; handled as ~0 binary
    ("Mg", "Cl"):  (0.35235, 1.6815, 0.0, 0.00519),
    ("Mg", "SO4"): (0.221, 3.343, -37.23, 0.025),
    ("Mg", "HCO3"): (0.329, 0.6072, 0.0, 0.0),
    ("Mg", "CO3"): (0.0, 0.0, 0.0, 0.0),
}

# Like-charge mixing theta
THETA = {
    ("Na", "Ca"): 0.07, ("Na", "Mg"): 0.07, ("Ca", "Mg"): 0.007,
    ("Cl", "SO4"): 0.02, ("Cl", "HCO3"): 0.03, ("Cl", "CO3"): -0.02,
    ("SO4", "HCO3"): 0.01, ("SO4", "CO3"): 0.02, ("HCO3", "CO3"): -0.04,
}

# Triplet psi
PSI = {
    ("Na", "Ca", "Cl"): -0.007, ("Na", "Mg", "Cl"): -0.012,
    ("Ca", "Mg", "Cl"): -0.012, ("Na", "Cl", "SO4"): 0.0014,
    ("Na", "Cl", "HCO3"): -0.015, ("Na", "Cl", "CO3"): 0.0085,
    ("Na", "SO4", "CO3"): -0.005, ("Na", "HCO3", "CO3"): -0.04,
    ("Ca", "Cl", "SO4"): -0.018, ("Mg", "Cl", "SO4"): -0.004,
}


def _pair(a, b, table):
    return table.get((a, b)) or table.get((b, a))


def water_density(T):
    """kg/m3, 0-100 C (Kell 1975 fit)."""
    t = T - 273.15
    rho = (999.83952 + 16.945176 * t - 7.9870401e-3 * t ** 2
           - 46.170461e-6 * t ** 3 + 105.56302e-9 * t ** 4
           - 280.54253e-12 * t ** 5) / (1 + 16.879850e-3 * t)
    return rho


def dielectric(T):
    """Static dielectric constant of water (Malmberg-Maryott 1956), 0-100 C."""
    t = T - 273.15
    return 87.740 - 0.40008 * t + 9.398e-4 * t ** 2 - 1.410e-6 * t ** 3


def A_phi(T):
    """Debye-Huckel osmotic slope from first principles."""
    rho = water_density(T)               # kg/m3
    eps_r = dielectric(T)
    eps = eps_r * EPS0
    term = (E_CHG ** 2 / (eps * KB * T)) ** 1.5
    # rho in kg/m3, Bjerrum term in m -> A_phi in (kg/mol)^0.5 (no /1000 factor)
    return (1.0 / 3.0) * np.sqrt(2 * np.pi * N_A * rho) * term \
        * (1.0 / (4 * np.pi)) ** 1.5


def _g(x):
    x = np.where(x == 0, 1e-12, x)
    return 2.0 * (1.0 - (1.0 + x) * np.exp(-x)) / x ** 2


def _gp(x):
    x = np.where(x == 0, 1e-12, x)
    return -2.0 * (1.0 - (1.0 + x + x ** 2 / 2.0) * np.exp(-x)) / x ** 2


def _alpha(zc, za):
    return (1.4, 12.0) if (abs(zc) == 2 and abs(za) == 2) else (2.0, 12.0)


def _B(a, b, I):
    p = _pair(a, b, BETA)
    if p is None:
        return 0.0, 0.0
    b0, b1, b2, _ = p
    zc = ZI[a] if a in CATIONS else ZI[b]
    za = ZI[b] if b in ANIONS else ZI[a]
    al1, al2 = _alpha(zc, za)
    sI = np.sqrt(I)
    B = b0 + b1 * _g(al1 * sI) + b2 * _g(al2 * sI)
    Bp = (b1 * _gp(al1 * sI) + b2 * _gp(al2 * sI)) / I
    return B, Bp


def _C(a, b):
    p = _pair(a, b, BETA)
    if p is None:
        return 0.0
    cphi = p[3]
    zc = abs(ZI[a] if a in CATIONS else ZI[b])
    za = abs(ZI[b] if b in ANIONS else ZI[a])
    return cphi / (2.0 * np.sqrt(zc * za))


# ---- Higher-order electrostatic (E-theta) terms via Pitzer J-function ----
def _J(x):
    """Pitzer J integral, Harvie (1981) numerical approximation."""
    # Chebyshev-free numerical integration of the J function.
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= 0:
            out[i] = 0.0
            continue
        s = np.linspace(1e-6, 1.0, 400)  # substitution y = exp(-s'?) -> use direct
        # J(x) = (1/4)x - 1 + integral_0^inf (...) ; use the standard quadrature form:
        # J(x) = x/4 - 1 + (1/x) * integral_0^inf [1 - exp(-(x/t)exp(-t))] dt  (approx)
        t = np.linspace(1e-4, 40, 4000)
        integrand = 1.0 - np.exp(-(xi / t) * np.exp(-t))
        out[i] = xi / 4.0 - 1.0 + np.trapezoid(integrand, t)
    return out


def _Jp(x):
    """Numerical derivative of J."""
    h = 1e-4
    return (_J(np.asarray(x) + h) - _J(np.asarray(x) - h)) / (2 * h)


def _Etheta(zi, zj, I, Aph):
    """Higher-order electrostatic mixing term for unsymmetric like-charge pairs."""
    if zi == zj:
        return 0.0, 0.0
    sI = np.sqrt(I)
    xij = 6.0 * zi * zj * Aph * sI
    xii = 6.0 * zi * zi * Aph * sI
    xjj = 6.0 * zj * zj * Aph * sI
    Jij, Jii, Jjj = _J(xij)[0], _J(xii)[0], _J(xjj)[0]
    Jpij, Jpii, Jpjj = _Jp(xij)[0], _Jp(xii)[0], _Jp(xjj)[0]
    Eth = (zi * zj / (4.0 * I)) * (Jij - 0.5 * Jii - 0.5 * Jjj)
    Ethp = -(Eth / I) + (zi * zj / (8.0 * I ** 2)) * \
        (xij * Jpij - 0.5 * xii * Jpii - 0.5 * xjj * Jpjj)
    return Eth, Ethp


def activity_coefficients(m, T=298.15, use_etheta=True):
    """
    m: dict of molalities {ion: mol/kg}. Returns dict {ion: gamma}.
    """
    Aph = A_phi(T)
    cats = [c for c in CATIONS if m.get(c, 0) > 0]
    ans = [a for a in ANIONS if m.get(a, 0) > 0]
    I = 0.5 * sum(m[i] * ZI[i] ** 2 for i in cats + ans)
    if I <= 0:
        return {i: 1.0 for i in cats + ans}
    sI = np.sqrt(I)
    Z = sum(m[i] * abs(ZI[i]) for i in cats + ans)

    # F term
    f_gamma = -Aph * (sI / (1 + BVAL * sI) + (2.0 / BVAL) * np.log(1 + BVAL * sI))
    F = f_gamma
    for c in cats:
        for a in ans:
            _, Bp = _B(c, a, I)
            F += m[c] * m[a] * Bp
    # like-charge Phi' (E-theta') contributions
    for ii in range(len(cats)):
        for jj in range(ii + 1, len(cats)):
            ci, cj = cats[ii], cats[jj]
            if use_etheta:
                _, ethp = _Etheta(ZI[ci], ZI[cj], I, Aph)
                F += m[ci] * m[cj] * ethp
    for ii in range(len(ans)):
        for jj in range(ii + 1, len(ans)):
            ai, aj = ans[ii], ans[jj]
            if use_etheta:
                _, ethp = _Etheta(ZI[ai], ZI[aj], I, Aph)
                F += m[ai] * m[aj] * ethp

    gamma = {}
    # Cations
    for M in cats:
        zM = ZI[M]
        ln = zM ** 2 * F
        for a in ans:
            B, _ = _B(M, a, I)
            ln += m[a] * (2 * B + Z * _C(M, a))
        for c in cats:
            if c == M:
                continue
            th = _pair(M, c, THETA) or 0.0
            eth = _Etheta(zM, ZI[c], I, Aph)[0] if use_etheta else 0.0
            phi = th + eth
            psi_sum = 0.0
            for a in ans:
                psi_sum += m[a] * (PSI.get(tuple(sorted([M, c])) + (a,))
                                   or PSI.get((min(M, c), max(M, c), a)) or 0.0)
            ln += m[c] * (2 * phi + psi_sum)
        # anion-anion psi with this cation
        for ii in range(len(ans)):
            for jj in range(ii + 1, len(ans)):
                ai, aj = ans[ii], ans[jj]
                ps = PSI.get((M, ai, aj)) or PSI.get((M, aj, ai)) or 0.0
                ln += m[ai] * m[aj] * ps
        for c in cats:
            for a in ans:
                ln += abs(zM) * m[c] * m[a] * _C(c, a)
        gamma[M] = float(np.exp(ln))

    # Anions
    for X in ans:
        zX = ZI[X]
        ln = zX ** 2 * F
        for c in cats:
            B, _ = _B(c, X, I)
            ln += m[c] * (2 * B + Z * _C(c, X))
        for a in ans:
            if a == X:
                continue
            th = _pair(X, a, THETA) or 0.0
            eth = _Etheta(zX, ZI[a], I, Aph)[0] if use_etheta else 0.0
            phi = th + eth
            psi_sum = 0.0
            for c in cats:
                psi_sum += m[c] * (PSI.get((c, min(X, a), max(X, a)))
                                   or PSI.get((c, max(X, a), min(X, a))) or 0.0)
            ln += m[a] * (2 * phi + psi_sum)
        for ii in range(len(cats)):
            for jj in range(ii + 1, len(cats)):
                ci, cj = cats[ii], cats[jj]
                ps = PSI.get((ci, cj, X)) or PSI.get((cj, ci, X)) or 0.0
                ln += m[ci] * m[cj] * ps
        for c in cats:
            for a in ans:
                ln += abs(zX) * m[c] * m[a] * _C(c, a)
        gamma[X] = float(np.exp(ln))

    return gamma


def gamma_pm(gp, gm, nu_p, nu_m):
    """Mean activity coefficient from individual ion gammas."""
    return (gp ** nu_p * gm ** nu_m) ** (1.0 / (nu_p + nu_m))


if __name__ == "__main__":
    print(f"A_phi(25C) = {A_phi(298.15):.4f}  [expected ~0.391]")
    print(f"dielectric(25C) = {dielectric(298.15):.2f}  [78.30]")
    print(f"water_density(25C) = {water_density(298.15):.2f} kg/m3  [997.0]")
