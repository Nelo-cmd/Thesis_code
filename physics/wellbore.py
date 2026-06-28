"""
Depth-resolved downhole carbonate-scaling simulator (methodology 3.4).

For one water sample plus an assumed well configuration, this computes the
pressure, temperature and pH profiles up the wellbore, performs carbonate
speciation with full Pitzer activities, applies the pressure-corrected
Plummer-Busenberg Ksp, and returns the depth-resolved calcite and aragonite
saturation-index profiles plus the summary features fed to the ML model.

Depth convention: d measured downward from wellhead. d=0 at wellhead (top),
d=TVD at bottomhole (bottom). P and T increase with d; pH rises toward the
wellhead as CO2 degasses (driven by the pressure drop).

Speciation uses full activities (a stronger form than the simplified concentration
equation in the draft):
    a(CO3) = K2(T) * gamma_HCO3 * m_HCO3 / a(H+),   a(H+) = 10^(-pH)
    IAP    = gamma_Ca * m_Ca * a(CO3)
    SI     = log10(IAP / Ksp_P(T,P))
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from physics import constants as C
from physics import pressure as PR
from physics import kinetics as KIN
from physics.pitzer import activity_coefficients

PSI_TO_BAR = 0.0689476

# Reference well configuration for the carbonate fields (documented typical values)
REF = dict(
    TVD_ft=9000.0,      # true vertical depth
    WHP_psi=500.0,      # wellhead pressure
    BHP_psi=4500.0,     # bottomhole pressure
    WHT_C=40.0,         # wellhead temperature
    BHT_C=85.0,         # bottomhole temperature (within Plummer-Busenberg 0-90 C)
    a_degas=0.5,        # CO2 degassing coefficient
    n_depth=60,         # depth discretisation points
    si_threshold=0.5,   # "high-risk" SI cutoff for zone length
)

# Per-well downhole-condition columns that, when present in a sample row, override
# REF so the simulation uses REAL measured conditions instead of one shared geometry.
PER_WELL_KEYS = ["TVD_ft", "WHP_psi", "BHP_psi", "WHT_C", "BHT_C"]


def config_from_row(row, base=REF):
    """Build a wellbore config from per-well columns when present, else fall back to REF.

    This is the hook for real per-well downhole data: supply any of TVD_ft, WHP_psi,
    BHP_psi, WHT_C, BHT_C per well and SI/kinetics stop being a single shared transform
    of surface chemistry — which is the only way physics can carry orthogonal signal.
    """
    cfg = dict(base)
    for k in PER_WELL_KEYS:
        if k in row and pd.notna(row[k]):
            cfg[k] = float(row[k])
    return cfg


def molalities(sample):
    """Build molality dict (mol/kg water) from a prepared-row Series/dict."""
    return {k: sample[f"m_{k}"] for k in
            ["Na", "Ca", "Mg", "Cl", "SO4", "HCO3"]}


def simulate(sample, cfg=None):
    """
    sample: dict/Series with m_* molalities, pH (measured ~ bottomhole), TDS_ppm.
    Returns dict with depth arrays and summary features.
    """
    if cfg is None:
        cfg = REF
    TVD = cfg["TVD_ft"]
    d = np.linspace(0.0, TVD, cfg["n_depth"])          # wellhead -> bottomhole

    g = (cfg["BHP_psi"] - cfg["WHP_psi"]) / TVD        # psi/ft
    G = (cfg["BHT_C"] - cfg["WHT_C"]) / TVD            # C/ft

    P_psi = cfg["WHP_psi"] + g * d                      # increases with depth
    T_C = cfg["WHT_C"] + G * d
    T_K = T_C + 273.15
    P_bar = P_psi * PSI_TO_BAR
    P_bh = cfg["BHP_psi"]

    # pH rises toward wellhead as CO2 degasses (P drops). Anchored at bottomhole.
    pH = sample["pH"] + cfg["a_degas"] * np.log10(P_bh / P_psi)

    m = molalities(sample)
    m_Ca = m["Ca"]
    m_HCO3 = m["HCO3"]

    # Activity coefficients vary smoothly with T (ionic strength is constant along
    # the well). Compute gamma on an 8-node T grid in ONE Pitzer call each, then
    # interpolate to all depth points. One call reads gCa, gCO3, gHCO3 together.
    T_nodes = np.linspace(T_K.min(), T_K.max(), 8)
    gCa_n, gCO3_n, gHCO3_n = [], [], []
    m_full = {**m, "CO3": 1e-6}        # trace CO3 to read its free-ion gamma
    for Tn in T_nodes:
        gam = activity_coefficients(m_full, T=Tn)
        gCa_n.append(gam.get("Ca", 1.0))
        gCO3_n.append(gam.get("CO3", 0.1))
        gHCO3_n.append(gam.get("HCO3", 0.6))
    gCa_d = np.interp(T_K, T_nodes, gCa_n)
    gCO3_d = np.interp(T_K, T_nodes, gCO3_n)
    gHCO3_d = np.interp(T_K, T_nodes, gHCO3_n)

    SI_cal = np.zeros_like(d)
    SI_arag = np.zeros_like(d)
    for i in range(len(d)):
        Tk = T_K[i]
        aH = 10.0 ** (-pH[i])
        a_CO3 = C.K2(Tk) * gHCO3_d[i] * m_HCO3 / aH
        IAP = (gCa_d[i] * m_Ca) * a_CO3
        logKc = PR.apply(C.logKsp_calcite(Tk), P_bar[i], Tk, "calcite")
        logKa = PR.apply(C.logKsp_aragonite(Tk), P_bar[i], Tk, "aragonite")
        SI_cal[i] = np.log10(IAP) - logKc
        SI_arag[i] = np.log10(IAP) - logKa

    # --- kinetic deposition-rate profile (PWP) along the well -----------------
    # reuse the depth-resolved activities; a(CO2) from carbonate speciation.
    a_H = 10.0 ** (-pH)
    a_HCO3 = gHCO3_d * m_HCO3
    a_CO2 = KIN.co2_activity(a_H, a_HCO3, T_K)
    dep = KIN.deposition_rate(SI_cal, a_H, a_CO2, a_H2O=1.0, T_K=T_K)

    res = _summarise(d, T_C, P_psi, pH, SI_cal, SI_arag, TVD, cfg["si_threshold"])
    res.update(_summarise_kinetics(d, dep))
    res["dep_profile"] = dep
    return res


def _summarise(d, T_C, P_psi, pH, SI_cal, SI_arag, TVD, thr):
    super_mask = SI_cal > 0
    if super_mask.any():
        onset_depth = float(d[super_mask].min())       # shallowest supersaturated TVD
    else:
        onset_depth = float(TVD)                        # no supersaturation -> base
    highrisk_mask = SI_cal > thr
    highrisk_len = float(d[highrisk_mask].max() - d[highrisk_mask].min()) \
        if highrisk_mask.sum() > 1 else 0.0
    dz = d[1] - d[0]
    return {
        "depth_ft": d, "T_C": T_C, "P_psi": P_psi, "pH_profile": pH,
        "SI_calcite": SI_cal, "SI_aragonite": SI_arag,
        # summary features for ML
        "SI_max_cal": float(SI_cal.max()),
        "SI_mean_cal": float(SI_cal.mean()),
        "SI_max_arag": float(SI_arag.max()),
        "onset_depth_ft": onset_depth,
        "supersat_fraction": float(super_mask.mean()),
        "highrisk_zone_ft": highrisk_len,
    }


def _summarise_kinetics(d, dep):
    """Summarise the depth-resolved PWP deposition rate into ML features.

    krate_integral is the depth-integrated rate (proportional to total scale
    deposited per unit area along the well) — the kinetic analogue of SI_max.
    """
    return {
        "krate_max": float(dep.max()),
        "krate_mean": float(dep.mean()),
        "krate_integral": float(np.trapezoid(dep, d)),
        "krate_peak_depth_ft": float(d[int(np.argmax(dep))]),
    }


def simulate_dataframe(df, cfg=None, per_well=False):
    """Run the simulator over every row; return df with appended physics features.

    per_well=True builds each row's config from its per-well columns (PER_WELL_KEYS)
    via config_from_row, falling back to REF for any missing value. This is how real
    measured downhole conditions enter the model. per_well=False keeps the original
    behaviour (one shared cfg/REF for all rows).
    """
    feats = ["SI_max_cal", "SI_mean_cal", "SI_max_arag",
             "onset_depth_ft", "supersat_fraction", "highrisk_zone_ft",
             "krate_max", "krate_mean", "krate_integral", "krate_peak_depth_ft"]
    rows = []
    for _, s in df.iterrows():
        row_cfg = config_from_row(s) if per_well else cfg
        r = simulate(s, row_cfg)
        rows.append({f: r[f] for f in feats})
    out = df.copy().reset_index(drop=True)
    fdf = pd.DataFrame(rows)
    for c in feats:
        out[c] = fdf[c].values
    return out


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("outputs/prepared.parquet")    # one scaling and one non-scaling example
    pos = df[df.Inspection_Result == 1].iloc[0]
    neg = df[df.Inspection_Result == 0].iloc[0]
    for label, s in [("SCALE(1)", pos), ("NOSCALE(0)", neg)]:
        r = simulate(s)
        print(f"\n=== {label}  Ca={s.Ca_ppm} HCO3={s.HCO3_ppm} pH={s.pH} TDS={s.TDS_ppm} ===")
        print(f"  SI_max_cal       = {r['SI_max_cal']:+.3f}")
        print(f"  SI_mean_cal      = {r['SI_mean_cal']:+.3f}")
        print(f"  onset_depth_ft   = {r['onset_depth_ft']:.0f}")
        print(f"  supersat_frac    = {r['supersat_fraction']:.3f}")
        print(f"  highrisk_zone_ft = {r['highrisk_zone_ft']:.0f}")
        print(f"  SI@wellhead={r['SI_calcite'][0]:+.3f}  SI@bottomhole={r['SI_calcite'][-1]:+.3f}")
