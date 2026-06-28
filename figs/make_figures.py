"""
Publication figures for Chapter 4 (the verified results).

  fig1_phreeqc_validation.png  — our calcite SI vs PHREEQC (r=0.997, constant offset)
  fig2_si_overlap.png          — SI_max distributions overlap across classes
                                 (the "necessary but not sufficient" result)
  fig3_simpson_paradox.png     — Ca/TDS relates to scaling OPPOSITELY by field
  fig4_physics_value.png       — AUC by feature set x regime (physics adds ~nothing)

Run from repo root:  python figs/make_figures.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figs")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})
BLUE, CRIMSON, GRAY = "#3b6ea5", "#a5343b", "#8a8a8a"
WP = os.path.join(ROOT, "outputs", "with_physics.parquet")


def fig1_phreeqc():
    v = pd.read_parquet(os.path.join(ROOT, "outputs", "phreeqc_validation.parquet")).dropna()
    x, y = v.SI_phreeqc.values, v.SI_ours.values
    r = np.corrcoef(x, y)[0, 1]; d = y - x
    slope, inter = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = [min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2]
    ax.plot(lim, lim, "--", color=GRAY, lw=1, label="1:1")
    ax.plot(lim, [slope * l + inter for l in lim], color=CRIMSON, lw=1.3,
            label=f"fit: y={slope:.3f}x{inter:+.3f}")
    ax.scatter(x, y, s=12, alpha=0.45, color=BLUE, edgecolor="none")
    ax.set(xlabel="calcite SI — PHREEQC (pitzer.dat)", ylabel="calcite SI — this work",
           title="SI engine validation vs PHREEQC", xlim=lim, ylim=lim)
    ax.text(0.04, 0.96, f"r = {r:.3f}\nbias = {d.mean():+.3f}\n"
            f"bias-corr. MAE = {np.abs(d - d.mean()).mean():.3f}\nn = {len(v)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig1_phreeqc_validation.png")); plt.close(fig)
    print("  fig1_phreeqc_validation.png")


def fig2_si_overlap():
    df = pd.read_parquet(WP)
    s1 = df[df.Inspection_Result == 1].SI_max_cal
    s0 = df[df.Inspection_Result == 0].SI_max_cal
    bins = np.linspace(df.SI_max_cal.min(), df.SI_max_cal.max(), 28)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(s1, bins, density=True, alpha=0.55, color=CRIMSON, label=f"scale (n={len(s1)})")
    ax.hist(s0, bins, density=True, alpha=0.55, color=BLUE, label=f"no scale (n={len(s0)})")
    ax.axvline(s1.median(), color=CRIMSON, lw=1.4, ls="--")
    ax.axvline(s0.median(), color=BLUE, lw=1.4, ls="--")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.annotate("saturation (SI=0)", xy=(0, ax.get_ylim()[1] * 0.9), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color="k")
    ax.set(xlabel="maximum calcite SI along wellbore", ylabel="density",
           title="SI distributions overlap across classes\n(SI is necessary but not sufficient)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig2_si_overlap.png")); plt.close(fig)
    print("  fig2_si_overlap.png")


def fig3_simpson():
    df = pd.read_parquet(WP)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for fld, color in [("A", CRIMSON), ("B", BLUE)]:
        sub = df[df.Field == fld]
        r = np.corrcoef(sub.Ca_TDS_norm, sub.Inspection_Result)[0, 1]
        q = pd.qcut(sub.Ca_TDS_norm, 5, duplicates="drop")
        rate = sub.groupby(q, observed=True).Inspection_Result.mean()
        cen = sub.groupby(q, observed=True).Ca_TDS_norm.mean()
        ax.plot(cen, rate, "o-", color=color, lw=1.6, ms=6,
                label=f"Field {fld}  (corr {r:+.2f}, pos rate {sub.Inspection_Result.mean():.2f})")
    rp = np.corrcoef(df.Ca_TDS_norm, df.Inspection_Result)[0, 1]
    ax.set(xlabel="Ca / TDS  (binned, per field)", ylabel="observed scaling rate",
           title=f"Simpson's paradox: Ca/TDS predicts scaling OPPOSITELY by field\n"
                 f"(pooled corr {rp:+.2f} masks the within-field reversal)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig3_simpson_paradox.png")); plt.close(fig)
    print("  fig3_simpson_paradox.png")


def fig4_physics_value():
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    df = pd.read_parquet(WP).reset_index(drop=True)
    y = df.Inspection_Result.values.astype(int)
    grp = (df.Field.astype(str) + "_" + df.Well_Number.astype(str)).values
    CHEM = ["Na_ppm", "Ca_ppm", "Mg_ppm", "SO4_ppm", "HCO3_ppm", "TDS_ppm", "pH",
            "Cl_ppm", "ionic_strength", "Ca_HCO3_ratio", "Mg_Ca_ratio", "SO4_Ca_ratio", "Ca_TDS_norm"]
    SETS = {"chemistry": CHEM, "+SI": CHEM + ["SI_max_cal", "SI_mean_cal"],
            "+kinetics": CHEM + ["SI_max_cal", "SI_mean_cal", "krate_max", "krate_mean", "krate_integral"]}

    def vf(folds, yy):
        return [(t, e) for t, e in folds if len(np.unique(yy[e])) == 2]

    def ev(frame, feats, yy, folds):
        X = frame[feats].values; a = []
        for tr, te in folds:
            spw = (yy[tr] == 0).sum() / max(1, (yy[tr] == 1).sum())
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric="logloss", random_state=0, scale_pos_weight=spw)
            m.fit(X[tr], yy[tr]); a.append(roc_auc_score(yy[te], m.predict_proba(X[te])[:, 1]))
        return np.array(a)

    pooled = vf(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(df, y)), y)
    grouped = vf(sum([list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + r).split(df, y, grp))
                      for r in range(10)], []), y)
    dfB = df[df.Field == "B"].reset_index(drop=True); yB = dfB.Inspection_Result.values.astype(int)
    grpB = dfB.Well_Number.values
    withinB = vf(sum([list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + r).split(dfB, yB, grpB))
                      for r in range(10)], []), yB)
    regimes = [("pooled\n(field-confounded)", df, y, pooled),
               ("grouped\n(leakage-free)", df, y, grouped),
               ("within Field B\n(decisive)", dfB, yB, withinB)]

    means = {s: [] for s in SETS}; stds = {s: [] for s in SETS}
    for _, frame, yy, folds in regimes:
        for s, feats in SETS.items():
            a = ev(frame, feats, yy, folds); means[s].append(a.mean()); stds[s].append(a.std())

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xpos = np.arange(len(regimes)); w = 0.26
    colors = {"chemistry": GRAY, "+SI": BLUE, "+kinetics": CRIMSON}
    for i, s in enumerate(SETS):
        ax.bar(xpos + (i - 1) * w, means[s], w, yerr=stds[s], capsize=3,
               color=colors[s], label=s, alpha=0.9)
    ax.axhline(0.5, color="k", lw=1, ls=":", label="chance")
    ax.set_xticks(xpos); ax.set_xticklabels([r[0] for r in regimes])
    ax.set(ylabel="ROC-AUC (XGBoost, mean ± std)", ylim=(0.45, 0.9),
           title="Physics adds no measurable value over chemistry\n(decisive test = within Field B)")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(os.path.join(FIG, "fig4_physics_value.png")); plt.close(fig)
    print("  fig4_physics_value.png")


if __name__ == "__main__":
    print("writing figures to figs/ ...")
    fig1_phreeqc(); fig2_si_overlap(); fig3_simpson(); fig4_physics_value()
    print("done.")
