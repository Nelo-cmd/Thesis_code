"""
Exploratory figures for the physics layer.

Produces two PNGs in figs/ that visualise what the simulator does on your data:

  fig1_wellbore_profiles.png
      Depth-resolved P, T, pH and SI profiles for one scaling and one
      non-scaling sample, plotted side-by-side as a wellbore log
      (depth on the y-axis, inverted so wellhead is at the top).

  fig2_si_distribution.png
      Histogram of SI_max by inspection class. Visually demonstrates the
      thermodynamic-paradox finding: the two distributions overlap heavily,
      which is why a thermodynamic-only model cannot classify the labels.

Run from inside thesis_code/:  python figs/explore_physics.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from physics.wellbore import simulate

# ---- styling (kept restrained, no rainbow palette) -------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
COL_SCALE = "#b03a2e"        # muted brick red    (positive class)
COL_NOSCALE = "#1f4e79"      # deep navy blue     (negative class)
COL_THR = "#7f8c8d"          # grey for thresholds

os.makedirs("figs", exist_ok=True)


def figure_1_profiles():
    """Side-by-side wellbore log: one scale, one no-scale sample."""
    df = pd.read_parquet("outputs/prepared.parquet")
    pos = df[df.Inspection_Result == 1].iloc[0]
    neg = df[df.Inspection_Result == 0].iloc[0]

    rp = simulate(pos)
    rn = simulate(neg)

    fig, axes = plt.subplots(1, 4, figsize=(11, 6), sharey=True)

    panels = [
        ("Pressure (psi)", "P_psi"),
        ("Temperature (°C)", "T_C"),
        ("pH", "pH_profile"),
        ("Saturation Index (calcite)", "SI_calcite"),
    ]
    for ax, (label, key) in zip(axes, panels):
        ax.plot(rp[key], rp["depth_ft"], color=COL_SCALE, lw=1.8,
                label="scale (positive)")
        ax.plot(rn[key], rn["depth_ft"], color=COL_NOSCALE, lw=1.8,
                label="no scale (negative)")
        ax.set_xlabel(label)
        if key == "SI_calcite":
            ax.axvline(0, color=COL_THR, ls="--", lw=0.8, label="SI = 0")

    axes[0].set_ylabel("Depth from wellhead (ft)")
    axes[0].invert_yaxis()             # wellhead at top, bottomhole at bottom
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[3].legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("Wellbore profiles — one scaling vs one non-scaling sample",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = "figs/fig1_wellbore_profiles.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def figure_2_si_distribution():
    """SI_max histogram split by inspection class."""
    df = pd.read_parquet("outputs/with_physics.parquet")
    pos = df[df.Inspection_Result == 1]["SI_max_cal"]
    neg = df[df.Inspection_Result == 0]["SI_max_cal"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(min(df.SI_max_cal.min(), 0),
                       df.SI_max_cal.max() + 0.1, 30)
    ax.hist(neg, bins=bins, color=COL_NOSCALE, alpha=0.55,
            label=f"no scale  (n = {len(neg)})", edgecolor="white", lw=0.4)
    ax.hist(pos, bins=bins, color=COL_SCALE, alpha=0.55,
            label=f"scale       (n = {len(pos)})", edgecolor="white", lw=0.4)
    ax.axvline(0, color=COL_THR, ls="--", lw=0.8)
    ax.text(0.02, ax.get_ylim()[1] * 0.97, "SI = 0\n(saturation)",
            color=COL_THR, fontsize=8, va="top")
    ax.set_xlabel("Maximum calcite saturation index along the wellbore")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of SI$_{max}$ by inspection class")
    ax.legend(loc="upper right", frameon=False)

    mu_pos, mu_neg = pos.mean(), neg.mean()
    ax.annotate(f"means:  scale = {mu_pos:+.2f}    no scale = {mu_neg:+.2f}",
                xy=(0.5, -0.20), xycoords="axes fraction",
                ha="center", fontsize=9, color="#444444")

    fig.tight_layout()
    out = "figs/fig2_si_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    figure_1_profiles()
    figure_2_si_distribution()