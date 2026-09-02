#!/usr/bin/env python3
"""
figE3_crosssectional.py
===============================================================================
[MANUSCRIPT] FigE3_crosssectional.pdf -- cross-sectional partial-correlation
heatmap, 3 microbiome measures (Shannon, richness, AgeAccel) x 4 phenotypes
(grip, SBP, waist, BMI), single-visit stratum, age+sex adjusted.

Regenerates the earlier two-row PNG (N_review_figures.py's FIG_R1) as a
vector PDF with the exact manuscript filename, now that the AgeAccel row
exists in crosssectional_matrix.csv (N3_ageaccel.py).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE3_crosssectional.py 2>&1 | tee figE3_crosssectional.log

Reads:  results_cross_lagged/review2/crosssectional_matrix.csv
Writes: figures/FigE3_crosssectional.pdf (+ .png)
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
REVIEW2_DIR = RESULTS_DIR / "review2"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROW_ORDER = ["shannon", "richness", "gut_ageaccel"]
ROW_LABELS = ["Shannon\ndiversity", "Species\nrichness", "Gut AgeAccel"]
COL_ORDER = ["grip", "sbp", "waist", "bmi"]
COL_LABELS = ["Grip\nstrength", "Systolic\nBP", "Waist\ncircumf.", "BMI"]


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE3_CROSSSECTIONAL: partial-r heatmap, 3 measures x 4 phenotypes")
    log("=" * 78)

    path = REVIEW2_DIR / "crosssectional_matrix.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run N3_crosssectional.py / "
                         f"N3_ageaccel.py first; this script only reads their output.")
    df = pd.read_csv(path)
    log(f"  Loaded {len(df)} rows from {path}")
    missing_rows = set(ROW_ORDER) - set(df["microbiome_var"].unique())
    if missing_rows:
        raise SystemExit(f"Missing microbiome_var rows: {missing_rows} -- "
                         f"crosssectional_matrix.csv must carry all 3 measures "
                         f"(run N3_ageaccel.py to add the AgeAccel row).")

    r_mat = np.full((3, 4), np.nan)
    p_mat = np.full((3, 4), np.nan)
    n_mat = np.full((3, 4), np.nan)
    for i, mv in enumerate(ROW_ORDER):
        for j, pv in enumerate(COL_ORDER):
            row = df[(df["microbiome_var"] == mv) & (df["phenotype"] == pv)]
            if len(row):
                r_mat[i, j] = row["partial_r"].iloc[0]
                p_mat[i, j] = row["p"].iloc[0]
                n_mat[i, j] = row["n"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.6),
                           gridspec_kw={"left": 0.22, "right": 0.90, "top": 0.86, "bottom": 0.14})
    vmax = np.nanmax(np.abs(r_mat))
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Partial r (age + sex adjusted)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for i in range(3):
        for j in range(4):
            r, p = r_mat[i, j], p_mat[i, j]
            if np.isnan(r):
                continue
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            text_color = "white" if abs(r) > vmax * 0.55 else "#222222"
            ax.text(j, i, f"{r:+.3f}{sig}\np={p:.2g}", ha="center", va="center",
                    fontsize=7.3, color=text_color, fontfamily="monospace")

    ax.set_xticks(range(4))
    ax.set_xticklabels(COL_LABELS, fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(3))
    ax.set_yticklabels(ROW_LABELS, fontsize=8.5)
    ax.set_title("Extended Data Fig. 3 — Cross-Sectional Partial Correlations\n"
                "(single-visit stratum, n=6,766–6,800, adj. age + sex)",
                fontsize=10, fontweight="bold", fontfamily="serif", pad=42)

    fig.text(0.02, 0.02, "* p<0.05  ** p<0.01.  Source: crosssectional_matrix.csv "
             "(N3_crosssectional.py + N3_ageaccel.py)", fontsize=6.2, color="#555555", style="italic")

    fig.savefig(FIG_DIR / "FigE3_crosssectional.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE3_crosssectional.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE3_crosssectional.pdf/.png")

    log("\n  Verification (key values):")
    for i, mv in enumerate(ROW_ORDER):
        for j, pv in enumerate(COL_ORDER):
            log(f"    {mv:<13} x {pv:<6}  r={r_mat[i,j]:+.4f}  p={p_mat[i,j]:.3g}  n={int(n_mat[i,j]) if not np.isnan(n_mat[i,j]) else 'NA'}")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
