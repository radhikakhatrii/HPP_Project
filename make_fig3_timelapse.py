#!/usr/bin/env python3
"""
Fig 3 - Between-visit movement in the microbiome measures and functional phenotypes.

Regenerates Fig3_timelapse.pdf using the OBSERVED quintile-to-quintile transitions
(visit 1 -> visit 2), replacing the earlier model-implied (bivariate-normal) version.

WHAT YOU NEED
-------------
A single per-participant CSV with one row per participant and paired visit-1 / visit-2
columns for the six measures below. Nothing leaves this script except the finished PDF
and a small text summary of retention percentages - no participant-level data is written out.

Edit COLS below to match your column names, then run:

    python make_fig3_timelapse.py  --csv paired_visits_full.csv  --out Fig3_timelapse.pdf

Only pandas, numpy and matplotlib are required.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# ----------------------------------------------------------------------------- #
# 1. COLUMN MAP  --  change the right-hand strings to your actual column names.
#    (v1 = first visit, v2 = second visit.)
# ----------------------------------------------------------------------------- #
COLS = {
    # panel a: microbiome ageing measures
    "AgeAccel":          ("age_accel_v1",  "age_accel_v2"),
    "Shannon diversity": ("shannon_v1",    "shannon_v2"),
    "Species richness":  ("richness_v1",   "richness_v2"),
    # panel b: functional phenotypes
    "Grip strength":              ("grip_v1",  "grip_v2"),
    "Systolic blood pressure":    ("sbp_v1",   "sbp_v2"),
    "Waist circumference":        ("waist_v1", "waist_v2"),
}

PANEL_A = ["AgeAccel", "Shannon diversity", "Species richness"]
PANEL_B = ["Grip strength", "Systolic blood pressure", "Waist circumference"]

# colour per measure (edit to taste / to match the other figures)
COLORS = {
    "AgeAccel":                 "#D28B47",  # instability orange
    "Shannon diversity":        "#4385C6",  # persistent blue
    "Species richness":         "#2F6FA5",  # estimate blue
    "Grip strength":            "#5B8C5A",
    "Systolic blood pressure":  "#B5651D",
    "Waist circumference":      "#8C6BB1",
}

N_Q = 5  # quintiles


# ----------------------------------------------------------------------------- #
# 2. Quintile transition matrix (observed)
# ----------------------------------------------------------------------------- #
def transition_matrix(v1, v2, n_q=N_Q):
    """Return (matrix, retention_pct, n) for paired vectors v1, v2.

    matrix[i, j] = proportion of all valid pairs starting in quintile i at
    visit 1 and ending in quintile j at visit 2 (rows are visit-1 quintiles).
    retention_pct = percentage on the diagonal (same quintile at both visits).
    """
    df = pd.DataFrame({"v1": v1, "v2": v2}).dropna()
    n = len(df)
    if n == 0:
        raise ValueError("No complete pairs after dropping missing values.")
    # rank-based quintiles, each visit scored on its own distribution
    q1 = pd.qcut(df["v1"].rank(method="first"), n_q, labels=False)
    q2 = pd.qcut(df["v2"].rank(method="first"), n_q, labels=False)
    mat = np.zeros((n_q, n_q))
    for i, j in zip(q1, q2):
        mat[int(i), int(j)] += 1
    mat /= mat.sum()
    retention_pct = 100.0 * np.trace(mat)
    return mat, retention_pct, n


# ----------------------------------------------------------------------------- #
# 3. Alluvial drawing
# ----------------------------------------------------------------------------- #
def _ribbon(ax, y0_bottom, y0_top, y1_bottom, y1_top, x0, x1, color, alpha):
    """Draw one curved ribbon between a visit-1 block and a visit-2 block."""
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0_bottom),
        (xm, y0_bottom), (xm, y1_bottom), (x1, y1_bottom),   # bottom edge (cubic)
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),            # top edge (cubic)
        (x0, y0_bottom),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=color,
                                   edgecolor="none", alpha=alpha, lw=0))


def draw_alluvial(ax, mat, retention_pct, color, title):
    """Draw a quintile flow (visit 1 left -> visit 2 right) on ax."""
    n_q = mat.shape[0]
    gap = 0.015                      # gap between stacked quintile blocks
    x0, x1 = 0.10, 0.90
    bar_w = 0.045

    row_tot = mat.sum(axis=1)        # visit-1 quintile sizes (~0.2 each)
    col_tot = mat.sum(axis=0)        # visit-2 quintile sizes

    # y positions of each block, stacked top (Q5) to bottom (Q1)
    def block_edges(totals):
        heights = totals * (1.0 - gap * (n_q - 1))
        edges = []
        y = 1.0
        for h in heights[::-1]:      # top-down
            edges.append((y - h, y)) # (bottom, top)
            y -= h + gap
        return edges[::-1]           # back to Q1..Q5 order

    left = block_edges(row_tot)
    right = block_edges(col_tot)

    # running fill cursors within each block
    left_cursor = [b for (b, t) in left]
    right_cursor = [b for (b, t) in right]

    # draw ribbons, heaviest (diagonal) last so they sit on top
    order = sorted([(i, j) for i in range(n_q) for j in range(n_q)],
                   key=lambda ij: mat[ij[0], ij[1]])
    for i, j in order:
        w = mat[i, j]
        if w <= 0:
            continue
        lb = left_cursor[i]; left_cursor[i] += w
        rb = right_cursor[j]; right_cursor[j] += w
        alpha = 0.85 if i == j else 0.25
        _ribbon(ax, lb, lb + w, rb, rb + w, x0 + bar_w, x1 - bar_w, color, alpha)

    # draw the quintile blocks
    for (b, t) in left:
        ax.add_patch(patches.Rectangle((x0, b), bar_w, t - b,
                                       facecolor=color, edgecolor="none"))
    for (b, t) in right:
        ax.add_patch(patches.Rectangle((x1 - bar_w, b), bar_w, t - b,
                                       facecolor=color, edgecolor="none"))

    ax.text(x0 + bar_w / 2, -0.03, "visit 1", ha="center", va="top", fontsize=7)
    ax.text(x1 - bar_w / 2, -0.03, "visit 2", ha="center", va="top", fontsize=7)
    ax.set_title(f"{title}\nsame-quintile: {retention_pct:.0f}%", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1.02)
    ax.axis("off")


# ----------------------------------------------------------------------------- #
# 4. Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-participant paired CSV")
    ap.add_argument("--out", default="Fig3_timelapse.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
    summary_lines = []

    for row, (panel, measures, tag) in enumerate(
        [("a", PANEL_A, "microbiome ageing measures"),
         ("b", PANEL_B, "functional phenotypes")]):
        for col, measure in enumerate(measures):
            c1, c2 = COLS[measure]
            missing = [c for c in (c1, c2) if c not in df.columns]
            if missing:
                raise KeyError(f"Missing column(s) {missing} for '{measure}'. "
                               f"Edit COLS at the top of the script.")
            mat, ret, n = transition_matrix(df[c1].values, df[c2].values)
            draw_alluvial(axes[row, col], mat, ret, COLORS[measure], measure)
            summary_lines.append(f"{panel}. {measure:<26s} n={n:5d}  "
                                 f"same-quintile={ret:.1f}%")
        # panel letter
        axes[row, 0].text(-0.02, 1.06, panel, transform=axes[row, 0].transAxes,
                          fontsize=13, fontweight="bold", va="bottom")

    fig.suptitle("Between-visit quintile movement (observed), median 2.88-year interval",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

    print("Saved:", args.out)
    print("\nRetention (same-quintile at both visits) - use these in the caption:")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
