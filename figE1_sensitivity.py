#!/usr/bin/env python3
"""
figE1_sensitivity.py
===============================================================================
[MANUSCRIPT] FigE1_sensitivity.pdf -- confounder-sensitivity dumbbell plot,
9 Path A tests, baseline (age+sex) vs fully-adjusted (+diet+smoking+alcohol+
activity+sequencing-depth+medication). 95% CIs shown are DERIVED here as
beta_std +/- 1.96*se from the two columns confounder_sensitivity.csv already
carries (beta_std, se) -- a standard, correct transformation, not a new
statistic and not fabricated.

Any prior figure that may have labelled this comparison an "equivalence
bound" was a mislabel: this is a sensitivity/confounding check, not an
equivalence test. This figure shows plain CIs and p-values only; the actual
TOST equivalence result lives in Fig5_bounds.pdf / FigS_equivalence_all18.pdf,
sourced from cross_lag_equivalence_bounds.csv, not duplicated here.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE1_sensitivity.py 2>&1 | tee figE1_sensitivity.log

Reads: results_cross_lagged/confounder_sensitivity.csv (produced by
       confounder_sensitivity.py -- run that first if missing)
Writes: figures/FigE1_sensitivity.pdf (+ .png)
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

_BASE_COLOR = "#9E9E9E"
_FULL_COLOR = "#4C72B0"
_INK = "#2B2B2B"


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE1_SENSITIVITY: baseline vs fully-adjusted, 9 Path A tests")
    log("=" * 78)

    path = RESULTS_DIR / "confounder_sensitivity.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run confounder_sensitivity.py first; "
                         f"this script only reads its output, it does not recompute anything.")
    df = pd.read_csv(path)
    df["ci_lo"] = df["beta_std"] - 1.96 * df["se"]
    df["ci_hi"] = df["beta_std"] + 1.96 * df["se"]

    tests = df[["microbiome_var", "outcome"]].drop_duplicates().values.tolist()
    n_tests = len(tests)
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, n_tests * 0.6)),
                           gridspec_kw={"left": 0.32, "right": 0.92, "top": 0.90, "bottom": 0.12})
    ax.axvline(0, color=_INK, lw=0.9, ls="--", alpha=0.4, zorder=1)

    labels = []
    for i, (mv, oc) in enumerate(tests):
        sub = df[(df["microbiome_var"] == mv) & (df["outcome"] == oc)]
        base = sub[sub["model"] == "baseline"]
        full = sub[sub["model"] == "fully_adjusted"]
        for rows, color, y_off, model_lbl in [(base, _BASE_COLOR, +0.13, "age+sex"),
                                              (full, _FULL_COLOR, -0.13, "fully adj.")]:
            if rows.empty:
                continue
            r = rows.iloc[0]
            y = i + y_off
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, lw=2.2,
                   solid_capstyle="round", zorder=3)
            sig = r["p"] < 0.05
            mk = "D" if sig else "o"
            ax.scatter([r["beta_std"]], [y], color=color, marker=mk,
                     s=60 if sig else 34, zorder=4, edgecolors=_INK, linewidths=0.5)
        labels.append(f"{mv} × {oc}")

    ax.set_yticks(range(n_tests))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised β", fontsize=9)
    handles = [
        plt.Line2D([0], [0], color=_BASE_COLOR, lw=2.2, label="Baseline (age+sex)"),
        plt.Line2D([0], [0], color=_FULL_COLOR, lw=2.2, label="Fully adjusted (+diet/smoking/"
                  "alcohol/activity/depth/meds)"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=_INK, markeredgecolor=_INK,
                  label="p<0.05", markersize=6),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Fig. E1 — Confounder Sensitivity: Baseline vs Fully-Adjusted  (HPP, 9 Path A tests)",
                fontsize=10.5, fontweight="bold", fontfamily="serif")
    fig.savefig(FIG_DIR / "FigE1_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE1_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE1_sensitivity.pdf/.png")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
