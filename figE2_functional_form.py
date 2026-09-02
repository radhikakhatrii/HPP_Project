#!/usr/bin/env python3
"""
figE2_functional_form.py
===============================================================================
[MANUSCRIPT] FigE2_form.pdf -- summary of the 72 functional-form tests across
4 families (change-to-change: 9, quadratic: 9, extreme-decile: 9, effect
modification/interactions: 45). Reads all 4 CSVs directly -- no
re-computation, no hand-entry.

Two panels: (a) a p-value strip per family (every individual test's p, so the
full 72-row granularity IS shown visually even though no single all-72-row
table exists in the ledger yet -- ROUND 6 Item 9 banked family-level counts
only); (b) FDR-significant count per family vs total, with the q<0.05 line
marked, confirming the "0 significant, null generalises" headline visually.

Any prior figure that mislabelled this an "equivalence bound" was a mislabel
(these are ordinary two-sided significance tests against a null of no
effect/no modification, not TOST-style equivalence tests) -- this figure
shows real p-values, q-values, and effect sizes only.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE2_functional_form.py 2>&1 | tee figE2_functional_form.log

Reads: results_cross_lagged/nonlinear_change.csv, nonlinear_quadratic.csv,
       nonlinear_extreme.csv, nonlinear_interactions.csv
       (all produced by nonlinear_interaction_effects.py -- run that first)
Writes: figures/FigE2_form.pdf (+ .png)
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
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = [
    ("nonlinear_change.csv", "Change-to-change\n(Δexposure→Δoutcome)", "p"),
    ("nonlinear_quadratic.csv", "Quadratic term\n(nonlinearity)", "p_sq"),
    ("nonlinear_extreme.csv", "Extreme decile\n(top+bottom 10%)", "p_extreme"),
    ("nonlinear_interactions.csv", "Effect modification\n(interactions)", "p"),
]

_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE2_FORM: 72 functional-form tests, 4 families")
    log("=" * 78)

    loaded = {}
    for fname, label, pcol in FAMILIES:
        path = RESULTS_DIR / fname
        if not path.exists():
            log(f"  SKIPPED family '{label}': {path} not found -- run "
                f"nonlinear_interaction_effects.py first.")
            continue
        df = pd.read_csv(path)
        loaded[label] = (df, pcol)
        n_sig = int((df["q_fdr"] < 0.05).sum())
        log(f"  {label.replace(chr(10), ' ')}: {len(df)} tests, {n_sig} FDR-significant, "
            f"smallest q={df['q_fdr'].min():.3f}")

    if not loaded:
        raise SystemExit("No nonlinear_*.csv files found -- run nonlinear_interaction_effects.py first.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"left": 0.07, "right": 0.97,
                                                "top": 0.85, "bottom": 0.20, "wspace": 0.25})

    # ---- Panel (a): p-value strip, one column per family, one dot per test ----
    for i, (label, (df, pcol)) in enumerate(loaded.items()):
        jitter = (np.random.default_rng(clp.RANDOM_SEED).uniform(-0.15, 0.15, len(df))
                 if len(df) > 1 else np.zeros(len(df)))
        ax1.scatter(np.full(len(df), i) + jitter, df[pcol], s=22, alpha=0.7,
                   color=_COLORS[i % len(_COLORS)], edgecolors="black", linewidths=0.3)
    ax1.axhline(0.05, color="#B0413E", lw=1.0, ls="--", alpha=0.7, label="nominal α=0.05")
    ax1.set_xticks(range(len(loaded)))
    ax1.set_xticklabels(list(loaded.keys()), fontsize=7.5)
    ax1.set_ylabel("Raw p-value", fontsize=9)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("(a) Every individual test's p-value, by family", fontsize=9.5,
                 fontweight="bold", loc="left")

    # ---- Panel (b): FDR-significant count vs total, per family ----
    fam_labels = list(loaded.keys())
    totals = [len(df) for df, _ in loaded.values()]
    sig_counts = [int((df["q_fdr"] < 0.05).sum()) for df, _ in loaded.values()]
    x = np.arange(len(fam_labels))
    ax2.bar(x, totals, color="#E0E0E0", edgecolor="black", linewidth=0.5, label="Total tests")
    ax2.bar(x, sig_counts, color="#B0413E", edgecolor="black", linewidth=0.5,
           label="FDR-significant (q<0.05)")
    for i, (tot, sig) in enumerate(zip(totals, sig_counts)):
        ax2.text(i, tot + 0.8, f"{sig}/{tot}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fam_labels, fontsize=7.5)
    ax2.set_ylabel("Number of tests", fontsize=9)
    ax2.legend(frameon=False, fontsize=7.5)
    total_all = sum(totals)
    sig_all = sum(sig_counts)
    ax2.set_title(f"(b) FDR-significant vs total  ({sig_all}/{total_all} overall)",
                 fontsize=9.5, fontweight="bold", loc="left")

    fig.suptitle("Fig. E2 — 72 Functional-Form Tests: Change, Nonlinearity, Extremes, "
                "Effect Modification  (HPP)",
                fontsize=11, fontweight="bold", fontfamily="serif")
    fig.savefig(FIG_DIR / "FigE2_form.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE2_form.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE2_form.pdf/.png")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
