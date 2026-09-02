#!/usr/bin/env python3
"""
figE5_enrichment.py
===============================================================================
[MANUSCRIPT] FigE5_enrichment.pdf -- (left) histogram of all 18 cross-lagged
P-values, split by direction (Path A / Path B); (right) QQ plot of the 18
P-values vs a uniform null, with the waist_circumference -> microbiome
Path-B points highlighted. Annotates pi0~=0.67 and the Path-A enrichment
binomial P=0.071 (both from N5_null_enrichment_pathB.py, Path-A family).

Regenerates the earlier N_review_figures.py FIG_R3 PNG as a vector PDF with
the exact manuscript filename.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE5_enrichment.py 2>&1 | tee figE5_enrichment.log

Reads:  results_cross_lagged/cross_lag_all_paths.csv (all 18 tests, primary spec)
        results_cross_lagged/review2/N5_pvalue_enrichment.csv (pi0, binomial P)
Writes: figures/FigE5_enrichment.pdf (+ .png)
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

_PATH_A = "#4C72B0"
_PATH_B = "#C44E52"
_INK = "#2B2B2B"


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE5_ENRICHMENT: 18-test p-value histogram + QQ vs uniform")
    log("=" * 78)

    path = RESULTS_DIR / "cross_lag_all_paths.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run cross_lagged_panel.py first.")
    raw = pd.read_csv(path)

    spec_col = next((c for c in raw.columns if "spec" in c.lower() or "label" in c.lower()), None)
    if spec_col:
        primary_val = next((v for v in raw[spec_col].value_counts().index
                            if "primary" in str(v).lower() or "base" in str(v).lower()),
                           raw[spec_col].value_counts().index[0])
        df = raw[raw[spec_col] == primary_val].copy()
    else:
        df = raw.copy()
    log(f"  Loaded {len(df)} rows (primary spec)")

    dir_col = next(c for c in df.columns if c.lower() == "direction")
    p_col = next(c for c in df.columns if c.lower() in ("p", "pval", "p_value"))
    exp_col = next(c for c in df.columns if "expos" in c.lower())
    out_col = next(c for c in df.columns if "outcom" in c.lower())

    pA = df[df[dir_col] == "A"][p_col].dropna().values
    pB = df[df[dir_col] == "B"][p_col].dropna().values
    p_all = np.concatenate([pA, pB])
    log(f"  Path A: {len(pA)} tests, Path B: {len(pB)} tests, total {len(p_all)}")

    enrich_path = REVIEW2_DIR / "N5_pvalue_enrichment.csv"
    pi0 = binom_p = None
    if enrich_path.exists():
        enrich = pd.read_csv(enrich_path).iloc[0]
        pi0 = float(enrich["storey_pi0"])
        binom_p = float(enrich["binomial_p_enrichment"])
        log(f"  Path-A enrichment: pi0={pi0:.3f}, binomial P={binom_p:.4f} "
            f"(source: {enrich_path})")
    else:
        log(f"  {enrich_path} not found -- pi0/binomial annotation skipped "
            f"(run N5_null_enrichment_pathB.py first)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"left": 0.08, "right": 0.97,
                                                "top": 0.85, "bottom": 0.14, "wspace": 0.27})

    # ---- Left: histogram of all 18 p-values, stacked by direction ----
    bins = np.linspace(0, 1, 11)
    ax1.hist([pA, pB], bins=bins, stacked=True, color=[_PATH_A, _PATH_B],
             edgecolor=_INK, linewidth=0.4,
             label=[f"Path A (n={len(pA)})", f"Path B (n={len(pB)})"])
    ax1.axhline(len(p_all) / 10, color=_INK, lw=1.0, ls="--", alpha=0.5,
               label="uniform-null expectation")
    ax1.set_xlabel("P-value", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.set_title("(a) Distribution of all 18 cross-lagged P-values", fontsize=9.5,
                 fontweight="bold", loc="left")
    if pi0 is not None:
        ax1.annotate(f"Path-A π₀≈{pi0:.2f}\nbinomial P={binom_p:.3f}",
                    xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=7.2,
                    color=_INK, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=_INK, lw=0.5, alpha=0.9))

    # ---- Right: QQ plot vs uniform, waist->microbiome highlighted ----
    p_sorted = np.sort(p_all)
    n = len(p_sorted)
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(np.clip(p_sorted, 1e-12, 1))

    is_waist = np.array([
        ("waist" in str(df.loc[df[p_col] == pv, exp_col].iloc[0]).lower()
         or "waist" in str(df.loc[df[p_col] == pv, out_col].iloc[0]).lower())
        for pv in p_sorted
    ])

    ax2.plot([0, expected.max() * 1.05], [0, expected.max() * 1.05], color=_INK, lw=0.9,
            ls="--", alpha=0.5, label="y = x (uniform null)")
    ax2.scatter(expected[~is_waist], observed[~is_waist], color=_PATH_A, s=32,
               edgecolors=_INK, linewidths=0.4, zorder=3, label="other tests")
    if is_waist.any():
        ax2.scatter(expected[is_waist], observed[is_waist], color=_PATH_B, s=55,
                   marker="D", edgecolors=_INK, linewidths=0.5, zorder=4,
                   label="waist_circumference ↔ microbiome")
    ax2.set_xlabel("Expected −log₁₀(P)", fontsize=9)
    ax2.set_ylabel("Observed −log₁₀(P)", fontsize=9)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2.set_title("(b) QQ plot vs uniform (all 18 tests)", fontsize=9.5,
                 fontweight="bold", loc="left")

    fig.suptitle("Extended Data Fig. 5 — P-Value Enrichment Across All 18 Cross-Lagged Tests",
                fontsize=11, fontweight="bold", fontfamily="serif")
    fig.text(0.5, 0.01, "Source: cross_lag_all_paths.csv (primary spec) + "
             "N5_pvalue_enrichment.csv (Path-A π₀, binomial test)",
             ha="center", fontsize=6.4, color="#555555", style="italic")

    fig.savefig(FIG_DIR / "FigE5_enrichment.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE5_enrichment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE5_enrichment.pdf/.png")
    log(f"\n  Verification: 18 tests total (A={len(pA)}, B={len(pB)}), "
        f"pi0={pi0}, binomial P={binom_p}")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
