#!/usr/bin/env python3
"""
figE4_pathA_forest.py
===============================================================================
[MANUSCRIPT] FigE4_pathA_forest.pdf -- 9 Path-A (microbiome(t1) -> phenotype(t2))
standardised beta + 95% CI, vs the +/-0.10 pre-registered equivalence margin.
Flags the two nominally-significant tests (richness->grip, gut_ageaccel->waist)
per N5_null_enrichment_pathB.py.

Regenerates the earlier N_review_figures.py FIG_R2 PNG as a vector PDF with
the exact manuscript filename.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE4_pathA_forest.py 2>&1 | tee figE4_pathA_forest.log

Reads:  results_cross_lagged/review2/N5_pathA_null_check.csv
Writes: figures/FigE4_pathA_forest.pdf (+ .png)
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
REVIEW2_DIR = RESULTS_DIR / "review2"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EQUIV = 0.10
_SIG_COLOR = "#B0413E"
_NULL_COLOR = "#4C72B0"
_INK = "#2B2B2B"


def _find_col(df: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE4_PATHA_FOREST: 9 Path-A tests vs +/-0.10 equivalence margin")
    log("=" * 78)

    path = REVIEW2_DIR / "N5_pathA_null_check.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run N5_null_enrichment_pathB.py first.")
    df = pd.read_csv(path)
    log(f"  Loaded {len(df)} Path-A rows from {path}")

    exp_col = next(c for c in df.columns if "expos" in c.lower())
    out_col = next(c for c in df.columns if "outcom" in c.lower())
    beta_col = next(c for c in df.columns if c.lower() in ("beta", "coef", "coefficient"))
    p_col = next(c for c in df.columns if c.lower() in ("p", "pval", "p_value"))
    q_col = _find_col(df, "q_bh_n5") or next((c for c in df.columns if "fdr" in c.lower()), None)

    df = df.sort_values(p_col).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, max(4.0, len(df) * 0.55)),
                           gridspec_kw={"left": 0.30, "right": 0.93, "top": 0.88, "bottom": 0.12})

    ax.axvspan(-EQUIV, EQUIV, color="#DDDDDD", alpha=0.5, zorder=0,
              label=f"Equivalence margin (±{EQUIV})")
    ax.axvline(0, color=_INK, lw=0.9, ls="--", alpha=0.5, zorder=1)

    labels = []
    for i, row in df.iterrows():
        sig = row[p_col] < 0.05
        color = _SIG_COLOR if sig else _NULL_COLOR
        ax.plot([row["ci_lo_95"], row["ci_hi_95"]], [i, i], color=color, lw=2.2,
               solid_capstyle="round", zorder=3)
        ax.scatter([row[beta_col]], [i], color=color, marker="D" if sig else "o",
                  s=65 if sig else 38, zorder=4, edgecolors=_INK, linewidths=0.5)
        p_txt = f"P={row[p_col]:.3f}" + (" *" if sig else "")
        x_anno = max(row["ci_hi_95"], EQUIV) + 0.006
        ax.text(x_anno, i, p_txt, va="center", fontsize=7, color=_INK)
        labels.append(f"{row[exp_col]} → {row[out_col]}")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised β (95% CI)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Extended Data Fig. 4 — Path A: Microbiome(t1) → Phenotype(t2)\n"
                "9 tests vs pre-registered ±0.10 equivalence margin",
                fontsize=10.5, fontweight="bold", fontfamily="serif")
    fig.text(0.02, 0.01, "Source: N5_pathA_null_check.csv (N5_null_enrichment_pathB.py). "
             "* nominally significant, p<0.05 (none FDR-significant).",
             fontsize=6.2, color="#555555", style="italic")

    fig.savefig(FIG_DIR / "FigE4_pathA_forest.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE4_pathA_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE4_pathA_forest.pdf/.png")

    log("\n  Verification (key values):")
    for _, row in df.iterrows():
        log(f"    {row[exp_col]} -> {row[out_col]}: beta={row[beta_col]:+.4f} "
            f"p={row[p_col]:.3g} q_BH={row[q_col]:.3g}" if q_col else "")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
