#!/usr/bin/env python3
"""
fig6_disease_combined.py
===============================================================================
[MANUSCRIPT] Fig6_disease.pdf -- two-panel disease figure: (a) disease-burden
partial correlations (AgeAccel vs 12 curated phenotypes + comorbidity count,
from longitudinal_microbiome_age.py's trait_state_disease.csv -- Round 6 Item
8 confirmed this script UNAFFECTED by any bug found this session), and
(b) incident-disease OR forest, post-EPV>=10 pruning (hypertension/richness,
the paper's old "largest signal", removed entirely -- EPV was exactly 9.0).

Panel (b) reuses cross_lagged_panel.py's own load_incident_diseases() is NOT
used here (that in-module version is diagnostic-only per ROUND 4 Item 3);
instead this reads incident_disease.csv directly, the canonical output of the
STANDALONE incident_disease.py script.

Panel (a)'s exact column names in trait_state_disease.csv were never read at
the code level this session (only inferred from a rendered summary table) --
column matching below is therefore DEFENSIVE (substring search, same idiom
already used throughout this codebase, e.g. sanity_check.py), and the script
logs exactly which column it matched, or skips panel (a) loudly rather than
guessing silently if nothing matches (Rule 4).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 fig6_disease_combined.py 2>&1 | tee fig6_disease_combined.log

Reads: results_longitudinal/trait_state_disease.csv (panel a, best-effort)
       results_cross_lagged/incident_disease.csv (panel b, required)
Writes: figures/Fig6_disease.pdf (+ .png)
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
LONG_DIR = Path("results_longitudinal")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

_SIG_COLOR = "#B0413E"
_NULL_COLOR = "#4C72B0"
_INK = "#2B2B2B"


def _find_col(df: pd.DataFrame, *substrings: str) -> str | None:
    """Case-insensitive substring match, first hit wins -- same defensive
    idiom used throughout this codebase for column-name discovery."""
    for c in df.columns:
        cl = c.lower()
        if all(s in cl for s in substrings):
            return c
    return None


def load_disease_burden() -> pd.DataFrame | None:
    path = LONG_DIR / "trait_state_disease.csv"
    if not path.exists():
        log(f"  Panel (a) SKIPPED: {path} not found. This is longitudinal_microbiome_age.py's "
            f"own output (Round 6 Item 8 confirmed that script unaffected by this session's "
            f"bugs) -- run it if this panel is wanted.")
        return None
    df = pd.read_csv(path)
    outcome_col = _find_col(df, "outcome") or _find_col(df, "phenotype") or df.columns[0]
    r_col = _find_col(df, "r", "mean") or _find_col(df, "r_2visit") or _find_col(df, "r_mean")
    p_col = _find_col(df, "p", "mean") or _find_col(df, "p_2visit") or _find_col(df, "p_mean")
    q_col = _find_col(df, "q") and _find_col(df, "mean")
    n_col = _find_col(df, "n")
    if r_col is None or p_col is None:
        log(f"  Panel (a) SKIPPED: could not confidently match r/p columns in {path}. "
            f"Columns found: {list(df.columns)}. Not guessing further -- fix the substring "
            f"match above once the real column names are known, rather than risk plotting "
            f"the wrong column.")
        return None
    log(f"  Panel (a): matched outcome='{outcome_col}', r='{r_col}', p='{p_col}', "
       f"q='{q_col}', n='{n_col}' in {path}")
    out = pd.DataFrame({
        "outcome": df[outcome_col], "r": df[r_col], "p": df[p_col],
        "n": df[n_col] if n_col else np.nan,
        "q": df[q_col] if q_col else np.nan,
    })
    return out


def load_incident() -> pd.DataFrame:
    path = RESULTS_DIR / "incident_disease.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run incident_disease.py first (the standalone, "
                         f"canonical script -- see MANUSCRIPT_UPDATE_LEDGER.md ROUND 4 Item 3 "
                         f"for why this file, not the in-module diagnostic, is authoritative).")
    df = pd.read_csv(path)
    # T2.1 dedup: remove capitalised duplicate (Abdominal_Adiposity vs abdominal_adiposity)
    # and the pre-filter diabetes definition (diabetes__before_filter vs diabetes)
    before = len(df)
    df = df[df["phenotype"] != "Abdominal_Adiposity"]
    df = df[df["phenotype"] != "diabetes__before_filter"]
    log(f"  [dedup] incident_disease.csv: {before} rows → {len(df)} after dropping "
        f"Abdominal_Adiposity duplicate and diabetes__before_filter (V19 changelog T2.1)")
    return df


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIG6_DISEASE: disease-burden + incident-disease (post-EPV pruning)")
    log("=" * 78)

    burden = load_disease_burden()
    incident = load_incident()
    log(f"  incident_disease.csv: {len(incident)} rows (post-EPV>=10 pruning already applied "
       f"by the script itself)")
    if "hypertension" in incident.get("phenotype", pd.Series(dtype=str)).str.lower().values:
        log("  *** WARNING: hypertension row found in incident_disease.csv -- expected it to "
            "be excluded (EPV=9.0 < 10). Check the script version that produced this file. ***")
    else:
        log("  Confirmed: hypertension is NOT present (excluded by the EPV>=10 filter, as expected).")

    n_panels = 2 if burden is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7),
                             gridspec_kw={"wspace": 0.35, "left": 0.10, "right": 0.96,
                                          "top": 0.88, "bottom": 0.10})
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    if burden is not None:
        ax = axes[ax_idx]; ax_idx += 1
        burden = burden.sort_values("p")
        for i, row in burden.reset_index(drop=True).iterrows():
            sig = pd.notna(row["q"]) and row["q"] < 0.05
            color = _SIG_COLOR if sig else _NULL_COLOR
            ax.barh(i, row["r"], color=color, alpha=0.8, edgecolor=_INK, linewidth=0.4)
            ax.text(row["r"] + (0.002 if row["r"] >= 0 else -0.002), i,
                   f"p={row['p']:.3f}", va="center",
                   ha="left" if row["r"] >= 0 else "right", fontsize=6.5, color=_INK)
        ax.axvline(0, color=_INK, lw=0.8)
        ax.set_yticks(range(len(burden)))
        ax.set_yticklabels(burden["outcome"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Partial r (AgeAccel vs phenotype, adj. age/sex/BMI)", fontsize=8.5)
        ax.set_title("(a) Disease burden (cross-sectional, 2-visit mean)", fontsize=9.5,
                    fontweight="bold", loc="left")

    ax = axes[ax_idx]
    incident = incident.copy()
    p_col = _find_col(incident, "p") or "p"
    or_col = _find_col(incident, "or") or "OR"
    lo_col = _find_col(incident, "ci", "low") or "CI low"
    hi_col = _find_col(incident, "ci", "high") or "CI high"
    q_col = _find_col(incident, "q") or "q (all tests)"
    incident = incident.sort_values(p_col)
    for i, row in incident.reset_index(drop=True).iterrows():
        sig = pd.notna(row.get(q_col, np.nan)) and row[q_col] < 0.05
        color = _SIG_COLOR if sig else _NULL_COLOR
        if lo_col in row and hi_col in row:
            ax.plot([row[lo_col], row[hi_col]], [i, i], color=color, lw=2.0, zorder=3)
        ax.scatter([row[or_col]], [i], color=color, marker="D" if sig else "o",
                  s=55 if sig else 32, zorder=4, edgecolors=_INK, linewidths=0.5)
    ax.axvline(1.0, color=_INK, lw=0.9, ls="--", alpha=0.5)
    label_col = _find_col(incident, "phenotype") or incident.columns[0]
    pred_col = _find_col(incident, "predictor")
    labels = ([f"{r[label_col]} ({r[pred_col]})" for _, r in incident.iterrows()]
             if pred_col else incident[label_col].tolist())
    ax.set_yticks(range(len(incident)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("OR (per SD)", fontsize=8.5)
    ax.set_title("(b) Incident disease, post-EPV≥10 pruning\n(hypertension removed: EPV=9.0)",
                fontsize=9.5, fontweight="bold", loc="left")

    fig.suptitle("Fig. 6 — Disease Burden and Incident Disease  (HPP, corrected pipeline)",
                fontsize=11.5, fontweight="bold", fontfamily="serif")
    fig.savefig(FIG_DIR / "Fig6_disease.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig6_disease.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/Fig6_disease.pdf/.png ({n_panels} panel{'s' if n_panels > 1 else ''})")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
