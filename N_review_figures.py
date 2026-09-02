#!/usr/bin/env python3
"""
N_review_figures.py
===============================================================================
Three figures for the Nastya review-round response letter.

  FIG_R1_crosssectional_heatmap.png
      Heatmap of partial r (controlling age+sex) between microbiome measures
      (Shannon, richness) and phenotypes (grip, SBP, waist, BMI) in the
      single-visit stratum (n ≈ 6,754–6,788).  Source: N3 results.

  FIG_R2_pathA_forest.png
      Forest plot of all 9 Path-A cross-lagged estimates with 95% CI and
      90% CI, overlaid with the ±0.10 SD equivalence bounds. Indicates which
      tests are nominally significant (p<0.05) and whether any survive BH.
      Source: N5 results / cross_lag_all_paths.csv.

  FIG_R3_pvalue_dist.png
      Distribution of all 18 cross-lagged p-values (Path-A + Path-B) against
      the uniform null.  Highlights the three waist → microbiome Path-B tests.
      Source: cross_lag_all_paths.csv.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python N_review_figures.py 2>&1 | tee N_review_figures.log

Writes PNGs into results_cross_lagged/review2/.
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import uniform as sp_uniform

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

RESULTS = Path(clp.RESULTS_DIR)
OUT     = RESULTS / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log     = clp.log

EQUIV   = 0.10   # ±0.10 SD equivalence margin


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


# ──────────────────────────────────────────────────────────────────────────────
def fig_crosssectional_heatmap() -> None:
    banner("FIG_R1: Cross-sectional partial correlation heatmap (N3)")

    csv = OUT / "crosssectional_matrix.csv"
    if not csv.exists():
        log(f"  {csv} not found — run N3_crosssectional.py first"); return

    df = pd.read_csv(csv)
    log(f"  Loaded {len(df)} rows from {csv.name}")

    # Pivot to matrix
    micro_order = ["shannon", "richness"]
    pheno_order = ["grip", "sbp", "waist", "bmi"]
    micro_labels = {"shannon": "Shannon\ndiversity", "richness": "Species\nrichness"}
    pheno_labels = {"grip": "Grip\n(kg)", "sbp": "SBP\n(mmHg)",
                    "waist": "Waist\n(cm)", "bmi": "BMI\n(kg/m²)"}

    # filter to available vars
    df = df[df["microbiome_var"].isin(micro_order) & df["phenotype"].isin(pheno_order)]
    mat_r = df.pivot(index="microbiome_var", columns="phenotype", values="partial_r")
    mat_p = df.pivot(index="microbiome_var", columns="phenotype", values="p")
    mat_n = df.pivot(index="microbiome_var", columns="phenotype", values="n")

    mat_r = mat_r.reindex(index=micro_order, columns=pheno_order)
    mat_p = mat_p.reindex(index=micro_order, columns=pheno_order)
    mat_n = mat_n.reindex(index=micro_order, columns=pheno_order)

    vmax = max(abs(mat_r.values[np.isfinite(mat_r.values)]).max(), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(mat_r.values, cmap="RdBu_r", norm=norm, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Partial r (controlling age + sex)", shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(len(pheno_order)))
    ax.set_xticklabels([pheno_labels[p] for p in pheno_order], fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(micro_order)))
    ax.set_yticklabels([micro_labels[m] for m in micro_order], fontsize=10)

    # annotate cells
    for i, mv in enumerate(micro_order):
        for j, pv in enumerate(pheno_order):
            r_val = mat_r.loc[mv, pv]
            p_val = mat_p.loc[mv, pv]
            n_val = mat_n.loc[mv, pv]
            if np.isfinite(r_val):
                sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
                txt = f"{r_val:+.3f}{sig}\nn={int(n_val):,}"
                color = "white" if abs(r_val) > vmax * 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, color=color, linespacing=1.4)

    ax.set_title("Cross-sectional partial correlations\n"
                 "(single-visit stratum; age + sex controlled)",
                 fontsize=11, pad=42)
    # footnote below figure using fig.text so it never overlaps axes
    fig.text(0.02, 0.01,
             "* p<0.05  ** p<0.01  |  Source: N3_crosssectional.py",
             fontsize=8, color="#555555", va="bottom")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = OUT / "FIG_R1_crosssectional_heatmap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def fig_pathA_forest() -> None:
    banner("FIG_R2: Path-A forest plot with equivalence bounds (N5)")

    csv = RESULTS / "cross_lag_all_paths.csv"
    if not csv.exists():
        log(f"  {csv} not found"); return

    raw = pd.read_csv(csv)
    spec_col = next((c for c in raw.columns if "spec" in c.lower()), None)
    if spec_col:
        spec_vals = raw[spec_col].value_counts()
        primary_val = next(
            (v for v in spec_vals.index if "primary" in str(v).lower()), spec_vals.index[0])
        base = raw[raw[spec_col] == primary_val].copy()
    else:
        base = raw.copy()

    pathA = base[base["direction"] == "A"].copy()
    if len(pathA) == 0:
        log("  No Path-A rows found."); return

    # Parse CI95 "(lo, hi)" strings
    ci_col = next((c for c in pathA.columns if "ci95" in c.lower()), None)
    if ci_col and pathA[ci_col].dtype == object:
        parsed = pathA[ci_col].str.strip("()[] ").str.split(",", expand=True).astype(float)
        pathA["ci_lo"] = parsed.iloc[:, 0].values
        pathA["ci_hi"] = parsed.iloc[:, 1].values
    elif "se" in pathA.columns:
        from scipy.stats import norm as sp_norm
        pathA["ci_lo"] = pathA["beta"] - sp_norm.ppf(0.975) * pathA["se"]
        pathA["ci_hi"] = pathA["beta"] + sp_norm.ppf(0.975) * pathA["se"]
    else:
        log("  Cannot compute CIs"); return

    # Labels
    exp_map = {"shannon": "Shannon", "richness": "Richness", "gut_ageaccel": "AgeAccel"}
    out_map = {"hand_grip_strength_right": "Grip", "grip": "Grip",
               "sitting_blood_pressure_systolic": "SBP", "sbp": "SBP",
               "waist_circumference": "Waist", "waist": "Waist"}
    exp_col = next(c for c in pathA.columns if "expos" in c.lower())
    out_col = next(c for c in pathA.columns if "outcom" in c.lower())

    pathA["label"] = (pathA[exp_col].map(exp_map).fillna(pathA[exp_col]) + " → " +
                      pathA[out_col].map(out_map).fillna(pathA[out_col]))
    pathA = pathA.sort_values("p")

    # Color by nominally significant
    colors = ["#D62728" if p < 0.05 else "#1F77B4" for p in pathA["p"]]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    n = len(pathA)

    # x limits first so we know the right edge for text placement
    lim = max(abs(pathA["ci_lo"].min()), abs(pathA["ci_hi"].max()), EQUIV + 0.02) + 0.04
    x_right = lim + 0.10
    ax.set_xlim(-lim, x_right)

    # Equivalence band
    ax.axvspan(-EQUIV, EQUIV, color="#EEEEEE", zorder=0)
    ax.axvline(0, color="#333333", lw=0.8, zorder=1)

    for i, (_, row) in enumerate(pathA.iterrows()):
        y = n - 1 - i
        ax.plot([row["ci_lo"], row["ci_hi"]], [y, y],
                color=colors[i], lw=1.8, zorder=2)
        ax.plot(row["beta"], y, "o", color=colors[i],
                markersize=6, zorder=3)
        p_str = f"p={row['p']:.3f}" if row["p"] >= 0.001 else f"p={row['p']:.2e}"
        # anchor text just past the right equivalence bound or CI, whichever is larger
        x_txt = max(row["ci_hi"], EQUIV) + 0.006
        ax.text(x_txt, y, p_str, va="center", fontsize=8,
                color=colors[i], clip_on=False)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(pathA["label"].tolist())), fontsize=10)
    ax.set_xlabel("Standardised β (95% CI)", fontsize=10)
    ax.set_title("Path-A: microbiome(t₁) → phenotype(t₂)\n"
                 "Cross-lagged estimates (n ≈ 1,769–1,778; age + sex controlled)",
                 fontsize=10.5, pad=8)
    ax.tick_params(axis="x", labelsize=9)

    red_patch  = mpatches.Patch(color="#D62728", label="p < 0.05 (nom.)")
    blue_patch = mpatches.Patch(color="#1F77B4", label="p ≥ 0.05")
    grey_patch = mpatches.Patch(color="#EEEEEE", label=f"±{EQUIV} equivalence zone")
    ax.legend(handles=[red_patch, blue_patch, grey_patch],
              loc="upper left", fontsize=9, framealpha=0.92)

    fig.text(0.02, 0.01,
             "Source: cross_lag_all_paths.csv (primary spec, age+sex)  |  N5_null_enrichment_pathB.py",
             fontsize=7.5, color="#555555", va="bottom")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUT / "FIG_R2_pathA_forest.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def fig_pvalue_dist() -> None:
    banner("FIG_R3: P-value distribution for all 18 cross-lagged tests (N5)")

    csv = RESULTS / "cross_lag_all_paths.csv"
    if not csv.exists():
        log(f"  {csv} not found"); return

    raw = pd.read_csv(csv)
    spec_col = next((c for c in raw.columns if "spec" in c.lower()), None)
    if spec_col:
        spec_vals = raw[spec_col].value_counts()
        pv = next(
            (v for v in spec_vals.index if "primary" in str(v).lower()), spec_vals.index[0])
        base = raw[raw[spec_col] == pv].copy()
    else:
        base = raw.copy()

    p_col   = next(c for c in base.columns if c.lower() == "p")
    dir_col = next(c for c in base.columns if c.lower() == "direction")
    exp_col = next(c for c in base.columns if "expos" in c.lower())
    out_col = next(c for c in base.columns if "outcom" in c.lower())

    all_p  = base[p_col].dropna().values
    pathA_p = base[base[dir_col] == "A"][p_col].dropna().values
    pathB_p = base[base[dir_col] == "B"][p_col].dropna().values

    # Highlight waist → microbiome Path-B rows
    waist_mask = (base[dir_col] == "B") & (
        base[exp_col].str.contains("waist|anthropo", case=False, na=False))
    waist_p = base.loc[waist_mask, p_col].dropna().values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: histogram of all 18 p-values
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    ax.hist(pathA_p, bins=bins, alpha=0.7, color="#2166AC", label=f"Path-A (n={len(pathA_p)})",
            edgecolor="white")
    ax.hist(pathB_p, bins=bins, alpha=0.6, color="#D6604D", label=f"Path-B (n={len(pathB_p)})",
            edgecolor="white", bottom=None)
    ax.axhline(len(all_p) / 10, color="#555555", lw=1.2, ls="--",
               label="Uniform null (expected)")
    ax.axvline(0.05, color="black", lw=0.8, ls=":")
    ax.set_xlabel("P-value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("P-value distribution\n(all 18 cross-lagged tests)", fontsize=10.5)
    # α label: place above x-axis at 0.05, after bars are drawn
    ax.annotate("α=0.05", xy=(0.05, 0), xytext=(0.09, ax.get_ylim()[1] * 0.82 if ax.get_ylim()[1] > 0 else 2),
                fontsize=8.5, color="black",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.legend(fontsize=9, framealpha=0.9)

    # Panel B: QQ-plot of p-values vs uniform
    ax2 = axes[1]
    p_sorted = np.sort(all_p)
    n_tests  = len(p_sorted)
    expected = (np.arange(1, n_tests + 1) - 0.5) / n_tests

    # Colour the waist-related Path-B tests distinctly
    waist_p_set = set(np.round(waist_p, 8))
    colors_qq = ["#D62728" if round(pv, 8) in waist_p_set else "#444444"
                 for pv in p_sorted]

    ax2.scatter(expected, p_sorted, c=colors_qq, s=48, zorder=3, edgecolors="none")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.0, zorder=2, label="Uniform null")
    ax2.set_xlabel("Expected p-value (uniform)", fontsize=10)
    ax2.set_ylabel("Observed p-value", fontsize=10)
    ax2.set_title("QQ-plot: observed vs uniform null\n"
                  f"(Storey π₀ ≈ 0.67; binomial enrichment p=0.071)", fontsize=10.5)
    red_dot  = mpatches.Patch(color="#D62728", label="waist → microbiome (Path-B)")
    grey_dot = mpatches.Patch(color="#444444", label="other tests")
    ax2.legend(handles=[red_dot, grey_dot], fontsize=9, framealpha=0.9)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.01,
             "Source: cross_lag_all_paths.csv (primary spec)  |  N5_null_enrichment_pathB.py",
             ha="center", fontsize=8, color="#555555", va="bottom")
    out_path = OUT / "FIG_R3_pvalue_dist.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    banner("N_review_figures: generating Nastya-review figures")
    fig_crosssectional_heatmap()
    fig_pathA_forest()
    fig_pvalue_dist()
    banner("DONE — three figures written to results_cross_lagged/review2/")
    log(f"  FIG_R1_crosssectional_heatmap.png")
    log(f"  FIG_R2_pathA_forest.png")
    log(f"  FIG_R3_pvalue_dist.png")


if __name__ == "__main__":
    main()
