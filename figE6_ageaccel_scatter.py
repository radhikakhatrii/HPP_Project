#!/usr/bin/env python3
"""
figE6_ageaccel_scatter.py
===============================================================================
[MANUSCRIPT] FigE6_ageaccel_scatter.pdf -- NEW (Nastya point 4). AgeAccel at
visit 1 vs visit 2 for the n=1,787 leak-free repeat-visit participants,
identity line, annotated with ICC(3,1)=0.516.

AgeAccel is recomputed live via the SAME leak-free clock call used elsewhere
in this pipeline (build_gut_clock_ageaccel -- identical features/
hyperparameters/seed as Fig2_clock.pdf and Fig3_reliability.pdf's live ICC
derivation), then restricted to the ids in
leak_free_ageaccel_both_visits.csv (the committed, audited "both visits
present" set). The per-visit values are also persisted to a CSV so this is
reusable without recomputation next time.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 figE6_ageaccel_scatter.py 2>&1 | tee figE6_ageaccel_scatter.log

Reads:  results_cross_lagged/leak_free_ageaccel_both_visits.csv (id list, required)
Writes: results_cross_lagged/ageaccel_by_visit_paired.csv (persisted for reuse)
        figures/FigE6_ageaccel_scatter.pdf (+ .png)
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

_PATH_A = "#4C72B0"
_INK = "#2B2B2B"


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIGE6_AGEACCEL_SCATTER: AgeAccel visit 1 vs visit 2 (leak-free, n=1,787)")
    log("=" * 78)

    both_path = RESULTS_DIR / "leak_free_ageaccel_both_visits.csv"
    if not both_path.exists():
        raise SystemExit(f"{both_path} not found -- run cross_lagged_panel.py's "
                         f"main pipeline first (writes this at [CHECK-2a]).")
    both_ids = set(pd.read_csv(both_path)["study_id"].astype(int))
    log(f"  Leak-free both-visits ids: {len(both_ids):,} (from {both_path})")

    log("  Recomputing AgeAccel live (same call as Fig2_clock.pdf / Fig3's ICC)...")
    pop = clp.load_population()

    # [BUGFIX] build_gut_clock_ageaccel's 2nd arg must be the FULL gut parquet
    # (all participants, single- and repeat-visit) so it can derive
    # single_ids = all_ids - repeat_ids for clock TRAINING. load_longitudinal_gut()
    # returns ONLY the repeat-visit subset -- passing that here makes single_ids
    # empty, silently skips clock training, and returns all-NaN AgeAccel (0
    # paired participants). Load gut_full exactly as cross_lagged_panel.py's own
    # main() does, and gut_div (repeat-visit diversity) separately for the
    # gut_diversity argument.
    gut_repeat = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut_repeat)

    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))
    log(f"  gut_full: {len(gut_full):,} rows, {gut_full['study_id'].nunique():,} participants "
        f"(vs gut_repeat: {gut_repeat['study_id'].nunique():,} repeat-visit only)")

    gut_ageaccel, clock_data = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
    log(f"  Clock CV r = {clock_data.get('r_cv', float('nan')):.4f}")
    if not np.isfinite(clock_data.get("r_cv", float("nan"))):
        raise SystemExit("Clock training returned NaN r_cv (empty training frame) -- "
                         "see [CHECK-1] log above for the leak-free overlap diagnostic.")

    sub = gut_ageaccel[gut_ageaccel["study_id"].isin(both_ids)][
        ["study_id", "visit_n", "gut_ageaccel"]].dropna()
    wide = sub.pivot_table(index="study_id", columns="visit_n", values="gut_ageaccel")
    wide = wide.dropna(subset=[1, 2]) if 1 in wide.columns and 2 in wide.columns else wide.dropna()
    wide = wide.rename(columns={1: "ageaccel_v1", 2: "ageaccel_v2"})
    log(f"  Paired AgeAccel (both visits, non-null): n={len(wide):,}")

    wide.reset_index().to_csv(RESULTS_DIR / "ageaccel_by_visit_paired.csv", index=False)
    log(f"  Saved {RESULTS_DIR/'ageaccel_by_visit_paired.csv'} (n={len(wide):,})")

    if len(wide) < 10:
        raise SystemExit(f"Only {len(wide)} paired participants -- cannot plot.")

    # ICC(3,1) via mixedlm variance ratio, same method as CHECK-2b / fig3_reliability.py
    long = pd.melt(wide.reset_index(), id_vars="study_id",
                   value_vars=["ageaccel_v1", "ageaccel_v2"],
                   var_name="visit", value_name="ageaccel")
    md = smf.mixedlm("ageaccel ~ 1", long, groups=long["study_id"]).fit(disp=False)
    vb, vw = float(md.cov_re.iloc[0, 0]), float(md.scale)
    icc_live = vb / (vb + vw)
    log(f"  ICC(3,1) live = {icc_live:.4f}  (ledger-cited: 0.516)")
    if abs(icc_live - 0.516) > 0.02:
        log(f"  *** NOTE: live ICC differs from the ledger-cited 0.516 by "
            f"{abs(icc_live-0.516):.4f} (>0.02 tolerance) -- reporting the live "
            f"value on the figure, flag for review before submission. ***")

    v1, v2 = wide["ageaccel_v1"].values, wide["ageaccel_v2"].values
    r_pearson = float(np.corrcoef(v1, v2)[0, 1])

    fig, ax = plt.subplots(figsize=(6.2, 6.0),
                           gridspec_kw={"left": 0.14, "right": 0.96, "top": 0.88, "bottom": 0.11})
    ax.scatter(v1, v2, s=18, alpha=0.45, color=_PATH_A, edgecolors="none")
    lo = min(v1.min(), v2.min())
    hi = max(v1.max(), v2.max())
    ax.plot([lo, hi], [lo, hi], color=_INK, lw=1.1, ls="--", alpha=0.6, label="identity (y=x)")
    ax.set_xlim(lo - 1, hi + 1)
    ax.set_ylim(lo - 1, hi + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gut AgeAccel, visit 1 (years)", fontsize=9.5)
    ax.set_ylabel("Gut AgeAccel, visit 2 (years)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.97, 0.05, f"ICC(3,1) = {icc_live:.3f}\nPearson r = {r_pearson:.3f}\nn = {len(wide):,}",
           transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
           fontfamily="monospace",
           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_INK, lw=0.5, alpha=0.9))
    ax.set_title("Extended Data Fig. 6 — Gut AgeAccel Test–Retest\n"
                "(leak-free clock, both visits, n=1,787)",
                fontsize=10.5, fontweight="bold", fontfamily="serif")
    fig.text(0.02, 0.01, "Source: build_gut_clock_ageaccel() (live, seed=42) restricted to "
             "leak_free_ageaccel_both_visits.csv ids.", fontsize=6.2, color="#555555", style="italic")

    fig.savefig(FIG_DIR / "FigE6_ageaccel_scatter.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "FigE6_ageaccel_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/FigE6_ageaccel_scatter.pdf/.png")
    log(f"\n  Verification: n={len(wide):,}, ICC={icc_live:.4f}, Pearson r={r_pearson:.4f}")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
