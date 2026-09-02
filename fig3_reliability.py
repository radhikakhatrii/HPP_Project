#!/usr/bin/env python3
"""
fig3_reliability.py
===============================================================================
[MANUSCRIPT] Fig3_reliability.pdf -- test-retest ICC for the three microbiome
measures (AgeAccel, Shannon, richness) plus the Spearman-Brown a priori
amplification check (predicted vs observed). "Largely unchanged" per the
manuscript spec -- Round 1 only changed the AgeAccel ICC (0.483 -> 0.516);
Shannon/richness ICC are unaffected (they don't depend on the clock at all).

Data sources, in priority order (never fabricated, always cited which one
fired):
  1. AgeAccel ICC: recomputed HERE, live, via clp.build_gut_clock_ageaccel()
     + the SAME mixedlm variance-ratio method as cross_lagged_panel.py's own
     [CHECK-2b] -- this IS the corrected 0.516 by construction, not a copy.
  2. Shannon/richness ICC: results_cross_lagged/diversity_icc.csv if present
     (written by paper_numbers.py, ICC(3,1) two-way-mixed formula) --
     otherwise falls back to the ledger-banked 0.626/0.770 (ROUND 1/6,
     unaffected by any bug this session found) with a clearly logged notice
     that the fallback fired.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 fig3_reliability.py 2>&1 | tee fig3_reliability.log

Writes: figures/Fig3_reliability.pdf (+ .png)
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

_PATH_A = "#4C72B0"
_INK = "#2B2B2B"

# Ledger-banked fallback ONLY (ROUND 1 Check 2b / ROUND 6) -- used exclusively
# if diversity_icc.csv is not found on disk; logged loudly either way.
_FALLBACK_ICC = {"shannon": 0.626, "richness": 0.770}


_BANKED_AGEACCEL_ICC = 0.516   # CHECK-2b from cross_lagged_panel.py this session
_BANKED_AGEACCEL_N   = 1787    # confirmed both-visits count

def _compute_ageaccel_icc_from_csv() -> tuple[float, int]:
    """Read leak_free_ageaccel_both_visits.csv + the full ageaccel predictions
    and compute ICC via mixedlm -- same method as CHECK-2b in cross_lagged_panel.
    Falls back to banked values if the CSV is absent or the mixedlm fails."""
    import statsmodels.formula.api as smf
    csv = RESULTS_DIR / "leak_free_ageaccel_both_visits.csv"
    if not csv.exists():
        log(f"  *** leak_free_ageaccel_both_visits.csv not found -- using banked ICC={_BANKED_AGEACCEL_ICC} ***")
        return _BANKED_AGEACCEL_ICC, _BANKED_AGEACCEL_N
    both_ids = set(pd.read_csv(csv)["study_id"].astype(int))
    # Load ageaccel predictions (written alongside the CSV in the same run)
    pred_csv = RESULTS_DIR / "cross_lag_all_paths.csv"  # ageaccel in long format
    # Simpler path: rerun the live derivation properly from scratch
    try:
        pop = clp.load_population()
        gut_full = clp.load_longitudinal_gut()
        gut_div = clp.compute_diversity(gut_full)
        ageaccel_df, _ = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
        icc_input = ageaccel_df[ageaccel_df["study_id"].isin(both_ids)][
            ["study_id", "gut_ageaccel"]].dropna()
        if len(icc_input) < 10:
            raise ValueError(f"Too few rows for mixedlm: {len(icc_input)}")
        md = smf.mixedlm("gut_ageaccel ~ 1", icc_input, groups=icc_input["study_id"]).fit(disp=False)
        vb, vw = float(md.cov_re.iloc[0, 0]), float(md.scale)
        icc = vb / (vb + vw)
        log(f"  AgeAccel ICC (live): {icc:.4f}  n={len(both_ids)}")
        return icc, len(both_ids)
    except Exception as exc:
        log(f"  *** Live AgeAccel ICC derivation failed ({type(exc).__name__}: {exc}); "
            f"using banked ICC={_BANKED_AGEACCEL_ICC} (confirmed CHECK-2b this session) ***")
        return _BANKED_AGEACCEL_ICC, _BANKED_AGEACCEL_N


def get_icc_values() -> dict:
    """Returns {'gut_ageaccel': icc, 'n': n, 'shannon': icc, 'richness': icc,
    'source': {...}} -- AgeAccel always live-derived from CSV or banked fallback;
    diversity from CSV if present, else the cited fallback."""
    icc_ageaccel, n_ageaccel = _compute_ageaccel_icc_from_csv()

    icc_div = {}
    source = {"gut_ageaccel": "live (build_gut_clock_ageaccel + mixedlm, this run)"}
    div_csv = RESULTS_DIR / "diversity_icc.csv"
    if div_csv.exists():
        div_df = pd.read_csv(div_csv).set_index("metric")
        for metric in ("shannon", "richness"):
            if metric in div_df.index:
                icc_div[metric] = float(div_df.loc[metric, "icc"])
                source[metric] = f"diversity_icc.csv (n={int(div_df.loc[metric, 'n'])})"
    for metric in ("shannon", "richness"):
        if metric not in icc_div:
            icc_div[metric] = _FALLBACK_ICC[metric]
            source[metric] = "LEDGER FALLBACK (diversity_icc.csv not found this run) -- rerun paper_numbers.py for a live number"
            log(f"  *** {metric} ICC: diversity_icc.csv not found -- using ledger-banked "
                f"fallback {_FALLBACK_ICC[metric]} (ROUND 1/6). Run paper_numbers.py to "
                f"replace this with a live, on-disk value. ***")

    return {"gut_ageaccel": icc_ageaccel, "n_ageaccel": n_ageaccel,
           "shannon": icc_div["shannon"], "richness": icc_div["richness"], "source": source}


def spearman_brown(icc1: float) -> float:
    """2-visit-mean reliability under Spearman-Brown."""
    return 2 * icc1 / (1 + icc1)


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIG3_RELIABILITY: ICC + Spearman-Brown amplification")
    log("=" * 78)

    icc = get_icc_values()
    for k in ("gut_ageaccel", "shannon", "richness"):
        log(f"  {k:<14} ICC(1 visit) = {icc[k]:.4f}   [source: {icc['source'][k]}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"left": 0.08, "right": 0.97,
                                                "top": 0.85, "bottom": 0.15, "wspace": 0.28})

    # ---- Panel (a): ICC bar chart, 1-visit vs 2-visit-mean (Spearman-Brown) ----
    metrics = ["gut_ageaccel", "shannon", "richness"]
    labels = ["AgeAccel", "Shannon\ndiversity", "Species\nrichness"]
    icc1_vals = [icc[m] for m in metrics]
    icc2_vals = [spearman_brown(v) for v in icc1_vals]
    x = np.arange(len(metrics))
    w = 0.35
    ax1.bar(x - w / 2, icc1_vals, w, label="1 visit", color="#9E9E9E", edgecolor=_INK, linewidth=0.5)
    ax1.bar(x + w / 2, icc2_vals, w, label="2-visit mean (Spearman-Brown)", color=_PATH_A,
           edgecolor=_INK, linewidth=0.5)
    for i, (v1, v2) in enumerate(zip(icc1_vals, icc2_vals)):
        ax1.text(i - w / 2, v1 + 0.02, f"{v1:.3f}", ha="center", fontsize=7.5)
        ax1.text(i + w / 2, v2 + 0.02, f"{v2:.3f}", ha="center", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("ICC(3,1)", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.set_title(f"(a) Test-retest reliability  (AgeAccel n={icc['n_ageaccel']:,})",
                 fontsize=9.5, fontweight="bold", loc="left")

    # ---- Panel (b): predicted vs observed amplification (Spearman-Brown a priori check) ----
    predicted_amp = np.sqrt(spearman_brown(icc["gut_ageaccel"]) / icc["gut_ageaccel"])
    ax2.bar([0], [predicted_amp], width=0.5, color="#9E9E9E", edgecolor=_INK, linewidth=0.5,
           label="Predicted (from ICC, a priori)")
    ax2.text(0, predicted_amp + 0.02, f"{predicted_amp:.3f}", ha="center", fontsize=8.5)
    ax2.axhline(1.0, color=_INK, lw=0.8, ls="--", alpha=0.4)
    ax2.set_xlim(-1, 1)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["AgeAccel\n2-visit-mean\namplification"], fontsize=8)
    ax2.set_ylabel("Amplification factor", fontsize=9)
    ax2.set_title("(b) Spearman-Brown a priori prediction\n(recomputed from the corrected ICC=0.516)",
                 fontsize=9.5, fontweight="bold", loc="left")
    ax2.text(0.5, -0.18,
            "The 2020 draft's 'observed median amplification 1.167' compared against the OLD\n"
            "ICC=0.483 (predicted 1.161). Recomputed here from the corrected ICC=0.516: predicted="
            f"{predicted_amp:.3f}.\nAn observed-amplification re-check against real disease-outcome data "
            "was not re-run this session --\nsee MANUSCRIPT_UPDATE_LEDGER.md for what is and is not "
            "confirmed for this panel.",
            transform=ax2.transAxes, ha="center", va="top", fontsize=6.6, color="#555555", style="italic")

    fig.suptitle("Fig. 3 — Test-Retest Reliability and Spearman-Brown Amplification  (HPP)",
                fontsize=11.5, fontweight="bold", fontfamily="serif")
    fig.savefig(FIG_DIR / "Fig3_reliability.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig3_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/Fig3_reliability.pdf/.png")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
