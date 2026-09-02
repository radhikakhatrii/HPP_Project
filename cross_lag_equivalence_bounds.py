#!/usr/bin/env python3
"""
cross_lag_equivalence_bounds.py
===============================================================================
M1 reframing item: equivalence-bounds analysis for the 18 primary-spec
(age+sex) cross-lag coefficients. Reads (does not recompute) the already-
committed cross_lag_all_paths.csv from cross_lagged_panel.py -- fast, no data
loading, no model refitting.

Two additions, BOTH explicitly disclosed as POST-HOC bounding exercises, not
pre-registered tests -- the SESOI was chosen on external grounds before this
script was written, but after the primary results already existed:

1. TOST (two one-sided tests) equivalence test against a pre-specified
   smallest effect of interest (SESOI) of |beta| = 0.10 standardised units --
   a conventional "small effect" threshold, chosen independently of these
   results, not derived from them. For each of the 18 coefficients: does the
   90% CI (equivalent to two one-sided 5% tests) fall entirely within
   [-0.10, +0.10]? The interesting rows are the 4 whose 95% CI excludes zero
   (Round 3): if those are ALSO TOST-equivalent to a null-at-0.10 effect,
   that is a "detectable but negligible" result -- statistically nonzero,
   and statistically smaller than a conventional small effect, at once.

2. Reliability-disattenuated bounds for gut_ageaccel, Path A ONLY
   (gut_ageaccel(t1) -> phenotype(t2), where gut_ageaccel is the regressor).
   Under classical measurement-error theory, unreliable measurement of a
   REGRESSOR attenuates its coefficient toward zero; unreliable measurement
   of an OUTCOME does not bias the point estimate (only inflates its SE) --
   so this is NOT applied to the 3 Path B rows, where gut_ageaccel is the
   outcome (y_t2). Uses the SAME simple sqrt(reliability) convention the
   paper's own subsec:sb already establishes for AgeAccel
   (beta_true ~ beta_observed / sqrt(ICC)), applied to a regression
   coefficient instead of a correlation -- not a new methodology.

   E1-E3 (Round 2) already largely settles the reliability question at the
   taxon level: S_all, a substantially MORE reliable signature (ICC=0.778 vs
   AgeAccel's 0.483/0.516), was STILL null. This section is framed as
   CONFIRMATORY of that finding, not independent new evidence.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 cross_lag_equivalence_bounds.py 2>&1 | tee cross_lag_equivalence_bounds.log

Writes:
    results_cross_lagged/cross_lag_equivalence_bounds.csv
    results_cross_lagged/cross_lag_ageaccel_disattenuated.csv
===============================================================================
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
log = clp.log

SESOI            = 0.10   # pre-specified smallest effect of interest, |beta| units
AGEACCEL_ICC     = 0.516  # corrected leak-free AgeAccel ICC (Round 1, Check 2b)
N_PRIMARY_PARAMS = 5      # y_t1, x_t1, age, sex, const -- primary-spec model


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def tost(beta: float, se: float, n: float, sesoi: float = SESOI,
        n_params: int = N_PRIMARY_PARAMS) -> dict:
    """Two one-sided tests for equivalence against +/- sesoi. Returns the two
    one-sided p-values, the overall TOST p (= the larger, limiting one), the
    90% CI, and whether that CI falls entirely within [-sesoi, +sesoi]."""
    df = n - n_params
    t_lower = (beta + sesoi) / se     # H0: true beta <= -sesoi
    t_upper = (beta - sesoi) / se     # H0: true beta >= +sesoi
    p_lower = 1 - t_dist.cdf(t_lower, df)
    p_upper = t_dist.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    tcrit_90 = t_dist.ppf(0.95, df)   # 90% two-sided = one-sided alpha 0.05 each side
    ci90_lo = beta - tcrit_90 * se
    ci90_hi = beta + tcrit_90 * se
    within_bound = (ci90_lo > -sesoi) and (ci90_hi < sesoi)
    return {"p_tost": p_tost, "p_lower": p_lower, "p_upper": p_upper,
            "ci90_lo": ci90_lo, "ci90_hi": ci90_hi,
            "equivalent_to_null_at_sesoi": bool(within_bound)}


def main() -> None:
    banner("EQUIVALENCE-BOUNDS ANALYSIS (M1 reframing) -- POST-HOC, not pre-registered")

    path = OUT / "cross_lag_all_paths.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run cross_lagged_panel.py first; "
                         f"this script only reads its output, it does not "
                         f"recompute anything.")
    df = pd.read_csv(path)
    df = df[df["covariate_spec"] == "primary"].copy()
    log(f"  loaded {len(df)} primary-spec rows from {path}")

    banner(f"1. TOST equivalence tests, SESOI = +/-{SESOI} standardised units")
    log("  Disclosure: SESOI chosen as a conventional 'small effect' threshold")
    log("  on external grounds, independent of these results -- but applied to")
    log("  already-observed estimates, so this is a POST-HOC bounding exercise,")
    log("  not a pre-registered equivalence test.")

    rows = []
    for _, r in df.iterrows():
        res = tost(r["beta"], r["se"], r["n"])
        ci95_excludes_zero = None
        ci95_lo = ci95_hi = np.nan
        try:
            ci = ast.literal_eval(r["ci95"])
            ci95_lo, ci95_hi = float(ci[0]), float(ci[1])
            ci95_excludes_zero = bool((ci[0] > 0) or (ci[1] < 0))
        except Exception:
            pass
        row = {"direction": r["direction"], "path": r["path"],
              "exposure": r["exposure"], "outcome": r["outcome"],
              "beta": r["beta"], "se": r["se"], "n": r["n"], "p": r["p"],
              "q_fdr": r["q_fdr"], "ci95_lo": ci95_lo, "ci95_hi": ci95_hi,
              "ci95_excludes_zero": ci95_excludes_zero,
              **res}
        rows.append(row)
        flag = ""
        if ci95_excludes_zero and res["equivalent_to_null_at_sesoi"]:
            flag = "  <-- DETECTABLE BUT NEGLIGIBLE (CI excl. 0 AND TOST-confirmed <SESOI)"
        log(f"  {r['path']:<55} beta={r['beta']:+.4f}  90% CI=[{res['ci90_lo']:+.4f}, "
            f"{res['ci90_hi']:+.4f}]  p_TOST={res['p_tost']:.4f}  "
            f"within_bound={res['equivalent_to_null_at_sesoi']}{flag}")

    n_equiv = sum(r["equivalent_to_null_at_sesoi"] for r in rows)
    log(f"\n  {n_equiv}/18 coefficients statistically distinguishable from a "
        f"|beta|>={SESOI} effect (90% CI entirely within +/-{SESOI}).")
    n_ci_excludes_zero = sum(1 for r in rows if r["ci95_excludes_zero"])
    n_detectable_negligible = sum(
        1 for r in rows if r["ci95_excludes_zero"] and r["equivalent_to_null_at_sesoi"])
    log(f"  {n_detectable_negligible} of the {n_ci_excludes_zero} CI-excludes-zero rows are ALSO "
        f"'detectable but negligible' (significantly nonzero AND significantly "
        f"smaller than {SESOI}).")

    banner("2. Reliability-disattenuated bounds -- gut_ageaccel Path A only")
    log("  Applies ONLY to gut_ageaccel(t1) -> phenotype(t2) (Path A), where")
    log("  gut_ageaccel is the REGRESSOR (x_t1). Classical measurement-error")
    log("  theory: unreliable measurement of a regressor attenuates its own")
    log("  coefficient toward zero; unreliable measurement of an OUTCOME does")
    log("  NOT bias the point estimate (only inflates its SE) -- so this is")
    log("  NOT applied to the 3 Path B rows (gut_ageaccel as y_t2).")
    log("  Same simple sqrt(reliability) convention as the paper's own")
    log("  subsec:sb, applied to a regression coefficient instead of a")
    log("  correlation -- an approximation, disclosed as such.")
    log(f"  ICC = {AGEACCEL_ICC} (corrected leak-free AgeAccel, Round 1 Check 2b); "
        f"sqrt(ICC) = {np.sqrt(AGEACCEL_ICC):.4f}")

    disatt_rows = []
    path_a_ageaccel = df[(df["exposure"] == "gut_ageaccel") & (df["direction"] == "A")]
    factor = 1 / np.sqrt(AGEACCEL_ICC)
    for _, r in path_a_ageaccel.iterrows():
        beta_disatt = r["beta"] * factor
        ci_obs = (np.nan, np.nan)
        ci_disatt = (np.nan, np.nan)
        try:
            ci_obs = ast.literal_eval(r["ci95"])
            ci_disatt = (ci_obs[0] * factor, ci_obs[1] * factor)
        except Exception:
            pass
        disatt_rows.append({
            "path": r["path"], "outcome": r["outcome"], "n": r["n"],
            "beta_observed": r["beta"], "beta_disattenuated": beta_disatt,
            "ci95_observed_lo": ci_obs[0], "ci95_observed_hi": ci_obs[1],
            "ci95_disattenuated_lo": ci_disatt[0],
            "ci95_disattenuated_hi": ci_disatt[1],
        })
        log(f"  {r['path']:<55} observed beta={r['beta']:+.4f}  ->  "
            f"disattenuated bound={beta_disatt:+.4f}  "
            f"(95% CI disattenuated: [{ci_disatt[0]:+.4f}, {ci_disatt[1]:+.4f}])")

    log("\n  Framed as CONFIRMATORY, not new evidence: E1-E3 (Round 2) already")
    log("  tested a substantially MORE reliable taxon-level signature (S_all,")
    log("  ICC=0.778 vs AgeAccel's 0.483/0.516) and it was STILL null (all 9")
    log("  signature-outcome tests p>=0.29, delta-R^2~0). These disattenuated")
    log("  AgeAccel bounds are consistent with, not independent of, that result.")

    equiv_df = pd.DataFrame(rows)
    equiv_df.to_csv(OUT / "cross_lag_equivalence_bounds.csv", index=False)
    log(f"\n  wrote {OUT/'cross_lag_equivalence_bounds.csv'}")

    disatt_df = pd.DataFrame(disatt_rows)
    disatt_df.to_csv(OUT / "cross_lag_ageaccel_disattenuated.csv", index=False)
    log(f"  wrote {OUT/'cross_lag_ageaccel_disattenuated.csv'}")

    plot_equivalence_bounds(equiv_df, OUT)

    banner("DONE")


def _forest_panel(ax, df, title, show_legend=True):
    """Shared forest-plot body: 95% CI (thick) + 90%/TOST equivalence band
    (shaded), beta marker diamond if CI95 excludes zero. Used for both the
    9-row main figure and the 18-row supplementary one so the two never
    diverge in how they draw the same underlying quantities."""
    n = len(df)
    ax.axvspan(-SESOI, SESOI, color="#E8E8E8", zorder=0,
              label=f"Equivalence band (|β|<{SESOI})")
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5, zorder=1)
    for i, row in df.reset_index(drop=True).iterrows():
        excl0 = bool(row["ci95_excludes_zero"])
        equiv = bool(row["equivalent_to_null_at_sesoi"])
        color = "#B0413E" if excl0 else "#4C72B0"
        # 95% CI -- the standard interval that determines "excludes zero"
        if np.isfinite(row.get("ci95_lo", np.nan)) and np.isfinite(row.get("ci95_hi", np.nan)):
            ax.plot([row["ci95_lo"], row["ci95_hi"]], [i, i],
                   color=color, lw=3.2, solid_capstyle="round", zorder=2, alpha=0.55)
        # 90% CI -- the TOST-equivalent interval
        ax.plot([row["ci90_lo"], row["ci90_hi"]], [i, i],
               color=color, lw=1.6, solid_capstyle="round", zorder=3)
        mk = "D" if excl0 else "o"
        ax.scatter([row["beta"]], [i], color=color, marker=mk,
                 s=70 if excl0 else 40, zorder=4, edgecolors="black", linewidths=0.5)
        if excl0 and equiv:
            ax.text(0.99, i, "detectable but negligible",
                   transform=ax.get_yaxis_transform(), ha="right",
                   va="center", fontsize=6.5, color=color, style="italic")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{r['direction']}: {r['exposure']}→{r['outcome']}" for _, r in df.iterrows()],
        fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised β  (thick=95% CI, thin=90% TOST-equivalent CI)", fontsize=8)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", fontfamily="serif")


def plot_equivalence_bounds(equiv_df: pd.DataFrame, out_dir: Path) -> None:
    """[MANUSCRIPT] Fig5_bounds.pdf -- the 9 Path-A microbiome->phenotype
    coefficients (matches tabE2_bounds / the old MDE-only power table this
    figure REPLACES), with 95% CIs, the +/-SESOI equivalence band, and the
    TOST verdict. A second file, FigS_equivalence_all18.pdf, carries the full
    18-row version (both directions) as supplementary material -- nothing
    from the original all-18 figure is lost, it is just no longer what
    Fig5_bounds itself shows, per the manuscript's own figure spec."""
    log("\n[Fig5] Equivalence-bounds forest plot(s)...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ---- Fig5_bounds.pdf: 9 Path A tests only ----
        path_a = equiv_df[equiv_df["direction"] == "A"].sort_values(
            ["exposure", "outcome"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.0, max(3.5, len(path_a) * 0.55)),
                               gridspec_kw={"left": 0.30, "right": 0.90,
                                            "top": 0.88, "bottom": 0.14})
        _forest_panel(ax, path_a,
                     f"Microbiome → Phenotype: 95% CIs and Equivalence to |β|<{SESOI}\n"
                     "(replaces the design-sensitivity MDE-only figure)")
        fig.savefig(out_dir / "Fig5_bounds.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "Fig5_bounds.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {out_dir}/Fig5_bounds.pdf/.png (9 Path-A tests)")

        # ---- Supplementary: all 18 (both directions) ----
        df18 = equiv_df.sort_values(["direction", "path"]).reset_index(drop=True)
        fig2, ax2 = plt.subplots(figsize=(8.5, max(4.5, len(df18) * 0.42)),
                                 gridspec_kw={"left": 0.32, "right": 0.90,
                                              "top": 0.90, "bottom": 0.10})
        _forest_panel(ax2, df18,
                     "Equivalence-Bounds Analysis: All 18 Primary-Spec Cross-Lag Coefficients\n"
                     f"vs. pre-specified SESOI = ±{SESOI} (post-hoc, supplementary)")
        fig2.savefig(out_dir / "FigS_equivalence_all18.pdf", dpi=300, bbox_inches="tight")
        fig2.savefig(out_dir / "FigS_equivalence_all18.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        log(f"  Saved: {out_dir}/FigS_equivalence_all18.pdf/.png (18 tests, supplementary)")
    except Exception as e:
        import traceback
        log(f"  Fig5_bounds plot failed: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
