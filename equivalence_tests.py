#!/usr/bin/env python3
"""
equivalence_tests.py
===============================================================================
Replaces the reliability-adjusted MDE argument with formal equivalence testing.

WHY THIS EXISTS
The manuscript currently bounds its null by dividing the minimum detectable
effect by the exposure ICC. That correction assumes a regression slope is
attenuated by exactly the reliability of its predictor, which is true for
simple regression with classical error in ONE variable. Our models are
multivariable, so the attenuation depends on the joint covariance structure
and the division is only an approximation.

Equivalence testing answers the question directly and without that assumption.
Rather than asking "could we have detected an effect", it asks "can we reject
the presence of an effect larger than delta". Two one-sided tests (TOST,
Schuirmann 1987; Lakens 2017) against bounds +/- delta:

    t_lower = (beta - (-delta)) / SE      tests beta > -delta
    t_upper = (beta -   delta ) / SE      tests beta <  delta
    p_TOST  = max(p_lower, p_upper)

If p_TOST < alpha, the effect lies inside (-delta, +delta) with confidence
1 - alpha. Inverting this gives the SMALLEST bound at which equivalence holds:

    delta_min = |beta| + z_(1-alpha) * SE

which is the outer edge of the (1 - 2*alpha) confidence interval. That single
number is the honest statement of what each test excludes, and it requires no
measurement-error correction at all.

Reliability then enters as a clearly separated SENSITIVITY analysis rather
than as the headline, since disattenuating an equivalence bound inherits the
same approximation we are trying to avoid relying on.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python equivalence_tests.py 2>&1 | tee equivalence_tests.log

Reads : results_cross_lagged/cross_lag_all_paths.csv
Writes: results_cross_lagged/equivalence_tests.csv
        results_cross_lagged/equivalence_tests.png
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
log = clp.log

ALPHA = 0.05
# Conventional benchmarks for a standardised effect. 0.10 is the usual
# "smallest effect size of interest" for a small standardised association;
# 0.05 is deliberately stringent.
BOUNDS = [0.05, 0.10, 0.15, 0.20]

# Exposure reliabilities, for the sensitivity column only.
ICC = {"shannon": 0.626, "richness": 0.770, "gut_ageaccel": 0.483}


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def tost(beta: float, se: float, delta: float, df: int) -> tuple[float, float, float]:
    """Two one-sided tests. Returns (p_lower, p_upper, p_tost)."""
    t_lo = (beta + delta) / se          # H0: beta <= -delta
    t_up = (beta - delta) / se          # H0: beta >=  delta
    p_lo = 1.0 - stats.t.cdf(t_lo, df)  # want beta > -delta
    p_up = stats.t.cdf(t_up, df)        # want beta <  delta
    return float(p_lo), float(p_up), float(max(p_lo, p_up))


def main() -> None:
    banner("EQUIVALENCE TESTING FOR THE CROSS-LAGGED NULL")

    path = OUT / "cross_lag_all_paths.csv"
    if not path.exists():
        raise SystemExit(f"FATAL: {path} not found. Run cross_lagged_panel.py first.")
    df = pd.read_csv(path)
    log(f"  loaded {len(df)} tests from {path.name}")
    log(f"  columns: {list(df.columns)}")
    if "direction" not in df.columns:
        raise SystemExit(
            "FATAL: no 'direction' column in cross_lag_all_paths.csv. "
            "The A/B split logic in this script depends on that exact name -- "
            "check the printed column list above and update the script rather "
            "than let the Path A/B breakdown silently go to zero again.")
    dircounts = df["direction"].value_counts().to_dict()
    log(f"  direction value counts: {dircounts}")
    if set(dircounts.keys()) != {"A", "B"}:
        raise SystemExit(
            f"FATAL: 'direction' column has unexpected values {list(dircounts)}, "
            f"expected exactly 'A' and 'B'. Fix before trusting any output below.")

    beta_col = "std_beta" if "std_beta" in df.columns else "beta"
    n_pred = 4  # exposure + autoregressive + age + sex
    df["df_resid"] = df["n"] - n_pred - 1

    rows = []
    for _, r in df.iterrows():
        b, se, dfree = float(r[beta_col]), float(r["se"]), int(r["df_resid"])
        mic = str(r.get("microbiome_var", ""))
        rec = {
            # BUG FIX: this CSV has TWO different columns that both sound like
            # "path" -- "direction" holds the literal A/B indicator, "path"
            # holds a descriptive string like "shannon(t1) -> grip(t2)" that
            # never equals "A" or "B". Filtering on the wrong one silently
            # zeroed out both Path A and Path B breakdowns while the ungrouped
            # total (which doesn't depend on this column) stayed correct --
            # exactly the impossible "0 + 0 = 17" pattern that exposed it.
            "direction": r.get("direction", ""),
            "path_label": r.get("path", ""),
            "exposure": r.get("exposure", ""),
            "outcome": r.get("outcome", ""),
            "microbiome_var": mic,
            "n": int(r["n"]),
            "beta_std": round(b, 5),
            "se": round(se, 5),
            "p_nhst": round(float(r["p"]), 6),
        }
        # smallest bound at which equivalence holds
        d_min = abs(b) + stats.t.ppf(1 - ALPHA, dfree) * se
        rec["delta_equiv_min"] = round(float(d_min), 5)
        # sensitivity: same bound expressed on the latent scale
        icc = ICC.get(mic)
        rec["icc_exposure"] = icc
        rec["delta_equiv_disattenuated"] = round(float(d_min / icc), 5) if icc else None
        # TOST at each conventional bound
        for d in BOUNDS:
            _, _, p = tost(b, se, d, dfree)
            rec[f"p_tost_{d:g}"] = round(p, 6)
            rec[f"equiv_{d:g}"] = bool(p < ALPHA)
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "equivalence_tests.csv", index=False)

    # ---------------------------------------------------------------- report
    banner("SMALLEST EQUIVALENCE BOUND PER TEST")
    log("  delta_equiv_min is the tightest interval within which the true effect")
    log("  can be placed at 95% confidence. It supersedes the MDE/ICC figure.\n")
    show = res[["direction", "microbiome_var", "outcome", "n", "beta_std", "se",
                "delta_equiv_min", "delta_equiv_disattenuated"]]
    log(show.to_string(index=False))

    for label, sub in (("Path A (microbiome -> aging)", res[res["direction"] == "A"]),
                       ("Path B (aging -> microbiome)", res[res["direction"] == "B"])):
        if sub.empty:
            continue
        lo, hi = sub["delta_equiv_min"].min(), sub["delta_equiv_min"].max()
        dlo = sub["delta_equiv_disattenuated"].min()
        dhi = sub["delta_equiv_disattenuated"].max()
        log(f"\n  {label}")
        log(f"    observed scale   : {lo:.3f} to {hi:.3f}")
        log(f"    latent scale     : {dlo:.3f} to {dhi:.3f}   (sensitivity only)")

    banner("EQUIVALENCE AT CONVENTIONAL BOUNDS")
    log("  Number of tests for which equivalence is established at alpha = 0.05.\n")
    log(f"  {'bound':<10}{'Path A':>10}{'Path B':>10}{'all 18':>10}")
    for d in BOUNDS:
        col = f"equiv_{d:g}"
        a = int(res[res['direction'] == 'A'][col].sum())
        b_ = int(res[res['direction'] == 'B'][col].sum())
        log(f"  +/-{d:<7g}{a:>10}{b_:>10}{int(res[col].sum()):>10}")

    log("\n  Reading: a test that is equivalent at +/-0.10 has ruled out any")
    log("  effect larger than a small standardised association. A test that is")
    log("  NOT equivalent even at +/-0.20 is simply uninformative -- neither")
    log("  significant nor bounded -- and should be described that way.")

    widest = f"equiv_{BOUNDS[-1]:g}"
    incon = res[(res["p_nhst"] > 0.05) & (~res[widest])]
    if len(incon):
        log(f"\n  {len(incon)} test(s) are neither significant nor equivalent at "
            f"+/-{BOUNDS[-1]:g}:")
        for _, r in incon.iterrows():
            log(f"    {r['direction']}  {r['microbiome_var']:<14} -> {r['outcome']:<34}"
                f" beta={r['beta_std']:+.4f}  delta_min={r['delta_equiv_min']:.3f}")

    # ---------------------------------------------------------------- figure
    try:
        pa = res[res["direction"] == "A"].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(pa) + 2.2))
        y = np.arange(len(pa))
        for i, r in pa.iterrows():
            lo = r["beta_std"] - 1.645 * r["se"]
            hi = r["beta_std"] + 1.645 * r["se"]
            ax.plot([lo, hi], [i, i], color="#1C7293", lw=2.6, solid_capstyle="butt")
            ax.scatter([r["beta_std"]], [i], color="#0D2B3E", s=42, zorder=5)
        for d, c in zip([0.10, 0.20], ["#C2410C", "#B8CBD6"]):
            ax.axvline(d, color=c, ls="--", lw=1.2)
            ax.axvline(-d, color=c, ls="--", lw=1.2)
            ax.text(d, len(pa) - 0.4, f"  ±{d:g}", color=c, fontsize=9, va="top")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['microbiome_var']} → {r['outcome']}"
                            for _, r in pa.iterrows()], fontsize=9)
        ax.set_xlabel("Standardised effect (90% CI, the interval used by TOST)")
        ax.set_title("Equivalence bounds, Path A (microbiome → aging)\n"
                     "An interval inside ±δ establishes equivalence at that bound",
                     fontsize=11)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(OUT / "equivalence_tests.png", dpi=150, bbox_inches="tight")
        plt.close()
        log(f"\n  wrote {OUT/'equivalence_tests.png'}")
    except Exception as exc:
        log(f"  figure failed: {exc}")

    log(f"  wrote {OUT/'equivalence_tests.csv'}")


if __name__ == "__main__":
    main()
