#!/usr/bin/env python3
"""
N5_null_enrichment_pathB.py
===============================================================================
Review points 1, 7, 8:

  (1) Null robustness: for every nominally null Path-A test, check 90% CI
      falls within ±0.10 standardised-effect equivalence margin and compare
      against BH-corrected threshold (q < 0.05).

  (7) P-value enrichment: is the distribution of Path-A p-values enriched
      below α=0.05 relative to a uniform null?  Binomial test + π₀ estimate.

  (8) Reverse-direction (Path-B) adiposity signal:
      Test whether adiposity(t₁) → microbiome-diversity(t₂) is significant,
      which would imply reverse causation confounds the forward interpretation.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 N5_null_enrichment_pathB.py 2>&1 | tee N5_null_enrichment_pathB.log

Reads:
    results_cross_lagged/cross_lag_all_paths.csv  (primary pipeline output)

Writes:
    results_cross_lagged/review2/N5_pathA_null_check.csv
    results_cross_lagged/review2/N5_pvalue_enrichment.csv
    results_cross_lagged/review2/N5_pathB_adiposity.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

RESULTS = Path(clp.RESULTS_DIR)
OUT     = RESULTS / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log     = clp.log

EQUIV_MARGIN = 0.10   # ±0.10 SD equivalence bound (pre-registered)
ALPHA        = 0.05


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main() -> None:
    banner("N5: NULL ROBUSTNESS + P-VALUE ENRICHMENT + PATH-B ADIPOSITY")

    # ── 0. Load main results ──────────────────────────────────────────────
    banner("0. Loading cross_lag_all_paths.csv")
    csv_path = RESULTS / "cross_lag_all_paths.csv"
    if not csv_path.exists():
        log(f"  ERROR: {csv_path} not found.")
        log("  Run the primary pipeline (cross_lagged_panel.py) first.")
        return

    raw = pd.read_csv(csv_path)
    log(f"  Loaded {len(raw):,} rows, columns: {list(raw.columns)}")

    # Identify spec column — take the primary/base spec (or all if only one)
    spec_col = next((c for c in raw.columns
                     if "spec" in c.lower() or "label" in c.lower()), None)
    if spec_col:
        spec_counts = raw[spec_col].value_counts()
        log(f"  spec values: {spec_counts.to_dict()}")
        # prefer 'primary'; fall back to 'base'; fall back to all rows
        primary_val = next(
            (v for v in spec_counts.index
             if "primary" in str(v).lower() or "base" in str(v).lower()),
            spec_counts.index[0])
        base = raw[raw[spec_col] == primary_val].copy()
        log(f"  Using spec='{primary_val}', rows: {len(base):,}")
    else:
        base = raw.copy()
        log(f"  No spec column — using all {len(base):,} rows")

    # Split Path-A and Path-B
    dir_col = next((c for c in base.columns if c.lower() == "direction"), None)
    if dir_col is None:
        log("  ERROR: no 'direction' column found.")
        return

    pathA = base[base[dir_col] == "A"].copy()
    pathB = base[base[dir_col] == "B"].copy()
    log(f"  Path-A rows: {len(pathA):,}   Path-B rows: {len(pathB):,}")

    # Identify p and beta columns
    p_col    = next(c for c in base.columns if c.lower() in ("p", "pval", "p_value"))
    beta_col = next(c for c in base.columns if c.lower() in ("beta", "coef", "coefficient"))
    se_col   = next((c for c in base.columns if c.lower() in ("se", "std_err", "stderr")), None)
    q_col    = next((c for c in base.columns if "fdr" in c.lower() or c.lower() == "q"), None)

    log(f"  Using columns: beta={beta_col}, p={p_col}, se={se_col}, q={q_col}")

    # Exposure / outcome columns
    exp_col = next(c for c in base.columns if "expos" in c.lower())
    out_col = next(c for c in base.columns if "outcom" in c.lower())

    # ── 1. Null robustness: equivalence check for Path-A ─────────────────
    banner("1. Path-A null robustness (review point 1)")

    if len(pathA) == 0:
        log("  ERROR: no Path-A rows found — check direction column values:")
        log(f"  {base[dir_col].value_counts().to_dict()}")
        return

    pathA = pathA.copy()

    # Use existing ci95 column if present (format "[lo, hi]"), else compute from se
    ci95_col = next((c for c in pathA.columns if "ci95" in c.lower() or c.lower() == "ci_95"), None)
    if ci95_col and pathA[ci95_col].dtype == object:
        # parse "[lo, hi]" strings
        parsed = pathA[ci95_col].str.strip("()[] ").str.split(",", expand=True).astype(float)
        pathA["ci_lo_95"] = parsed.iloc[:, 0].values
        pathA["ci_hi_95"] = parsed.iloc[:, 1].values
        se_from_ci = (pathA["ci_hi_95"] - pathA["ci_lo_95"]) / (2 * norm.ppf(0.975))
        pathA["ci_lo_90"] = pathA[beta_col] - norm.ppf(0.95) * se_from_ci
        pathA["ci_hi_90"] = pathA[beta_col] + norm.ppf(0.95) * se_from_ci
    elif se_col and se_col in pathA.columns:
        pathA["ci_lo_90"] = pathA[beta_col] - norm.ppf(0.95) * pathA[se_col]
        pathA["ci_hi_90"] = pathA[beta_col] + norm.ppf(0.95) * pathA[se_col]
        pathA["ci_lo_95"] = pathA[beta_col] - norm.ppf(0.975) * pathA[se_col]
        pathA["ci_hi_95"] = pathA[beta_col] + norm.ppf(0.975) * pathA[se_col]
    else:
        z = np.where(pathA[p_col] > 0, norm.ppf(1 - pathA[p_col] / 2), 50.0)
        se_est = np.abs(pathA[beta_col].values) / np.maximum(z, 1e-9)
        pathA["ci_lo_90"] = pathA[beta_col] - norm.ppf(0.95) * se_est
        pathA["ci_hi_90"] = pathA[beta_col] + norm.ppf(0.95) * se_est
        pathA["ci_lo_95"] = pathA[beta_col] - norm.ppf(0.975) * se_est
        pathA["ci_hi_95"] = pathA[beta_col] + norm.ppf(0.975) * se_est

    # BH correction on Path-A p-values
    _, q_bh, _, _ = multipletests(pathA[p_col].fillna(1), method="fdr_bh")
    pathA["q_bh_n5"] = q_bh

    # Equivalence: both bounds of 90% CI inside (−EQUIV, +EQUIV) → equivalent to null
    pathA["equiv_null"] = (
        (pathA["ci_lo_90"] > -EQUIV_MARGIN) &
        (pathA["ci_hi_90"] <  EQUIV_MARGIN))
    pathA["nom_sig"] = pathA[p_col] < ALPHA
    pathA["fdr_sig"] = pathA["q_bh_n5"] < ALPHA

    log(f"\n  Path-A summary ({len(pathA)} tests):")
    log(f"  Nominally significant (p<0.05):   {pathA['nom_sig'].sum()}")
    log(f"  FDR-significant (q<0.05, BH):     {pathA['fdr_sig'].sum()}")
    log(f"  Equivalent to null (±{EQUIV_MARGIN} margin): {pathA['equiv_null'].sum()}")

    log(f"\n  {'Exposure':<15} {'Outcome':<8} {'β':>8} {'p':>10} {'q_BH':>8} {'CI90':^22} {'equiv?'}")
    log(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*22} {'-'*7}")
    for _, row in pathA.sort_values(p_col).iterrows():
        equiv = "YES" if row["equiv_null"] else "NO "
        sig   = "**" if row["fdr_sig"] else ("*" if row["nom_sig"] else "  ")
        log(f"  {str(row[exp_col]):<15} {str(row[out_col]):<8} "
            f"{row[beta_col]:>+8.4f} {row[p_col]:>10.3g} {row['q_bh_n5']:>8.3g} "
            f"[{row['ci_lo_90']:>+7.4f},{row['ci_hi_90']:>+7.4f}]  {equiv} {sig}")

    pathA.to_csv(OUT / "N5_pathA_null_check.csv", index=False)
    log(f"\n  Wrote {OUT/'N5_pathA_null_check.csv'}")

    # ── 2. P-value enrichment (review point 7) ───────────────────────────
    banner("2. P-value enrichment for Path-A (review point 7)")

    p_vals = pathA[p_col].dropna().values
    n_tests = len(p_vals)
    n_below_05 = int((p_vals < 0.05).sum())
    n_below_01 = int((p_vals < 0.01).sum())
    n_below_001 = int((p_vals < 0.001).sum())

    # Binomial test: under uniform null, expect 5% below 0.05
    binom_05 = binomtest(n_below_05, n_tests, 0.05, alternative="greater")
    # π₀ estimate (Storey's method, simplified): proportion of p > 0.5
    pi0_est = min(1.0, 2 * (p_vals > 0.5).mean())

    log(f"  Total Path-A tests: {n_tests}")
    log(f"  p < 0.05: {n_below_05} / {n_tests} = {n_below_05/n_tests:.1%}  "
        f"(expected {0.05*n_tests:.1f} under null)")
    log(f"  p < 0.01: {n_below_01} / {n_tests} = {n_below_01/n_tests:.1%}  "
        f"(expected {0.01*n_tests:.1f} under null)")
    log(f"  p < 0.001: {n_below_001} / {n_tests} = {n_below_001/n_tests:.1%}  "
        f"(expected {0.001*n_tests:.1f} under null)")
    log(f"  Binomial test (p<0.05 enrichment): p = {binom_05.pvalue:.4g}")
    log(f"  Storey π₀ estimate: {pi0_est:.3f}  "
        f"({'most tests are null' if pi0_est > 0.8 else 'some true signal present'})")

    enrich_df = pd.DataFrame([{
        "n_tests": n_tests,
        "n_p_lt_005": n_below_05,
        "n_p_lt_001": n_below_01,
        "n_p_lt_0001": n_below_001,
        "frac_p_lt_005": round(n_below_05 / n_tests, 4),
        "expected_p_lt_005_under_null": round(0.05 * n_tests, 2),
        "binomial_p_enrichment": round(binom_05.pvalue, 6),
        "storey_pi0": round(pi0_est, 4),
        "n_fdr_sig_q005": int(pathA["fdr_sig"].sum()),
    }])
    enrich_df.to_csv(OUT / "N5_pvalue_enrichment.csv", index=False)
    log(f"  Wrote {OUT/'N5_pvalue_enrichment.csv'}")

    # P-value distribution summary
    quantiles = np.quantile(p_vals, [0.1, 0.25, 0.5, 0.75, 0.9])
    log(f"  P-value quantiles [10th, 25th, 50th, 75th, 90th]:")
    log(f"  {[round(q,3) for q in quantiles]}")

    # ── 3. Path-B adiposity → microbiome (review point 8) ────────────────
    banner("3. Path-B: adiposity(t₁) → microbiome-diversity(t₂) (review point 8)")

    # Path-B: exposure = phenotype (grip/sbp/waist), outcome = microbiome var
    adiposity_outs = {"waist", "bmi", "sbp", "grip"}
    micro_exps     = {"shannon", "richness", "gut_ageaccel", "ageaccel"}

    # In Path-B, the roles are swapped: exposure=phenotype, outcome=microbiome
    # OR: direction=="B" means outcome(t1)→microbiome(t2)
    # Filter for paths where EITHER the outcome is a microbiome var (direction B)
    # or exposure is adiposity/phenotype
    pb = pathB.copy()

    if len(pb) == 0:
        log("  No Path-B rows found.")
        log(f"  All direction values in base: {base[dir_col].value_counts().to_dict()}")
        pd.DataFrame().to_csv(OUT / "N5_pathB_adiposity.csv", index=False)
    else:
        # Adiposity-relevant Path-B: exposures that are adiposity phenotypes → microbiome
        pb["exp_lower"] = pb[exp_col].str.lower()
        pb["out_lower"] = pb[out_col].str.lower()

        # Path B: outcome(t1) → microbiome(t2) [direction=B means microbiome is outcome]
        # OR: exposure=phenotype, outcome=microbiome (depends on pipeline convention)
        adip_pb = pb[
            pb["exp_lower"].str.contains("waist|bmi|sbp|grip|blood|anthrop", na=False) |
            pb["out_lower"].str.contains("waist|bmi|sbp|grip|blood|anthrop", na=False)
        ].copy()

        if len(adip_pb) == 0:
            # Show all Path-B to help identify naming
            log("  No adiposity Path-B found with name filter — showing all Path-B:")
            log(f"\n  {pb[[exp_col, out_col, beta_col, p_col]].to_string(index=False)}")
            adip_pb = pb.copy()

        # BH correction on all Path-B
        _, q_bh_b, _, _ = multipletests(pb[p_col].fillna(1), method="fdr_bh")
        pb["q_bh_n5"] = q_bh_b
        adip_pb = pb.copy()   # show all Path-B

        log(f"\n  All Path-B results ({len(adip_pb)} tests):")
        log(f"  {'Exposure':<20} {'Outcome':<20} {'β':>8} {'p':>10} {'q_BH':>8} {'sig'}")
        log(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*4}")
        for _, row in adip_pb.sort_values(p_col).iterrows():
            sig = "**" if row["q_bh_n5"] < 0.05 else ("*" if row[p_col] < 0.05 else "  ")
            log(f"  {str(row[exp_col]):<20} {str(row[out_col]):<20} "
                f"{row[beta_col]:>+8.4f} {row[p_col]:>10.3g} {row['q_bh_n5']:>8.3g} {sig}")

        n_pathB_sig = int((pb[p_col] < 0.05).sum())
        n_pathB_fdr = int((pb["q_bh_n5"] < 0.05).sum())
        log(f"\n  Path-B nominally significant: {n_pathB_sig} / {len(pb)}")
        log(f"  Path-B FDR significant:       {n_pathB_fdr} / {len(pb)}")

        adip_pb.to_csv(OUT / "N5_pathB_adiposity.csv", index=False)
        log(f"  Wrote {OUT/'N5_pathB_adiposity.csv'}")

    banner("N5 COMPLETE")
    log("""
  Summary:
    Review point 1 (null robustness):
      See N5_pathA_null_check.csv — which paths have 90% CI inside ±0.10 margin?
    Review point 7 (p-value enrichment):
      See N5_pvalue_enrichment.csv — binomial test + π₀ estimate.
    Review point 8 (reverse adiposity):
      See N5_pathB_adiposity.csv — Path-B coefficients and significance.
""")


if __name__ == "__main__":
    main()
