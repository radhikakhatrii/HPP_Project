#!/usr/bin/env python3
"""
build_table1_cohort.py
===============================================================================
Phase D (tab1_cohort) support script. Computes the demographic/characteristics
summary for the CORRECTED analytic strata (training n=6,807; longitudinal
n=1,823; leak-free AgeAccel-both-visits n=1,787) that tab1_cohort.tex needs and
that no committed result file currently carries. Reuses cross_lagged_panel's
own loaders exclusively -- same leak-free discipline, same primary age+sex
spec, deterministic (RANDOM_SEED=42 inherited from clp).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 build_table1_cohort.py 2>&1 | tee build_table1_cohort.log

Writes: results_cross_lagged/table1_cohort_characteristics.csv
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
log = clp.log


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def describe(df: pd.DataFrame, label: str, age_col="age_at_visit", sex_col="sex_encoded",
            bmi_col=None) -> dict:
    n = len(df)
    row = {"stratum": label, "n": n}
    if age_col in df.columns:
        row["age_mean"] = round(float(df[age_col].mean()), 2)
        row["age_sd"] = round(float(df[age_col].std()), 2)
        row["age_n"] = int(df[age_col].notna().sum())
    if sex_col in df.columns:
        # sex_encoded convention: confirm mapping direction against clp.load_population()
        # (0/1 -- treat as-is; report both the raw mean and n for transparency rather than
        # assuming which value means "female")
        row["sex_encoded_mean"] = round(float(df[sex_col].mean()), 4)
        row["sex_encoded_n"] = int(df[sex_col].notna().sum())
    if bmi_col and bmi_col in df.columns:
        row["bmi_mean"] = round(float(df[bmi_col].mean()), 2)
        row["bmi_sd"] = round(float(df[bmi_col].std()), 2)
        row["bmi_n"] = int(df[bmi_col].notna().sum())
    log(f"  {label:<45} n={n:,}  " +
        ("age=%.2f+/-%.2f (n=%d)  " % (row.get("age_mean", float('nan')),
                                      row.get("age_sd", float('nan')),
                                      row.get("age_n", 0)) if age_col in df.columns else "") +
        ("sex_encoded_mean=%.4f (n=%d)" % (row.get("sex_encoded_mean", float('nan')),
                                          row.get("sex_encoded_n", 0)) if sex_col in df.columns else ""))
    return row


def main() -> None:
    banner("TABLE 1 COHORT CHARACTERISTICS -- corrected strata")

    pop = clp.load_population()
    gut_full = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut_full)

    repeat_ids = set(gut_full.groupby("study_id")["visit"].nunique()
                     .pipe(lambda s: s[s >= 2]).index)
    single_ids = set(pop["study_id"]) - repeat_ids

    rows = []

    # --- CLOCK-TRAINING STRATUM (single-visit, n should land at 6,807) ---
    train_pop = pop[pop["study_id"].isin(single_ids)]
    rows.append(describe(train_pop, "Clock-training stratum (single-visit)"))

    # --- LONGITUDINAL STRATUM (repeat-visit, n should land at 1,823 with diversity) ---
    long_ids = set(gut_div["study_id"].unique())
    long_pop = pop[pop["study_id"].isin(long_ids)]
    rows.append(describe(long_pop, "Longitudinal stratum (repeat-visit, diversity both visits)"))

    # --- LEAK-FREE AgeAccel BOTH VISITS (n should land at 1,787) ---
    both_path = OUT / "leak_free_ageaccel_both_visits.csv"
    if both_path.exists():
        both_ids = set(pd.read_csv(both_path)["study_id"])
        ageaccel_pop = pop[pop["study_id"].isin(both_ids)]
        rows.append(describe(ageaccel_pop, "Leak-free AgeAccel, both visits (n=1,787 expected)"))
    else:
        log(f"  SKIPPED leak-free AgeAccel stratum: {both_path} not found. "
            f"Run cross_lagged_panel.py first (it writes this file at CHECK-2a).")

    # --- ANALYTIC SAMPLES per primary-spec cross-lag test (n=1,769/1,782/1,778) ---
    # Reconstruct via the same covariate-completeness gate cross_lag_ols() uses, on
    # age+sex only (PRIMARY_COVARIATES), for each of the 3 outcome datasets, using
    # shannon as the representative exposure (richness/gut_ageaccel share the same
    # covariate-completeness gate and land within a few participants of this by
    # construction -- see exclusion_funnel.csv for the exact per-exposure breakdown
    # already banked in ROUND 6's Table A9 answer).
    try:
        outcomes = clp.load_functional_outcomes()
        for ds_name, out_df in outcomes.items():
            wide_x = clp.make_wide(gut_div, "shannon")
            meas = next((c for c in out_df.columns
                        if c not in ("study_id", "visit", "visit_n")), None)
            wide_y = clp.make_wide(out_df, meas)
            merged = wide_x.join(wide_y, how="inner").join(
                pop.set_index("study_id")[clp.PRIMARY_COVARIATES], how="left")
            need = [f"shannon_t1", f"{meas}_t1", f"{meas}_t2"] + clp.PRIMARY_COVARIATES
            have = [c for c in need if c in merged.columns]
            complete = merged[have].apply(pd.to_numeric, errors="coerce").dropna()
            ids_in_sample = complete.index
            sample_pop = pop[pop["study_id"].isin(ids_in_sample)]
            rows.append(describe(sample_pop, f"Analytic sample -- {ds_name} (primary spec)"))
    except Exception as exc:
        log(f"  Analytic-sample reconstruction failed: {type(exc).__name__}: {exc} "
            f"-- skipped, not fabricated. The exclusion_funnel.csv numbers already "
            f"banked (ROUND 6) remain the authoritative per-test n's regardless of "
            f"whether this diagnostic block succeeds.")

    # --- FULL HPP POPULATION (reference row) ---
    rows.append(describe(pop, "Full HPP population (reference)"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "table1_cohort_characteristics.csv", index=False)
    log(f"\n  wrote {OUT / 'table1_cohort_characteristics.csv'}")
    banner("DONE")


if __name__ == "__main__":
    main()
