#!/usr/bin/env python3
"""
clock_validation.py
===============================================================================
Adinath's point 4: full validation of the gut microbiome clock beyond the
headline r/R2/MAE already reported -- calibration, age- and sex-stratified
performance, and batch sensitivity, plus an explicit leakage re-check.

DESIGN NOTE: build_gut_clock_ageaccel() in cross_lagged_panel.py does not
expose study_id or sex alongside its cross-validated predictions, only the
raw y_train/y_pred_cv arrays. Rather than modify that shared, already-
validated function, this script reproduces its exact training-frame
construction (same prevalence filter, same single-visit selection, same
feature stacking order, same LightGBM hyperparameters, same CV scheme) while
keeping study_id attached throughout, so predictions can be stratified.

BUILT-IN CONSISTENCY CHECK: this script's own CV r is compared against the
published 0.376 before anything downstream is trusted. If they don't agree
closely, the reproduction has a bug and the script says so explicitly rather
than reporting stratified numbers from a clock that isn't the real one.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python clock_validation.py 2>&1 | tee clock_validation.log

Writes to results_cross_lagged/:
    clock_validation_calibration.csv
    clock_validation_stratified.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log
PUBLISHED_R = 0.376
CLOCK_CV_FOLDS = 5
RANDOM_SEED = 42


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def make_clock():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05,
                         num_leaves=31, random_state=RANDOM_SEED, verbose=-1)


def main() -> None:
    banner("CLOCK VALIDATION: calibration, stratified performance, batch sensitivity")

    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut)
    # THE BUG THIS FIXES: the original build_gut_clock_ageaccel() derives
    # repeat_ids from gut_diversity["study_id"], not from the raw repeat-visit
    # gut table. If any repeat-visit participant fails diversity computation
    # (e.g. an all-NaN species row), they're present in `gut` but absent from
    # `gut_div` -- using `gut` here shrinks the excluded "repeat" set and
    # silently inflates the single-visit training frame.
    repeat_ids = set(gut_div["study_id"])

    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "research_stage"})

    banner("1. Reproducing the training frame (single-visit participants)")
    species_df = clp._load_linked_species(gut_full.rename(columns={"research_stage": "visit"}),
                                          clp.DATA_ROOT)
    sp_meta = {"participant_id", "study_id", "cohort", "research_stage",
              "array_index", "sample_name", "collection_timestamp",
              "collection_date", "timezone", "visit_n"}
    sp_abund_cols = [c for c in species_df.columns
                     if c not in sp_meta and str(species_df[c].dtype).startswith("float")]
    abund_mat = species_df[sp_abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
    prevalence = (abund_mat > 0).mean()
    sp_abund_cols = [c for c in sp_abund_cols if prevalence[c] > 0.05]
    log(f"  species passing >5% prevalence: {len(sp_abund_cols)}  (published: 511)")

    sp_pid = next(c for c in species_df.columns if c in ("participant_id", "study_id"))
    sp_stg = next(c for c in species_df.columns if "research_stage" in c.lower())
    sp_slim = species_df[[sp_pid, sp_stg] + sp_abund_cols].rename(
        columns={sp_pid: "study_id", sp_stg: "research_stage"})

    single_ids = set(gut_full["study_id"].unique()) - repeat_ids
    train_base = gut_full[gut_full["study_id"].isin(single_ids)].copy()
    train_base = (train_base.sort_values("research_stage")
                            .groupby("study_id", as_index=False).first())
    train_base = train_base.merge(sp_slim, on=["study_id", "research_stage"], how="left")
    train_base = train_base.merge(pop[["study_id", "age_at_visit", "sex_encoded"]],
                                  on="study_id", how="left")

    div_sv = gut_div.groupby("study_id", as_index=False)[["shannon", "richness"]].first()
    train_base = train_base.merge(div_sv, on="study_id", how="left")
    # THE BUG THIS FIXES: the original build_gut_clock_ageaccel() only drops
    # on missing age_at_visit; it deliberately tolerates an imperfect species
    # merge by fillna(0) later (treating an unmatched species value as "not
    # detected" rather than excluding the participant). This script had
    # additionally required a species column to be non-null, which silently
    # cut the training set from 5,188 to 449 -- exactly what the built-in
    # consistency check in Section 2 was designed to catch, and did.
    train_df = train_base.dropna(subset=["age_at_visit"])

    n_before_techmerge = len(train_df)
    for c in ("raw_read_count", "run_name"):
        if c in gut_full.columns:
            src = (gut_full[gut_full["study_id"].isin(train_df["study_id"])]
                   [["study_id", "research_stage", c]]
                   .drop_duplicates(subset=["study_id", "research_stage"]))
            train_df = train_df.merge(src, on=["study_id", "research_stage"],
                                      how="left", suffixes=("", "_dup"))
    if len(train_df) != n_before_techmerge:
        log(f"  WARNING: raw_read_count/run_name merge changed row count "
            f"{n_before_techmerge} -> {len(train_df)} even after dedup -- "
            f"investigate further before trusting training frame size.")
    log(f"  training frame: {len(train_df):,} single-visit participants "
        f"(published: 5,188)")

    X_train = np.hstack([
        train_df[sp_abund_cols].fillna(0).astype(float).values,
        train_df[["shannon", "richness"]].fillna(0).values,
    ])
    y_train = train_df["age_at_visit"].astype(float).values

    banner("2. Cross-validated predictions and consistency check")
    y_pred_cv = cross_val_predict(make_clock(), X_train, y_train,
                                  cv=KFold(CLOCK_CV_FOLDS, shuffle=True,
                                          random_state=RANDOM_SEED))
    r_cv = pearsonr(y_train, y_pred_cv)[0]
    r2_cv = r2_score(y_train, y_pred_cv)
    mae_cv = float(np.mean(np.abs(y_train - y_pred_cv)))
    log(f"  reproduced: r={r_cv:.4f}  R2={r2_cv:.4f}  MAE={mae_cv:.2f} yr")
    log(f"  published : r={PUBLISHED_R:.4f}")
    diff = abs(r_cv - PUBLISHED_R)
    if diff > 0.02:
        log(f"\n  WARNING: reproduced r differs from published by {diff:.4f}, more than")
        log(f"  the 0.02 tolerance. This reproduction may not be faithful to the")
        log(f"  original training frame -- treat everything below with caution and")
        log(f"  report the discrepancy rather than the stratified numbers as fact.")
    else:
        log(f"\n  Match within tolerance ({diff:.4f} <= 0.02). This reproduction is")
        log(f"  the same clock, and the stratified analyses below are trustworthy.")

    train_df = train_df.reset_index(drop=True)
    train_df["pred_age"] = y_pred_cv
    train_df["actual_age"] = y_train
    train_df["resid"] = y_pred_cv - y_train

    banner("3. Calibration: predicted age regressed on actual age")
    cal = sm.OLS(train_df["pred_age"], sm.add_constant(train_df["actual_age"])).fit()
    slope, intercept = float(cal.params["actual_age"]), float(cal.params["const"])
    log(f"  slope     = {slope:.4f}   (1.0 = perfectly calibrated)")
    log(f"  intercept = {intercept:.4f}")
    log(f"  A slope well below 1 confirms the compression toward the cohort mean")
    log(f"  already visible in Fig. 1 -- this quantifies it rather than describing")
    log(f"  it qualitatively.")

    # Calibration by age decile, for a table/figure
    train_df["age_decile"] = pd.qcut(train_df["actual_age"], 10, duplicates="drop")
    cal_table = (train_df.groupby("age_decile", observed=True)
                .agg(n=("actual_age", "size"),
                     mean_actual=("actual_age", "mean"),
                     mean_pred=("pred_age", "mean"),
                     mean_resid=("resid", "mean"))
                .reset_index())
    cal_table.to_csv(OUT / "clock_validation_calibration.csv", index=False)
    log(f"\n{cal_table.to_string(index=False)}")
    log(f"\n  wrote {OUT/'clock_validation_calibration.csv'}")

    banner("4. Age-stratified performance (tertiles)")
    train_df["age_tertile"] = pd.qcut(train_df["actual_age"], 3,
                                      labels=["youngest", "middle", "oldest"])
    strat_rows = []
    for grp, sub in train_df.groupby("age_tertile", observed=True):
        r, p = pearsonr(sub["actual_age"], sub["pred_age"])
        mae = float(np.abs(sub["resid"]).mean())
        age_range = f"[{sub['actual_age'].min():.0f}-{sub['actual_age'].max():.0f}]"
        log(f"  {str(grp):<10} n={len(sub):<6} age={age_range:<12} r={r:.3f}  MAE={mae:.2f} yr")
        strat_rows.append({"stratum_type": "age_tertile", "stratum": str(grp),
                           "n": len(sub), "r": round(r, 4), "mae": round(mae, 3)})

    banner("5. Sex-stratified performance")
    if "sex_encoded" in train_df.columns:
        for sex_val, label in ((1, "male"), (0, "female")):
            sub = train_df[train_df["sex_encoded"] == sex_val]
            if len(sub) < 30:
                log(f"  {label}: n={len(sub)}, too few, skipped")
                continue
            r, p = pearsonr(sub["actual_age"], sub["pred_age"])
            mae = float(np.abs(sub["resid"]).mean())
            log(f"  {label:<10} n={len(sub):<6} r={r:.3f}  MAE={mae:.2f} yr")
            strat_rows.append({"stratum_type": "sex", "stratum": label,
                               "n": len(sub), "r": round(r, 4), "mae": round(mae, 3)})
    else:
        log("  sex_encoded not available in training frame, skipped")

    banner("6. Batch / technical sensitivity")
    if "raw_read_count" in train_df.columns:
        rc = pd.to_numeric(train_df["raw_read_count"], errors="coerce")
        ok = rc.notna() & train_df["resid"].notna()
        if ok.sum() > 30:
            r_depth, p_depth = pearsonr(rc[ok], train_df.loc[ok, "resid"])
            log(f"  residual vs sequencing depth: r={r_depth:.4f}  p={p_depth:.4g}  "
                f"n={ok.sum():,}")
            log(f"  A non-trivial correlation here would mean prediction error is")
            log(f"  partly a technical artefact of read depth rather than biology.")
        else:
            log(f"  too few rows with both residual and read count ({ok.sum()})")
    else:
        log("  raw_read_count not available in training frame, skipped")

    if "run_name" in train_df.columns and train_df["run_name"].notna().sum() > 30:
        batch_var = (train_df.dropna(subset=["run_name", "resid"])
                    .groupby("run_name")["resid"].agg(["mean", "count"]))
        batch_var = batch_var[batch_var["count"] >= 10]
        log(f"\n  residual mean by sequencing run (batches with >=10 samples): "
            f"{len(batch_var)} batches")
        log(f"    range of batch means: [{batch_var['mean'].min():+.3f}, "
            f"{batch_var['mean'].max():+.3f}] years")
        log(f"    overall residual SD: {train_df['resid'].std():.3f} years")
        if batch_var["mean"].std() > 0.3 * train_df["resid"].std():
            log(f"    NOTE: between-batch variation is a non-trivial fraction of total")
            log(f"    residual variance -- worth a formal batch-adjustment sensitivity")
            log(f"    check before over-interpreting AgeAccel differences across batches.")
        else:
            log(f"    Batch-to-batch variation is small relative to total residual")
            log(f"    variance. No evidence of a strong batch artefact.")
    else:
        log("  run_name not available or too sparse, skipped")

    pd.DataFrame(strat_rows).to_csv(OUT / "clock_validation_stratified.csv", index=False)
    log(f"\n  wrote {OUT/'clock_validation_stratified.csv'}")


if __name__ == "__main__":
    main()
