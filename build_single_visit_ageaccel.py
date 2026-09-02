#!/usr/bin/env python3
"""
build_single_visit_ageaccel.py
===============================================================================
Compute cross-validated AgeAccel for the single-visit (clock-training) stratum.

The primary clock in cross_lagged_panel.py trains on single-visit participants
and predicts on repeat-visit participants (held-out).  To answer N3 Part B
("does AgeAccel add incremental value over Shannon+richness?"), we need
AgeAccel for those same single-visit participants.  We get unbiased estimates
via SAME K-fold cross-validated predictions used to evaluate the clock.

Identical to build_gut_clock_ageaccel() in cross_lagged_panel.py:
  • Species features: prevalence >5% (same 511 species)
  • Appended: Shannon, richness
  • Model: LightGBM (same as primary clock)
  • CV: CLOCK_CV_FOLDS-fold (5), RANDOM_SEED=42

AgeAccel = CV-predicted biological age − chronological age.

Saves:
    results_cross_lagged/single_visit_ageaccel.csv
        columns: study_id, chron_age, bio_age_cv, gut_ageaccel

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python build_single_visit_ageaccel.py 2>&1 | tee build_single_visit_ageaccel.log
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, KFold

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp
from cross_lagged_panel import _load_linked_species   # same helper as main clock

OUT_CSV = Path(clp.RESULTS_DIR) / "single_visit_ageaccel.csv"
log     = clp.log
SEED    = clp.RANDOM_SEED
CV      = clp.CLOCK_CV_FOLDS


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main() -> None:
    banner("BUILD SINGLE-VISIT AGEACCEL (CV predictions on training stratum)")

    # ── 1. Identify single-visit stratum (same as build_gut_clock_ageaccel) ─
    banner("1. Stratum identification")
    gut      = clp.load_longitudinal_gut()
    pop      = clp.load_population()
    div_long = clp.compute_diversity(gut)
    repeat_ids = set(div_long["study_id"].unique())
    log(f"  Repeat-visit stratum: {len(repeat_ids):,}")

    gut_full = pd.read_parquet(
        clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id","study_id"))
    stg_col = next((c for c in gut_full.columns if "research_stage" in c.lower()), None)
    gut_full = gut_full.rename(columns={pid_col: "study_id"})
    if stg_col and stg_col != "research_stage":
        gut_full = gut_full.rename(columns={stg_col: "research_stage"})

    single_ids = set(gut_full["study_id"].unique()) - repeat_ids
    log(f"  Single-visit stratum: {len(single_ids):,}")

    # ── 2. Species features (same prevalence filter) ──────────────────────
    banner("2. Loading species features (prevalence >5%)")
    species_df = _load_linked_species(gut_full, clp.DATA_ROOT)
    if species_df is None:
        log("  ERROR: species parquet not found."); return

    sp_meta = {"participant_id","study_id","cohort","research_stage",
               "array_index","sample_name","collection_timestamp",
               "collection_date","timezone","visit_n"}
    sp_abund_cols = [c for c in species_df.columns
                     if c not in sp_meta
                     and str(species_df[c].dtype).startswith("float")]
    abund_all  = species_df[sp_abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
    prevalence = (abund_all > 0).mean()
    sp_abund_cols = [c for c in sp_abund_cols if prevalence[c] > 0.05]
    log(f"  Species passing prevalence >5%: {len(sp_abund_cols):,}")

    if species_df.index.names != [None]:
        species_df = species_df.reset_index()
    sp_pid   = next((c for c in species_df.columns if c in ("participant_id","study_id")), None)
    sp_stage = next((c for c in species_df.columns if "research_stage" in c.lower()), None)
    sp_slim  = species_df[[sp_pid, sp_stage] + sp_abund_cols].copy()
    sp_slim  = sp_slim.rename(columns={sp_pid: "study_id", sp_stage: "research_stage"})

    # ── 3. Training frame (single-visit, one row each, lowest stage) ──────
    banner("3. Building training frame")
    train_base = gut_full[gut_full["study_id"].isin(single_ids)].copy()
    if "research_stage" in train_base.columns:
        train_base = (train_base.sort_values("research_stage")
                                .groupby("study_id", as_index=False).first())
    else:
        train_base = train_base.groupby("study_id", as_index=False).first()

    train_base = train_base.merge(
        pop[["study_id","age_at_visit","birth_year"]].copy(), on="study_id", how="left")

    # per-visit age from collection date (mirrors B3 fix in build_gut_clock_ageaccel)
    visit_age = clp._compute_visit_age(train_base)
    train_base["age_at_visit"] = visit_age.where(
        visit_age.notna(), train_base["age_at_visit"])

    # attach species
    if "research_stage" in train_base.columns:
        train_base = train_base.merge(sp_slim, on=["study_id","research_stage"], how="left")
    else:
        sp_sv = sp_slim.groupby("study_id", as_index=False).first()
        train_base = train_base.merge(
            sp_sv.drop(columns=["research_stage"]), on="study_id", how="left")

    # attach Shannon/richness from the longitudinal diversity frame (same source
    # as build_gut_clock_ageaccel line 891)
    div_sv = div_long[div_long["study_id"].isin(single_ids)][
        ["study_id","shannon","richness"]].groupby("study_id", as_index=False).first()
    train_base = train_base.merge(div_sv, on="study_id", how="left")

    train_df = train_base.dropna(subset=["age_at_visit"]).copy()
    log(f"  Training frame: {len(train_df):,} participants")

    # leak-free check
    overlap = set(train_df["study_id"]) & repeat_ids
    if overlap:
        log(f"  FAIL CHECK-1: {len(overlap)} participants leaked!"); return
    log(f"  CHECK-1 PASS: 0 participants leaked.")

    # ── 4. Feature matrix ─────────────────────────────────────────────────
    banner("4. Building feature matrix")
    sp_present = [c for c in sp_abund_cols if c in train_df.columns]
    X_sp  = train_df[sp_present].fillna(0).astype(float).values
    X_div = train_df[["shannon","richness"]].fillna(0).astype(float).values
    X     = np.hstack([X_sp, X_div])
    y     = train_df["age_at_visit"].astype(float).values
    ids   = train_df["study_id"].values
    log(f"  X: {X.shape}  (species={len(sp_present)}, +2 diversity)")
    log(f"  y (age): mean={y.mean():.1f} SD={y.std():.1f} yr")

    # ── 5. Cross-validated predictions (same as primary clock) ────────────
    banner(f"5. {CV}-fold CV predictions (seed={SEED})")
    try:
        from lightgbm import LGBMRegressor
        clock = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=SEED, verbose=-1)
        log("  Model: LightGBM (same as primary clock)")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        clock = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        log("  Model: HistGradientBoostingRegressor (LightGBM not available)")

    kf = KFold(n_splits=CV, shuffle=True, random_state=SEED)
    y_pred_cv = cross_val_predict(clock, X, y, cv=kf)

    r_cv   = float(pearsonr(y, y_pred_cv)[0])
    r2_cv  = float(r2_score(y, y_pred_cv))
    mae_cv = float(np.mean(np.abs(y - y_pred_cv)))
    log(f"  CV r={r_cv:.4f}  R²={r2_cv:.4f}  MAE={mae_cv:.2f} yr")
    log(f"  (Primary clock reports r≈0.380 — this should be close)")

    # ── 6. Compute AgeAccel and save ──────────────────────────────────────
    banner("6. Computing AgeAccel and saving")
    result = pd.DataFrame({
        "study_id":    ids,
        "chron_age":   y,
        "bio_age_cv":  y_pred_cv,
        "gut_ageaccel": y_pred_cv - y,
    })
    result.to_csv(OUT_CSV, index=False)
    log(f"  Saved {len(result):,} rows to {OUT_CSV}")
    log(f"  AgeAccel: mean={result['gut_ageaccel'].mean():.4f}  "
        f"SD={result['gut_ageaccel'].std():.4f}  "
        f"range=[{result['gut_ageaccel'].min():.2f}, {result['gut_ageaccel'].max():.2f}]")

    banner("DONE — now re-run N3_crosssectional.py for Part B")


if __name__ == "__main__":
    main()
