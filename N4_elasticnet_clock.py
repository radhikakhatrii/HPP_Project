#!/usr/bin/env python3
"""
N4_elasticnet_clock.py
===============================================================================
Review point 9: Elastic-net baseline clock

Trains an elastic-net regularised regression on EXACTLY the same:
  • single-visit training stratum (same participants as the primary LightGBM clock)
  • species feature set (prevalence >5%, then +Shannon, +richness)
  • 5-fold cross-validation (CLOCK_CV_FOLDS from cross_lagged_panel.py)
  • RANDOM_SEED = 42

Reports CV Pearson r, R², MAE and saves per-participant AgeAccel for the
repeat-visit stratum.  The goal is to show the AgeAccel signal is not
artefactual to the specific ML algorithm.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 N4_elasticnet_clock.py 2>&1 | tee N4_elasticnet_clock.log

Writes:
    results_cross_lagged/review2/elasticnet_clock_metrics.csv
    results_cross_lagged/review2/elasticnet_ageaccel_repeat.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT  = Path(clp.RESULTS_DIR) / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log  = clp.log
SEED = clp.RANDOM_SEED
CV   = clp.CLOCK_CV_FOLDS


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main() -> None:
    banner("N4: ELASTIC-NET BASELINE CLOCK (review point 9)")

    # ── 1. Load exactly what the primary clock uses ───────────────────────
    banner("1. Loading data (same as primary LightGBM clock)")
    gut     = clp.load_longitudinal_gut()          # repeat-visit gut frame
    pop     = clp.load_population()
    div_long = clp.compute_diversity(gut)
    repeat_ids = set(div_long["study_id"].unique())
    log(f"  Repeat-visit (longitudinal) stratum: {len(repeat_ids):,}")

    gut_full = pd.read_parquet(
        clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next((c for c in gut_full.columns if "research_stage" in c.lower()), None)
    gut_full = gut_full.rename(columns={pid_col: "study_id"})
    if stg_col and stg_col != "research_stage":
        gut_full = gut_full.rename(columns={stg_col: "research_stage"})

    all_ids    = set(gut_full["study_id"].unique())
    single_ids = all_ids - repeat_ids
    log(f"  Single-visit (training) stratum: {len(single_ids):,}")

    # ── 2. Species features (same prevalence filter as primary clock) ─────
    banner("2. Loading species features (prevalence >5%)")
    from cross_lagged_panel import _load_linked_species  # internal helper

    species_df = _load_linked_species(gut_full, clp.DATA_ROOT)
    if species_df is None:
        log("  ERROR: species parquet not found — cannot build ElasticNet clock.")
        return

    sp_meta = {"participant_id", "study_id", "cohort", "research_stage",
               "array_index", "sample_name", "collection_timestamp",
               "collection_date", "timezone", "visit_n"}
    sp_abund_cols = [c for c in species_df.columns
                     if c not in sp_meta
                     and str(species_df[c].dtype).startswith("float")]

    abund_all = species_df[sp_abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
    prevalence = (abund_all > 0).mean()
    sp_abund_cols = [c for c in sp_abund_cols if prevalence[c] > 0.05]
    log(f"  Species passing prevalence >5%: {len(sp_abund_cols):,}")

    # flatten species_df to study_id + research_stage + features
    if species_df.index.names != [None]:
        species_df = species_df.reset_index()
    sp_pid   = next((c for c in species_df.columns if c in ("participant_id","study_id")), None)
    sp_stage = next((c for c in species_df.columns if "research_stage" in c.lower()), None)
    sp_slim  = species_df[[sp_pid, sp_stage] + sp_abund_cols].copy()
    sp_slim  = sp_slim.rename(columns={sp_pid: "study_id", sp_stage: "research_stage"})

    # ── 3. Training frame ─────────────────────────────────────────────────
    banner("3. Building training frame (single-visit participants)")
    train_base = gut_full[gut_full["study_id"].isin(single_ids)].copy()
    if "research_stage" in train_base.columns:
        train_base = (train_base.sort_values("research_stage")
                                .groupby("study_id", as_index=False).first())
    else:
        train_base = train_base.groupby("study_id", as_index=False).first()

    train_base = train_base.merge(
        pop[["study_id", "age_at_visit"]].copy(), on="study_id", how="left")

    # attach species abundances
    if "research_stage" in train_base.columns:
        train_base = train_base.merge(sp_slim, on=["study_id","research_stage"], how="left")
    else:
        sp_sv = sp_slim.groupby("study_id", as_index=False).first()
        train_base = train_base.merge(
            sp_sv.drop(columns=["research_stage"]), on="study_id", how="left")

    # attach diversity
    div_sv = div_long[div_long["study_id"].isin(single_ids)][
        ["study_id","shannon","richness"]].groupby("study_id", as_index=False).first()
    train_base = train_base.merge(div_sv, on="study_id", how="left")

    train_df = train_base.dropna(subset=["age_at_visit"])
    log(f"  Training frame: {len(train_df):,} participants")

    # leak-free check
    overlap = set(train_df["study_id"]) & repeat_ids
    if overlap:
        log(f"  FAIL CHECK-1: {len(overlap)} participants leaked into training!")
        return
    log(f"  CHECK-1 PASS: 0 participants leaked from longitudinal stratum.")

    # ── 4. Feature matrix ─────────────────────────────────────────────────
    banner("4. Building feature matrix")
    sp_present = [c for c in sp_abund_cols if c in train_df.columns]
    X_sp = train_df[sp_present].fillna(0).astype(float).values
    X_div = train_df[["shannon","richness"]].fillna(0).astype(float).values
    X_train = np.hstack([X_sp, X_div])
    y_train = train_df["age_at_visit"].astype(float).values
    log(f"  X shape: {X_train.shape}  (species={len(sp_present)}, +shannon, +richness)")
    log(f"  y (age) mean={y_train.mean():.1f} yr  SD={y_train.std():.1f} yr")

    # ── 5. ElasticNet 5-fold CV ───────────────────────────────────────────
    banner(f"5. ElasticNet {CV}-fold cross-validation (seed={SEED})")

    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
    enet_cv = ElasticNetCV(l1_ratio=l1_ratios, cv=CV,
                           max_iter=2000, random_state=SEED, n_jobs=-1)
    pipe = Pipeline([("scaler", StandardScaler()), ("enet", enet_cv)])

    kf = KFold(n_splits=CV, shuffle=True, random_state=SEED)
    y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=kf)

    r_cv  = float(pearsonr(y_train, y_pred_cv)[0])
    r2_cv = float(1 - np.sum((y_train - y_pred_cv)**2) /
                      np.sum((y_train - y_train.mean())**2))
    mae_cv = float(np.mean(np.abs(y_train - y_pred_cv)))
    log(f"  ElasticNet CV r   = {r_cv:.4f}")
    log(f"  ElasticNet CV R²  = {r2_cv:.4f}")
    log(f"  ElasticNet CV MAE = {mae_cv:.2f} yr")

    # Fit on full training set for prediction
    pipe.fit(X_train, y_train)
    # Best hyperparams (from the inner CV of ElasticNetCV)
    best_alpha   = float(pipe.named_steps["enet"].alpha_)
    best_l1ratio = float(pipe.named_steps["enet"].l1_ratio_)
    log(f"  Best alpha={best_alpha:.5f}, l1_ratio={best_l1ratio:.2f}")
    n_nonzero = int(np.sum(pipe.named_steps["enet"].coef_ != 0))
    log(f"  Non-zero coefficients: {n_nonzero} / {len(pipe.named_steps['enet'].coef_)}")

    # ── 6. AgeAccel for repeat-visit stratum ─────────────────────────────
    banner("6. Predicting ElasticNet AgeAccel for repeat-visit stratum")
    repeat_df = div_long[div_long["study_id"].isin(repeat_ids)].copy()
    repeat_df = repeat_df.merge(pop[["study_id","age_at_visit"]], on="study_id", how="left")

    # attach species for repeat-visit
    fg_stage = next((c for c in gut_full.columns if c == "research_stage"), None)
    if fg_stage:
        repeat_gut = gut_full[gut_full["study_id"].isin(repeat_ids)].copy()
        if "visit" in repeat_df.columns:
            repeat_gut2 = repeat_gut.rename(columns={"research_stage": "visit"})
            sp_repeat = sp_slim.rename(columns={"research_stage": "visit"})
            repeat_df = repeat_df.merge(
                sp_repeat[["study_id","visit"] + sp_present], on=["study_id","visit"], how="left")
        else:
            sp_repeat_sv = sp_slim.groupby("study_id", as_index=False).first()
            repeat_df = repeat_df.merge(
                sp_repeat_sv.drop(columns=["research_stage"]), on="study_id", how="left")
    else:
        sp_repeat_sv = sp_slim.groupby("study_id", as_index=False).first()
        repeat_df = repeat_df.merge(
            sp_repeat_sv.drop(columns=["research_stage"]), on="study_id", how="left")

    sp_in_repeat = [c for c in sp_present if c in repeat_df.columns]
    X_rep_sp  = repeat_df[sp_in_repeat].fillna(0).astype(float).values
    X_rep_div = repeat_df[["shannon","richness"]].fillna(0).astype(float).values

    # Pad missing species columns with zeros
    if len(sp_in_repeat) < len(sp_present):
        n_missing = len(sp_present) - len(sp_in_repeat)
        X_rep_sp = np.hstack([X_rep_sp, np.zeros((len(repeat_df), n_missing))])
    X_repeat = np.hstack([X_rep_sp, X_rep_div])

    y_rep_pred = pipe.predict(X_repeat)
    repeat_df["enet_bio_age"]   = y_rep_pred
    repeat_df["enet_ageaccel"]  = repeat_df["enet_bio_age"] - repeat_df["age_at_visit"]

    r_repeat = float(pearsonr(
        repeat_df["age_at_visit"].dropna(),
        repeat_df.loc[repeat_df["age_at_visit"].notna(), "enet_bio_age"])[0])
    log(f"  Repeat-visit: n={len(repeat_df):,}, Pearson r (bio_age vs chron_age) = {r_repeat:.4f}")
    log(f"  AgeAccel mean={repeat_df['enet_ageaccel'].mean():.3f}  "
        f"SD={repeat_df['enet_ageaccel'].std():.3f}")

    out_accel = repeat_df[["study_id","visit_n","enet_ageaccel"]].copy()
    out_accel.to_csv(OUT / "elasticnet_ageaccel_repeat.csv", index=False)
    log(f"  Wrote {OUT/'elasticnet_ageaccel_repeat.csv'}")

    # ── 7. Save metrics ───────────────────────────────────────────────────
    banner("7. Saving metrics")
    metrics = pd.DataFrame([{
        "model":           "ElasticNet",
        "training_n":      len(y_train),
        "n_features":      X_train.shape[1],
        "n_species":       len(sp_present),
        "cv_folds":        CV,
        "cv_r":            round(r_cv,  4),
        "cv_r2":           round(r2_cv, 4),
        "cv_mae_yr":       round(mae_cv, 3),
        "best_alpha":      round(best_alpha, 6),
        "best_l1_ratio":   round(best_l1ratio, 2),
        "n_nonzero_coef":  n_nonzero,
        "repeat_r":        round(r_repeat, 4),
        "ageaccel_mean":   round(float(repeat_df["enet_ageaccel"].mean()), 4),
        "ageaccel_sd":     round(float(repeat_df["enet_ageaccel"].std()),  4),
        "random_seed":     SEED,
    }])
    metrics.to_csv(OUT / "elasticnet_clock_metrics.csv", index=False)
    log(f"  Wrote {OUT/'elasticnet_clock_metrics.csv'}")

    banner("N4 COMPLETE")
    log(f"""
  Summary:
    Training n     : {len(y_train):,}
    Features       : {X_train.shape[1]:,} ({len(sp_present)} species + shannon + richness)
    CV r           : {r_cv:.4f}
    CV R²          : {r2_cv:.4f}
    CV MAE         : {mae_cv:.2f} yr
    Non-zero coefs : {n_nonzero}
    Repeat-visit r : {r_repeat:.4f}

  These metrics characterise the elastic-net as a baseline biological age clock.
  Comparison with the primary LightGBM clock's CV metrics in cross_lagged_panel.py
  quantifies how much the signal depends on algorithm choice.
""")


if __name__ == "__main__":
    main()
