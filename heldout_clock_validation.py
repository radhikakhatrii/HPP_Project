#!/usr/bin/env python3
"""
heldout_clock_validation.py
===============================================================================
AQ-18: Held-out evaluation of the frozen gut microbiome age clock.

The clock is trained on the single-visit stratum (n ≈ 6,807) — identically
to the main pipeline in build_gut_clock_ageaccel() — and applied WITHOUT
retraining to the longitudinal stratum at visit 1. The two strata are
disjoint by construction (CHECK-1 in the main pipeline). This gives an
independent, held-out estimate of clock performance on a population the
model has never seen.

Metrics computed and banked:
  n                  — participants in held-out evaluation
  r                  — Pearson(predicted, chronological age)
  r_sq               — r²
  mae                — mean |predicted − chronological| in years
  slope              — OLS slope (predicted ~ chronological age)
  intercept          — OLS intercept (years)
  mae_null           — MAE of the stratum-mean-age null model
  perm_pct           — percentile of observed r among 10,000 shuffled-age runs

Provenance: every value from this run's live computation.
Expected: r ≈ 0.38 (near CV r), slope ≈ 0.15–0.20 (classical attenuation).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 heldout_clock_validation.py 2>&1 | tee heldout_clock_validation.log

Writes: results_cross_lagged/heldout_clock_validation.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

RANDOM_SEED   = clp.RANDOM_SEED       # 42
N_PERMUTATIONS = 10_000
OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main() -> None:
    banner("AQ-18: HELD-OUT CLOCK VALIDATION (longitudinal stratum, visit 1)")

    # ------------------------------------------------------------------ data
    banner("1. Loading data")
    gut      = clp.load_longitudinal_gut()          # repeat-visit stratum only
    pop      = clp.load_population()
    gut_div  = clp.compute_diversity(gut)

    repeat_ids = set(gut_div["study_id"].unique())
    log(f"  Longitudinal stratum: {len(repeat_ids):,} participants")

    gut_full = pd.read_parquet(
        clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col  = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col  = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    # --------------------------------------------------------- species matrix
    banner("2. Building species feature matrix")
    species_df = clp._load_linked_species(gut_full, clp.DATA_ROOT)

    sp_meta = {"participant_id", "study_id", "cohort", "research_stage",
               "array_index", "sample_name", "collection_timestamp",
               "collection_date", "timezone", "visit_n"}

    if species_df is not None:
        if species_df.index.names != [None]:
            species_df = species_df.reset_index()
        sp_abund_cols = [c for c in species_df.columns
                         if c not in sp_meta
                         and str(species_df[c].dtype).startswith("float")]
        abund_mat = species_df[sp_abund_cols].apply(
            pd.to_numeric, errors="coerce").fillna(0).abs()
        prevalence = (abund_mat > 0).mean()
        sp_abund_cols = [c for c in sp_abund_cols if prevalence[c] > 0.05]
        log(f"  {len(sp_abund_cols)} species pass prevalence >5%")
        feature_mode = "species"

        pid_sp  = next(c for c in species_df.columns if c in ("participant_id", "study_id"))
        stg_sp  = next(c for c in species_df.columns if "research_stage" in c.lower())
        sp_slim = species_df[[pid_sp, stg_sp] + sp_abund_cols].copy()
        sp_slim = sp_slim.rename(columns={pid_sp: "study_id",
                                          stg_sp: "research_stage"})
    else:
        sp_abund_cols = []
        sp_slim = None
        feature_mode = "diversity_only"
        log("  Species parquet not found — falling back to diversity only")

    # ------------------------------------------------------- training frame
    banner("3. Building training frame (single-visit stratum)")
    single_ids = set(gut_full["study_id"].unique()) - repeat_ids
    train_base = (gut_full[gut_full["study_id"].isin(single_ids)]
                  .sort_values("visit")
                  .groupby("study_id", as_index=False).first())
    train_base = train_base.merge(
        pop[["study_id", "age_at_visit", "birth_year", "sex_encoded"]],
        on="study_id", how="left")

    # Per-visit age (B3 fix: use actual collection date, not birth_year alone)
    visit_age = clp._compute_visit_age(train_base)
    train_base["age_at_visit"] = visit_age.where(
        visit_age.notna(), train_base["age_at_visit"])

    # attach species
    if sp_slim is not None and "visit" in train_base.columns:
        train_base = train_base.merge(
            sp_slim.rename(columns={"research_stage": "visit"}),
            on=["study_id", "visit"], how="left")
    elif sp_slim is not None:
        sv_sp = sp_slim.groupby("study_id", as_index=False).first()
        train_base = train_base.merge(
            sv_sp.drop(columns=["research_stage"]), on="study_id", how="left")

    # attach diversity
    div_sv = (gut_div[gut_div["study_id"].isin(single_ids)]
              [["study_id", "shannon", "richness"]]
              .groupby("study_id", as_index=False).first())
    train_base = train_base.merge(div_sv, on="study_id", how="left")
    train_df = train_base.dropna(subset=["age_at_visit"])

    # CHECK-1: no overlap with longitudinal stratum
    overlap = set(train_df["study_id"]) & repeat_ids
    if overlap:
        log(f"  [CHECK-1] FAIL: {len(overlap)} participants leaked into training!")
        raise SystemExit("Leak detected — aborting held-out evaluation.")
    log(f"  [CHECK-1] PASS: training n={len(train_df):,}, "
        f"longitudinal n={len(repeat_ids):,}, overlap=0")

    # feature matrix
    if sp_abund_cols and all(c in train_df.columns for c in sp_abund_cols):
        X_train = train_df[sp_abund_cols].fillna(0).astype(float).values
        if "shannon" in train_df.columns:
            X_train = np.hstack([X_train,
                                 train_df[["shannon", "richness"]].fillna(0).values])
    elif "shannon" in train_df.columns:
        X_train = train_df[["shannon", "richness"]].fillna(0).values
        sp_abund_cols = []
        feature_mode = "diversity_only"
    else:
        X_train = np.zeros((len(train_df), 2))
        sp_abund_cols = []
        feature_mode = "diversity_only"

    y_train = train_df["age_at_visit"].astype(float).values
    log(f"  Training frame: n={len(y_train):,}, mode={feature_mode}")

    # ----------------------------------------------------------------- clock
    banner("4. Training clock (same hyperparameters as main pipeline)")
    try:
        from lightgbm import LGBMRegressor
        clock = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=RANDOM_SEED, verbose=-1)
        log("  Classifier: LGBMRegressor (n_estimators=300, lr=0.05, leaves=31)")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        clock = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_SEED)
        log("  Classifier: HistGradientBoostingRegressor (LightGBM not available)")

    clock.fit(X_train, y_train)
    log("  Clock fitted.")

    # ------------------------------------------------- held-out prediction
    banner("5. Applying clock to longitudinal stratum (visit 1 only)")
    # visit-1 rows in gut_div
    long_v1 = gut_div[gut_div["visit_n"] == 1].copy()
    long_v1 = long_v1.merge(
        pop[["study_id", "age_at_visit", "birth_year"]], on="study_id", how="left")

    # per-visit age: attach collection dates from gut_full (same B3 fix)
    date_col = clp._find_collection_date_col(gut_full)
    if date_col is not None and "visit" in gut_full.columns:
        date_slim = (gut_full[gut_full["visit_n"] == 1]
                     [["study_id", date_col]]
                     .drop_duplicates("study_id"))
        long_v1 = long_v1.merge(date_slim, on="study_id", how="left")
        va = clp._compute_visit_age(long_v1)
        long_v1["age_at_visit"] = va.where(va.notna(), long_v1["age_at_visit"])

    # attach species at visit 1.
    # gut_div (source of long_v1) already carries the raw 'visit' stage label —
    # compute_diversity() preserves it at line 739 — so no extra merge needed
    # to acquire it. Merging gut_v1_stage would collide and rename it visit_x/y.
    if sp_slim is not None and "visit" in long_v1.columns:
        long_v1 = long_v1.merge(
            sp_slim.rename(columns={"research_stage": "visit"}),
            on=["study_id", "visit"], how="left")
    elif sp_slim is not None:
        # fallback: gut_div had no 'visit' — get stage label from gut_full
        gut_v1_stage = (gut_full[gut_full["visit_n"] == 1]
                        [["study_id", "visit"]]
                        .drop_duplicates("study_id"))
        long_v1 = long_v1.merge(gut_v1_stage, on="study_id", how="left")
        long_v1 = long_v1.merge(
            sp_slim.rename(columns={"research_stage": "visit"}),
            on=["study_id", "visit"], how="left")

    long_v1 = long_v1.dropna(subset=["age_at_visit"])
    log(f"  Longitudinal stratum, visit 1, age available: n={len(long_v1):,}")

    # build prediction feature matrix
    if sp_abund_cols and all(c in long_v1.columns for c in sp_abund_cols):
        species_matched = long_v1[sp_abund_cols].notna().any(axis=1).values
        X_heldout = long_v1[sp_abund_cols].fillna(0).astype(float).values
        X_heldout = np.hstack([X_heldout,
                               long_v1[["shannon", "richness"]].fillna(0).values])
        n_unmatched = int((~species_matched).sum())
        if n_unmatched:
            log(f"  WARNING: {n_unmatched} visit-1 rows have no species match "
                f"and will be excluded from metrics (not zero-filled).")
    else:
        X_heldout = long_v1[["shannon", "richness"]].fillna(0).values
        species_matched = np.ones(len(long_v1), dtype=bool)

    bio_age_pred = clock.predict(X_heldout).astype(float)
    bio_age_pred[~species_matched] = np.nan

    long_v1 = long_v1.reset_index(drop=True)
    long_v1["bio_age_pred"] = bio_age_pred
    eval_df = long_v1[["study_id", "age_at_visit", "bio_age_pred"]].dropna()
    log(f"  Participants with valid held-out prediction: n={len(eval_df):,}")

    # ---------------------------------------------------------------- metrics
    banner("6. Computing held-out metrics")
    y_true = eval_df["age_at_visit"].astype(float).values
    y_pred = eval_df["bio_age_pred"].astype(float).values
    n_eval = len(eval_df)

    r, r_pval = pearsonr(y_pred, y_true)
    r_sq = r ** 2
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # OLS: predicted ~ chronological (slope < 1 = attenuation; expected ~0.15–0.20)
    ols = sm.OLS(y_pred, sm.add_constant(y_true)).fit()
    slope     = float(ols.params[1])
    intercept = float(ols.params[0])

    # null model: predict stratum mean age for everyone
    mean_age = float(y_true.mean())
    mae_null = float(np.mean(np.abs(mean_age - y_true)))

    # permutation test: shuffle chronological ages, recompute r, 10,000×
    rng = np.random.default_rng(RANDOM_SEED)
    perm_r = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        y_shuffled = rng.permutation(y_true)
        perm_r[i] = pearsonr(y_pred, y_shuffled)[0]
    perm_pct = float(np.mean(perm_r < r)) * 100.0   # % of permutations beaten

    log(f"\n  ── HELD-OUT METRICS ──────────────────────────────────────────────")
    log(f"  n                         = {n_eval:,}")
    log(f"  Pearson r                 = {r:.4f}")
    log(f"  r²                        = {r_sq:.4f}")
    log(f"  MAE                       = {mae:.3f} yr")
    log(f"  OLS slope  (pred ~ chron) = {slope:.4f}")
    log(f"  OLS intercept             = {intercept:.3f} yr")
    log(f"  MAE_null (mean-age model) = {mae_null:.3f} yr")
    log(f"  Permutation pct (10k, seed={RANDOM_SEED}): r exceeds {perm_pct:.2f}% of shuffles")
    log(f"  ──────────────────────────────────────────────────────────────────")

    # ------------------------------------------------------------------ save
    row = pd.DataFrame([{
        "n":           n_eval,
        "r":           round(r,         6),
        "r_sq":        round(r_sq,      6),
        "mae":         round(mae,       4),
        "slope":       round(slope,     6),
        "intercept":   round(intercept, 4),
        "mae_null":    round(mae_null,  4),
        "perm_pct":    round(perm_pct,  2),
        "n_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "feature_mode": feature_mode,
    }])
    out_path = OUT / "heldout_clock_validation.csv"
    row.to_csv(out_path, index=False)
    log(f"\n  [AQ-18] wrote {out_path}")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
