#!/usr/bin/env python3
"""
N3_ageaccel.py
===============================================================================
Nastya review point 3 — AgeAccel on the single-visit stratum.

Task A  Apply the frozen leak-free clock (same spec as build_gut_clock_ageaccel)
        to the ~6,807 single-visit participants at visit 1.
        Residualise predicted age on chronological age (OLS) → AgeAccel.
        Save: results_cross_lagged/review2/single_visit_ageaccel.csv
              columns: participant_id, chron_age, sex, predicted_age, ageaccel

Task B  Partial Pearson r (adjusting age + sex) of AgeAccel with grip, SBP,
        waist, BMI. Same stratum, same per-phenotype n as the Shannon/richness
        rows in crosssectional_matrix.csv.
        Appends AgeAccel rows to crosssectional_matrix.csv.

Task C  Nested model comparison per phenotype:
          Model 1: phenotype ~ Shannon + richness + age + sex
          Model 2: phenotype ~ Shannon + richness + AgeAccel + age + sex
        Reports adj R², ΔR², AgeAccel std-beta, 95% CI, P.
        Save: results_cross_lagged/review2/ageaccel_incremental.csv

Deliverable
        N3_AGEACCEL_NUMBERS.md — paste-ready changelog.

Hard rules
  • Uses the same clock hyperparameters, same features, same training stratum
    as build_gut_clock_ageaccel() — do not retrain with different settings.
  • Cross-validated predictions (cross_val_predict) to avoid in-sample bias.
  • AgeAccel = OLS residual of predicted_age on chron_age (same as in meth:ageaccel).
  • Never touches the Shannon/richness rows already in crosssectional_matrix.csv.

Usage:
    python N3_ageaccel.py 2>&1 | tee N3_ageaccel.log
===============================================================================
"""
from __future__ import annotations

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, KFold

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp
from cross_lagged_panel import _load_linked_species

SEED  = clp.RANDOM_SEED   # 42
CV    = clp.CLOCK_CV_FOLDS # 5
OUT   = Path(clp.RESULTS_DIR) / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log   = clp.log


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def partial_r(df: pd.DataFrame, x: str, y: str,
              covariates: list[str]) -> tuple[float, float, int]:
    """Partial Pearson r via residuals-on-residuals (same as N3_crosssectional.py)."""
    cols = [x, y] + covariates
    sub  = df[cols].dropna().astype(float)
    n    = len(sub)
    if n < 30:
        return np.nan, np.nan, n

    def resid(col: str) -> np.ndarray:
        formula = f"Q('{col}') ~ " + " + ".join(f"Q('{c}')" for c in covariates)
        try:
            return smf.ols(formula, data=sub).fit().resid.values
        except Exception:
            return np.full(n, np.nan)

    rx, ry = resid(x), resid(y)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r, p = pearsonr(rx[mask], ry[mask])
    return float(r), float(p), int(mask.sum())


def load_pheno(path: Path, candidates: list[str], label: str,
               keep_ids: set) -> pd.DataFrame | None:
    """Load one phenotype parquet, visit 1 only, one row per participant."""
    import pyarrow.parquet as pq
    if not path.exists():
        log(f"  {label}: parquet not found at {path}")
        return None
    try:
        cols = pq.read_schema(path).names
    except Exception as e:
        log(f"  {label}: schema error — {e}")
        return None
    pid   = next((c for c in cols if c in ("participant_id", "study_id")), None)
    stage = next((c for c in cols if "research_stage" in c.lower()), None)
    meas  = next((c for c in candidates if c in cols), None)
    if not (pid and stage and meas):
        log(f"  {label}: expected cols not found (pid={pid} stage={stage} meas={meas})")
        return None
    df = pd.read_parquet(path, columns=[pid, stage, meas])
    if pid not in df.columns:
        df = df.reset_index()
    df = df.rename(columns={pid: "study_id", stage: "visit"})
    df[meas] = pd.to_numeric(df[meas], errors="coerce")
    df = df[df["study_id"].isin(keep_ids)]
    df["visit_n"] = df.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))
    df = df[df["visit_n"] == 1].sort_values("visit").groupby("study_id", as_index=False).first()
    log(f"  {label}: {len(df):,} participants at visit 1")
    return df[["study_id", meas]].rename(columns={meas: label})


# ── Task A: AgeAccel on single-visit stratum ─────────────────────────────────

def task_a() -> pd.DataFrame | None:
    """Return DataFrame[participant_id, chron_age, sex, predicted_age, ageaccel]."""
    banner("TASK A — Apply frozen clock to single-visit stratum")

    # 1. Stratum identification (mirrors build_gut_clock_ageaccel exactly)
    gut      = clp.load_longitudinal_gut()
    pop      = clp.load_population()
    div_long = clp.compute_diversity(gut)
    repeat_ids = set(div_long["study_id"].unique())
    log(f"  Repeat-visit stratum: {len(repeat_ids):,}")

    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next((c for c in gut_full.columns if "research_stage" in c.lower()), None)
    gut_full = gut_full.rename(columns={pid_col: "study_id"})
    if stg_col and stg_col != "research_stage":
        gut_full = gut_full.rename(columns={stg_col: "research_stage"})

    single_ids = set(gut_full["study_id"].unique()) - repeat_ids
    log(f"  Single-visit stratum: {len(single_ids):,}")

    # 2. Species features (same prevalence filter as primary clock)
    species_df = _load_linked_species(gut_full, clp.DATA_ROOT)
    if species_df is None:
        log("  ERROR: species parquet not found — cannot train clock.")
        return None

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

    # 3. Training frame (mirrors build_gut_clock_ageaccel lines 854–891)
    if "research_stage" in gut_full.columns:
        train_base = (gut_full[gut_full["study_id"].isin(single_ids)]
                      .copy()
                      .sort_values("research_stage")
                      .groupby("study_id", as_index=False).first())
    else:
        train_base = gut_full[gut_full["study_id"].isin(single_ids)].copy()

    train_base = train_base.merge(
        pop[["study_id","age_at_visit","birth_year","sex_encoded"]].copy(),
        on="study_id", how="left")

    visit_age = clp._compute_visit_age(train_base)
    train_base["age_at_visit"] = visit_age.where(visit_age.notna(), train_base["age_at_visit"])

    # attach species
    if "research_stage" in train_base.columns:
        train_base = train_base.merge(sp_slim, on=["study_id","research_stage"], how="left")
    else:
        sp_sv = sp_slim.groupby("study_id", as_index=False).first()
        train_base = train_base.merge(
            sp_sv.drop(columns=["research_stage"]), on="study_id", how="left")

    # attach Shannon/richness (same source as the primary clock)
    div_sv = div_long[div_long["study_id"].isin(single_ids)][
        ["study_id","shannon","richness"]].groupby("study_id", as_index=False).first()
    train_base = train_base.merge(div_sv, on="study_id", how="left")

    train_df = train_base.dropna(subset=["age_at_visit"]).copy()
    log(f"  Training frame n = {len(train_df):,}")

    # Leak-free check
    _overlap = set(train_df["study_id"]) & repeat_ids
    if _overlap:
        log(f"  FAIL CHECK-1: {len(_overlap)} participants leaked!"); return None
    log("  CHECK-1 PASS: 0 participants leaked.")

    # 4. Feature matrix
    sp_present = [c for c in sp_abund_cols if c in train_df.columns]
    X_sp  = train_df[sp_present].fillna(0).astype(float).values
    X_div = train_df[["shannon","richness"]].fillna(0).astype(float).values
    X     = np.hstack([X_sp, X_div])
    y     = train_df["age_at_visit"].astype(float).values
    ids   = train_df["study_id"].values
    sex   = (train_df["sex_encoded"].astype(float).values
             if "sex_encoded" in train_df.columns else np.full(len(train_df), np.nan))
    log(f"  Feature matrix: {X.shape}")

    # 5. CV predictions (frozen clock spec: LightGBM n_est=300 lr=0.05 leaves=31)
    banner("  5-fold CV predictions (same spec as primary clock)")
    try:
        from lightgbm import LGBMRegressor
        clock = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=SEED, verbose=-1)
        log("  Model: LightGBM")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        clock = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
        log("  Model: HistGradientBoostingRegressor (LightGBM unavailable)")

    kf = KFold(n_splits=CV, shuffle=True, random_state=SEED)
    y_pred_cv = cross_val_predict(clock, X, y, cv=kf)

    r_cv  = float(pearsonr(y, y_pred_cv)[0])
    r2_cv = float(r2_score(y, y_pred_cv))
    mae   = float(np.mean(np.abs(y - y_pred_cv)))
    log(f"  CV r={r_cv:.4f}  R²={r2_cv:.4f}  MAE={mae:.2f} yr")

    # 6. Residualise predicted age on chronological age (OLS) → AgeAccel
    #    This is the exact same residualisation as meth:ageaccel (and as applied
    #    to the repeat-visit stratum in the primary analysis).
    log("  Residualising predicted age on chronological age (OLS)...")
    resid_df = pd.DataFrame({"pred": y_pred_cv, "chron": y})
    ols_fit  = smf.ols("pred ~ chron", data=resid_df).fit()
    ageaccel = ols_fit.resid.values
    r_accel_age = float(pearsonr(ageaccel, y)[0])
    log(f"  AgeAccel mean={ageaccel.mean():.4f} SD={ageaccel.std():.4f}")
    log(f"  r(AgeAccel, chron_age) = {r_accel_age:.4f}  "
        f"[sanity: should be ~0 after residualisation]")
    if abs(r_accel_age) > 0.05:
        log("  WARNING: AgeAccel still correlates with age — check residualisation.")

    result = pd.DataFrame({
        "participant_id": ids,
        "chron_age":      y,
        "sex":            sex,
        "predicted_age":  y_pred_cv,
        "ageaccel":       ageaccel,
    })
    out_path = OUT / "single_visit_ageaccel.csv"
    result.to_csv(out_path, index=False)
    log(f"  Saved {len(result):,} rows to {out_path}")
    return result


# ── Task B + C: partial correlations + incremental R² ───────────────────────

def tasks_bc(sv_accel: pd.DataFrame) -> dict:
    """Run partial correlations (B) and nested models (C). Returns numbers dict."""
    banner("TASK B — AgeAccel partial correlations with phenotypes")

    # Population frame for age + sex covariates
    pop = clp.load_population()
    sv_ids = set(sv_accel["participant_id"].unique())

    pop_sv = pop[pop["study_id"].isin(sv_ids)][
        ["study_id","age_at_visit","sex_encoded"]].copy()
    pop_sv = pop_sv.rename(columns={"age_at_visit": "age", "sex_encoded": "sex"})

    # Phenotypes at visit 1
    pheno_specs = {
        "grip": {
            "path": clp.DATA_ROOT / "hand_grip" / "hand_grip.parquet",
            "candidates": ["hand_grip_strength_right","hand_grip_strength_left",
                           "grip_strength","dominant_hand_grip"],
        },
        "sbp": {
            "path": clp.DATA_ROOT / "blood_pressure" / "blood_pressure.parquet",
            "candidates": ["sitting_blood_pressure_systolic",
                           "lying_blood_pressure_systolic",
                           "standing_one_min_blood_pressure_systolic"],
        },
        "waist": {
            "path": clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
            "candidates": ["waist_circumference","waist_cm"],
        },
        "bmi": {
            "path": clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
            "candidates": ["bmi","body_mass_index"],
        },
    }

    pheno_frames = {}
    for label, spec in pheno_specs.items():
        df = load_pheno(spec["path"], spec["candidates"], label, sv_ids)
        if df is not None:
            pheno_frames[label] = df

    # Shannon + richness for single-visit stratum (for Task C)
    # Must compute from gut_full filtered to sv_ids — div_long only has repeat-visit participants.
    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id","study_id"))
    gut_full = gut_full.rename(columns={pid_col: "study_id"})

    gut_sv_all = gut_full[gut_full["study_id"].isin(sv_ids)].copy()
    stg_col_sv = next((c for c in gut_sv_all.columns if "research_stage" in c.lower()), None)
    if stg_col_sv:
        gut_sv_all = gut_sv_all.rename(columns={stg_col_sv: "visit"})
    if "visit" in gut_sv_all.columns and "visit_n" not in gut_sv_all.columns:
        gut_sv_all["visit_n"] = gut_sv_all.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))
    div_sv_full = clp.compute_diversity(gut_sv_all)
    if "visit_n" not in div_sv_full.columns:
        div_sv_full["visit_n"] = div_sv_full.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))
    div_sv = (div_sv_full[div_sv_full["visit_n"] == 1]
              .groupby("study_id", as_index=False).first()
              [["study_id","shannon","richness"]])
    log(f"  Shannon/richness for single-visit: {len(div_sv):,} participants")

    # Analytic frame
    accel_slim = sv_accel[["participant_id","ageaccel"]].rename(
        columns={"participant_id": "study_id", "ageaccel": "gut_ageaccel"})
    analytic = pop_sv.merge(accel_slim, on="study_id", how="inner")
    analytic = analytic.merge(div_sv[["study_id","shannon","richness"]], on="study_id", how="left")
    for label, df in pheno_frames.items():
        analytic = analytic.merge(df, on="study_id", how="left")
    log(f"  Analytic frame: {len(analytic):,} rows  cols: {list(analytic.columns)}")

    pheno_vars = [c for c in ("grip","sbp","waist","bmi") if c in analytic.columns]
    covariates = ["age","sex"]

    # ── Task B: partial correlations ──────────────────────────────────────
    pcorr_rows = []
    log(f"\n  {'Phenotype':<8}  {'partial_r':>10}  {'p':>12}  {'n':>7}")
    log(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*7}")
    for pv in pheno_vars:
        r, p, n = partial_r(analytic, "gut_ageaccel", pv, covariates)
        sig = "**" if (not np.isnan(p) and p < 0.01) else (
              "*"  if (not np.isnan(p) and p < 0.05) else "")
        log(f"  {pv:<8}  {r:>+10.4f}  {p:>12.3g}  {n:>7,}  {sig}")
        pcorr_rows.append({
            "microbiome_var": "gut_ageaccel",
            "phenotype":      pv,
            "partial_r":      round(r, 6) if not np.isnan(r) else np.nan,
            "p":              p,
            "n":              n,
            "covariates":     "age+sex",
        })

    # Append to existing crosssectional_matrix.csv (do not touch existing rows)
    matrix_path = OUT / "crosssectional_matrix.csv"
    if matrix_path.exists():
        existing = pd.read_csv(matrix_path)
        # remove any prior gut_ageaccel rows to avoid duplicates
        existing = existing[existing["microbiome_var"] != "gut_ageaccel"]
        updated = pd.concat([existing, pd.DataFrame(pcorr_rows)], ignore_index=True)
    else:
        updated = pd.DataFrame(pcorr_rows)
    updated.to_csv(matrix_path, index=False)
    log(f"\n  Updated {matrix_path} ({len(updated)} rows total)")

    # ── Task C: incremental R² ────────────────────────────────────────────
    banner("TASK C — AgeAccel incremental R² over Shannon + richness")
    incr_rows = []
    log(f"\n  {'Phenotype':<8}  {'R2_div':>7}  {'R2_full':>7}  "
        f"{'dR2':>7}  {'beta_std':>8}  {'ci_lo':>7}  {'ci_hi':>7}  "
        f"{'P':>10}  n")
    log(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  "
        f"{'-'*7}  {'-'*7}  {'-'*10}  ----")

    for pv in pheno_vars:
        cols = [pv,"shannon","richness","gut_ageaccel","age","sex"]
        sub  = analytic[cols].dropna().astype(float)
        n    = len(sub)
        if n < 50:
            log(f"  {pv}: n={n} too small, skipping")
            continue

        # Standardise AgeAccel within this complete-case sample (std-beta)
        sub = sub.copy()
        sub["gut_ageaccel_z"] = (sub["gut_ageaccel"] - sub["gut_ageaccel"].mean()) / sub["gut_ageaccel"].std()

        m1 = smf.ols(f"{pv} ~ shannon + richness + age + sex", data=sub).fit()
        m2 = smf.ols(f"{pv} ~ shannon + richness + gut_ageaccel_z + age + sex", data=sub).fit()

        r2_base = float(m1.rsquared_adj)
        r2_full = float(m2.rsquared_adj)
        delta   = r2_full - r2_base
        beta    = float(m2.params.get("gut_ageaccel_z", np.nan))
        ci      = m2.conf_int().loc["gut_ageaccel_z"] if "gut_ageaccel_z" in m2.params else [np.nan, np.nan]
        ci_lo   = float(ci[0])
        ci_hi   = float(ci[1])
        p_val   = float(m2.pvalues.get("gut_ageaccel_z", np.nan))

        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
        log(f"  {pv:<8}  {r2_base:>7.4f}  {r2_full:>7.4f}  "
            f"{delta:>+7.4f}  {beta:>+8.4f}  {ci_lo:>+7.4f}  {ci_hi:>+7.4f}  "
            f"{p_val:>10.3g}  {n:,}  {sig}")

        incr_rows.append({
            "phenotype":          pv,
            "n":                  n,
            "R2_diversity":       round(r2_base, 6),
            "R2_plus_ageaccel":   round(r2_full, 6),
            "delta_R2":           round(delta, 6),
            "ageaccel_beta":      round(beta, 6),
            "ci_low":             round(ci_lo, 6),
            "ci_high":            round(ci_hi, 6),
            "P":                  p_val,
            "base_model":         "shannon+richness+age+sex",
            "full_model":         "shannon+richness+ageaccel_z+age+sex",
        })

    incr_df = pd.DataFrame(incr_rows)
    incr_df.to_csv(OUT / "ageaccel_incremental.csv", index=False)
    log(f"\n  Saved {OUT / 'ageaccel_incremental.csv'}")

    # Verdict
    n_sig   = sum(1 for r in incr_rows if r["P"] < 0.05)
    max_dR2 = max((r["delta_R2"] for r in incr_rows), default=np.nan)
    if n_sig == 0:
        verdict = (f"AgeAccel adds NO incremental predictive value over Shannon + richness "
                   f"cross-sectionally (p<0.05 in 0/{len(incr_rows)} phenotype models; "
                   f"max ΔR²={max_dR2:+.4f}).")
    else:
        verdict = (f"AgeAccel adds incremental value in {n_sig}/{len(incr_rows)} phenotype(s) "
                   f"(max ΔR²={max_dR2:+.4f}).")

    log(f"\n  VERDICT: {verdict}")
    return {"pcorr": pcorr_rows, "incr": incr_rows, "verdict": verdict}


# ── Write N3_AGEACCEL_NUMBERS.md ─────────────────────────────────────────────

def write_changelog(sv_accel: pd.DataFrame, numbers: dict) -> None:
    banner("Writing N3_AGEACCEL_NUMBERS.md")
    today = datetime.date.today().isoformat()
    lines = [
        f"# N3_AGEACCEL_NUMBERS.md",
        f"",
        f"Generated {today} by N3_ageaccel.py",
        f"Source files: results_cross_lagged/review2/",
        f"",
        f"---",
        f"",
        f"## Task A — Single-visit AgeAccel (n = {len(sv_accel):,})",
        f"",
        f"| Column | Value | Source |",
        f"|--------|-------|--------|",
        f"| n (single-visit stratum) | {len(sv_accel):,} | single_visit_ageaccel.csv |",
        f"| AgeAccel mean | {sv_accel['ageaccel'].mean():.4f} yr | single_visit_ageaccel.csv → ageaccel |",
        f"| AgeAccel SD | {sv_accel['ageaccel'].std():.4f} yr | single_visit_ageaccel.csv → ageaccel |",
        f"| r(AgeAccel, chron_age) | {pearsonr(sv_accel['ageaccel'], sv_accel['chron_age'])[0]:.4f} | sanity check — expect ~0 |",
        f"",
        f"---",
        f"",
        f"## Task B — AgeAccel partial r (controlling age + sex)",
        f"",
        f"Paste-ready for Table E12 AgeAccel row and workbook sheet 23_N3_CrossSectional (AQ cells).",
        f"",
        f"| Phenotype | partial r | P | n | Source |",
        f"|-----------|-----------|---|---|--------|",
    ]
    for row in numbers["pcorr"]:
        pv = row["phenotype"]
        r  = row["partial_r"]
        p  = row["p"]
        n  = row["n"]
        r_str = f"{r:+.4f}" if not (isinstance(r, float) and np.isnan(r)) else "NA"
        p_str = f"{p:.3g}"  if not (isinstance(p, float) and np.isnan(p)) else "NA"
        lines.append(f"| {pv} | {r_str} | {p_str} | {n:,} | crosssectional_matrix.csv → partial_r |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Task C — AgeAccel incremental R² over Shannon + richness",
        f"",
        f"All betas are standardised (z-scored AgeAccel within each complete-case sample).",
        f"",
        f"| Phenotype | R²_diversity | R²_+AgeAccel | ΔR² | AgeAccel β (std) | 95% CI | P | n | Source |",
        f"|-----------|-------------|-------------|-----|-----------------|--------|---|---|--------|",
    ]
    for row in numbers["incr"]:
        ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        lines.append(
            f"| {row['phenotype']} "
            f"| {row['R2_diversity']:.4f} "
            f"| {row['R2_plus_ageaccel']:.4f} "
            f"| {row['delta_R2']:+.4f} "
            f"| {row['ageaccel_beta']:+.4f} "
            f"| {ci} "
            f"| {row['P']:.3g} "
            f"| {row['n']:,} "
            f"| ageaccel_incremental.csv |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Verdict",
        f"",
        f"{numbers['verdict']}",
        f"",
        f"---",
        f"",
        f"## Still outstanding (figures — separate brief)",
        f"",
        f"- Fig2_clock.pdf — predicted-vs-measured age (needs participant-level predictions)",
        f"- FigE6_ageaccel_scatter.pdf — AgeAccel visit 1 vs visit 2 scatter (n=1,787, ICC 0.516)",
        f"- property_map.pdf — 248-taxa property map",
        f"- Fig1_flow.pdf — relabel panel d age-encoding tile",
        f"- FigE3_crosssectional.pdf — regenerate to add AgeAccel row (after Tasks B/C)",
        f"",
    ]

    md_path = Path("N3_AGEACCEL_NUMBERS.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"  Wrote {md_path.resolve()}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sv_accel = task_a()
    if sv_accel is None:
        log("\nTask A failed — cannot proceed to B/C.")
        return

    numbers = tasks_bc(sv_accel)
    write_changelog(sv_accel, numbers)
    banner("ALL DONE — N3_AGEACCEL_NUMBERS.md written")


if __name__ == "__main__":
    main()
