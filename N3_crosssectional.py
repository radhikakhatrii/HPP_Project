#!/usr/bin/env python3
"""
N3_crosssectional.py
===============================================================================
Review point 3: Cross-sectional partial correlations + AgeAccel incremental
value over Shannon/richness.

All analyses on the single-visit stratum (n ≈ 6,807) — the same participants
used to train the clock. One measurement per person (visit 1 baseline).

Part A — Partial correlations (controlling for chronological age + sex):
    For each of {Shannon, richness, AgeAccel} × {grip, SBP, waist, BMI}:
    partial r, P, n

Part B — AgeAccel incremental value:
    For each phenotype:
      Model 1: phenotype ~ Shannon + richness + age + sex   → adjusted R²
      Model 2: phenotype ~ Shannon + richness + AgeAccel + age + sex → adjusted R²
      ΔR², AgeAccel partial β, SE, P

    If AgeAccel adds nothing over diversity, it says so plainly.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 N3_crosssectional.py 2>&1 | tee N3_crosssectional.log

Reads:
    results_cross_lagged/  (for single-visit IDs via existing diversity+clock results)
    hpp_datasets/...parquets (grip, blood_pressure, anthropometrics)

Writes:
    results_cross_lagged/review2/crosssectional_matrix.csv
    results_cross_lagged/review2/ageaccel_incremental.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT  = Path(clp.RESULTS_DIR) / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log  = clp.log


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def partial_r(df: pd.DataFrame, x: str, y: str,
              covariates: list[str]) -> tuple[float, float, int]:
    """
    Partial Pearson r between x and y, controlling for covariates.
    Uses residuals-on-residuals approach (equivalent to partial correlation).
    Returns (r, p, n).
    """
    cols = [x, y] + covariates
    sub  = df[cols].dropna().astype(float)
    n    = len(sub)
    if n < 30:
        return np.nan, np.nan, n

    def resid(outcome: str) -> np.ndarray:
        formula = f"Q('{outcome}') ~ " + " + ".join(
            f"Q('{c}')" for c in covariates)
        try:
            return smf.ols(formula, data=sub).fit().resid.values
        except Exception:
            return np.full(n, np.nan)

    rx = resid(x)
    ry = resid(y)
    mask = np.isfinite(rx) & np.isfinite(ry)
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum())
    r, p = pearsonr(rx[mask], ry[mask])
    return float(r), float(p), int(mask.sum())


def adj_r2(model_fit) -> float:
    return float(model_fit.rsquared_adj)


def load_parquet_cols(parquet_path: Path, meas_candidates: list[str],
                      label: str, keep_ids: set | None = None,
                      visit_n_keep: int = 1) -> pd.DataFrame | None:
    """Read only id + stage + measurement; filter to keep_ids at visit 1."""
    import pyarrow.parquet as pq, time
    if not parquet_path.exists():
        log(f"  {label}: not found")
        return None
    try:
        schema_cols = pq.read_schema(parquet_path).names
    except OSError as e:
        log(f"  {label}: schema error — {e}")
        return None

    pid   = next((c for c in schema_cols if c in ("participant_id", "study_id")), None)
    stage = next((c for c in schema_cols if "research_stage" in c.lower()), None)
    meas  = next((c for c in meas_candidates if c in schema_cols), None)
    if not pid or not stage or not meas:
        log(f"  {label}: cols not found — pid={pid}, stage={stage}, meas={meas}")
        return None

    log(f"  {label}: reading [{pid}, {stage}, {meas}]")
    t0 = time.time()
    try:
        df = pd.read_parquet(parquet_path, columns=[pid, stage, meas])
    except OSError as e:
        log(f"  {label}: I/O error — {e}")
        return None
    log(f"    {len(df):,} rows in {time.time()-t0:.2f}s")

    if pid not in df.columns:
        df = df.reset_index()
    if pid not in df.columns:
        return None
    df = df.rename(columns={pid: "study_id", stage: "visit"})
    df[meas] = pd.to_numeric(df[meas], errors="coerce")
    if keep_ids is not None:
        df = df[df["study_id"].isin(keep_ids)]
    df["visit_n"] = df.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))
    df = df[df["visit_n"] == visit_n_keep].copy()
    # one row per participant (take first if duplicates)
    df = df.sort_values("visit").groupby("study_id", as_index=False).first()
    df = df.rename(columns={meas: label})
    log(f"    → {len(df):,} participants at visit {visit_n_keep}")
    return df[["study_id", label]]


def main() -> None:
    banner("N3: CROSS-SECTIONAL PARTIAL CORRELATIONS + AGEACCEL INCREMENTAL VALUE")

    # ── 1. Single-visit stratum IDs ───────────────────────────────────────
    banner("1. Identifying single-visit stratum (n ≈ 6,807)")
    # Load gut parquet to split single vs repeat visitors
    gut = clp.load_longitudinal_gut()            # repeat-visit gut data
    pop = clp.load_population()

    div_long  = clp.compute_diversity(gut)       # Shannon/richness for repeat stratum
    repeat_ids = set(div_long["study_id"].unique())
    log(f"  Repeat-visit (longitudinal) stratum: {len(repeat_ids):,}")

    # Full gut parquet for all participants
    gut_full = pd.read_parquet(
        clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    gut_full = gut_full.rename(columns={pid_col: "study_id"})
    all_gut_ids = set(gut_full["study_id"].unique())
    single_ids  = all_gut_ids - repeat_ids
    log(f"  Single-visit stratum: {len(single_ids):,}")

    # ── 2. Diversity + AgeAccel for single-visit stratum ─────────────────
    banner("2. Computing Shannon, richness, AgeAccel for single-visit stratum")

    # Diversity: compute_diversity on single-visit rows
    # compute_diversity (line 706) requires gut to already have 'visit' and
    # 'visit_n' columns — same setup that load_longitudinal_gut() does internally.
    gut_sv = gut_full[gut_full["study_id"].isin(single_ids)].copy()
    stg_col = next((c for c in gut_sv.columns if "research_stage" in c.lower()), None)
    if stg_col:
        gut_sv = gut_sv.rename(columns={stg_col: "visit"})
    if "visit" in gut_sv.columns and "visit_n" not in gut_sv.columns:
        gut_sv["visit_n"] = gut_sv.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))
    div_sv = clp.compute_diversity(gut_sv)
    # take visit 1 only (all single-visit participants have visit_n == 1 anyway)
    if "visit_n" not in div_sv.columns:
        div_sv["visit_n"] = div_sv.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))
    div_sv = div_sv[div_sv["visit_n"] == 1].copy()
    div_sv = div_sv.groupby("study_id", as_index=False).first()
    log(f"  Diversity computed for {len(div_sv):,} single-visit participants")

    # AgeAccel: load from the saved cross-lagged results (all_paths has ageaccel col)
    # Check if there's a saved per-participant AgeAccel file
    ageaccel_path = Path(clp.RESULTS_DIR) / "single_visit_ageaccel.csv"
    if not ageaccel_path.exists():
        # Fall back: try frozen_signature_pipeline or compute via build_gut_clock
        log(f"  {ageaccel_path} not found — attempting to build AgeAccel via clock")
        try:
            # build_gut_clock_ageaccel returns the training frame with ageaccel column
            result = clp.build_gut_clock_ageaccel(gut_full, pop)
            if result is not None and "gut_ageaccel" in result.columns:
                sv_accel = result[result["study_id"].isin(single_ids)][
                    ["study_id", "gut_ageaccel"]].copy()
                sv_accel.to_csv(ageaccel_path, index=False)
                log(f"  AgeAccel computed for {len(sv_accel):,} participants")
            else:
                sv_accel = None
                log("  build_gut_clock_ageaccel did not return gut_ageaccel — "
                    "AgeAccel incremental analysis will be skipped")
        except Exception as e:
            log(f"  Cannot compute AgeAccel: {e}")
            sv_accel = None
    else:
        sv_accel = pd.read_csv(ageaccel_path)
        # normalise column name
        if "gut_ageaccel" not in sv_accel.columns:
            accel_col = next((c for c in sv_accel.columns if "accel" in c.lower()), None)
            if accel_col:
                sv_accel = sv_accel.rename(columns={accel_col: "gut_ageaccel"})
        sv_accel = sv_accel[sv_accel["study_id"].isin(single_ids)]
        log(f"  AgeAccel loaded: {len(sv_accel):,} participants")

    # ── 3. Population covariates ──────────────────────────────────────────
    banner("3. Population covariates (age, sex)")
    pop_sv = pop[pop["study_id"].isin(single_ids)][
        ["study_id", "age_at_visit", "sex_encoded"]].copy()
    pop_sv = pop_sv.rename(columns={"age_at_visit": "age", "sex_encoded": "sex"})
    log(f"  Population covariate rows: {len(pop_sv):,}")

    # ── 4. Phenotypes at visit 1 ──────────────────────────────────────────
    banner("4. Phenotypes (grip, SBP, waist, BMI) at visit 1")
    pheno_specs = {
        "grip": {
            "path": clp.DATA_ROOT / "hand_grip" / "hand_grip.parquet",
            "candidates": ["hand_grip_strength_right", "hand_grip_strength_left",
                           "grip_strength", "dominant_hand_grip"],
        },
        "sbp": {
            "path": clp.DATA_ROOT / "blood_pressure" / "blood_pressure.parquet",
            "candidates": ["sitting_blood_pressure_systolic",
                           "lying_blood_pressure_systolic",
                           "standing_one_min_blood_pressure_systolic"],
        },
        "waist": {
            "path": clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
            "candidates": ["waist_circumference", "waist_cm"],
        },
        "bmi": {
            "path": clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
            "candidates": ["bmi", "body_mass_index"],
        },
    }

    pheno_frames = {}
    for label, spec in pheno_specs.items():
        df = load_parquet_cols(spec["path"], spec["candidates"], label,
                               keep_ids=single_ids, visit_n_keep=1)
        if df is not None:
            pheno_frames[label] = df

    # ── 5. Merge into analytic frame ──────────────────────────────────────
    banner("5. Building analytic frame")
    analytic = pop_sv.copy()
    analytic = analytic.merge(
        div_sv[["study_id", "shannon", "richness"]], on="study_id", how="left")
    if sv_accel is not None and "gut_ageaccel" in sv_accel.columns:
        analytic = analytic.merge(
            sv_accel[["study_id", "gut_ageaccel"]], on="study_id", how="left")
    for label, df in pheno_frames.items():
        analytic = analytic.merge(df, on="study_id", how="left")

    log(f"  Analytic frame: {len(analytic):,} rows, columns: {list(analytic.columns)}")

    micro_vars   = [c for c in ("shannon", "richness", "gut_ageaccel")
                    if c in analytic.columns]
    pheno_vars   = [c for c in ("grip", "sbp", "waist", "bmi")
                    if c in analytic.columns]
    covariates   = ["age", "sex"]
    log(f"  Microbiome vars: {micro_vars}")
    log(f"  Phenotype vars:  {pheno_vars}")

    # ── 6. Part A: partial correlations ──────────────────────────────────
    banner("6. Part A — Partial correlations (controlling age + sex)")
    partcorr_rows = []
    log(f"  {'Microbiome':<15} {'Phenotype':<8}  {'partial_r':>10}  {'p':>12}  {'n':>6}")
    log(f"  {'-'*15} {'-'*8}  {'-'*10}  {'-'*12}  {'-'*6}")
    for mv in micro_vars:
        for pv in pheno_vars:
            r, p, n = partial_r(analytic, mv, pv, covariates)
            sig = "**" if (not np.isnan(p) and p < 0.01) else (
                  "*"  if (not np.isnan(p) and p < 0.05) else "")
            log(f"  {mv:<15} {pv:<8}  {r:>+10.4f}  {p:>12.3g}  {n:>6,}  {sig}")
            partcorr_rows.append({
                "microbiome_var": mv,
                "phenotype":      pv,
                "partial_r":      round(r, 6) if not np.isnan(r) else np.nan,
                "p":              p,
                "n":              n,
                "covariates":     "+".join(covariates),
            })

    partcorr_df = pd.DataFrame(partcorr_rows)
    partcorr_df.to_csv(OUT / "crosssectional_matrix.csv", index=False)
    log(f"\n  Wrote {OUT/'crosssectional_matrix.csv'}")

    # ── 7. Part B: AgeAccel incremental value ─────────────────────────────
    banner("7. Part B — AgeAccel incremental R² over Shannon + richness")

    if "gut_ageaccel" not in analytic.columns:
        log("  AgeAccel not available — skipping incremental analysis")
        log("  (run build_gut_clock_ageaccel or provide single_visit_ageaccel.csv)")
        return

    incr_rows = []
    log(f"\n  {'Phenotype':<8}  {'R2_base':>8}  {'R2_full':>8}  "
        f"{'delta_R2':>9}  {'beta_accel':>11}  {'p_accel':>10}")
    log(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*10}")

    for pv in pheno_vars:
        sub = analytic[[pv, "shannon", "richness", "gut_ageaccel",
                         "age", "sex"]].dropna().astype(float)
        n   = len(sub)
        if n < 50:
            log(f"  {pv}: too few observations ({n}) — skipping")
            continue

        # Model 1: base (Shannon + richness + age + sex)
        m1 = smf.ols(
            f"{pv} ~ shannon + richness + age + sex", data=sub).fit()
        r2_base = adj_r2(m1)

        # Model 2: full (+ AgeAccel)
        m2 = smf.ols(
            f"{pv} ~ shannon + richness + gut_ageaccel + age + sex",
            data=sub).fit()
        r2_full     = adj_r2(m2)
        delta_r2    = r2_full - r2_base
        beta_accel  = float(m2.params.get("gut_ageaccel", np.nan))
        se_accel    = float(m2.bse.get("gut_ageaccel",   np.nan))
        p_accel     = float(m2.pvalues.get("gut_ageaccel", np.nan))

        sig = "**" if p_accel < 0.01 else ("*" if p_accel < 0.05 else "")
        log(f"  {pv:<8}  {r2_base:>8.4f}  {r2_full:>8.4f}  "
            f"{delta_r2:>+9.4f}  {beta_accel:>+11.4f}  {p_accel:>10.3g}  {sig}")

        incr_rows.append({
            "phenotype":    pv,
            "n":            n,
            "adj_r2_base":  round(r2_base,   6),
            "adj_r2_full":  round(r2_full,   6),
            "delta_adj_r2": round(delta_r2,  6),
            "beta_ageaccel": round(beta_accel, 6),
            "se_ageaccel":   round(se_accel,   6),
            "p_ageaccel":    p_accel,
            "base_model":   "shannon+richness+age+sex",
            "full_model":   "shannon+richness+gut_ageaccel+age+sex",
        })

    incr_df = pd.DataFrame(incr_rows)
    incr_df.to_csv(OUT / "ageaccel_incremental.csv", index=False)
    log(f"\n  Wrote {OUT/'ageaccel_incremental.csv'}")

    # Verdict
    banner("VERDICT — does AgeAccel add anything over Shannon + richness?")
    if len(incr_rows):
        sig_count = sum(1 for r in incr_rows if r["p_ageaccel"] < 0.05)
        max_delta = max(r["delta_adj_r2"] for r in incr_rows)
        log(f"  AgeAccel p<0.05 in {sig_count}/{len(incr_rows)} phenotype models")
        log(f"  Maximum ΔR² = {max_delta:+.4f}")
        if sig_count == 0:
            log("  Conclusion: AgeAccel adds NO incremental predictive value over "
                "Shannon + richness cross-sectionally.")
        else:
            log(f"  Conclusion: AgeAccel adds incremental value in "
                f"{sig_count} phenotype(s) — see ageaccel_incremental.csv.")

    log("\n[DONE] N3 complete.")


if __name__ == "__main__":
    main()
