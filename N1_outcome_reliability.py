#!/usr/bin/env python3
"""
N1_outcome_reliability.py
===============================================================================
Review point 6: Outcome test-retest reliability and outcome-disattenuated
cross-lagged coefficients.

The three outcomes in the primary cross-lagged analysis (right-hand grip
strength, seated systolic blood pressure, waist circumference) have no
reported ICC, even though both visits exist. Noisy outcomes inflate SEs and
may explain why nothing survived correction. This script:

  1. Computes ICC(3,1), Pearson retest r, and n for each of the three outcomes
     in the longitudinal participants used in the cross-lagged models.

  2. For each of the 9 Path-A tests (microbiome(t1) → outcome(t2)), computes
     outcome-disattenuated β and 95% CI by dividing observed β and CI bounds
     by √(outcome ICC), mirroring the exposure disattenuation already reported.

Disattenuation formula:
    β_disatt  = β_obs  / sqrt(ICC_outcome)
    CI_lo_disatt = CI_lo / sqrt(ICC_outcome)   [same divisor for both bounds]
    CI_hi_disatt = CI_hi / sqrt(ICC_outcome)

ICC(3,1) estimator: two-way mixed, single measures, absolute agreement.
Uses pingouin.intraclass_corr if available; falls back to the ANOVA-based
analytic formula.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 N1_outcome_reliability.py 2>&1 | tee N1_outcome_reliability.log

Reads:  results_cross_lagged/cross_lag_all_paths.csv
Writes: results_cross_lagged/review2/outcome_reliability.csv
        results_cross_lagged/review2/crosslag_outcome_disattenuated.csv
===============================================================================
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR) / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log

PRIMARY_SPEC = "age+sex"   # covariate_spec label used in the primary 18-test run


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


# ── ICC(3,1) helper ────────────────────────────────────────────────────────
def compute_icc31(df: pd.DataFrame, subject_col: str, value_col: str,
                  rater_col: str) -> tuple[float, int]:
    """
    ICC(3,1): two-way mixed, single measures, absolute agreement.
    Uses pingouin if available; falls back to ANOVA-formula.
    Returns (icc, n_subjects).
    """
    sub = df[[subject_col, rater_col, value_col]].dropna()
    # keep only subjects with EXACTLY two rater observations
    counts = sub.groupby(subject_col)[value_col].count()
    sub = sub[sub[subject_col].isin(counts[counts == 2].index)]
    n = sub[subject_col].nunique()
    if n < 30:
        log(f"    WARNING: only {n} subjects with both visits — ICC unreliable")

    try:
        import pingouin as pg
        icc_tbl = pg.intraclass_corr(data=sub, targets=subject_col,
                                     raters=rater_col, ratings=value_col)
        icc = float(icc_tbl[icc_tbl["Type"] == "ICC3"]["ICC"].iloc[0])
        log(f"    ICC(3,1) via pingouin: {icc:.4f}  (n={n:,} subjects)")
        return icc, n
    except Exception:
        pass

    # ANOVA-formula fallback
    grand = sub[value_col].mean()
    subj_means = sub.groupby(subject_col)[value_col].mean()
    # between-subjects SS
    k = 2  # visits
    ss_between = k * ((subj_means - grand) ** 2).sum()
    ms_between = ss_between / (n - 1)
    # within-subjects (rater + error)
    sub_w = sub.merge(subj_means.rename("subj_mean"), left_on=subject_col,
                      right_index=True)
    ss_within = ((sub_w[value_col] - sub_w["subj_mean"]) ** 2).sum()
    ms_within  = ss_within / (n * (k - 1))
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    log(f"    ICC(3,1) via ANOVA formula: {icc:.4f}  (n={n:,} subjects)")
    return icc, n


def parse_ci95(val) -> tuple[float, float]:
    """Parse ci95 stored as '(lo, hi)' string or already a tuple/list."""
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return float(val[0]), float(val[1])
    try:
        parsed = ast.literal_eval(str(val))
        return float(parsed[0]), float(parsed[1])
    except Exception:
        return np.nan, np.nan


def main() -> None:
    banner("N1: OUTCOME TEST-RETEST RELIABILITY")

    # ── 1. Load existing 18-test cross-lagged results ─────────────────────
    banner("1. Loading primary cross-lagged results")
    primary_csv = Path(clp.RESULTS_DIR) / "cross_lag_all_paths.csv"
    if not primary_csv.exists():
        raise SystemExit(f"{primary_csv} not found — run cross_lagged_panel.py first.")
    all_results = pd.read_csv(primary_csv)

    # Primary spec: the age+sex covariate set used in Table 2
    spec_col = next((c for c in all_results.columns
                     if "covariate" in c.lower()), None)
    if spec_col and all_results[spec_col].nunique() > 1:
        primary = all_results[all_results[spec_col].str.contains(
            PRIMARY_SPEC, case=False, na=False)].copy()
        log(f"  Filtered to covariate_spec containing '{PRIMARY_SPEC}': "
            f"{len(primary)} rows (from {len(all_results)} total)")
    else:
        primary = all_results.copy()
        log(f"  Using all rows ({len(primary)}) — single covariate spec or col not found")

    path_a = primary[primary["direction"] == "A"].copy()
    log(f"  Path-A tests: {len(path_a)}")

    # ── 2. Load functional outcomes — minimal direct loader ───────────────
    banner("2. Loading functional outcomes (column-filtered direct read)")

    # Get the analytic sample IDs from the already-committed both-visits CSV.
    both_visits_path = Path(clp.RESULTS_DIR) / "leak_free_ageaccel_both_visits.csv"
    if both_visits_path.exists():
        longitudinal_ids = set(pd.read_csv(both_visits_path)["study_id"])
        log(f"  Longitudinal participants (leak_free_ageaccel_both_visits.csv): "
            f"{len(longitudinal_ids):,}")
    else:
        log("  WARNING: leak_free_ageaccel_both_visits.csv not found — "
            "using all participants with two outcome visits")
        longitudinal_ids = None

    def _load_outcome_fast(parquet_path: Path,
                           meas_candidates: list[str],
                           label: str) -> pd.DataFrame | None:
        """Read only id + stage + measurement columns; filter to longitudinal IDs.
        Reads row-group by row-group with progress + timing so a slow read is
        visibly progressing rather than indistinguishable from a hang."""
        import time
        import pyarrow.parquet as pq

        try:
            exists = parquet_path.exists()
        except OSError as e:
            log(f"  {label}: filesystem error checking {parquet_path} — {e}")
            log(f"    (FUSE/NFS mount is broken — ask your admin to remount)")
            return None
        if not exists:
            log(f"  {label}: {parquet_path} not found")
            return None

        t0 = time.time()
        try:
            pf = pq.ParquetFile(parquet_path)
        except OSError as e:
            log(f"  {label}: cannot open parquet file — {e}")
            log(f"    (FUSE/NFS mount is broken — ask your admin to remount)")
            return None
        schema_cols = pf.schema_arrow.names
        n_rows = pf.metadata.num_rows
        n_rg   = pf.num_row_groups
        size_mb = parquet_path.stat().st_size / 1e6
        log(f"  {label}: {parquet_path.name}  "
            f"({size_mb:,.1f} MB, {n_rows:,} rows, {n_rg} row groups, "
            f"{len(schema_cols)} cols)  [metadata read {time.time()-t0:.2f}s]")

        pid   = next((c for c in schema_cols if c in ("participant_id", "study_id")), None)
        stage = next((c for c in schema_cols if "research_stage" in c.lower()), None)
        meas  = next((c for c in meas_candidates if c in schema_cols), None)
        if not pid or not stage or not meas:
            log(f"  {label}: could not find id/stage/meas in schema — "
                f"pid={pid}, stage={stage}, meas={meas}")
            log(f"    schema sample: {schema_cols[:20]}")
            return None
        log(f"  {label}: reading cols [{pid}, {stage}, {meas}] "
            f"across {n_rg} row group(s)...")

        t_read = time.time()
        try:
            df = pd.read_parquet(parquet_path, columns=[pid, stage, meas])
        except OSError as e:
            log(f"  {label}: I/O error reading parquet — {e}")
            log(f"    (FUSE/NFS mount is broken — ask your admin to remount)")
            return None
        log(f"  {label}: read complete in {time.time()-t_read:.2f}s "
            f"({len(df):,} raw rows)")

        # participant_id may be the parquet index rather than a column
        if pid not in df.columns:
            df = df.reset_index()
        if pid not in df.columns:
            log(f"  {label}: {pid} not found after reset_index — "
                f"columns: {list(df.columns[:10])}")
            return None
        df = df.rename(columns={pid: "study_id", stage: "visit"})
        df[meas] = pd.to_numeric(df[meas], errors="coerce")
        if longitudinal_ids is not None:
            df = df[df["study_id"].isin(longitudinal_ids)]
        df["visit_n"] = df.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))
        df = df[df["visit_n"].isin([1, 2])].rename(columns={meas: label})
        log(f"    → {len(df):,} rows, {df['study_id'].nunique():,} participants "
            f"after ID/visit filtering  [total {time.time()-t0:.2f}s]")
        return df[["study_id", "visit_n", label]]

    # Column priority lists (from prior full run — these won in load_functional_outcomes)
    outcome_specs = {
        "hand_grip": {
            "path": clp.DATA_ROOT / "hand_grip" / "hand_grip.parquet",
            "label": "grip",
            "candidates": ["hand_grip_strength_right", "hand_grip_strength_left",
                           "grip_strength", "dominant_hand_grip",
                           "finger_grip_strength_right", "finger_grip_strength_left"],
        },
        "blood_pressure": {
            "path": clp.DATA_ROOT / "blood_pressure" / "blood_pressure.parquet",
            "label": "sbp",
            "candidates": ["sitting_blood_pressure_systolic",
                           "lying_blood_pressure_systolic",
                           "standing_one_min_blood_pressure_systolic",
                           "systolic", "systolic_bp"],
        },
        "anthropometrics": {
            "path": clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
            "label": "waist",
            "candidates": ["waist_circumference", "waist_cm", "bmi"],
        },
    }

    outcome_frames: dict[str, pd.DataFrame] = {}
    for ds_name, spec in outcome_specs.items():
        frame = _load_outcome_fast(spec["path"], spec["candidates"], spec["label"])
        if frame is not None:
            outcome_frames[ds_name] = (frame, spec["label"])

    # For each outcome, build a long-format two-visit frame restricted to
    # participants who also have gut data at both visits.
    # Columns: study_id, visit_n (1 or 2), value
    OUTCOME_MAP = {
        "hand_grip":      "grip",
        "blood_pressure": "sbp",
        "anthropometrics": "waist",
    }

    icc_rows = []
    outcome_iccs: dict[str, float] = {}   # dataset name → ICC

    for ds_name, label in OUTCOME_MAP.items():
        entry = outcome_frames.get(ds_name)
        if entry is None:
            log(f"  {ds_name}: not available — skipping")
            continue
        df, meas_col = entry   # df already has columns: study_id, visit_n, <label>
        if df.empty:
            log(f"  {ds_name}: empty — skipping")
            continue

        log(f"\n  {ds_name} (col={meas_col})")

        # Already filtered to longitudinal IDs in _load_outcome_fast
        sub = df[["study_id", "visit_n", meas_col]].copy()
        sub[meas_col] = pd.to_numeric(sub[meas_col], errors="coerce")
        sub = sub.dropna(subset=[meas_col])

        # visit 1 and 2 only
        sub = sub[sub["visit_n"].isin([1, 2])]

        # Pearson retest r (on wide format)
        v1 = sub[sub["visit_n"] == 1].set_index("study_id")[meas_col].rename("v1")
        v2 = sub[sub["visit_n"] == 2].set_index("study_id")[meas_col].rename("v2")
        pair = pd.concat([v1, v2], axis=1).dropna()
        n_pair = len(pair)
        if n_pair >= 10:
            retest_r, retest_p = pearsonr(pair["v1"], pair["v2"])
        else:
            retest_r, retest_p = np.nan, np.nan
        log(f"    Paired n={n_pair:,}  Pearson retest r={retest_r:.4f}  p={retest_p:.3g}")

        # ICC(3,1)
        icc, n_icc = compute_icc31(sub, "study_id", meas_col, "visit_n")
        outcome_iccs[ds_name] = icc

        icc_rows.append({
            "outcome_label": label,
            "dataset": ds_name,
            "measurement_column": meas_col,
            "n_paired": n_pair,
            "pearson_retest_r": round(retest_r, 6),
            "pearson_retest_p": retest_p,
            "icc31": round(icc, 6),
            "icc31_sqrt": round(np.sqrt(max(icc, 0)), 6),
        })

    icc_df = pd.DataFrame(icc_rows)
    icc_df.to_csv(OUT / "outcome_reliability.csv", index=False)
    log(f"\n  Wrote {OUT/'outcome_reliability.csv'}")

    banner("OUTCOME ICCs (summary)")
    for r in icc_rows:
        log(f"  {r['outcome_label']:<8}  ICC(3,1)={r['icc31']:.4f}  "
            f"√ICC={r['icc31_sqrt']:.4f}  "
            f"retest_r={r['pearson_retest_r']:.4f}  n={r['n_paired']:,}")

    # ── 3. Outcome-disattenuated coefficients for Path-A tests ────────────
    banner("3. Outcome-disattenuated Path-A coefficients")
    log("  Formula: β_disatt = β_obs / √(ICC_outcome)")
    log("  CI bounds divided by the same factor (√ICC > 0, so sign preserved)")

    disatt_rows = []
    for _, row in path_a.iterrows():
        ds  = row.get("dataset", "")
        icc = outcome_iccs.get(ds, np.nan)
        sqrt_icc = np.sqrt(icc) if (not np.isnan(icc) and icc > 0) else np.nan

        ci_lo, ci_hi = parse_ci95(row.get("ci95", "(nan,nan)"))
        beta_obs = float(row.get("beta", np.nan))
        se_obs   = float(row.get("se",   np.nan))

        if not np.isnan(sqrt_icc):
            beta_disatt  = beta_obs  / sqrt_icc
            ci_lo_disatt = ci_lo     / sqrt_icc
            ci_hi_disatt = ci_hi     / sqrt_icc
            se_disatt    = se_obs    / sqrt_icc
        else:
            beta_disatt = ci_lo_disatt = ci_hi_disatt = se_disatt = np.nan

        # Does CI shift? Check whether disattenuated CI excludes zero when
        # observed CI includes it (or vice versa for the ±0.10 margin)
        obs_excludes_zero = (not np.isnan(ci_lo)) and (ci_lo > 0 or ci_hi < 0)
        dis_excludes_zero = (not np.isnan(ci_lo_disatt)) and (
            ci_lo_disatt > 0 or ci_hi_disatt < 0)
        obs_within_01 = (not np.isnan(ci_lo)) and abs(ci_lo) < 0.10 and abs(ci_hi) < 0.10
        dis_within_01 = (not np.isnan(ci_lo_disatt)) and (
            abs(ci_lo_disatt) < 0.10 and abs(ci_hi_disatt) < 0.10)

        log(f"  {row.get('path','')[:55]:<55} "
            f"β_obs={beta_obs:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}]  "
            f"β_dis={beta_disatt:+.4f} [{ci_lo_disatt:+.4f},{ci_hi_disatt:+.4f}]  "
            f"√ICC={sqrt_icc:.4f}")

        disatt_rows.append({
            "path":          row.get("path", ""),
            "direction":     "A",
            "microbiome_var": row.get("microbiome_var", ""),
            "outcome":       row.get("outcome", ""),
            "dataset":       ds,
            "n":             int(row.get("n", 0)) if not pd.isna(row.get("n")) else np.nan,
            "beta_obs":      round(beta_obs,  6),
            "se_obs":        round(se_obs,    6),
            "ci_lo_obs":     round(ci_lo,     6),
            "ci_hi_obs":     round(ci_hi,     6),
            "p":             float(row.get("p",     np.nan)),
            "q_fdr":         float(row.get("q_fdr", np.nan)),
            "outcome_icc31": round(icc,       6) if not np.isnan(icc) else np.nan,
            "sqrt_icc":      round(sqrt_icc,  6) if not np.isnan(sqrt_icc) else np.nan,
            "beta_disatt":   round(beta_disatt,  6) if not np.isnan(beta_disatt) else np.nan,
            "se_disatt":     round(se_disatt,    6) if not np.isnan(se_disatt) else np.nan,
            "ci_lo_disatt":  round(ci_lo_disatt, 6) if not np.isnan(ci_lo_disatt) else np.nan,
            "ci_hi_disatt":  round(ci_hi_disatt, 6) if not np.isnan(ci_hi_disatt) else np.nan,
            "obs_ci_excludes_zero": obs_excludes_zero,
            "disatt_ci_excludes_zero": dis_excludes_zero,
            "obs_ci_within_pm010": obs_within_01,
            "disatt_ci_within_pm010": dis_within_01,
        })

    disatt_df = pd.DataFrame(disatt_rows)
    disatt_df.to_csv(OUT / "crosslag_outcome_disattenuated.csv", index=False)
    log(f"\n  Wrote {OUT/'crosslag_outcome_disattenuated.csv'}")

    # ── 4. Summary verdict ─────────────────────────────────────────────────
    banner("VERDICT")
    for ds, lbl in OUTCOME_MAP.items():
        icc = outcome_iccs.get(ds, np.nan)
        if np.isnan(icc):
            continue
        sqrt_icc = np.sqrt(max(icc, 0))
        sub_dis = disatt_df[disatt_df["dataset"] == ds]
        n_new_zero = int(
            sub_dis["disatt_ci_excludes_zero"].sum() -
            sub_dis["obs_ci_excludes_zero"].sum())
        n_leaves_margin = int(
            sub_dis["obs_ci_within_pm010"].sum() -
            sub_dis["disatt_ci_within_pm010"].sum())
        log(f"  {lbl}: ICC={icc:.3f}, √ICC={sqrt_icc:.3f}")
        log(f"    Disattenuation inflates all β by ÷{sqrt_icc:.3f} ({1/sqrt_icc:.3f}×)")
        log(f"    Tests where CI newly excludes zero: {n_new_zero}")
        log(f"    Tests where CI leaves ±0.10 margin: {n_leaves_margin}")

    log("\n[DONE] N1 complete.")


if __name__ == "__main__":
    main()
