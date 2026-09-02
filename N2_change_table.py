#!/usr/bin/env python3
"""
N2_change_table.py
===============================================================================
Review point 5: Between-visit change table + grip-decline reality check.

For the 3 microbiome measures (Shannon, richness, AgeAccel) and 3 phenotypes
(grip, SBP, waist), reports visit-1 mean±SD, visit-2 mean±SD, mean Δ±SD,
and visit-to-visit Pearson r.

For grip specifically: confirms the ≥10% loss rate and estimates how much of
that fraction is expected from measurement noise alone, using grip ICC(3,1)
from N1 (0.8933).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 N2_change_table.py 2>&1 | tee N2_change_table.log

Reads:
    results_cross_lagged/leak_free_ageaccel_both_visits.csv  (microbiome vars)
    results_cross_lagged/review2/outcome_reliability.csv     (grip ICC from N1)
    hpp_datasets/hand_grip/hand_grip.parquet
    hpp_datasets/blood_pressure/blood_pressure.parquet
    hpp_datasets/anthropometrics/anthropometrics.parquet

Writes:
    results_cross_lagged/review2/change_summary.csv
    results_cross_lagged/review2/grip_decline_vs_noise.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT  = Path(clp.RESULTS_DIR) / "review2"
OUT.mkdir(parents=True, exist_ok=True)
log  = clp.log
RANDOM_SEED = clp.RANDOM_SEED   # 42
N_SIM = 100_000                  # simulations for noise-expected loss rate


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


# ── fast parquet loader (same pattern as N1) ──────────────────────────────
def load_parquet_cols(parquet_path: Path,
                      meas_candidates: list[str],
                      label: str,
                      keep_ids: set | None = None) -> pd.DataFrame | None:
    import pyarrow.parquet as pq, time
    if not parquet_path.exists():
        log(f"  {label}: not found at {parquet_path}")
        return None
    try:
        schema_cols = pq.read_schema(parquet_path).names
    except OSError as e:
        log(f"  {label}: cannot read schema — {e}")
        return None

    pid   = next((c for c in schema_cols if c in ("participant_id", "study_id")), None)
    stage = next((c for c in schema_cols if "research_stage" in c.lower()), None)
    meas  = next((c for c in meas_candidates if c in schema_cols), None)
    if not pid or not stage or not meas:
        log(f"  {label}: missing cols — pid={pid}, stage={stage}, meas={meas}")
        log(f"    schema sample: {schema_cols[:20]}")
        return None

    log(f"  {label}: reading [{pid}, {stage}, {meas}]")
    t0 = time.time()
    try:
        df = pd.read_parquet(parquet_path, columns=[pid, stage, meas])
    except OSError as e:
        log(f"  {label}: I/O error — {e}")
        return None
    log(f"    read {len(df):,} rows in {time.time()-t0:.2f}s")

    if pid not in df.columns:
        df = df.reset_index()
    if pid not in df.columns:
        log(f"  {label}: {pid} missing after reset_index")
        return None

    df = df.rename(columns={pid: "study_id", stage: "visit"})
    df[meas] = pd.to_numeric(df[meas], errors="coerce")
    if keep_ids is not None:
        df = df[df["study_id"].isin(keep_ids)]
    df["visit_n"] = df.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))
    df = df[df["visit_n"].isin([1, 2])].rename(columns={meas: label})
    log(f"    → {df['study_id'].nunique():,} participants after filtering")
    return df[["study_id", "visit_n", label]]


def change_stats(wide: pd.DataFrame, col: str) -> dict:
    """Given wide frame with columns v1, v2, return summary stats."""
    v1 = wide["v1"].dropna()
    v2 = wide["v2"].dropna()
    pair = wide[["v1", "v2"]].dropna()
    delta = pair["v2"] - pair["v1"]
    r, p  = pearsonr(pair["v1"], pair["v2"]) if len(pair) >= 10 else (np.nan, np.nan)
    return {
        "variable":   col,
        "n_v1":       len(v1),
        "mean_v1":    round(v1.mean(), 4),
        "sd_v1":      round(v1.std(),  4),
        "n_v2":       len(v2),
        "mean_v2":    round(v2.mean(), 4),
        "sd_v2":      round(v2.std(),  4),
        "n_paired":   len(pair),
        "mean_delta": round(delta.mean(), 4),
        "sd_delta":   round(delta.std(),  4),
        "visit_r":    round(r, 6),
        "visit_r_p":  p,
    }


def to_wide(df_long: pd.DataFrame, val_col: str) -> pd.DataFrame:
    """Long (study_id, visit_n, val) → wide (study_id, v1, v2)."""
    v1 = df_long[df_long["visit_n"] == 1].set_index("study_id")[val_col].rename("v1")
    v2 = df_long[df_long["visit_n"] == 2].set_index("study_id")[val_col].rename("v2")
    return pd.concat([v1, v2], axis=1).reset_index()


def main() -> None:
    banner("N2: BETWEEN-VISIT CHANGE TABLE + GRIP-DECLINE REALITY CHECK")

    # ── 1. Longitudinal IDs ───────────────────────────────────────────────
    banner("1. Loading longitudinal participant IDs")
    both_path = Path(clp.RESULTS_DIR) / "leak_free_ageaccel_both_visits.csv"
    if not both_path.exists():
        raise SystemExit(f"{both_path} not found — run cross_lagged_panel.py first.")
    both_df = pd.read_csv(both_path)
    log(f"  Columns in leak_free_ageaccel_both_visits.csv: {list(both_df.columns)}")
    longitudinal_ids = set(both_df["study_id"])
    log(f"  Longitudinal participants: {len(longitudinal_ids):,}")

    # ── 2. Microbiome measures per visit ──────────────────────────────────
    banner("2. Microbiome measures (Shannon, richness, AgeAccel)")

    # Determine what columns are available in the both-visits CSV
    micro_cols_available = [c for c in both_df.columns
                            if c in ("shannon", "richness", "gut_ageaccel",
                                     "age_accel", "ageaccel", "microbiome_age_accel")]
    log(f"  Microbiome cols in CSV: {micro_cols_available}")

    micro_long_frames = []

    if micro_cols_available:
        # CSV has diversity values — use them directly
        rename_map = {
            "gut_ageaccel": "AgeAccel", "age_accel": "AgeAccel",
            "ageaccel": "AgeAccel", "microbiome_age_accel": "AgeAccel",
            "shannon": "shannon", "richness": "richness",
        }
        micro_df = both_df[["study_id"] + micro_cols_available].copy()
        micro_df = micro_df.rename(columns=rename_map)
        if "visit_n" in both_df.columns:
            micro_df["visit_n"] = both_df["visit_n"]
        elif "visit" in both_df.columns:
            micro_df["visit_n"] = both_df.groupby("study_id")["visit"].transform(
                lambda x: x.rank(method="dense").astype(int))
        else:
            micro_df["visit_n"] = micro_df.groupby("study_id").cumcount() + 1

        for mc in [c for c in ("shannon", "richness", "AgeAccel")
                   if c in micro_df.columns]:
            sub = micro_df[["study_id", "visit_n", mc]].dropna(subset=[mc])
            sub = sub[sub["visit_n"].isin([1, 2])]
            micro_long_frames.append((mc, sub))
            log(f"  {mc}: {sub['study_id'].nunique():,} participants, {len(sub):,} rows")
    else:
        # CSV only has study_id — compute Shannon/richness from gut parquet directly
        log("  No diversity cols in CSV — computing Shannon/richness via compute_diversity()")
        log("  (AgeAccel change requires clock application to visit 2 — skipped here)")
        try:
            gut = clp.load_longitudinal_gut()
            div = clp.compute_diversity(gut)
            # filter to analytic longitudinal IDs
            div = div[div["study_id"].isin(longitudinal_ids)].copy()
            div["visit_n"] = div.groupby("study_id")["visit"].transform(
                lambda x: x.rank(method="dense").astype(int))
            div = div[div["visit_n"].isin([1, 2])]
            for mc in ("shannon", "richness"):
                if mc in div.columns:
                    sub = div[["study_id", "visit_n", mc]].dropna(subset=[mc])
                    micro_long_frames.append((mc, sub))
                    log(f"  {mc}: {sub['study_id'].nunique():,} participants, "
                        f"{len(sub):,} rows")
        except Exception as e:
            log(f"  WARNING: compute_diversity() failed — {e}")
            log("  Microbiome change stats will be omitted from change table")

    # ── 3. Phenotype outcomes ─────────────────────────────────────────────
    banner("3. Phenotype outcomes (grip, SBP, waist)")
    phenotype_specs = {
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
    }

    pheno_frames: dict[str, pd.DataFrame] = {}
    for label, spec in phenotype_specs.items():
        df = load_parquet_cols(spec["path"], spec["candidates"], label, longitudinal_ids)
        if df is not None:
            pheno_frames[label] = df

    # ── 4. Change table ───────────────────────────────────────────────────
    banner("4. Building change table")
    rows = []

    # Microbiome measures
    for mc, sub in micro_long_frames:
        wide = to_wide(sub, mc)
        rows.append(change_stats(wide, mc))
        log(f"  {mc}: Δ={rows[-1]['mean_delta']:+.4f}±{rows[-1]['sd_delta']:.4f}  "
            f"r={rows[-1]['visit_r']:.4f}  n={rows[-1]['n_paired']:,}")

    # Phenotypes
    for label, df in pheno_frames.items():
        wide = to_wide(df, label)
        rows.append(change_stats(wide, label))
        log(f"  {label}: Δ={rows[-1]['mean_delta']:+.4f}±{rows[-1]['sd_delta']:.4f}  "
            f"r={rows[-1]['visit_r']:.4f}  n={rows[-1]['n_paired']:,}")

    change_df = pd.DataFrame(rows)
    change_df.to_csv(OUT / "change_summary.csv", index=False)
    log(f"\n  Wrote {OUT/'change_summary.csv'}")

    # ── 5. Grip-decline reality check ─────────────────────────────────────
    banner("5. Grip-decline reality check")

    # Grip ICC from N1
    icc_path = OUT / "outcome_reliability.csv"
    if icc_path.exists():
        icc_df = pd.read_csv(icc_path)
        grip_row = icc_df[icc_df["outcome_label"] == "grip"]
        grip_icc = float(grip_row["icc31"].iloc[0]) if len(grip_row) else 0.8933
    else:
        grip_icc = 0.8933   # banked from N1
    log(f"  Grip ICC(3,1) = {grip_icc:.4f}  (from N1)")

    if "grip" not in pheno_frames:
        log("  grip data not available — skipping grip-decline analysis")
        return

    grip_wide = to_wide(pheno_frames["grip"], "grip").dropna()
    n_grip = len(grip_wide)

    # Observed ≥10% loss
    pct_change = (grip_wide["v2"] - grip_wide["v1"]) / grip_wide["v1"] * 100
    n_loss_10  = int((pct_change <= -10).sum())
    frac_loss_10_obs = n_loss_10 / n_grip

    log(f"\n  Observed grip % change: mean={pct_change.mean():+.2f}%  "
        f"SD={pct_change.std():.2f}%")
    log(f"  Participants losing ≥10% grip: {n_loss_10:,} / {n_grip:,} "
        f"= {frac_loss_10_obs*100:.1f}%")

    # Noise-expected ≥10% loss under null of NO true change
    # Under null: Δ ~ N(0, σ²_within), where σ²_within = 2·σ²_total·(1-ICC)
    # % change ~ N(0, [σ_within / mean_v1 × 100]²)
    mean_grip = float(grip_wide["v1"].mean())
    sd_grip   = float(grip_wide["v1"].std())
    sd_diff_null = sd_grip * np.sqrt(2 * (1 - grip_icc))
    sd_pctchange_null = sd_diff_null / mean_grip * 100

    # Analytic fraction (exact under normality)
    frac_loss_10_null_analytic = float(norm.cdf(-10, loc=0, scale=sd_pctchange_null))

    # Simulation-based fraction (draws pairs under null)
    rng = np.random.default_rng(RANDOM_SEED)
    true_vals  = rng.choice(grip_wide["v1"].values, size=N_SIM, replace=True)
    sigma_e    = sd_grip * np.sqrt(1 - grip_icc)   # per-measurement SD_error
    v1_sim     = true_vals + rng.normal(0, sigma_e, N_SIM)
    v2_sim     = true_vals + rng.normal(0, sigma_e, N_SIM)
    pct_sim    = (v2_sim - v1_sim) / v1_sim * 100
    frac_loss_10_null_sim = float((pct_sim <= -10).mean())

    log(f"\n  Under pure measurement-noise null (ICC={grip_icc:.4f}):")
    log(f"    SD of noise-only % change    = {sd_pctchange_null:.2f}%")
    log(f"    Noise-expected ≥10% loss (analytic) = "
        f"{frac_loss_10_null_analytic*100:.1f}%")
    log(f"    Noise-expected ≥10% loss (simulation, n={N_SIM:,}, seed={RANDOM_SEED}) = "
        f"{frac_loss_10_null_sim*100:.1f}%")
    log(f"\n  Observed 42% vs noise-expected {frac_loss_10_null_analytic*100:.1f}%:")
    excess = frac_loss_10_obs - frac_loss_10_null_analytic
    log(f"    Excess attributable to true decline = "
        f"{excess*100:.1f} pp  ({excess/frac_loss_10_obs*100:.0f}% of observed rate)")

    # Distribution summary
    pct_bins = [-100, -30, -20, -10, -5, 0, 5, 10, 20, 30, 100]
    counts = pd.cut(pct_change, bins=pct_bins).value_counts().sort_index()
    log("\n  Grip % change distribution:")
    for interval, cnt in counts.items():
        log(f"    {str(interval):>20}  {cnt:>5,}  ({cnt/n_grip*100:.1f}%)")

    # Save
    grip_noise_row = pd.DataFrame([{
        "n_grip_paired":              n_grip,
        "mean_grip_v1":               round(mean_grip, 3),
        "sd_grip_v1":                 round(sd_grip, 3),
        "mean_pct_change":            round(pct_change.mean(), 3),
        "sd_pct_change":              round(pct_change.std(), 3),
        "n_loss_ge10pct":             n_loss_10,
        "frac_loss_ge10pct_obs":      round(frac_loss_10_obs, 6),
        "grip_icc31":                 round(grip_icc, 6),
        "sd_noise_pct_change":        round(sd_pctchange_null, 4),
        "frac_loss_ge10pct_null_analytic": round(frac_loss_10_null_analytic, 6),
        "frac_loss_ge10pct_null_sim":      round(frac_loss_10_null_sim, 6),
        "n_sim":                      N_SIM,
        "random_seed":                RANDOM_SEED,
        "excess_pp":                  round(excess, 6),
        "pct_of_observed_rate_noise": round(frac_loss_10_null_analytic / frac_loss_10_obs, 4),
    }])
    grip_noise_row.to_csv(OUT / "grip_decline_vs_noise.csv", index=False)
    log(f"\n  Wrote {OUT/'grip_decline_vs_noise.csv'}")
    log("\n[DONE] N2 complete.")


if __name__ == "__main__":
    main()
