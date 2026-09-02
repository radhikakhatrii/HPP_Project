#!/usr/bin/env python3
"""
paper_numbers.py
================================================================================
One-shot script to compute every additional number needed to strengthen the
cross-lagged panel paper, beyond what cross_lagged_panel.py's main() already
writes to results_cross_lagged/.

Run AFTER cross_lagged_panel.py has completed at least once (so that
results_cross_lagged/cross_lag_all_paths.csv exists).

    conda activate pheno
    cd ~/studies/HPP_Prototype   # (or wherever cross_lagged_panel.py lives)
    python3 paper_numbers.py

Produces, in order:
  1. Visit interval (median / IQR / range, in years) between t1 and t2
  2. Minimum detectable effect (MDE) at 80% power, per cross-lag test
  3. Test-retest ICC(3,1) for Shannon diversity and species richness
  4. Corrected age mean/SD (fixes the "nan ± nan" row in table1_characteristics.csv)

All numbers are printed to the console AND written to
results_cross_lagged/paper_numbers_summary.txt for easy copy/paste or
screenshotting.
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cross_lagged_panel as clp  # reuses the exact same loaders as the main pipeline

OUT_DIR = clp.RESULTS_DIR if hasattr(clp, "RESULTS_DIR") else Path("results_cross_lagged")
OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

lines: list[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def section(title: str) -> None:
    emit("\n" + "=" * 72)
    emit(f"  {title}")
    emit("=" * 72)


# --------------------------------------------------------------------------- #
# 1. Visit interval
# --------------------------------------------------------------------------- #
section("1. Visit interval (t1 -> t2)")
try:
    gut_long = clp.load_longitudinal_gut()
    gut_long["collection_date"] = pd.to_datetime(gut_long["collection_date"])

    t1 = gut_long[gut_long["visit_n"] == 1][["study_id", "collection_date"]].rename(
        columns={"collection_date": "date_t1"})
    t2 = gut_long[gut_long["visit_n"] == 2][["study_id", "collection_date"]].rename(
        columns={"collection_date": "date_t2"})
    merged = t1.merge(t2, on="study_id")
    merged["interval_years"] = (merged["date_t2"] - merged["date_t1"]).dt.days / 365.25

    emit(f"  n participants with both visits dated: {len(merged):,}")
    emit(f"  median interval: {merged['interval_years'].median():.2f} yr")
    emit(f"  IQR: [{merged['interval_years'].quantile(0.25):.2f}, "
         f"{merged['interval_years'].quantile(0.75):.2f}] yr")
    emit(f"  min / max: {merged['interval_years'].min():.2f} / "
         f"{merged['interval_years'].max():.2f} yr")
    merged.to_csv(OUT_DIR / "visit_intervals.csv", index=False)
    emit(f"  wrote {OUT_DIR / 'visit_intervals.csv'}")
except Exception as exc:
    emit(f"  FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 2. Minimum detectable effect (MDE) at 80% power
# --------------------------------------------------------------------------- #
section("2. Minimum detectable effect (80% power), per cross-lag test")
crosslag_path = OUT_DIR / "cross_lag_all_paths.csv"
try:
    if not crosslag_path.exists():
        emit(f"  SKIPPED: {crosslag_path} not found. "
             f"Run cross_lagged_panel.py first.")
    else:
        df = pd.read_csv(crosslag_path)
        # ci95 was written as a Python tuple string, e.g. "(-0.031, 0.033)"
        ci = df["ci95"].astype(str).str.strip("()").str.split(",", expand=True)
        df["ci_lo"] = pd.to_numeric(ci[0], errors="coerce")
        df["ci_hi"] = pd.to_numeric(ci[1], errors="coerce")
        df["se"] = (df["ci_hi"] - df["ci_lo"]) / (2 * 1.96)
        df["mde_80pct_power"] = df["se"] * (1.96 + 0.84)  # two-sided alpha=.05, power=.80

        cols = ["path", "exposure", "outcome", "n", "std_beta", "se", "mde_80pct_power"]
        cols = [c for c in cols if c in df.columns]
        emit(df[cols].to_string(index=False))
        df.to_csv(OUT_DIR / "mde_by_test.csv", index=False)
        emit(f"\n  wrote {OUT_DIR / 'mde_by_test.csv'}")
        emit("\n  Interpretation: compare |std_beta| (observed effect) against "
             "mde_80pct_power (smallest effect the design could reliably detect). "
             "Observed effects well below MDE support a genuine null rather than "
             "an underpowered test.")
except Exception as exc:
    emit(f"  FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 3. Test-retest ICC for Shannon diversity and species richness
# --------------------------------------------------------------------------- #
section("3. Test-retest ICC(3,1) for Shannon diversity and richness")
try:
    gut_full = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut_full)

    def icc_2way(t1_vals: np.ndarray, t2_vals: np.ndarray) -> float:
        """ICC(3,1), consistency, two fixed raters (visit 1 vs visit 2)."""
        x = np.column_stack([t1_vals, t2_vals])
        n = x.shape[0]
        row_means = x.mean(axis=1)
        col_means = x.mean(axis=0)
        grand_mean = x.mean()
        ss_rows = 2 * np.sum((row_means - grand_mean) ** 2)
        ss_cols = n * np.sum((col_means - grand_mean) ** 2)
        ss_total = np.sum((x - grand_mean) ** 2)
        ss_error = ss_total - ss_rows - ss_cols
        ms_rows = ss_rows / (n - 1)
        ms_error = ss_error / (n - 1)
        return float((ms_rows - ms_error) / (ms_rows + ms_error))

    icc_rows = []
    for col in ["shannon", "richness"]:
        wide = clp.make_wide(gut_div, col).dropna()
        icc = icc_2way(wide[f"{col}_t1"].values, wide[f"{col}_t2"].values)
        pearson_r = float(np.corrcoef(wide[f"{col}_t1"], wide[f"{col}_t2"])[0, 1])
        emit(f"  {col:<10}: ICC(3,1) = {icc:.3f}, Pearson r = {pearson_r:.3f}, "
             f"n = {len(wide):,}")
        icc_rows.append({"metric": col, "icc": icc, "pearson_r": pearson_r,
                          "n": len(wide)})
    pd.DataFrame(icc_rows).to_csv(OUT_DIR / "diversity_icc.csv", index=False)
    emit(f"\n  wrote {OUT_DIR / 'diversity_icc.csv'}")
except Exception as exc:
    emit(f"  FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 4. Corrected age mean/SD (fixes "nan +/- nan" in table1_characteristics.csv)
# --------------------------------------------------------------------------- #
section("4. Corrected age mean / SD")
try:
    pop = clp.load_population()
    mean_age = float(pop["age_at_visit"].mean())
    sd_age = float(pop["age_at_visit"].std())
    n_age = int(pop["age_at_visit"].notna().sum())
    emit(f"  Age (years): {mean_age:.1f} +/- {sd_age:.1f}  (n={n_age:,})")
    emit("  -> manually replace the 'Age (years): nan +/- nan' row in "
         "table1_characteristics.csv with this line.")
except Exception as exc:
    emit(f"  FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Save combined summary
# --------------------------------------------------------------------------- #
summary_path = OUT_DIR / "paper_numbers_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nAll output also written to: {summary_path}")
