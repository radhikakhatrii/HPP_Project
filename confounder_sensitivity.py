#!/usr/bin/env python3
"""
confounder_sensitivity.py
===============================================================================
Adinath's point 3: does the Path A null survive adjustment for confounders and
technical factors beyond age and sex?

Baseline model (matches Table 2):
    aging(t2) ~ aging(t1) + microbiome(t1) + age + sex

Fully-adjusted model, same 9 Path A tests, adds:
    + diet quality score      (hpp_prototype._build_diet_score)
    + smoking status          (lifestyle_and_environment)
    + alcohol frequency       (lifestyle_and_environment)
    + physical activity       (hpp_prototype._build_pa)
    + sequencing depth        (raw_read_count, gut_microbiome.parquet)
    + any medication (binary) (medications.parquet)

We report both models side by side per test, not the adjusted model alone --
if the exposure coefficient and its significance are essentially unchanged
under full adjustment, that is the actual sensitivity claim. A shrinking
sample size from listwise deletion across seven more covariates is itself
worth watching, so n is reported for both models.

Reuses cross_lagged_panel.py's loaders for the microbiome/outcome side, and
hpp_prototype.py's lifestyle-covariate helpers so the diet/activity
construction is identical to what the rest of the project already uses --
this deliberately does NOT reimplement diet-score or PA logic from scratch.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python confounder_sensitivity.py 2>&1 | tee confounder_sensitivity.log

Writes: results_cross_lagged/confounder_sensitivity.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp
import hpp_prototype as hproto

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log

PATH_A_TESTS = [
    ("shannon", "hand_grip"), ("shannon", "blood_pressure"), ("shannon", "anthropometrics"),
    ("richness", "hand_grip"), ("richness", "blood_pressure"), ("richness", "anthropometrics"),
    ("gut_ageaccel", "hand_grip"), ("gut_ageaccel", "blood_pressure"), ("gut_ageaccel", "anthropometrics"),
]


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def fit(wide: pd.DataFrame, x_col: str, y_col: str, cov_cols: list[str],
        label: str) -> dict | None:
    need = [f"{x_col}_t1", f"{x_col}_t2", f"{y_col}_t1", f"{y_col}_t2"] + cov_cols
    have = [c for c in need if c in wide.columns]
    if len(have) < len(need):
        log(f"    {label}: missing columns {set(need) - set(have)}, skipped")
        return None
    sub = wide[have].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    if n < 200:
        log(f"    {label}: n={n} after listwise deletion, skipped")
        return None
    z = sub.copy()
    for c in z.columns:
        mu, sd = z[c].mean(), z[c].std()
        if sd and sd > 0:
            z[c] = (z[c] - mu) / sd
    X = sm.add_constant(z[[f"{y_col}_t1", f"{x_col}_t1"] + cov_cols].astype(float))
    y = z[f"{y_col}_t2"].astype(float)
    res = sm.OLS(y, X).fit()
    b = float(res.params[f"{x_col}_t1"])
    se = float(res.bse[f"{x_col}_t1"])
    p = float(res.pvalues[f"{x_col}_t1"])
    log(f"    {label:<12} n={n:<5} beta={b:+.4f}  SE={se:.4f}  p={p:.4f}")
    return {"model": label, "n": n, "beta_std": round(b, 5), "se": round(se, 5),
            "p": round(p, 6)}


def diagnose_sign_flip(long_df: pd.DataFrame, outcomes: dict,
                       covars_all: pd.DataFrame, full_cov: list[str],
                       microbiome_var: str, dataset: str) -> pd.DataFrame:
    """
    Investigate a baseline-vs-fully-adjusted sign flip: does it
    track the SAMPLE (listwise deletion across the added covariates changes
    WHO is in the model) or a SPECIFIC CONFOUNDER (one covariate's inclusion
    drives the sign change)? Two tests:
      (a) Refit the baseline model (age+sex only) restricted to exactly the
          same participants who survive the fully-adjusted model's
          complete-case filter. If the flip already appears here, with NO
          extra covariates in the model at all, it's a sample-composition
          artifact from listwise deletion, not genuine confounding.
      (b) Refit age+sex+X for each extra covariate individually. If the flip
          only appears once a SPECIFIC X is added, that names the confounder.
    """
    banner(f"DIAGNOSTIC: {microbiome_var} -> {dataset} sign flip")
    value_col = microbiome_var
    wide_x = clp.make_wide(long_df, value_col)
    out_df = outcomes.get(dataset)
    meas = next((c for c in out_df.columns
                if c not in ("study_id", "visit", "visit_n")), None)
    wide_y = clp.make_wide(out_df, meas)
    wide_y = wide_y.rename(columns={f"{meas}_t1": "outcome_t1",
                                    f"{meas}_t2": "outcome_t2"})
    wide = wide_x.join(wide_y, how="inner").join(covars_all, how="left")
    wide = wide.rename(columns={f"{value_col}_t1": f"{microbiome_var}_t1",
                                f"{value_col}_t2": f"{microbiome_var}_t2"})

    rows = []
    r_base_full = fit(wide, microbiome_var, "outcome", ["age", "sex"], "baseline_full_n")
    if r_base_full:
        rows.append(r_base_full)

    need_full = ([f"{microbiome_var}_t1", f"{microbiome_var}_t2",
                 "outcome_t1", "outcome_t2"] + full_cov)
    sub_full_adj = wide[[c for c in need_full if c in wide.columns]].apply(
        pd.to_numeric, errors="coerce").dropna()
    full_adj_ids = set(sub_full_adj.index)

    r_full = fit(wide, microbiome_var, "outcome", full_cov, "fully_adjusted")
    if r_full:
        rows.append(r_full)

    # (a) baseline restricted to the SAME complete-case ids as fully-adjusted
    wide_restricted = wide.loc[wide.index.isin(full_adj_ids)]
    r_base_restricted = fit(wide_restricted, microbiome_var, "outcome", ["age", "sex"],
                            "baseline_restricted_to_fulladj_n")
    if r_base_restricted:
        rows.append(r_base_restricted)

    # (b) age+sex+X for each extra covariate individually
    extra_covs = [c for c in full_cov if c not in ("age", "sex")]
    for c in extra_covs:
        r = fit(wide, microbiome_var, "outcome", ["age", "sex", c], f"age_sex_plus_{c}")
        if r:
            rows.append(r)

    diag_df = pd.DataFrame(rows)
    fname = f"{microbiome_var}_{dataset}_signflip_diagnostic.csv"
    diag_df.to_csv(OUT / fname, index=False)
    log(f"  wrote {OUT/fname}")

    log("\n  VERDICT:")
    if r_base_full is None:
        log("  baseline_full_n model failed to fit -- cannot diagnose.")
        return diag_df
    sign_full = np.sign(r_base_full["beta_std"])
    if r_base_restricted is not None:
        sign_restricted = np.sign(r_base_restricted["beta_std"])
        if sign_restricted != sign_full:
            log(f"  Baseline model restricted to the SAME n={r_base_restricted['n']} "
                f"sample as fully-adjusted ALREADY flips sign "
                f"(beta={r_base_restricted['beta_std']:+.4f}) with NO extra covariates "
                f"in the model at all. VERDICT: SAMPLE-COMPOSITION ARTIFACT from "
                f"listwise deletion across the added covariates, not genuine "
                f"confounding by any of them.")
        else:
            log(f"  Baseline model restricted to the same n={r_base_restricted['n']} "
                f"sample keeps the same sign (beta={r_base_restricted['beta_std']:+.4f}) "
                f"as the full-n baseline -- NOT purely a sample-composition artifact. "
                f"Checking individual covariates below for the driver:")
    else:
        log("  baseline_restricted_to_fulladj_n model failed to fit (n too small) -- "
            "cannot separate sample effect from confounding cleanly.")
    for r in rows:
        if r["model"].startswith("age_sex_plus_"):
            flipped = np.sign(r["beta_std"]) != sign_full
            marker = " <-- SIGN FLIPS HERE, candidate confounder" if flipped else ""
            log(f"    {r['model']:<28} beta={r['beta_std']:+.4f}  p={r['p']:.4f}  "
                f"n={r['n']}{marker}")
    return diag_df


def main() -> None:
    banner("CONFOUNDER AND TECHNICAL-FACTOR SENSITIVITY MODELS")

    gut = clp.load_longitudinal_gut()
    pop = clp.load_population()
    gut_div = clp.compute_diversity(gut)

    # build_gut_clock_ageaccel needs the FULL gut table (single-visit +
    # repeat-visit participants) to train the leak-free clock -- NOT the
    # repeat-visit-only `gut` from load_longitudinal_gut(). Passing `gut`
    # here would silently train and score on the same restricted population,
    # defeating the leak-free design. Loaded exactly as main() does it.
    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    ageaccel_df, _ = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
    outcomes = clp.load_functional_outcomes()

    # ---- baseline covariates: age + sex, matching Table 2 ----
    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].drop_duplicates("study_id")

    # ---- lifestyle: diet, smoking, alcohol, activity, via hpp_prototype ----
    banner("Loading lifestyle covariates (reusing hpp_prototype.py logic)")
    lif = hproto._load("lifestyle")
    lif_small = lif[["study_id"] + [c for c in
                     ("activity_vigorous_days_weekly", "activity_vigorous_minutes_daily",
                      "activity_moderate_days_weekly", "activity_moderate_minutes_daily",
                      "activity_walking_10min_days_weekly", "activity_walking_minutes_daily",
                      "smoking_current_status", "alcohol_current_frequency")
                     if c in lif.columns]].drop_duplicates("study_id")
    lif_small["pa_weekly_min"] = hproto._build_pa(lif_small)
    diet = hproto._build_diet_score()
    log(f"  diet_score: n={diet['diet_score'].notna().sum() if 'diet_score' in diet.columns else 0:,}")

    lifestyle_cov = (lif_small[["study_id", "pa_weekly_min", "smoking_current_status",
                                "alcohol_current_frequency"]]
                     .merge(diet[["study_id", "diet_score"]] if "diet_score" in diet.columns
                            else pd.DataFrame(columns=["study_id", "diet_score"]),
                            on="study_id", how="left"))
    for c in ("smoking_current_status", "alcohol_current_frequency"):
        lifestyle_cov[c] = pd.to_numeric(lifestyle_cov[c], errors="coerce")

    # ---- sequencing depth, visit 1, directly from the gut table ----
    banner("Loading sequencing depth (raw_read_count, visit 1)")
    depth = gut[gut["visit_n"] == 1][["study_id"]].copy()
    if "raw_read_count" in gut.columns:
        depth["seq_depth"] = pd.to_numeric(
            gut[gut["visit_n"] == 1].set_index("study_id")["raw_read_count"], errors="coerce"
        ).reindex(depth["study_id"]).values
        log(f"  seq_depth: n={depth['seq_depth'].notna().sum():,}")
    else:
        depth["seq_depth"] = np.nan
        log("  raw_read_count not found in gut table; seq_depth will be all-NaN "
            "and dropped from the adjusted model automatically by listwise deletion.")
    depth = depth.drop_duplicates("study_id")

    # ---- any medication, binary, from medications.parquet ----
    banner("Loading medication-use flag")
    try:
        meds = clp._load_parquet_flat(clp.DATA_ROOT / "medications" / "medications.parquet")
        m_pid = next((c for c in meds.columns if c in ("participant_id", "study_id")), None)
        if m_pid and m_pid != "study_id":
            meds = meds.rename(columns={m_pid: "study_id"})
        meds["on_medication"] = (~meds["medication"].astype(str)
                                 .str.lower().str.contains("no medications taken",
                                                           na=False)).astype(float)
        med_flag = meds.groupby("study_id", as_index=False)["on_medication"].max()
        log(f"  on_medication: n={len(med_flag):,}, "
            f"{int(med_flag['on_medication'].sum()):,} flagged positive")
    except Exception as exc:
        log(f"  medication load failed ({exc}); on_medication will be all-NaN.")
        med_flag = pd.DataFrame(columns=["study_id", "on_medication"])

    covars_all = (demo
                  .merge(lifestyle_cov, on="study_id", how="left")
                  .merge(depth, on="study_id", how="left")
                  .merge(med_flag, on="study_id", how="left"))
    covars_all = covars_all.rename(columns={"age_at_visit": "age", "sex_encoded": "sex"})
    covars_all = covars_all.set_index("study_id")

    FULL_COV = ["age", "sex", "diet_score", "smoking_current_status",
               "alcohol_current_frequency", "pa_weekly_min", "seq_depth",
               "on_medication"]
    for c in FULL_COV:
        n_ok = covars_all[c].notna().sum() if c in covars_all.columns else 0
        log(f"  coverage: {c:<26} n={n_ok:,}")

    banner("FITTING BASELINE vs FULLY-ADJUSTED MODELS, 9 PATH A TESTS")
    rows = []
    for microbiome_var, dataset in PATH_A_TESTS:
        if microbiome_var == "gut_ageaccel":
            long_df = ageaccel_df.rename(columns={"gut_ageaccel": "gut_ageaccel"})
            value_col = "gut_ageaccel"
        else:
            long_df = gut_div
            value_col = microbiome_var
        wide_x = clp.make_wide(long_df, value_col)

        out_df = outcomes.get(dataset)
        if out_df is None or out_df.empty:
            log(f"  {microbiome_var} x {dataset}: outcome table missing, skipped")
            continue
        meas = next((c for c in out_df.columns
                    if c not in ("study_id", "visit", "visit_n")), None)
        wide_y = clp.make_wide(out_df, meas)
        wide_y = wide_y.rename(columns={f"{meas}_t1": "outcome_t1",
                                        f"{meas}_t2": "outcome_t2"})

        wide = wide_x.join(wide_y, how="inner").join(covars_all, how="left")
        wide = wide.rename(columns={f"{value_col}_t1": f"{microbiome_var}_t1",
                                    f"{value_col}_t2": f"{microbiome_var}_t2"})

        log(f"\n  --- {microbiome_var} -> {dataset} ---")
        base = fit(wide, microbiome_var, "outcome", ["age", "sex"], "baseline")
        full = fit(wide, microbiome_var, "outcome", FULL_COV, "fully-adjusted")
        for r, label in ((base, "baseline"), (full, "fully_adjusted")):
            if r:
                r["microbiome_var"] = microbiome_var
                r["outcome"] = dataset
                rows.append(r)

    if rows:
        res = pd.DataFrame(rows)
        res.to_csv(OUT / "confounder_sensitivity.csv", index=False)
        banner("SUMMARY: DOES THE COEFFICIENT SURVIVE FULL ADJUSTMENT?")
        piv = res.pivot_table(index=["microbiome_var", "outcome"], columns="model",
                              values=["beta_std", "p", "n"])
        log(piv.to_string())
        log(f"\n  wrote {OUT/'confounder_sensitivity.csv'}")
        log("\n  Read each pair: if beta_std and p are similar between baseline and")
        log("  fully_adjusted, the null (or the effect) is robust to these")
        log("  confounders. A material n drop in fully_adjusted reflects listwise")
        log("  deletion across 6 added covariates and should be reported alongside")
        log("  any coefficient change, since a smaller n also widens the SE.")

        # richness -> hand_grip flipped sign and tightened under
        # full adjustment (baseline beta=+0.0227 p=0.0146 n=1,769 vs
        # fully-adjusted beta=-0.0331 p=0.0010 n=1,557) -- diagnose whether
        # that's sample composition or a specific confounder.
        diagnose_sign_flip(gut_div, outcomes, covars_all, FULL_COV,
                           "richness", "hand_grip")


if __name__ == "__main__":
    main()
