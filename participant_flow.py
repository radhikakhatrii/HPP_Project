#!/usr/bin/env python3
"""
participant_flow.py
===============================================================================
Adinath's point 2: a participant-flow diagram and a formal comparison of
included versus excluded participants at every stage of the pipeline.

This is the single largest open structural question in the paper. Each
cross-lagged test uses only 509-522 of the 1,823 repeat-visit participants,
because it additionally requires the specific phenotype at both visits. If
that 509-522 differs systematically from the people it excludes -- e.g. if
returners with complete data are healthier -- every effect estimate is biased
toward the null we report, and that bias has never been checked.

This script does two things:

  1. FLOW: counts participants surviving each successive inclusion criterion,
     from the full HPP population down to each individual cross-lagged test's
     analytic sample. Written as a table, ready to become a CONSORT-style
     flow diagram figure.

  2. INCLUDED vs EXCLUDED: for each of the three cross-lagged phenotypes
     (hand grip, blood pressure, anthropometrics), compares baseline age,
     sex, BMI, and baseline microbiome diversity between participants
     INCLUDED in that test and those in the 1,823-person repeat-visit pool
     but EXCLUDED from that specific test. Standardised mean differences
     (Cohen's d for continuous, Cohen's h for proportions) are reported
     alongside t-tests / chi-square tests, since with n in the hundreds a
     p-value alone under-communicates a real but modest imbalance.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python participant_flow.py 2>&1 | tee participant_flow.log

Writes to results_cross_lagged/:
    participant_flow_counts.csv
    participant_flow_included_vs_excluded.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def compare_group(included: pd.DataFrame, excluded: pd.DataFrame,
                  label: str) -> list[dict]:
    rows = []
    for col, kind in [("age", "continuous"), ("bmi_t1", "continuous"),
                      ("shannon_t1", "continuous"), ("richness_t1", "continuous")]:
        if col not in included.columns or col not in excluded.columns:
            continue
        a = included[col].dropna().values.astype(float)
        b = excluded[col].dropna().values.astype(float)
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = cohens_d(a, b)
        rows.append({
            "test": label, "variable": col,
            "n_included": len(a), "mean_included": round(float(a.mean()), 3),
            "sd_included": round(float(a.std()), 3),
            "n_excluded": len(b), "mean_excluded": round(float(b.mean()), 3),
            "sd_excluded": round(float(b.std()), 3),
            "cohens_d": round(d, 4), "t": round(float(t), 4), "p": round(float(p), 5),
        })
    if "sex" in included.columns and "sex" in excluded.columns:
        a = included["sex"].dropna()
        b = excluded["sex"].dropna()
        if len(a) > 5 and len(b) > 5:
            p1, p2 = float(a.mean()), float(b.mean())
            table = [[int((a == 1).sum()), int((a == 0).sum())],
                    [int((b == 1).sum()), int((b == 0).sum())]]
            chi2, p, _, _ = stats.chi2_contingency(table)
            rows.append({
                "test": label, "variable": "sex (proportion coded 1)",
                "n_included": len(a), "mean_included": round(p1, 4),
                "sd_included": None,
                "n_excluded": len(b), "mean_excluded": round(p2, 4),
                "sd_excluded": None,
                "cohens_d": round(cohens_h(p1, p2), 4),
                "t": round(float(chi2), 4), "p": round(float(p), 5),
            })
    return rows


def main() -> None:
    banner("PARTICIPANT FLOW AND SELECTION-BIAS CHECK")

    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    outcomes = clp.load_functional_outcomes()
    gut_div = clp.compute_diversity(gut)

    flow = []

    def record(label, ids):
        flow.append({"stage": label, "n_participants": len(set(ids))})
        log(f"  {label:<62} n = {len(set(ids)):,}")

    banner("1. FLOW COUNTS")
    all_pop_ids = set(pop["study_id"])
    record("Full HPP population (age/sex table)", all_pop_ids)

    gut_ids = set(gut["study_id"])
    record("Repeat-visit gut microbiome pairs", gut_ids)

    div_ids = set(gut_div["study_id"])
    record("With computable diversity at >=1 visit", div_ids)

    div_both = (gut_div.groupby("study_id")["visit_n"].nunique() >= 2)
    div_both_ids = set(div_both[div_both].index)
    record("With diversity at BOTH visits", div_both_ids)

    has_age = set(pop.dropna(subset=["age_at_visit"])["study_id"])
    has_sex = set(pop.dropna(subset=["sex_encoded"])["study_id"])
    age_sex_ids = div_both_ids & has_age & has_sex
    record("...and non-missing age AND sex", age_sex_ids)
    log("  (every cross-lagged model in Table 2 controls for age and sex, so")
    log("   this stage gates ALL downstream tests, not just this row -- fixing")
    log("   a bug in the first version of this script where the outcome-")
    log("   availability stage below was computed independently of this one,")
    log("   which let counts rise from 523 back up to ~1,800: impossible in a")
    log("   real funnel, and the tell that something upstream was wrong.)")

    outcome_both_ids = {}
    for name, df in outcomes.items():
        if df is None or df.empty:
            continue
        meas = next((c for c in df.columns
                    if c not in ("study_id", "visit", "visit_n")), None)
        if meas is None:
            continue
        w = df.dropna(subset=[meas])
        both = (w.groupby("study_id")["visit_n"].nunique() >= 2)
        # Cascade through age_sex_ids, not div_both_ids: this is what makes it
        # the actual model-ready analytic sample rather than an inflated count
        # that no regression in the paper ever actually used.
        ids = set(both[both].index) & age_sex_ids
        outcome_both_ids[name] = ids
        record(f"...AND {name} at both visits (this test's analytic sample)", ids)
        log(f"    [cross-check against Table 2] paper reports n=509/521/522 for "
            f"grip/BP/anthropometrics respectively -- compare against the count above")

    pd.DataFrame(flow).to_csv(OUT / "participant_flow_counts.csv", index=False)
    log(f"\n  wrote {OUT/'participant_flow_counts.csv'}")

    banner("2. BASELINE CHARACTERISTICS: INCLUDED vs EXCLUDED")
    log("  'Included' = analytic sample for that test. 'Excluded' = repeat-visit")
    log("  participants (with diversity at both visits) who did NOT make it into")
    log("  that specific test. A significant, sizeable difference means the null")
    log("  results are conditioned on a non-representative subset.\n")

    # Build a covariate frame: age, sex, baseline BMI, baseline diversity,
    # for every repeat-visit participant with diversity at both visits.
    base_div = (gut_div[gut_div["visit_n"] == 1][["study_id", "shannon", "richness"]]
                .rename(columns={"shannon": "shannon_t1", "richness": "richness_t1"}))
    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].rename(
        columns={"age_at_visit": "age", "sex_encoded": "sex"})

    # Load BMI directly from the raw anthropometrics table rather than reusing
    # outcomes["anthropometrics"] -- that dict holds whichever column WON the
    # most-non-null contest among ["bmi","body_mass_index","waist_circumference",
    # "waist_cm"], and Table 1 confirms the pipeline selected waist
    # circumference, not BMI. Depending on that dict here would have silently
    # left bmi_t1 empty for every comparison.
    bmi_t1 = pd.DataFrame(columns=["study_id", "bmi_t1"])
    try:
        anthro_raw = clp._load_parquet_flat(
            clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet")
        a_pid = next((c for c in anthro_raw.columns
                     if c in ("participant_id", "study_id")), None)
        a_stg = next((c for c in anthro_raw.columns
                     if "research_stage" in c.lower()), None)
        bmi_col = next((c for c in anthro_raw.columns
                        if c.lower() in ("bmi", "body_mass_index")), None)
        if a_pid and a_stg and bmi_col:
            if a_pid != "study_id" and "study_id" in anthro_raw.columns:
                anthro_raw = anthro_raw.drop(columns=["study_id"])
            anthro_raw = anthro_raw.rename(columns={a_pid: "study_id", a_stg: "visit"})
            anthro_raw["bmi"] = pd.to_numeric(anthro_raw[bmi_col], errors="coerce")
            anthro_raw = anthro_raw[anthro_raw["bmi"].between(12, 70)]
            vis = gut[["study_id", "visit", "visit_n"]].drop_duplicates()
            anthro_raw = anthro_raw.merge(vis, on=["study_id", "visit"], how="inner")
            bmi_t1 = (anthro_raw[anthro_raw["visit_n"] == 1][["study_id", "bmi"]]
                     .rename(columns={"bmi": "bmi_t1"})
                     .drop_duplicates("study_id"))
            log(f"  BMI loaded independently: n={len(bmi_t1):,} at visit 1")
        else:
            log(f"  BMI columns not resolved (pid={a_pid}, stage={a_stg}, "
                f"bmi={bmi_col}); bmi_t1 will be empty in the comparison.")
    except Exception as exc:
        log(f"  BMI load failed ({exc}); bmi_t1 will be empty in the comparison.")

    covars = (pd.DataFrame({"study_id": list(div_both_ids)})
              .merge(base_div, on="study_id", how="left")
              .merge(demo, on="study_id", how="left")
              .merge(bmi_t1, on="study_id", how="left"))

    all_rows = []
    for name, incl_ids in outcome_both_ids.items():
        excl_ids = div_both_ids - incl_ids
        incl_df = covars[covars["study_id"].isin(incl_ids)]
        excl_df = covars[covars["study_id"].isin(excl_ids)]
        log(f"\n  --- {name}:  included n={len(incl_df)}  excluded n={len(excl_df)} ---")
        rows = compare_group(incl_df, excl_df, name)
        for r in rows:
            flag = " <-- |d|>=0.2, worth noting" if abs(r["cohens_d"]) >= 0.2 else ""
            log(f"    {r['variable']:<28} included={r['mean_included']:>7}  "
                f"excluded={r['mean_excluded']:>7}  d={r['cohens_d']:+.3f}  "
                f"p={r['p']:.4f}{flag}")
        all_rows.extend(rows)

    if all_rows:
        res = pd.DataFrame(all_rows)
        res.to_csv(OUT / "participant_flow_included_vs_excluded.csv", index=False)
        log(f"\n  wrote {OUT/'participant_flow_included_vs_excluded.csv'}")

        banner("SUMMARY")
        n_sig = int((res["p"] < 0.05).sum())
        n_meaningful = int((res["cohens_d"].abs() >= 0.2).sum())
        log(f"  {n_sig} of {len(res)} comparisons significant at p<0.05")
        log(f"  {n_meaningful} of {len(res)} comparisons have |Cohen's d| >= 0.2 "
            f"(small-or-larger imbalance)")
        if n_meaningful == 0:
            log("\n  No comparison shows a meaningful imbalance. This is the strongest")
            log("  version of the answer to the selection-bias question: included and")
            log("  excluded participants look alike on every characteristic checked.")
        else:
            log("\n  At least one meaningful imbalance was found -- see the flagged rows")
            log("  above. Report these explicitly rather than treating the sample as")
            log("  representative by assumption.")


if __name__ == "__main__":
    main()
