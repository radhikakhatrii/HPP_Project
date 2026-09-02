#!/usr/bin/env python3
"""
prospective_outcomes.py
===============================================================================
point 8: redesign the prospective disease analysis so ONLY baseline
(t1) microbiome predicts genuinely SUBSEQUENT (t2) outcomes -- never using any
t2 microbiome measurement, and never using an outcome already present at t1
as anything but an exclusion criterion (so "incident" means truly new).

Four outcome families, all t1 -> t2, all predicted from gut_ageaccel(t1),
shannon(t1), richness(t1) separately, adjusted for age + sex:

  1. MULTIMORBIDITY PROGRESSION: change in distinct-diagnosis count
     (medical_conditions), continuous (OLS) and binarised "count increased"
     (logistic).
  2. HOSPITALIZATION (incident): presence in medical_procedures/
     follow_up_medical.parquet at t2 among participants with no record at t1.
  3. MEDICATION INITIATION (incident): not on any medication at t1 who are on
     medication at t2 (medications.parquet) -- only run if the table exposes
     a per-visit field; skipped explicitly (not approximated) otherwise.
  4. FUNCTIONAL DECLINE: hand-grip strength drop of >=10% from t1 to t2 (a
     conventional sarcopenia-relevant threshold), logistic.

Reuses cross_lagged_panel.py's population/gut/clock loaders. Does NOT
reimplement incident_disease.py's curated-phenotype sweep (already done,
separately, for the 18 curated conditions) -- this script targets the four
outcome types Adinath asked for that incident_disease.py does not cover.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python prospective_outcomes.py 2>&1 | tee prospective_outcomes.log

Writes: results_cross_lagged/prospective_outcomes.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log

PREDICTORS = ["gut_ageaccel", "shannon", "richness"]


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else s * 0.0


def derive_visit_n(df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    """Same pattern used throughout cross_lagged_panel.py: dense rank of
    research_stage per participant. THE BUG THIS FIXES: an earlier version of
    this function preferred a per-row collection_date/collection_timestamp
    over research_stage. That's correct for tables with one row per real
    clinical visit, but wrong for tables that can have MULTIPLE rows per real
    visit (e.g. one row per diagnosis in medical_conditions, one row per drug
    in medications) -- each row there carries its own distinct timestamp, so
    date-based cumcount silently assigned every row its own "visit",
    collapsing every (study_id, visit_n) group to size 1 and making
    multimorbidity/medication counts meaningless. research_stage is shared by
    every row recorded at the same real visit, so ranking on it is the
    correct grouping key whenever it's available; only fall back to
    collection_date if no research_stage column exists at all."""
    stg_col = next((c for c in df.columns if "research_stage" in c.lower()), None)
    date_col = next((c for c in df.columns
                     if c in ("collection_date", "collection_timestamp")), None)
    if stg_col:
        df["visit_n"] = df.groupby(pid_col)[stg_col].transform(
            lambda x: x.rank(method="dense").astype(int))
    elif date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([pid_col, date_col])
        df["visit_n"] = df.groupby(pid_col).cumcount() + 1
    else:
        df["visit_n"] = 1
    return df


def fit_model(df: pd.DataFrame, predictor: str, outcome: str, covars: list[str],
             label: str, binary: bool) -> dict | None:
    cols = [predictor, outcome] + covars
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    if n < 150:
        log(f"    {label:<44} n={n} < 150, skipped")
        return None
    z = sub.copy()
    z[predictor] = zscore(z[predictor])
    for c in covars:
        if c != "sex_encoded":
            z[c] = zscore(z[c])
    X = sm.add_constant(z[[predictor] + covars].astype(float))
    y = z[outcome].astype(float)
    try:
        if binary:
            # [TASK 3] The flat "n_events < 20" floor didn't account for how
            # many parameters the model actually fits (predictor + covars +
            # const). With covars=[age,sex] that's 4 params, so n_events=20
            # only gives EPV=5.0 -- half the standard <10 fragility floor --
            # and was tabulated with no EPV disclosed at all. Replaced with a
            # genuine EPV>=10 requirement on the SMALLER class (logistic
            # regression fragility is governed by whichever class is rarer),
            # matching incident_disease.py's convention, and the model is
            # EXCLUDED (not fit-then-flagged) when it fails.
            n_events = int(y.sum())
            n_params = 1 + len(covars) + 1  # predictor + covariates + constant
            min_class = min(n_events, n - n_events)
            epv = min_class / n_params
            if epv < 10:
                log(f"    {label:<44} n={n} n_events={n_events} EPV={epv:.1f} "
                    f"(<10, smaller class) -- EXCLUDED as fragile, not tabulated")
                return None
            fit = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
            eff = float(np.exp(fit.params[predictor]))
            p = float(fit.pvalues[predictor])
            log(f"    {label:<44} n={n:<5} n_events={n_events:<5} EPV={epv:.1f}  "
                f"OR={eff:.3f}  p={p:.4f}")
            return {"test": label, "predictor": predictor, "n": n,
                   "n_events": n_events, "epv": round(epv, 1), "effect": round(eff, 5),
                   "effect_type": "OR", "p": round(p, 6)}
        else:
            fit = sm.OLS(y, X).fit()
            eff = float(fit.params[predictor])
            p = float(fit.pvalues[predictor])
            log(f"    {label:<44} n={n:<5} beta={eff:+.4f}  p={p:.4f}")
            return {"test": label, "predictor": predictor, "n": n,
                   "n_events": np.nan, "effect": round(eff, 5),
                   "effect_type": "beta_std", "p": round(p, 6)}
    except Exception as exc:
        log(f"    {label:<44} model failed: {exc}")
        return None


def main() -> None:
    banner("PROSPECTIVE OUTCOMES: baseline microbiome -> genuinely subsequent events "
          "(Adinath point 8)")

    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut)

    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg_col = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    ageaccel_df, _ = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)

    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].drop_duplicates("study_id")
    covars = ["age_at_visit", "sex_encoded"]

    # ---- baseline predictor frame: gut_ageaccel_t1, shannon_t1, richness_t1 ----
    div_t1 = (gut_div[gut_div["visit_n"] == 1]
             .groupby("study_id", as_index=False)[["shannon", "richness"]].first())
    acc_t1 = (ageaccel_df[ageaccel_df["visit_n"] == 1]
             .groupby("study_id", as_index=False)["gut_ageaccel"].first()
             if "visit_n" in ageaccel_df.columns else
             ageaccel_df.groupby("study_id", as_index=False)["gut_ageaccel"].first())
    base = (div_t1.merge(acc_t1, on="study_id", how="outer")
                  .merge(demo, on="study_id", how="left"))
    log(f"  baseline predictor frame: n={len(base):,} "
        f"(shannon={base['shannon'].notna().sum():,}, "
        f"gut_ageaccel={base['gut_ageaccel'].notna().sum():,})")

    all_rows = []

    # ============================================================ 1. multimorbidity
    banner("1. MULTIMORBIDITY PROGRESSION (count of new incident diagnoses "
          "across curated_phenotypes, t1 -> t2)")
    # Raw medical_conditions.parquet was tried first and rejected: its
    # research_stage labels mix a comprehensive baseline intake (mean 4.06
    # diagnosis rows/participant) with lightweight follow-up phone calls
    # (mean ~1.1-1.5 rows), and visit_n is a per-participant relative rank so
    # "visit_n==2" maps to a different real-world stage for different
    # participants -- raw row-counts are not comparable across visits and
    # produced a spurious mean DECREASE of -2.02 conditions. Instead, this
    # reuses incident_disease.py's already-validated, per-condition,
    # strictly-incident (prevalent-excluded) logic on curated_phenotypes, and
    # sums incident flags across conditions per participant.
    try:
        NEG_KEYWORDS = ["not ", "non ", "non-", "never", "absent", "negative",
                       "no ", "healthy", "normal", "free"]
        NEG_EXACT = {"no", "none", "false", "0", "negative", "absent",
                    "normal", "healthy", "never", "nil", "n"}

        def dedupe_columns(df, label=""):
            if not df.columns.duplicated().any():
                return df
            seen, new_names = {}, []
            for c in df.columns:
                if c in seen:
                    seen[c] += 1
                    new_names.append(f"__drop_{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_names.append(c)
            df = df.copy()
            df.columns = new_names
            return df[[c for c in df.columns if not c.startswith("__drop_")]]

        universe = set(base["study_id"])
        incident_frames = []
        pheno_dir = clp.DATA_ROOT / "curated_phenotypes"
        for fpath in sorted(pheno_dir.glob("*.parquet")):
            name = fpath.stem
            try:
                df = pd.read_parquet(fpath)
                if df.index.names != [None]:
                    df = df.reset_index()
                df = dedupe_columns(df, label=f"{name} (pre-rename)")
                pid2 = (next((c for c in df.columns if c == "participant_id"), None)
                       or next((c for c in df.columns if c == "study_id"), None))
                stage2 = next((c for c in df.columns if "research_stage" in c.lower()), None)
                if not pid2 or not stage2:
                    continue
                if pid2 == "participant_id" and "study_id" in df.columns:
                    df = df.drop(columns=["study_id"])
                df = df.rename(columns={pid2: "study_id", stage2: "visit"})
                df = dedupe_columns(df, label=f"{name} (post-rename)")
                df["visit_n"] = df.groupby("study_id")["visit"].transform(
                    lambda x: x.rank(method="dense").astype(int))

                excl = {"study_id", "visit", "visit_n", "cohort", "array_index",
                       "collection_timestamp", "collection_date", "timezone"}
                pheno_col = next((c for c in df.columns
                                 if c not in excl and "curated" in c.lower()), None)
                if pheno_col is None:
                    pheno_col = next((c for c in df.columns if c not in excl), None)
                if pheno_col is None:
                    continue

                dtype_str = str(df[pheno_col].dtype)
                sample_vals = df[pheno_col].dropna().unique()
                has_explicit_false = any(str(v).strip().lower() in ("false", "0")
                                         for v in sample_vals)
                is_boolean = (dtype_str == "bool"
                             or set(str(v).lower() for v in sample_vals) <= {"true", "false"})

                if is_boolean and has_explicit_false:
                    df["is_pos"] = (df[pheno_col].astype(str).str.lower()
                                    .map({"true": 1, "false": 0}))
                    df = df.dropna(subset=["is_pos"])
                    df["is_pos"] = df["is_pos"].astype(int)
                    t1 = df[df["visit_n"] == 1][["study_id", "is_pos"]].rename(
                        columns={"is_pos": "pos_t1"}).drop_duplicates("study_id")
                    t2 = df[df["visit_n"] == 2][["study_id", "is_pos"]].rename(
                        columns={"is_pos": "pos_t2"}).drop_duplicates("study_id")
                    wide = t1.merge(t2, on="study_id", how="inner")
                    wide["incident"] = ((wide["pos_t1"] == 0) & (wide["pos_t2"] == 1)).astype(int)
                    wide = wide[wide["pos_t1"] == 0]
                elif is_boolean:
                    present_t1 = set(df[df["visit_n"] == 1]["study_id"])
                    present_t2 = set(df[df["visit_n"] == 2]["study_id"])
                    at_risk = universe - present_t1
                    incident_s = at_risk & present_t2
                    wide = pd.DataFrame({"study_id": list(at_risk)})
                    wide["incident"] = wide["study_id"].isin(incident_s).astype(int)
                elif dtype_str in ("object", "category") or df[pheno_col].dtype.name == "category":
                    def is_negative(v):
                        v_low = str(v).strip().lower()
                        return v_low in NEG_EXACT or any(kw in v_low for kw in NEG_KEYWORDS)
                    neg_labels = {v for v in sample_vals if is_negative(v)}
                    if not neg_labels:
                        continue
                    df["is_pos"] = np.where(
                        df[pheno_col].isna(), np.nan,
                        (~df[pheno_col].isin(neg_labels)).astype(float))
                    df = df.dropna(subset=["is_pos"])
                    df["is_pos"] = df["is_pos"].astype(int)
                    t1 = df[df["visit_n"] == 1][["study_id", "is_pos"]].rename(
                        columns={"is_pos": "pos_t1"}).drop_duplicates("study_id")
                    t2 = df[df["visit_n"] == 2][["study_id", "is_pos"]].rename(
                        columns={"is_pos": "pos_t2"}).drop_duplicates("study_id")
                    wide = t1.merge(t2, on="study_id", how="inner")
                    wide["incident"] = ((wide["pos_t1"] == 0) & (wide["pos_t2"] == 1)).astype(int)
                    wide = wide[wide["pos_t1"] == 0]
                else:
                    continue

                if wide["incident"].sum() < 10:
                    continue
                incident_frames.append(
                    wide[["study_id", "incident"]].rename(columns={"incident": f"inc_{name}"}))
            except Exception as exc:
                log(f"    {name}: failed ({exc})")
                continue

        log(f"  usable curated_phenotype conditions: {len(incident_frames)}")
        if not incident_frames:
            log("  no usable incident-condition frames -- multimorbidity outcome skipped.")
        else:
            from functools import reduce
            allinc = reduce(lambda a, b: a.merge(b, on="study_id", how="outer"), incident_frames)
            inc_cols = [c for c in allinc.columns if c.startswith("inc_")]
            allinc["n_new_diagnoses"] = allinc[inc_cols].sum(axis=1, skipna=True)
            allinc["n_conditions_at_risk"] = allinc[inc_cols].notna().sum(axis=1)
            allinc["any_new_diagnosis"] = (allinc["n_new_diagnoses"] > 0).astype(float)
            log(f"  combined frame: n={len(allinc):,}, "
                f"mean n_new_diagnoses={allinc['n_new_diagnoses'].mean():.2f}, "
                f"any_new_diagnosis rate={allinc['any_new_diagnosis'].mean():.3f}")
            # require reasonable coverage across conditions so a participant who
            # was only at-risk for 1 of N conditions doesn't get treated the same
            # as one at risk for all of them
            min_coverage = max(3, len(inc_cols) // 3)
            mm = allinc[allinc["n_conditions_at_risk"] >= min_coverage].copy()
            log(f"  requiring at-risk coverage >= {min_coverage}/{len(inc_cols)} conditions: "
                f"n={len(mm):,} retained")
            merged = base.merge(
                mm[["study_id", "n_new_diagnoses", "any_new_diagnosis"]],
                on="study_id", how="inner")
            for pred in PREDICTORS:
                if pred not in merged.columns:
                    continue
                r = fit_model(merged, pred, "n_new_diagnoses", covars,
                             f"{pred} -> multimorbidity accrual (continuous)", binary=False)
                if r: all_rows.append(r)
                r = fit_model(merged, pred, "any_new_diagnosis", covars,
                             f"{pred} -> any new diagnosis (binary)", binary=True)
                if r: all_rows.append(r)
    except Exception as exc:
        log(f"  multimorbidity analysis failed: {exc}")

    # ============================================================ 2. hospitalization
    banner("2. HOSPITALIZATION (incident, t1 -> t2 presence)")
    try:
        hosp_path = clp.DATA_ROOT / "medical_procedures" / "follow_up_medical.parquet"
        if not hosp_path.exists():
            log(f"  {hosp_path} not found; hospitalization outcome skipped "
                f"(not approximated).")
        else:
            hosp = pd.read_parquet(hosp_path)
            if hosp.index.names != [None]:
                hosp = hosp.reset_index()
            h_pid = next(c for c in hosp.columns if c in ("participant_id", "study_id"))
            hosp = hosp.rename(columns={h_pid: "study_id"})
            log(f"  follow_up_medical columns: {list(hosp.columns)}")
            hosp_col = next((c for c in hosp.columns
                            if "hospital" in c.lower() or "admission" in c.lower()), None)
            hosp = derive_visit_n(hosp, "study_id")
            if hosp_col:
                log(f"  using explicit column: {hosp_col}")
                vals = hosp[hosp_col].dropna().unique()
                is_bool = str(hosp[hosp_col].dtype) == "bool" or set(
                    str(v).lower() for v in vals) <= {"true", "false"}
                if is_bool:
                    hosp["is_hosp"] = hosp[hosp_col].astype(str).str.lower().map(
                        {"true": 1, "false": 0})
                else:
                    hosp["is_hosp"] = pd.to_numeric(hosp[hosp_col], errors="coerce")
                t1p = set(hosp[(hosp["visit_n"] == 1) & (hosp["is_hosp"] == 1)]["study_id"])
                t2p = set(hosp[(hosp["visit_n"] == 2) & (hosp["is_hosp"] == 1)]["study_id"])
            else:
                log("  no explicit hospitalization flag column; treating row "
                    "PRESENCE in this table per visit as a hospitalization record "
                    "(presence-only encoding, same convention as incident_disease.py).")
                t1p = set(hosp[hosp["visit_n"] == 1]["study_id"])
                t2p = set(hosp[hosp["visit_n"] == 2]["study_id"])
            universe = set(base["study_id"])
            at_risk = universe - t1p
            incident_s = at_risk & t2p
            hh = pd.DataFrame({"study_id": list(at_risk)})
            hh["hosp_incident"] = hh["study_id"].isin(incident_s).astype(float)
            log(f"  at-risk (no hospitalization record at t1): n={len(hh):,}, "
                f"incident: {int(hh['hosp_incident'].sum())} "
                f"({100*hh['hosp_incident'].mean():.1f}%)")
            merged = base.merge(hh, on="study_id", how="inner")
            for pred in PREDICTORS:
                if pred not in merged.columns:
                    continue
                r = fit_model(merged, pred, "hosp_incident", covars,
                             f"{pred} -> incident hospitalization", binary=True)
                if r: all_rows.append(r)
    except Exception as exc:
        log(f"  hospitalization analysis failed: {exc}")

    # ============================================================ 3. medication initiation
    banner("3. MEDICATION INITIATION (incident, t1 -> t2)")
    try:
        med_path = clp.DATA_ROOT / "medications" / "medications.parquet"
        meds = pd.read_parquet(med_path)
        if meds.index.names != [None]:
            meds = meds.reset_index()
        m_pid = next(c for c in meds.columns if c in ("participant_id", "study_id"))
        meds = meds.rename(columns={m_pid: "study_id"})
        stg_present = any("research_stage" in c.lower() for c in meds.columns)
        date_present = any(c in ("collection_date", "collection_timestamp") for c in meds.columns)
        if not (stg_present or date_present):
            log(f"  medications.parquet has no visit/date column "
                f"({list(meds.columns)}); medication-initiation outcome cannot "
                f"be computed per-visit and is skipped (not approximated).")
        else:
            meds = derive_visit_n(meds, "study_id")
            # DIAGNOSTIC: see the raw free-text values driving the "no medications"
            # match before trusting it -- an implausibly high initiation rate
            # usually means some other "none" phrasing isn't being caught.
            raw_vals = meds["medication"].dropna().astype(str)
            log(f"  raw 'medication' value_counts (top 15):\n"
                f"{raw_vals.value_counts().head(15).to_string()}")
            meds["on_med"] = (~meds["medication"].astype(str).str.lower()
                              .str.contains("no medications taken", na=False)).astype(float)
            m1 = (meds[meds["visit_n"] == 1].groupby("study_id")["on_med"].max()
                  .rename("med_t1").reset_index())
            m2 = (meds[meds["visit_n"] == 2].groupby("study_id")["on_med"].max()
                  .rename("med_t2").reset_index())
            log(f"  on_med rate at t1: {m1['med_t1'].mean():.3f} (n={len(m1):,}); "
                f"at t2: {m2['med_t2'].mean():.3f} (n={len(m2):,})")
            mw = m1.merge(m2, on="study_id", how="inner")
            mw = mw[mw["med_t1"] == 0].copy()
            mw["med_initiated"] = mw["med_t2"]
            log(f"  med-naive-at-t1 raw medication values at t2 that flip to "
                f"'initiated' (sample of 10 distinct study_ids' t2 rows):")
            sample_ids = mw[mw["med_initiated"] == 1]["study_id"].head(10)
            sample_rows = meds[(meds["study_id"].isin(sample_ids)) & (meds["visit_n"] == 2)]
            log(sample_rows[["study_id", "medication"]].to_string())
            log(f"  medication-naive at t1: n={len(mw):,}, initiated by t2: "
                f"{int(mw['med_initiated'].sum())} ({100*mw['med_initiated'].mean():.1f}%)")
            merged = base.merge(mw, on="study_id", how="inner")
            for pred in PREDICTORS:
                if pred not in merged.columns:
                    continue
                r = fit_model(merged, pred, "med_initiated", covars,
                             f"{pred} -> medication initiation", binary=True)
                if r: all_rows.append(r)
    except FileNotFoundError:
        log("  medications.parquet not found; medication-initiation outcome skipped.")
    except Exception as exc:
        log(f"  medication-initiation analysis failed: {exc}")

    # ============================================================ 4. functional decline
    banner("4. FUNCTIONAL DECLINE (hand-grip strength drop >=10%, t1 -> t2)")
    try:
        outcomes = clp.load_functional_outcomes()
        grip = outcomes.get("hand_grip")
        if grip is None or grip.empty:
            log("  hand_grip outcome table unavailable; functional-decline outcome skipped.")
        else:
            meas = next(c for c in grip.columns if c not in ("study_id", "visit", "visit_n"))
            wide = clp.make_wide(grip, meas)
            wide = wide.rename(columns={f"{meas}_t1": "grip_t1", f"{meas}_t2": "grip_t2"})
            wide = wide.reset_index()
            wide = wide.dropna(subset=["grip_t1", "grip_t2"])
            wide = wide[wide["grip_t1"] > 0]
            wide["pct_change"] = (wide["grip_t2"] - wide["grip_t1"]) / wide["grip_t1"]
            wide["decline"] = (wide["pct_change"] <= -0.10).astype(float)
            log(f"  n={len(wide):,}, declined >=10%: {int(wide['decline'].sum())} "
                f"({100*wide['decline'].mean():.1f}%)")
            merged = base.merge(wide[["study_id", "decline"]], on="study_id", how="inner")
            for pred in PREDICTORS:
                if pred not in merged.columns:
                    continue
                r = fit_model(merged, pred, "decline", covars,
                             f"{pred} -> functional decline (grip -10%)", binary=True)
                if r: all_rows.append(r)
    except Exception as exc:
        log(f"  functional-decline analysis failed: {exc}")

    # ---------------------------------------------------------------- summary
    banner("SUMMARY")
    if not all_rows:
        log("  No models produced results -- check the section-by-section log above.")
        return
    res = pd.DataFrame(all_rows)
    _, res["q_fdr"], _, _ = multipletests(res["p"], alpha=0.05, method="fdr_bh")
    res["significant"] = res["q_fdr"] < 0.05
    res.to_csv(OUT / "prospective_outcomes.csv", index=False)
    log(res.sort_values("p").to_string(index=False))
    log(f"\n  {len(res)} tests run across 4 outcome families")
    log(f"  FDR-significant (q<0.05): {int(res['significant'].sum())}")
    log(f"\n  wrote {OUT/'prospective_outcomes.csv'}")


if __name__ == "__main__":
    main()
