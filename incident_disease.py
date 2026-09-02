#!/usr/bin/env python3
"""
Incident disease analysis: gut microbiome diversity (t1) → new disease onset (t2).

This is the strongest causal claim available in longitudinal HPP data:
  - Prevalent disease at t1 is EXCLUDED (not baseline association)
  - Only participants disease-FREE at t1 who DEVELOP disease by t2 are cases
  - This is prospective prediction, not cross-sectional association

n ≈ 1,800 per phenotype (confirmed: all 1,823 gut repeat participants
have both visits of curated phenotype data).
"""
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Reuses cross_lagged_panel.py's corrected load_population() for age/sex
# rather than a standalone reimplementation -- see the age-covariate block
# below for why.
import cross_lagged_panel as clp

DATA_ROOT   = Path(os.environ.get("HPP_DATA_ROOT",  Path.home() / "studies" / "hpp_datasets"))
RESULTS_DIR = Path(os.environ.get("HPP_RESULTS_DIR", Path.home() / "studies" / "HPP_Prototype" / "results_cross_lagged"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_INCIDENT = 30    # minimum incident cases to run model
ALPHA_FDR    = 0.05

def log(m): print(m, flush=True)

def dedupe_columns(df, label=""):
    """Collapse duplicate column names, keeping the first occurrence.

    Needed at TWO points, not one: once on raw load (index levels that also
    exist as data columns), and AGAIN after the study_id/visit rename, because
    the rename itself can create a fresh collision whenever pid2 is not
    literally "participant_id" (e.g. it's already "study_id"), which the old
    single pre-rename dedup pass could not have anticipated.
    """
    if not df.columns.duplicated().any():
        return df
    dup_names = sorted(set(df.columns[df.columns.duplicated()]))
    log(f"    {label}: deduplicating columns {dup_names} (keeping first occurrence)")
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


def bh_fdr(pvals):
    pv = np.asarray(pvals, dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    rank = np.empty(n); rank[order] = np.arange(1, n+1)
    q = np.minimum(1.0, pv * n / rank)
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return q

# ── Step 1: Load gut microbiome Shannon diversity at visit 1 ──
log("Loading gut microbiome diversity at t1...")
gut = pd.read_parquet(DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
if gut.index.names != [None]: gut = gut.reset_index()
pid   = next(c for c in gut.columns if c in ("participant_id","study_id"))
stage = next(c for c in gut.columns if "research_stage" in c.lower())
gut   = gut.rename(columns={pid: "study_id", stage: "visit"})

# Keep repeat-visit participants only
counts = gut.groupby("study_id")["visit"].nunique()
repeat_ids = set(counts[counts >= 2].index)
gut = gut[gut["study_id"].isin(repeat_ids)].copy()

# Order visits by collection date where available, matching cross_lagged_panel.py.
# Dense-ranking the research_stage string sorts alphabetically, which is not
# guaranteed to be chronological and would silently swap t1/t2.
date_col = next((c for c in gut.columns
                 if c in ("collection_date", "collection_timestamp")), None)
if date_col:
    gut[date_col] = pd.to_datetime(gut[date_col], errors="coerce")
    gut = gut.sort_values(["study_id", date_col])
    gut["visit_n"] = gut.groupby("study_id").cumcount() + 1
    log(f"  visit_n assigned by {date_col}")
else:
    log("  WARNING: no collection date column; falling back to research_stage "
        "rank. Verify that stage labels sort chronologically.")
    gut["visit_n"] = gut.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

# Load species parquet for Shannon
sp_col = next((c for c in gut.columns if "species" in c.lower() and "parquet" in c.lower()), None)
sp_path_val = gut[sp_col].dropna().iloc[0]
gut_dir = DATA_ROOT / "gut_microbiome"
sp_path = gut_dir / str(sp_path_val).lstrip("./")
log(f"  Loading species from: {sp_path}")
sp = pd.read_parquet(sp_path)
if sp.index.names != [None]: sp = sp.reset_index()

sp_meta = {"participant_id","study_id","cohort","research_stage",
           "array_index","sample_name","collection_timestamp","collection_date","timezone"}
abund_cols = [c for c in sp.columns if c not in sp_meta
              and str(sp[c].dtype).startswith("float")]
log(f"  Species columns: {len(abund_cols)}")

abund = sp[abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
row_sums = abund.sum(axis=1)
sp = sp.copy()
sp["richness"] = (abund > 0).sum(axis=1).astype(float)
p = abund.div(row_sums.replace(0, np.nan), axis=0)
with np.errstate(divide="ignore", invalid="ignore"):
    lp = np.where(p > 0, np.log(p), 0.0)
sp["shannon"] = -(p.values * lp).sum(axis=1)
sp.loc[row_sums == 0, "shannon"] = np.nan

sp_pid   = next(c for c in ("participant_id","study_id") if c in sp.columns)
sp_stage = next(c for c in sp.columns if "research_stage" in c.lower())
# Select only the needed columns BEFORE renaming, so a pre-existing "study_id"
# elsewhere in `sp` (distinct from sp_pid) can never collide with the rename.
sp_small = sp[[sp_pid, sp_stage, "shannon", "richness"]].rename(
    columns={sp_pid: "study_id", sp_stage: "visit"})
sp_small = dedupe_columns(sp_small, label="species diversity (post-rename)")

# Join diversity onto gut visit table
gut_div = gut[["study_id","visit","visit_n"]].merge(sp_small, on=["study_id","visit"], how="left")
gut_div = gut_div.groupby(["study_id","visit_n"], as_index=False).agg(
    {"shannon":"mean","richness":"mean"})

gut_t1 = gut_div[gut_div["visit_n"] == 1][["study_id","shannon","richness"]].copy()
gut_t1 = gut_t1.dropna(subset=["shannon"])
log(f"  Gut t1 participants with Shannon: {len(gut_t1):,}")
log(f"  Shannon mean={gut_t1['shannon'].mean():.3f} sd={gut_t1['shannon'].std():.3f}")

# ── Step 2: Sex + age covariates ───────────────────────────────
# Age comes from cross_lagged_panel.py's load_population(), which backfills
# birth year from population.parquet.year_of_birth (near-complete coverage)
# and uses genuine per-visit age wherever a collection date is available.
# Reimplementing age loading independently here would risk silently drifting
# from that corrected source.
log("\nLoading sex + age covariates (via cross_lagged_panel.load_population, "
    "the corrected source)...")
pop_demo = clp.load_population()[["study_id", "age_at_visit", "sex_encoded"]].rename(
    columns={"age_at_visit": "age", "sex_encoded": "sex"})
log(f"  pop_demo: {len(pop_demo):,} rows, "
    f"sex non-null: {pop_demo['sex'].notna().sum():,}, "
    f"age non-null: {pop_demo['age'].notna().sum():,}")

# ── Step 2b: Baseline BMI covariate ────────────────────────────
# Diversity-hypertension associations in the literature routinely attenuate
# once BMI is adjusted for, since diet and adiposity sit upstream of both the
# microbiome and blood pressure. Without this, any diversity -> hypertension
# result is not defensible as more than a confound.
log("\nLoading baseline BMI covariate...")
try:
    anthro = pd.read_parquet(DATA_ROOT / "anthropometrics" / "anthropometrics.parquet")
    if anthro.index.names != [None]: anthro = anthro.reset_index()
    anthro_pid = next((c for c in anthro.columns if c in ("participant_id", "study_id")), None)
    anthro_stage = next((c for c in anthro.columns if "research_stage" in c.lower()), None)
    bmi_col = next((c for c in anthro.columns
                    if c.lower() in ("bmi", "body_mass_index")
                    or c.lower().endswith("__bmi")), None)
    if anthro_pid and anthro_stage and bmi_col:
        anthro = anthro.rename(columns={anthro_pid: "study_id", anthro_stage: "visit"})
        anthro = dedupe_columns(anthro, label="anthropometrics (post-rename)")
        date_col = next((c for c in anthro.columns
                         if c in ("collection_date", "collection_timestamp")), None)
        if date_col:
            anthro[date_col] = pd.to_datetime(anthro[date_col], errors="coerce")
            anthro = anthro.sort_values(["study_id", date_col])
            anthro["visit_n"] = anthro.groupby("study_id").cumcount() + 1
        else:
            anthro["visit_n"] = anthro.groupby("study_id")["visit"].transform(
                lambda x: x.rank(method="dense").astype(int))
        bmi_t1 = (anthro[anthro["visit_n"] == 1][["study_id", bmi_col]]
                  .rename(columns={bmi_col: "bmi_t1"})
                  .dropna(subset=["bmi_t1"]))
        bmi_t1["bmi_t1"] = pd.to_numeric(bmi_t1["bmi_t1"], errors="coerce")
        bmi_t1 = bmi_t1.dropna(subset=["bmi_t1"])
        bmi_t1 = bmi_t1[bmi_t1["bmi_t1"].between(12, 70)]  # plausible BMI range
        bmi_t1 = bmi_t1.drop_duplicates("study_id")
        log(f"  BMI col='{bmi_col}', n at t1 with plausible BMI: {len(bmi_t1):,}")
        pop_demo = pop_demo.merge(bmi_t1, on="study_id", how="left")
    else:
        log(f"  BMI column not resolved (pid={anthro_pid}, stage={anthro_stage}, "
            f"bmi_col={bmi_col}); proceeding WITHOUT BMI adjustment.")
except Exception as exc:
    log(f"  BMI load failed ({exc}); proceeding WITHOUT BMI adjustment.")

# Phenotypes where BMI is the outcome or a near-tautological proxy for it.
# Adjusting for BMI when predicting these would be circular. Compared via
# .lower() at the usage site since HPP ships both "abdominal_adiposity" and
# "Abdominal_Adiposity" as separate files for the same phenotype.
BMI_CIRCULAR_OUTCOMES = {"bmi", "abdominal_adiposity"}

# ── Step 3: Incident disease logistic regression ──────────────
log("\nRunning incident disease analysis...")
pheno_dir = DATA_ROOT / "curated_phenotypes"
results = []

for fpath in sorted(pheno_dir.glob("*.parquet")):
    name = fpath.stem
    try:
        df = pd.read_parquet(fpath)
        if df.index.names != [None]: df = df.reset_index()
        # Guarantee unique column names — reset_index creates duplicates when
        # both participant_id and study_id are index levels AND data columns
        df = dedupe_columns(df, label=f"{name} (pre-rename)")
        pid2   = next((c for c in df.columns if c == "participant_id"), None) or \
                 next((c for c in df.columns if c == "study_id"), None)
        stage2 = next((c for c in df.columns if "research_stage" in c.lower()), None)
        if not pid2 or not stage2: continue
        # If participant_id exists alongside study_id (cohort), drop cohort study_id first
        if pid2 == "participant_id" and "study_id" in df.columns:
            df = df.drop(columns=["study_id"])
        df = df.rename(columns={pid2: "study_id", stage2: "visit"})
        # Rename itself can create a NEW collision (e.g. pid2 was already
        # "study_id" and a distinct second "study_id"-named column exists, or
        # stage2's target "visit" collides with an existing "visit" column).
        # This is the fix for "The column label 'study_id' is not unique.",
        # which was silently killing every categorical phenotype before.
        df = dedupe_columns(df, label=f"{name} (post-rename)")
        df["visit_n"] = df.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int))

        # Find phenotype value column
        excl = {"study_id","visit","visit_n","cohort","array_index",
                "collection_timestamp","collection_date","timezone"}
        pheno_col = next((c for c in df.columns
                          if c not in excl and "curated" in c.lower()), None)
        if pheno_col is None:
            pheno_col = next((c for c in df.columns if c not in excl), None)
        if pheno_col is None: continue

        # Determine encoding type and define disease-positive label
        dtype_str = str(df[pheno_col].dtype)
        sample_vals = df[pheno_col].dropna().unique()

        # Negative label: values that mean "disease absent"
        NEG_KEYWORDS = ["not ", "non ", "non-", "never", "absent", "negative",
                        "no ", "healthy", "normal", "free"]
        # Exact tokens that the substring rules above miss. "None" contains
        # neither "non " nor "non-", so without this it read as disease-positive.
        NEG_EXACT = {"no", "none", "false", "0", "negative", "absent",
                     "normal", "healthy", "never", "nil", "n"}

        has_explicit_false = any(str(v).strip().lower() in ("false", "0")
                                 for v in sample_vals)
        is_boolean = (dtype_str == "bool"
                      or set(str(v).lower() for v in sample_vals) <= {"true", "false"})

        if is_boolean and has_explicit_false:
            # Boolean but fully enumerated: both True and False are recorded,
            # so presence in the table says nothing. Treat as categorical.
            log(f"  {name}: boolean with explicit False -> value-based logic")
            df["is_pos"] = (df[pheno_col].astype(str).str.lower()
                            .map({"true": 1, "false": 0}))
            df = df.dropna(subset=["is_pos"])
            df["is_pos"] = df["is_pos"].astype(int)
            t1 = df[df["visit_n"]==1][["study_id","is_pos"]].rename(
                columns={"is_pos":"pos_t1"}).drop_duplicates("study_id")
            t2 = df[df["visit_n"]==2][["study_id","is_pos"]].rename(
                columns={"is_pos":"pos_t2"}).drop_duplicates("study_id")
            wide = t1.merge(t2, on="study_id", how="inner")
            wide["incident"] = ((wide["pos_t1"]==0) & (wide["pos_t2"]==1)).astype(int)
            n_prevalent = int((wide["pos_t1"]==1).sum())
            n_incident  = int(wide["incident"].sum())
            n_at_risk   = int((wide["pos_t1"]==0).sum())
            wide = wide[wide["pos_t1"] == 0]

        elif is_boolean:
            # Presence-only boolean: absent from table = disease-free
            # Use all gut repeat participants as universe
            universe = set(gut_t1["study_id"])
            present_t1 = set(df[df["visit_n"]==1]["study_id"])
            present_t2 = set(df[df["visit_n"]==2]["study_id"])
            at_risk    = universe - present_t1          # disease-free at t1
            incident_s = at_risk & present_t2           # developed by t2
            wide = pd.DataFrame({"study_id": list(universe)})
            wide["incident"] = wide["study_id"].isin(incident_s).astype(int)
            wide["at_risk"]  = wide["study_id"].isin(at_risk).astype(int)
            wide = wide[wide["at_risk"] == 1].drop(columns=["at_risk"])
            n_prevalent = len(universe & present_t1)
            n_incident  = len(incident_s)
            n_at_risk   = len(at_risk)

        elif dtype_str in ("object","category") or df[pheno_col].dtype.name == "category":
            # Categorical: find negative label
            def is_negative(v):
                v_low = str(v).strip().lower()
                if v_low in NEG_EXACT:
                    return True
                return any(kw in v_low for kw in NEG_KEYWORDS)

            neg_labels = {v for v in sample_vals if is_negative(v)}
            pos_labels = {v for v in sample_vals if not is_negative(v)}
            if not neg_labels:
                log(f"  {name}: no negative label found in {list(sample_vals)[:5]}, skip")
                continue
            log(f"  {name}: neg={neg_labels}, pos={pos_labels}")

            # NaN must not become "positive": isin() is False for NaN, so a
            # bare ~isin() would silently label every missing value diseased.
            df["is_pos"] = np.where(
                df[pheno_col].isna(), np.nan,
                (~df[pheno_col].isin(neg_labels)).astype(float))
            df = df.dropna(subset=["is_pos"])
            df["is_pos"] = df["is_pos"].astype(int)
            t1 = df[df["visit_n"]==1][["study_id","is_pos"]].rename(
                columns={"is_pos":"pos_t1"}).drop_duplicates("study_id")
            t2 = df[df["visit_n"]==2][["study_id","is_pos"]].rename(
                columns={"is_pos":"pos_t2"}).drop_duplicates("study_id")
            wide = t1.merge(t2, on="study_id", how="inner")
            wide["incident"] = ((wide["pos_t1"]==0) & (wide["pos_t2"]==1)).astype(int)
            n_prevalent = int((wide["pos_t1"]==1).sum())
            n_incident  = int(wide["incident"].sum())
            n_at_risk   = int((wide["pos_t1"]==0).sum())
            # Restrict to the at-risk set. Without this, participants who
            # already had the disease at t1 carry incident=0 and are used as
            # CONTROLS, which contradicts the incident design in the docstring
            # and biases the odds ratio toward the null.
            wide = wide[wide["pos_t1"] == 0]

        else:
            # Continuous: skip (no natural disease-free threshold)
            log(f"  {name}: continuous dtype={dtype_str}, skip")
            continue

        if n_incident < MIN_INCIDENT:
            log(f"  {name}: n_incident={n_incident} < {MIN_INCIDENT}, skip")
            continue

        # Merge gut diversity t1
        merged = wide.merge(gut_t1, on="study_id", how="inner")
        # Optionally merge demographics
        if len(pop_demo) > 0 and "study_id" in pop_demo.columns:
            merged = merged.merge(pop_demo, on="study_id", how="left")

        merged = merged.dropna(subset=["shannon","incident"])
        n = len(merged)
        if n < 200:
            log(f"  {name}: n={n} after merge, skip")
            continue

        inc_rate = merged["incident"].mean()
        log(f"  {name}: n={n:,}, n_incident={merged['incident'].sum()}, "
            f"incidence_rate={inc_rate:.3f}, n_prevalent={n_prevalent}")

        # Full covariate set BMI is excluded when the outcome IS an adiposity
        # measure (bmi, abdominal_adiposity) -- adjusting for it there would
        # be circular, not a genuine confounder adjustment.
        full_cov = []
        if "sex" in merged.columns and merged["sex"].notna().sum() > 100:
            full_cov.append("sex")
        if "age" in merged.columns and merged["age"].notna().sum() > 100:
            full_cov.append("age")
        if ("bmi_t1" in merged.columns
                and merged["bmi_t1"].notna().sum() > 100
                and name.lower() not in BMI_CIRCULAR_OUTCOMES):
            full_cov.append("bmi_t1")
        elif "bmi_t1" in merged.columns and name.lower() in BMI_CIRCULAR_OUTCOMES:
            log(f"    {name}: BMI adjustment skipped (outcome is BMI-derived; "
                f"adjusting would be circular)")

        # Standardise continuous predictors only. Standardising the binary sex
        # indicator would make its odds ratio uninterpretable.
        for c in ["shannon", "richness"] + [x for x in full_cov if x != "sex"]:
            if c in merged.columns:
                mu, sd = merged[c].mean(), merged[c].std()
                if sd and sd > 0:
                    merged[c] = (merged[c] - mu) / sd

        # Age has ~61% coverage in this cohort (sociodemographics does not span
        # the full population), so requiring the FULL covariate set silently
        # throws away tests that would otherwise run fine -- this is what
        # happened to hypertension (n=670 -> n=160 -> below the n=200 floor,
        # silently skipped, no result at all). A fixed adjustment level is
        # therefore the wrong policy. Instead we try progressively lighter
        # covariate sets and use the first one that clears n>=200, recording
        # EXACTLY which set was used so nothing is silently downgraded.
        candidate_sets = []
        if full_cov:
            candidate_sets.append(list(full_cov))
        if "age" in full_cov and len(full_cov) > 1:
            candidate_sets.append([c for c in full_cov if c != "age"])
        if "bmi_t1" in full_cov and len(full_cov) > 1:
            candidate_sets.append([c for c in full_cov if c != "bmi_t1"])
        if "sex" in full_cov:
            candidate_sets.append(["sex"])
        candidate_sets.append([])
        # de-duplicate while preserving priority order (fullest adjustment first)
        seen_sets, ordered_sets = set(), []
        for cs in candidate_sets:
            key = tuple(sorted(cs))
            if key not in seen_sets:
                seen_sets.add(key)
                ordered_sets.append(cs)

        # One model per microbiome predictor. The previous version put only
        # `shannon` in the design matrix but looped over ["shannon","richness"],
        # so richness was silently dropped by the params-index guard and never
        # actually tested despite appearing in the output schema.
        for predictor in [c for c in ("shannon", "richness") if c in merged.columns]:
            cov, model_df = None, None
            for cs in ordered_sets:
                trial_cols = [predictor] + cs + ["incident"]
                trial_df = merged[trial_cols].dropna()
                if len(trial_df) >= 200:
                    cov, model_df = cs, trial_df
                    break
            if model_df is None:
                log(f"    {name}/{predictor}: no covariate set (down to none) "
                    f"reached n>=200, skipped entirely")
                continue
            if cov != full_cov:
                log(f"    {name}/{predictor}: full covariate set {full_cov} gave "
                    f"n<200; FELL BACK to {cov if cov else '[none]'} (n={len(model_df)}). "
                    f"This test is adjusted for LESS confounding than the others.")

            n_inc_model = int(model_df["incident"].sum())
            epv = n_inc_model / (1 + len(cov) + 1)  # events per variable, incl. predictor
            # [TASK 3] Previously this only logged a warning and still fit,
            # tabulated and FDR-corrected the model alongside legitimate ones
            # -- tabulate-then-disclaim, which also let fragile near-noise
            # p-values distort the across-the-family FDR correction for every
            # other test. Now excluded outright: not fit, not tabulated, not
            # part of the FDR family.
            if epv < 10:
                log(f"    {name}/{predictor}: EPV={epv:.1f} (< 10 rule-of-thumb "
                    f"floor) -- EXCLUDED as fragile, not tabulated")
                continue
            X = sm.add_constant(model_df[[predictor] + cov].astype(float))
            y = model_df["incident"].astype(int)
            try:
                fit = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=200)
                or_val = np.exp(fit.params[predictor])
                p_val  = fit.pvalues[predictor]
                ci     = np.exp(fit.conf_int().loc[predictor].tolist())
                results.append({
                    "phenotype": name,
                    "predictor": predictor,
                    "n": len(model_df),
                    "n_incident": n_inc_model,
                    "n_prevalent_excluded": int(n_prevalent),
                    "epv": round(epv, 1),
                    "incidence_rate": round(inc_rate, 4),
                    "OR": round(or_val, 4),
                    "CI_lo": round(ci[0], 4),
                    "CI_hi": round(ci[1], 4),
                    "p": round(p_val, 6),
                    "covariates": "+".join(cov) if cov else "none",
                    "full_adjustment": (cov == full_cov),
                })
            except Exception as e:
                log(f"    {name}/{predictor}: logit failed - {e}")

    except Exception as e:
        log(f"  {name}: ERROR — {e}")

if not results:
    log("\nNo results produced — check errors above.")
else:
    df_res = pd.DataFrame(results)
    # FDR separately per predictor
    for pred in df_res["predictor"].unique():
        mask = df_res["predictor"] == pred
        df_res.loc[mask, "q_fdr"] = bh_fdr(df_res.loc[mask, "p"].values)
    # Per-predictor FDR above; also correct across the whole family, which is
    # the stricter and more defensible statement now that both predictors run.
    df_res["q_fdr_all"] = bh_fdr(df_res["p"].values)
    df_res["significant"] = df_res["q_fdr"] < ALPHA_FDR
    df_res["significant_all"] = df_res["q_fdr_all"] < ALPHA_FDR
    df_res = df_res.sort_values(["predictor","p"])

    out_path = RESULTS_DIR / "incident_disease.csv"
    df_res.to_csv(out_path, index=False)
    log(f"\n{'='*60}")
    log(f"INCIDENT DISEASE RESULTS")
    log(f"{'='*60}")
    log(f"Total phenotypes tested: {df_res['phenotype'].nunique()}")
    log(f"Total tests: {len(df_res)} "
        f"({df_res['predictor'].nunique()} predictors x "
        f"{df_res['phenotype'].nunique()} phenotypes)")
    log(f"FDR-significant, within predictor (q<{ALPHA_FDR}): "
        f"{int(df_res['significant'].sum())}")
    log(f"FDR-significant, across all tests  (q<{ALPHA_FDR}): "
        f"{int(df_res['significant_all'].sum())}")
    log("")
    log(f"{'phenotype':<24}{'predictor':<10}{'OR':>8}  {'95% CI':<18}"
        f"{'p':>9}{'q(within)':>11}{'q(all)':>9}{'n_inc':>7}{'EPV':>7}  flag")
    for _, row in df_res.sort_values(["predictor", "p"]).iterrows():
        sig = "***" if row["significant"] else ("*" if row["p"] < 0.05 else "")
        thin = " [EPV<10]" if row.get("epv", 99) < 10 else ""
        fell_back = "" if row.get("full_adjustment", True) else " [PARTIAL ADJ]"
        log(f"  {row['phenotype']:<22}{row['predictor']:<10}{row['OR']:>8.3f}  "
            f"[{row['CI_lo']:.3f}-{row['CI_hi']:.3f}]"
            f"{row['p']:>9.4f}{row['q_fdr']:>11.4f}{row['q_fdr_all']:>9.4f}"
            f"{row['n_incident']:>7}{row.get('epv', float('nan')):>7.1f}  "
            f"{row.get('covariates','')}  {sig}{thin}{fell_back}")
    log("\n  q(within)  = FDR across the 9 phenotypes for that predictor alone")
    log("  q(all)     = FDR across all 18 tests (the standard used elsewhere in the paper)")
    log("  EPV        = events per model variable; <10 is the standard fragility floor")

    # ── Plot ──────────────────────────────────────────────────
    shannon_res = df_res[df_res["predictor"]=="shannon"].sort_values("OR")
    fig, ax = plt.subplots(figsize=(10, max(5, len(shannon_res)*0.45)))
    colors = ["#E53935" if s else "#78909C" for s in shannon_res["significant"]]
    for i, (_, row) in enumerate(shannon_res.iterrows()):
        ax.plot([row["CI_lo"], row["CI_hi"]], [i,i], color=colors[i], lw=1.5)
        marker = "*" if row["significant"] else "o"
        ax.scatter([row["OR"]], [i], color=colors[i], zorder=5, s=60, marker=marker)
        ax.text(max(row["CI_hi"], 1.5)+0.02, i,
                f"n={row['n_incident']}  p={row['p']:.3f}",
                va="center", fontsize=7.5)
    ax.axvline(1.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(shannon_res)))
    ax.set_yticklabels(shannon_res["phenotype"], fontsize=9)
    ax.set_xlabel("OR per SD Shannon diversity (higher OR = more diverse → lower risk)")
    ax.set_title(
        f"Gut Microbiome Diversity → Incident Disease (HPP longitudinal)\n"
        f"n≈1,800, controlling for baseline disease status | ★ = FDR q<0.05",
        fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "incident_disease_OR.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\nSaved: {RESULTS_DIR}/incident_disease.csv")
    log(f"Saved: {RESULTS_DIR}/incident_disease_OR.png")
