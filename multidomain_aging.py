#!/usr/bin/env python3
"""
multidomain_aging.py
===============================================================================
Adinath's point 5: build a multidomain biological-aging score from the
clinical data already in HPP, then test whether baseline microbiome AgeAccel
predicts subsequent CHANGE in it.

This is the direct answer to the limitation the manuscript already concedes:
that grip strength, systolic blood pressure and waist circumference are a
coarse three-phenotype proxy for "aging". Here we use seven physiological
domains built from verified columns in blood_tests, blood_pressure,
anthropometrics and hand_grip.

DOMAINS (higher score = more aged, by construction)
  metabolic     glucose, HbA1c, triglycerides, HDL(-)
  renal         creatinine, urea, urine ACR
  hepatic       ALT, AST, GGT, alkaline phosphatase, albumin(-)
  inflammatory  WBC, neutrophil %, lymphocyte %(-), RDW, ferritin
  cardiovascular systolic blood pressure
  body_comp     BMI, waist circumference
  physical      grip strength(-)

TWO DESIGN POINTS THAT MATTER

1. Standardisation uses VISIT-1 parameters applied to BOTH visits. Z-scoring
   each visit separately would force the mean change to zero and destroy
   exactly the signal we are trying to measure.

2. The composite is VALIDATED before it is used: a genuine aging score must
   correlate with chronological age. If it does not, nothing downstream of it
   means anything, and the script says so rather than proceeding quietly.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python multidomain_aging.py 2>&1 | tee multidomain_aging.log

Writes to results_cross_lagged/:
    multidomain_score_components.csv
    multidomain_aging_results.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log

# (column, sign) -- sign +1 means HIGHER value = MORE aged.
# Column names verified against the blood_tests schema dump.
DOMAINS: dict[str, list[tuple[str, float]]] = {
    "metabolic": [
        ("bt__glucose_float_value", +1),
        ("bt__hba1c_float_value", +1),
        ("bt__triglycerides_float_value", +1),
        ("bt__hdl_cholesterol_float_value", -1),
    ],
    "renal": [
        ("bt__creatinine_float_value", +1),
        ("bt__urea_float_value", +1),
        ("urine__acr_float_value", +1),
    ],
    "hepatic": [
        ("bt__alt_float_value", +1),
        ("bt__ast_float_value", +1),
        ("bt__ggt_float_value", +1),
        ("bt__alkaline_phosphatase_float_value", +1),
        ("bt__albumin_float_value", -1),
    ],
    "inflammatory": [
        ("bt__wbc_float_value", +1),
        ("bt__neutrophils_%_float_value", +1),
        ("bt__lymphocytes_%_float_value", -1),
        ("bt__rdw_float_value", +1),
        ("bt__ferritin_float_value", +1),
    ],
}
# Domains sourced from the functional-outcome tables rather than blood_tests
FUNCTIONAL_DOMAINS = {
    "cardiovascular": ("blood_pressure", +1),
    "body_comp": ("anthropometrics", +1),
    "physical": ("hand_grip", -1),
}


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def winsorise(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    """Clip extreme lab values before z-scoring. A single implausible
    creatinine would otherwise dominate an entire domain score."""
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def main() -> None:
    banner("MULTIDOMAIN BIOLOGICAL AGING SCORE")

    gut = clp.load_longitudinal_gut()
    pop = clp.load_population()
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

    vis = gut[["study_id", "visit", "visit_n"]].drop_duplicates()

    # ---------------------------------------------------------------- labs
    banner("1. Loading blood tests and mapping to visits")
    bt = clp._load_parquet_flat(clp.DATA_ROOT / "blood_tests" / "blood_tests.parquet")
    b_pid = next((c for c in bt.columns if c in ("participant_id", "study_id")), None)
    b_stg = next((c for c in bt.columns if "research_stage" in c.lower()), None)
    b_date = next((c for c in bt.columns
                  if c in ("collection_date", "collection_timestamp")), None)
    if b_pid != "study_id" and "study_id" in bt.columns:
        bt = bt.drop(columns=["study_id"])
    bt = bt.rename(columns={b_pid: "study_id", b_stg: "visit"})
    log(f"  blood_tests raw: {len(bt):,} rows, date col = {b_date}")

    # THE BUG THIS REPLACES: joining on literal (study_id, visit) string
    # equality assumes blood_tests encodes research_stage identically to the
    # gut table. It doesn't -- that join produced ZERO matches, silently
    # zeroing every lab-based domain and only surfacing at the validation
    # gate below. Fix: derive blood_tests' OWN visit_n by date order (the
    # same pattern already used for anthropometrics/BMI elsewhere in this
    # project), then join gut and blood_tests by visit ORDER, not by
    # matching the raw stage label text.
    if b_date is None:
        raise SystemExit("FATAL: blood_tests has no collection_date/timestamp "
                         "column, cannot derive visit order independently.")
    bt[b_date] = pd.to_datetime(bt[b_date], errors="coerce")
    n_bad_date = bt[b_date].isna().sum()
    if n_bad_date:
        log(f"  WARNING: {n_bad_date:,} blood_tests rows have unparseable dates, dropped")
    bt = bt.dropna(subset=[b_date])
    bt = bt.sort_values(["study_id", b_date])
    bt["visit_n"] = bt.groupby("study_id").cumcount() + 1
    log(f"  blood_tests visit_n assigned by date order: "
        f"{(bt['visit_n'] <= 2).sum():,} rows at visit 1 or 2")

    bt = bt.merge(vis[["study_id", "visit_n"]].drop_duplicates(), on=["study_id", "visit_n"],
                 how="inner")
    log(f"  blood_tests rows mapped to gut visits (by visit ORDER): {len(bt):,}")
    if len(bt) == 0:
        log("  STILL ZERO after the fix -- blood_tests and gut_microbiome may use")
        log("  incompatible visit numbering (e.g. blood drawn at extra call-only")
        log("  stages that don't correspond 1:1 with stool collection visits).")
        log("  Falling back to NEAREST-DATE matching within a 90-day window.")
        bt_all = clp._load_parquet_flat(clp.DATA_ROOT / "blood_tests" / "blood_tests.parquet")
        bt_all = bt_all.rename(columns={b_pid: "study_id"})
        bt_all[b_date] = pd.to_datetime(bt_all[b_date], errors="coerce")
        bt_all = bt_all.dropna(subset=[b_date])
        gut_dates = gut.copy()
        gdate_col = next((c for c in gut.columns
                          if c in ("collection_date", "collection_timestamp")), None)
        if gdate_col:
            gut_dates[gdate_col] = pd.to_datetime(gut_dates[gdate_col], errors="coerce")
            merged_rows = []
            for sid, grp in bt_all.groupby("study_id"):
                gsub = gut_dates[gut_dates["study_id"] == sid]
                for _, brow in grp.iterrows():
                    if gsub.empty:
                        continue
                    diffs = (gsub[gdate_col] - brow[b_date]).abs()
                    if diffs.min() <= pd.Timedelta(days=90):
                        best = gsub.loc[diffs.idxmin()]
                        rec = brow.to_dict()
                        rec["visit_n"] = best["visit_n"]
                        merged_rows.append(rec)
            if merged_rows:
                bt = pd.DataFrame(merged_rows)
                log(f"  nearest-date fallback matched: {len(bt):,} rows")

    component_report = []

    for dom, comps in DOMAINS.items():
        present = [(c, s) for c, s in comps if c in bt.columns]
        missing = [c for c, _ in comps if c not in bt.columns]
        log(f"\n  domain '{dom}': {len(present)}/{len(comps)} components present")
        for c in missing:
            log(f"    MISSING: {c}")
        for c, s in present:
            n_ok = pd.to_numeric(bt[c], errors="coerce").notna().sum()
            log(f"    {c:<44} sign={s:+.0f}  n={n_ok:,}")
            component_report.append({"domain": dom, "component": c, "sign": s,
                                     "n_nonnull": int(n_ok), "source": "blood_tests"})

    # ------------------------------------------------- functional domains
    banner("2. Loading functional-outcome domains")
    outcomes = clp.load_functional_outcomes()
    func_long = {}
    for dom, (ds_name, sign) in FUNCTIONAL_DOMAINS.items():
        df = outcomes.get(ds_name)
        if df is None or df.empty:
            log(f"  domain '{dom}': source table '{ds_name}' unavailable, skipped")
            continue
        meas = next((c for c in df.columns
                    if c not in ("study_id", "visit", "visit_n")), None)
        if meas is None:
            continue
        sub = df[["study_id", "visit_n", meas]].rename(columns={meas: dom})
        func_long[dom] = (sub, sign, meas)
        log(f"  domain '{dom}': from {ds_name}.{meas}, sign={sign:+.0f}, "
            f"n={sub[dom].notna().sum():,}")
        component_report.append({"domain": dom, "component": meas, "sign": sign,
                                 "n_nonnull": int(sub[dom].notna().sum()),
                                 "source": ds_name})

    pd.DataFrame(component_report).to_csv(
        OUT / "multidomain_score_components.csv", index=False)
    log(f"\n  wrote {OUT/'multidomain_score_components.csv'}")

    # ------------------------------------------------- build domain scores
    banner("3. Building domain z-scores (visit-1 parameters applied to BOTH visits)")
    log("  Standardising on visit-1 mean/SD and applying those same parameters to")
    log("  visit 2. Z-scoring each visit separately would force mean change to")
    log("  zero and erase the very signal being measured.\n")

    domain_scores = []   # list of (study_id, visit_n, domain, score)

    for dom, comps in DOMAINS.items():
        present = [(c, s) for c, s in comps if c in bt.columns]
        if not present:
            continue
        parts = []
        for c, sign in present:
            vals = pd.to_numeric(bt[c], errors="coerce")
            v1_mask = bt["visit_n"] == 1
            v1 = winsorise(vals[v1_mask].dropna())
            mu, sd = v1.mean(), v1.std()
            if not sd or sd == 0:
                continue
            z = sign * (winsorise(vals) - mu) / sd
            parts.append(z.rename(c))
        if not parts:
            continue
        mat = pd.concat(parts, axis=1)
        score = mat.mean(axis=1, skipna=True)
        n_comp = mat.notna().sum(axis=1)
        d = pd.DataFrame({"study_id": bt["study_id"], "visit_n": bt["visit_n"],
                          "domain": dom, "score": score, "n_components": n_comp})
        d = d[d["n_components"] >= 1]
        domain_scores.append(d)
        log(f"  {dom:<16} rows={len(d):,}  mean={d['score'].mean():+.3f}  "
            f"sd={d['score'].std():.3f}")

    for dom, (sub, sign, meas) in func_long.items():
        vals = pd.to_numeric(sub[dom], errors="coerce")
        v1 = winsorise(vals[sub["visit_n"] == 1].dropna())
        mu, sd = v1.mean(), v1.std()
        if not sd or sd == 0:
            continue
        z = sign * (winsorise(vals) - mu) / sd
        d = pd.DataFrame({"study_id": sub["study_id"], "visit_n": sub["visit_n"],
                          "domain": dom, "score": z, "n_components": 1})
        d = d.dropna(subset=["score"])
        domain_scores.append(d)
        log(f"  {dom:<16} rows={len(d):,}  mean={d['score'].mean():+.3f}  "
            f"sd={d['score'].std():.3f}")

    if not domain_scores:
        raise SystemExit("FATAL: no domain scores could be built.")

    ds = pd.concat(domain_scores, ignore_index=True)
    ds = ds.groupby(["study_id", "visit_n", "domain"], as_index=False)["score"].mean()

    # ------------------------------------------------- composite
    banner("4. Composite score")
    comp = ds.groupby(["study_id", "visit_n"], as_index=False).agg(
        composite=("score", "mean"), n_domains=("domain", "nunique"))
    log(f"  person-visits with >=1 domain : {len(comp):,}")
    for k in sorted(comp["n_domains"].unique()):
        log(f"    exactly {k} domain(s): {(comp['n_domains'] == k).sum():,}")

    MIN_DOMAINS = 4
    comp = comp[comp["n_domains"] >= MIN_DOMAINS]
    log(f"\n  requiring >= {MIN_DOMAINS} domains: {len(comp):,} person-visits retained")

    # ------------------------------------------------- VALIDATION
    banner("5. VALIDATION: does the composite actually track chronological age?")
    v1 = comp[comp["visit_n"] == 1].merge(
        pop[["study_id", "age_at_visit"]], on="study_id", how="left").dropna(
        subset=["age_at_visit", "composite"])
    if len(v1) < 100:
        raise SystemExit("FATAL: too few participants to validate the composite.")
    r, p = stats.pearsonr(v1["age_at_visit"], v1["composite"])
    n_val = len(v1)
    log(f"  correlation with chronological age at visit 1: r = {r:.4f} "
        f"(p = {p:.4g}, n = {n_val:,})")
    # M1: bank the validation r to a CSV so it is traceable (V19 changelog M1)
    val_row = pd.DataFrame([{"analysis": "primary_composite_7domain",
                             "r_pearson": round(r, 6), "p": p,
                             "n": n_val, "min_domains": MIN_DOMAINS,
                             "n_domains_total": 7,
                             "threshold_r": 0.15,
                             "threshold_passed": r >= 0.15}])
    val_path = OUT / "multidomain_composite_validation.csv"
    val_row.to_csv(val_path, index=False)
    log(f"  [M1] validation r banked to {val_path}")
    if r < 0.15:
        log("\n  WARNING: the composite barely tracks chronological age. It is not")
        log("  behaving as an aging score, and every result below should be")
        log("  treated as uninterpretable until the component/sign choices are")
        log("  revisited. Reporting the downstream tests anyway, clearly flagged.")
    else:
        log("  -> The composite behaves as an aging measure. Downstream tests are")
        log("     interpretable.")

    # ------------------------------------------------- wide + change
    banner("6. Testing microbiome AgeAccel against change in the composite")
    # THE BUG THIS REPLACES: a generic pivot_table(columns="visit_n") picks up
    # ANY value present, and a small number of participants have a THIRD gut
    # visit -- leaving a stray, unrenamed column literally named 3 that later
    # collided across frames in .join() with "columns overlap". Fixed by
    # using clp.make_wide()'s own pattern: filter to visit_n==1 and
    # visit_n==2 explicitly, exactly like the rest of this project already
    # does, rather than pivoting generically and hoping only {1,2} appear.
    wide = clp.make_wide(comp, "composite")
    wide = wide.rename(columns={"composite_t1": "comp_t1",
                                "composite_t2": "comp_t2"}).dropna(
        subset=["comp_t1", "comp_t2"])
    wide["delta_comp"] = wide["comp_t2"] - wide["comp_t1"]
    log(f"  participants with composite at BOTH visits: {len(wide):,}")
    log(f"  delta composite: mean={wide['delta_comp'].mean():+.4f}  "
        f"sd={wide['delta_comp'].std():.4f}")

    acc = clp.make_wide(ageaccel_df, "gut_ageaccel")
    acc = acc.rename(columns={"gut_ageaccel_t1": "accel_t1",
                              "gut_ageaccel_t2": "accel_t2"})

    div_w = clp.make_wide(gut_div, "shannon").join(
        clp.make_wide(gut_div, "richness"), how="outer")

    # interval
    dcol = next((c for c in gut.columns
                if c in ("collection_date", "collection_timestamp")), None)
    if dcol:
        g = gut.copy()
        g[dcol] = pd.to_datetime(g[dcol], errors="coerce")
        t1 = g[g["visit_n"] == 1].groupby("study_id")[dcol].min()
        t2 = g[g["visit_n"] == 2].groupby("study_id")[dcol].min()
        iv = ((t2 - t1).dt.days / 365.25).rename("interval_years")
    else:
        iv = pd.Series(dtype=float, name="interval_years")

    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].drop_duplicates(
        "study_id").set_index("study_id")
    m = (wide.join(acc, how="left").join(div_w, how="left")
             .join(iv, how="left").join(demo, how="left"))
    m["delta_accel"] = m["accel_t2"] - m["accel_t1"]
    log(f"  analysis frame: {len(m):,}")

    def fit(df, y, x, covars, label):
        cols = [y, x] + covars
        sub = df[[c for c in cols if c in df.columns]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 100:
            log(f"    {label}: n={len(sub)}, skipped")
            return None
        z = sub.copy()
        for c in z.columns:
            if c == "sex_encoded":
                continue
            sd = z[c].std()
            if sd and sd > 0:
                z[c] = (z[c] - z[c].mean()) / sd
        X = sm.add_constant(z[[x] + [c for c in covars if c in z.columns]].astype(float))
        res = sm.OLS(z[y].astype(float), X).fit()
        b, se, pv = float(res.params[x]), float(res.bse[x]), float(res.pvalues[x])
        ci = [float(v) for v in res.conf_int().loc[x].tolist()]
        log(f"    {label:<54} beta={b:+.4f}  SE={se:.4f}  p={pv:.4g}  n={len(sub):,}")
        return {"model": label, "outcome": y, "exposure": x, "n": len(sub),
                "beta_std": round(b, 5), "se": round(se, 5), "p": pv,
                "ci_lo": round(ci[0], 5), "ci_hi": round(ci[1], 5),
                "mde_80pct": round(se * 2.80, 5),
                "delta_equiv_min": round(abs(b) + 1.645 * se, 5)}

    base = [c for c in ("age_at_visit", "sex_encoded", "comp_t1", "interval_years")
            if c in m.columns]
    rows = []
    log("\n  -- PRIMARY: does baseline microbiome age predict later change? --")
    rows.append(fit(m, "delta_comp", "accel_t1", base,
                    "AgeAccel(t1) -> change in composite"))
    rows.append(fit(m, "delta_comp", "shannon_t1", base,
                    "Shannon(t1) -> change in composite"))
    rows.append(fit(m, "delta_comp", "richness_t1", base,
                    "Richness(t1) -> change in composite"))
    log("\n  -- change-to-change --")
    rows.append(fit(m, "delta_comp", "delta_accel", base,
                    "change in AgeAccel <-> change in composite"))
    log("\n  -- REVERSE: does baseline aging predict later microbiome age? --")
    rows.append(fit(m, "accel_t2", "comp_t1",
                    [c for c in base if c != "comp_t1"] + ["accel_t1"],
                    "composite(t1) -> AgeAccel(t2), adj. AgeAccel(t1)"))
    log("\n  -- cross-sectional anchor --")
    rows.append(fit(m, "comp_t1", "accel_t1",
                    [c for c in base if c != "comp_t1"],
                    "AgeAccel(t1) <-> composite(t1)"))

    rows = [r for r in rows if r]
    if rows:
        res = pd.DataFrame(rows)
        res["q_fdr"] = clp.bh_fdr(res["p"].values)
        res["significant"] = res["q_fdr"] < 0.05
        res.to_csv(OUT / "multidomain_aging_results.csv", index=False)
        banner("RESULTS")
        log(res[["model", "n", "beta_std", "se", "p", "q_fdr",
                 "delta_equiv_min", "significant"]].to_string(index=False))
        log(f"\n  composite-vs-age validation r = {r:.4f}")
        log("\n  delta_equiv_min is the tightest bound at which equivalence holds")
        log("  (95% confidence). Read a null row as 'effects larger than this are")
        log("  excluded', not merely 'nothing found'.")
        log(f"\n  wrote {OUT/'multidomain_aging_results.csv'}")

    # ------------------------------------------------------------------ M2
    # Domain-exclusion sensitivity: exclude body_comp and physical; keep
    # metabolic, renal, hepatic, inflammatory, cardiovascular (5 domains).
    # Reviewer concern: Shannon/richness -> delta_composite signal may be
    # driven by adiposity-linked domains (body_comp, physical). Lower
    # MIN_DOMAINS threshold proportionally: 4/7 -> 3/5.
    banner("M2. SENSITIVITY: 5-domain composite (excl. body_comp, physical)")
    EXCL_DOMAINS = {"body_comp", "physical"}
    MIN_DOMAINS_REDUCED = 3
    ds_red = ds[~ds["domain"].isin(EXCL_DOMAINS)]
    comp_red = ds_red.groupby(["study_id", "visit_n"], as_index=False).agg(
        composite=("score", "mean"), n_domains=("domain", "nunique"))
    log(f"  person-visits with >=1 of 5 remaining domains: {len(comp_red):,}")
    for k in sorted(comp_red["n_domains"].unique()):
        log(f"    exactly {k} domain(s): {(comp_red['n_domains'] == k).sum():,}")
    comp_red = comp_red[comp_red["n_domains"] >= MIN_DOMAINS_REDUCED]
    log(f"  requiring >= {MIN_DOMAINS_REDUCED} of 5 domains: {len(comp_red):,} person-visits")

    # Validate reduced composite against chronological age
    v1_red = comp_red[comp_red["visit_n"] == 1].merge(
        pop[["study_id", "age_at_visit"]], on="study_id", how="left").dropna(
        subset=["age_at_visit", "composite"])
    if len(v1_red) >= 50:
        r_red, p_red = stats.pearsonr(v1_red["age_at_visit"], v1_red["composite"])
        log(f"  reduced composite vs age: r = {r_red:.4f} (p = {p_red:.4g}, n = {len(v1_red):,})")
        val_row_red = pd.DataFrame([{"analysis": "reduced_composite_5domain",
                                     "r_pearson": round(r_red, 6), "p": p_red,
                                     "n": len(v1_red), "min_domains": MIN_DOMAINS_REDUCED,
                                     "n_domains_total": 5,
                                     "threshold_r": 0.15,
                                     "threshold_passed": r_red >= 0.15}])
        val_all = pd.read_csv(val_path)
        val_all = pd.concat([val_all, val_row_red], ignore_index=True)
        val_all.to_csv(val_path, index=False)
        log(f"  reduced validation r appended to {val_path}")
    else:
        log("  too few participants to validate reduced composite, skipping")

    wide_red = clp.make_wide(comp_red, "composite")
    wide_red = wide_red.rename(columns={"composite_t1": "comp_t1",
                                        "composite_t2": "comp_t2"}).dropna(
        subset=["comp_t1", "comp_t2"])
    wide_red["delta_comp"] = wide_red["comp_t2"] - wide_red["comp_t1"]
    log(f"  participants with reduced composite at BOTH visits: {len(wide_red):,}")

    base_red = [c for c in ("age_at_visit", "sex_encoded", "comp_t1", "interval_years")
                if c in m.columns]
    m_red = (wide_red.join(acc, how="left").join(div_w, how="left")
                     .join(iv, how="left").join(demo, how="left"))
    m_red["delta_comp"] = m_red["comp_t2"] - m_red["comp_t1"]

    log("\n  -- SENSITIVITY models (5-domain composite, excl. body_comp + physical) --")
    rows_red = []
    rows_red.append(fit(m_red, "delta_comp", "shannon_t1", base_red,
                        "M2 Shannon(t1) -> delta_comp_5domain"))
    rows_red.append(fit(m_red, "delta_comp", "richness_t1", base_red,
                        "M2 Richness(t1) -> delta_comp_5domain"))
    rows_red.append(fit(m_red, "delta_comp", "accel_t1", base_red,
                        "M2 AgeAccel(t1) -> delta_comp_5domain"))
    rows_red = [r for r in rows_red if r]
    if rows_red:
        res_red = pd.DataFrame(rows_red)
        res_red["q_fdr"] = clp.bh_fdr(res_red["p"].values)
        res_red["significant"] = res_red["q_fdr"] < 0.05
        red_path = OUT / "multidomain_aging_sensitivity_reduced.csv"
        res_red.to_csv(red_path, index=False)
        banner("M2 SENSITIVITY RESULTS")
        log(res_red[["model", "n", "beta_std", "se", "ci_lo", "ci_hi",
                     "p", "q_fdr", "significant"]].to_string(index=False))
        log(f"\n  [M2] wrote {red_path}")
        n_sig = res_red["significant"].sum()
        log(f"  [M2] {n_sig}/3 models significant (q<0.05) in 5-domain composite. "
            f"{'Diversity signal SURVIVES exclusion of body_comp+physical.' if n_sig > 0 else 'Diversity signal COLLAPSES without body_comp+physical.'}")
    else:
        log("  M2: no rows fitted, check sample sizes.")


if __name__ == "__main__":
    main()
