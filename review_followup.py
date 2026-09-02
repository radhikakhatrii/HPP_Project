#!/usr/bin/env python3
"""
review_followup.py
===============================================================================
Responses to the three technical points raised in the HPP review call.

  A. LEAKAGE CHECK. The question asked was whether the age predictor was
     trained on the baseline visit of the same people it later predicts. It is
     not: training uses only participants who have a SINGLE visit, one row each,
     so repeat-visit participants are held out entirely. Because there is one
     row per training participant, ordinary k-fold is already person-disjoint.
     This section proves that rather than asserting it, by re-running the clock
     under GroupKFold(groups=study_id) and confirming the estimate is unchanged.

  B. THE SUGGESTED ALTERNATIVE. Train across ALL participants with person-
     grouped cross-validation, taking out-of-fold predictions for every sample.
     This is leak-free by construction, uses a larger training set, and -- the
     reason it matters here -- yields AgeAccel for every repeat participant at
     both visits instead of the 523 the current disjoint design produces.

  C. DELTA-BMI. Self-reported disease is a weak endpoint. BMI is objectively
     measured at both visits, continuous, and far better powered. This section
     tests microbiome age acceleration against change in BMI, in both temporal
     directions.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python review_followup.py 2>&1 | tee review_followup.log

Writes to results_cross_lagged/:
    followup_leakage_check.csv
    followup_grouped_clock.csv
    followup_delta_bmi.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log
SEED = getattr(clp, "RANDOM_SEED", 42)


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def make_clock():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05,
                         num_leaves=31, random_state=SEED, verbose=-1)


def icc31(a, b) -> float:
    x = np.column_stack([a, b]).astype(float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ss_t = ((x - gm) ** 2).sum()
    ms_e = (ss_t - (n - 1) * ms_r - (k - 1) * ms_c) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


# --------------------------------------------------------------------------- #
# Shared: build the species feature matrix keyed by (study_id, visit_n)
# --------------------------------------------------------------------------- #
def build_feature_matrix(gut: pd.DataFrame):
    """Returns (feat, species_cols) with one row per (study_id, visit_n)."""
    sp = clp._load_linked_species(gut, clp.DATA_ROOT)
    if sp is None:
        raise SystemExit("FATAL: species parquet not found.")

    meta = {"participant_id", "study_id", "cohort", "research_stage",
            "array_index", "sample_name", "collection_timestamp",
            "collection_date", "timezone", "visit_n"}
    abund = [c for c in sp.columns
             if c not in meta and str(sp[c].dtype).startswith("float")]
    A = sp[abund].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    prev = (A > 0).mean(axis=0)
    keep = prev[prev > 0.05].index.tolist()
    log(f"  species at >5% prevalence: {len(keep)}")

    pid = next(c for c in ("participant_id", "study_id") if c in sp.columns)
    stg = next(c for c in sp.columns if "research_stage" in c.lower())
    feat = A[keep].copy()
    feat["study_id"] = sp[pid].values
    feat["research_stage"] = sp[stg].values

    # attach visit_n from the gut visit table (date-ordered upstream)
    vis = gut[["study_id", "visit", "visit_n"]].drop_duplicates()
    vis = vis.rename(columns={"visit": "research_stage"})
    feat = feat.merge(vis, on=["study_id", "research_stage"], how="left")
    feat = feat.groupby(["study_id", "visit_n"], as_index=False)[keep].mean()
    log(f"  feature matrix: {feat.shape[0]:,} rows x {len(keep)} species")
    return feat, keep


# =========================================================================== #
def main() -> None:
    banner("SETUP")
    gut = clp.load_longitudinal_gut()          # repeat-visit participants only
    pop = clp.load_population()

    # Full gut table (all participants, not just repeat) -- reuse clp's loader
    # by reading the parquet the same way it does.
    gut_all = clp._load_parquet_flat(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    pid_a = next((c for c in gut_all.columns if c in ("participant_id", "study_id")), None)
    stg_a = next((c for c in gut_all.columns if "research_stage" in c.lower()), None)
    if pid_a and pid_a != "study_id":
        if "study_id" in gut_all.columns:
            gut_all = gut_all.drop(columns=["study_id"])
        gut_all = gut_all.rename(columns={pid_a: "study_id"})
    gut_all = gut_all.rename(columns={stg_a: "visit"})

    repeat_ids = set(gut["study_id"].unique())
    all_ids = set(gut_all["study_id"].unique())
    single_ids = all_ids - repeat_ids
    log(f"  participants total       : {len(all_ids):,}")
    log(f"  repeat-visit participants: {len(repeat_ids):,}")
    log(f"  single-visit participants: {len(single_ids):,}")

    feat, species = build_feature_matrix(gut)

    # ---------------------------------------------------------------------- #
    banner("A.  LEAKAGE CHECK  --  is the training set person-disjoint?")
    # Rebuild the training frame exactly as cross_lagged_panel.py does.
    sp_all = clp._load_linked_species(gut_all, clp.DATA_ROOT)
    meta = {"participant_id", "study_id", "cohort", "research_stage",
            "array_index", "sample_name", "collection_timestamp",
            "collection_date", "timezone", "visit_n"}
    A_all = sp_all[[c for c in sp_all.columns
                    if c not in meta and str(sp_all[c].dtype).startswith("float")]]
    A_all = A_all.apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    keep_all = [c for c in species if c in A_all.columns]
    pid_s = next(c for c in ("participant_id", "study_id") if c in sp_all.columns)
    stg_s = next(c for c in sp_all.columns if "research_stage" in c.lower())

    train_df = A_all[keep_all].copy()
    train_df["study_id"] = sp_all[pid_s].values
    train_df["research_stage"] = sp_all[stg_s].values
    train_df = train_df[train_df["study_id"].isin(single_ids)]
    # one row per participant, lowest research_stage
    train_df = (train_df.sort_values("research_stage")
                        .groupby("study_id", as_index=False).first())
    train_df = train_df.merge(pop[["study_id", "age_at_visit"]], on="study_id", how="left")
    train_df = train_df.dropna(subset=["age_at_visit"])

    n_people = train_df["study_id"].nunique()
    n_rows = len(train_df)
    overlap = len(set(train_df["study_id"]) & repeat_ids)
    log(f"  training rows            : {n_rows:,}")
    log(f"  distinct participants    : {n_people:,}")
    log(f"  rows per participant     : {n_rows / max(n_people,1):.3f}")
    log(f"  overlap with repeat set  : {overlap}")
    if overlap == 0 and n_rows == n_people:
        log("  -> Training set contains ONE row per person and NO repeat-visit")
        log("     participant. Ordinary k-fold is therefore already person-disjoint.")
    else:
        log("  -> WARNING: training set is not one-row-per-person or overlaps the")
        log("     repeat set. The leakage concern would then be live.")

    # Explicit numeric coercion. groupby(...).first() and the pop merge can both
    # leave object dtype behind, and np.corrcoef fails opaquely on object arrays
    # with "'float' object has no attribute 'shape'".
    log(f"  dtype before coercion: features={train_df[keep_all].dtypes.unique()[:3]}, "
        f"age={train_df['age_at_visit'].dtype}")
    X = train_df[keep_all].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y = pd.to_numeric(train_df["age_at_visit"], errors="coerce").to_numpy(dtype=np.float64)
    groups = train_df["study_id"].to_numpy()
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    log(f"  after coercion: X={X.shape} {X.dtype}, y={y.shape} {y.dtype}, "
        f"finite y={ok.sum():,}")

    rows_a = []
    for name, cv in [("KFold (as published)", KFold(5, shuffle=True, random_state=SEED)),
                     ("GroupKFold by participant", GroupKFold(n_splits=5))]:
        kw = {"groups": groups} if "Group" in name else {}
        pred = cross_val_predict(make_clock(), X, y, cv=cv, **kw)
        r = float(np.corrcoef(y, pred)[0, 1])
        rows_a.append({"cv_scheme": name, "n": len(y), "r": round(r, 4),
                       "r2_score": round(float(r2_score(y, pred)), 4),
                       "mae": round(float(np.mean(np.abs(y - pred))), 3)})
        log(f"  {name:<28} r={r:.4f}  R2={r2_score(y,pred):.4f}  "
            f"MAE={np.mean(np.abs(y-pred)):.2f}")
    pd.DataFrame(rows_a).to_csv(OUT / "followup_leakage_check.csv", index=False)
    d = abs(rows_a[0]["r"] - rows_a[1]["r"])
    log(f"\n  difference in r: {d:.4f}")
    log("  A negligible difference confirms the published estimate carries no")
    log("  within-person leakage, because there was never more than one sample")
    log("  per training participant to leak.")

    # ---------------------------------------------------------------------- #
    banner("B.  SUGGESTED DESIGN  --  person-grouped CV across ALL participants")
    log("  Train on everyone, grouping folds by participant so that both visits")
    log("  of a repeat participant always land in the same held-out fold. Every")
    log("  prediction is then out-of-fold and leak-free.")

    all_df = A_all[keep_all].copy()
    all_df["study_id"] = sp_all[pid_s].values
    all_df["research_stage"] = sp_all[stg_s].values
    vis = gut[["study_id", "visit", "visit_n"]].drop_duplicates().rename(
        columns={"visit": "research_stage"})
    all_df = all_df.merge(vis, on=["study_id", "research_stage"], how="left")
    # single-visit participants have no visit_n from the repeat table -> visit 1
    all_df["visit_n"] = all_df["visit_n"].fillna(1).astype(int)
    all_df = all_df.groupby(["study_id", "visit_n"], as_index=False)[keep_all].mean()
    all_df = all_df.merge(pop[["study_id", "age_at_visit"]], on="study_id", how="left")
    all_df = all_df.dropna(subset=["age_at_visit"])
    log(f"  samples in grouped design: {len(all_df):,} "
        f"from {all_df['study_id'].nunique():,} participants")

    Xg = all_df[keep_all].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    yg = pd.to_numeric(all_df["age_at_visit"], errors="coerce").to_numpy(dtype=np.float64)
    gg = all_df["study_id"].to_numpy()
    ok_g = np.isfinite(yg)
    Xg, yg, gg = Xg[ok_g], yg[ok_g], gg[ok_g]
    all_df = all_df[ok_g].reset_index(drop=True)
    log(f"  grouped design after coercion: X={Xg.shape} {Xg.dtype}, "
        f"y finite={ok_g.sum():,}")
    pred_g = cross_val_predict(make_clock(), Xg, yg,
                               cv=GroupKFold(n_splits=5), groups=gg)
    r_g = float(np.corrcoef(yg, pred_g)[0, 1])
    log(f"  clock (grouped, all data): r={r_g:.4f}  "
        f"R2={r2_score(yg,pred_g):.4f}  MAE={np.mean(np.abs(yg-pred_g)):.2f}")

    all_df["pred_age"] = pred_g
    ols = sm.OLS(all_df["pred_age"].astype(float),
                 sm.add_constant(all_df["age_at_visit"].astype(float))).fit()
    all_df["age_accel"] = ols.resid

    wide = all_df.pivot_table(index="study_id", columns="visit_n",
                              values="age_accel", aggfunc="mean")
    wide = wide.rename(columns={1: "accel_v1", 2: "accel_v2"})
    both = wide.dropna(subset=["accel_v1", "accel_v2"]) if "accel_v2" in wide.columns \
        else pd.DataFrame()
    log(f"\n  repeat participants with AgeAccel at BOTH visits: {len(both):,}")
    log(f"  (the current disjoint design yields 523)")
    if len(both) > 50:
        icc_new = icc31(both["accel_v1"].values, both["accel_v2"].values)
        r_new = float(np.corrcoef(both["accel_v1"], both["accel_v2"])[0, 1])
        log(f"  retest r  = {r_new:.4f}")
        log(f"  ICC(3,1)  = {icc_new:.4f}   (published value from the disjoint "
            f"design: 0.483)")
        pd.DataFrame([{
            "design": "person-grouped CV, all participants",
            "n_samples": len(all_df), "n_participants": all_df["study_id"].nunique(),
            "clock_r": round(r_g, 4), "clock_mae": round(float(np.mean(np.abs(yg-pred_g))), 3),
            "n_pairs_both_visits": len(both),
            "retest_r": round(r_new, 4), "icc_3_1": round(icc_new, 4),
        }]).to_csv(OUT / "followup_grouped_clock.csv", index=False)
        both.reset_index().to_csv(OUT / "followup_grouped_ageaccel.csv", index=False)
        log(f"  wrote followup_grouped_clock.csv and followup_grouped_ageaccel.csv")
    else:
        log("  Too few paired participants recovered; inspect visit_n assignment.")
        both = pd.DataFrame()

    # ---------------------------------------------------------------------- #
    banner("C.  DELTA-BMI  --  objectively measured, continuous endpoint")
    anthro = clp._load_parquet_flat(clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet")
    a_pid = next((c for c in anthro.columns if c in ("participant_id", "study_id")), None)
    a_stg = next((c for c in anthro.columns if "research_stage" in c.lower()), None)
    bmi_col = next((c for c in anthro.columns if c.lower() in ("bmi", "body_mass_index")), None)
    log(f"  id={a_pid}  stage={a_stg}  bmi={bmi_col}")
    if not (a_pid and a_stg and bmi_col):
        log("  FATAL: could not resolve BMI columns; section C skipped.")
        return
    if a_pid != "study_id" and "study_id" in anthro.columns:
        anthro = anthro.drop(columns=["study_id"])
    anthro = anthro.rename(columns={a_pid: "study_id", a_stg: "research_stage"})
    anthro["bmi"] = pd.to_numeric(anthro[bmi_col], errors="coerce")
    anthro = anthro[anthro["bmi"].between(12, 70)]
    anthro = anthro.merge(vis, on=["study_id", "research_stage"], how="inner")
    bmi_w = anthro.pivot_table(index="study_id", columns="visit_n",
                               values="bmi", aggfunc="mean")
    bmi_w = bmi_w.rename(columns={1: "bmi_v1", 2: "bmi_v2"}).dropna(
        subset=["bmi_v1", "bmi_v2"])
    bmi_w["delta_bmi"] = bmi_w["bmi_v2"] - bmi_w["bmi_v1"]
    log(f"  participants with BMI at both visits: {len(bmi_w):,}")
    log(f"  delta BMI: mean={bmi_w['delta_bmi'].mean():+.3f}  "
        f"sd={bmi_w['delta_bmi'].std():.3f}  "
        f"range=[{bmi_w['delta_bmi'].min():+.1f}, {bmi_w['delta_bmi'].max():+.1f}]")

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

    # Prefer the recovered grouped AgeAccel (larger n); fall back if unavailable.
    if len(both) > 50:
        accel = both.reset_index()[["study_id", "accel_v1", "accel_v2"]]
        src = "person-grouped CV (section B)"
    else:
        ga = clp.build_gut_clock_ageaccel(clp.compute_diversity(gut), gut_all, pop)[0]
        w = ga.pivot_table(index="study_id", columns="visit_n",
                           values="gut_ageaccel", aggfunc="mean")
        accel = w.rename(columns={1: "accel_v1", 2: "accel_v2"}).dropna().reset_index()
        src = "published disjoint design"
    log(f"  AgeAccel source: {src}  (n={len(accel):,})")

    m = (bmi_w.reset_index()[["study_id", "bmi_v1", "bmi_v2", "delta_bmi"]]
         .merge(accel, on="study_id", how="inner")
         .merge(iv.reset_index(), on="study_id", how="left")
         .merge(pop[["study_id", "age_at_visit", "sex_encoded"]], on="study_id", how="left"))
    m["delta_accel"] = m["accel_v2"] - m["accel_v1"]
    log(f"  analysis frame: {len(m):,} participants")

    def fit(df, y, x, covars, label):
        cols = [y, x] + covars
        sub = df[[c for c in cols if c in df.columns]].dropna()
        if len(sub) < 100:
            log(f"  {label}: n={len(sub)}, skipped")
            return None
        z = sub.copy()
        for c in z.columns:
            if c == "sex_encoded":
                continue
            sd = z[c].std()
            if sd and sd > 0:
                z[c] = (z[c] - z[c].mean()) / sd
        Xd = sm.add_constant(z[[x] + [c for c in covars if c in z.columns]].astype(float))
        res = sm.OLS(z[y].astype(float), Xd).fit()
        b, se, p = float(res.params[x]), float(res.bse[x]), float(res.pvalues[x])
        ci = [float(v) for v in res.conf_int().loc[x].tolist()]
        log(f"  {label:<52} beta={b:+.4f}  SE={se:.4f}  p={p:.4g}  n={len(sub):,}")
        return {"model": label, "outcome": y, "exposure": x, "n": len(sub),
                "beta_std": round(b, 5), "se": round(se, 5), "p": p,
                "ci_lo": round(ci[0], 5), "ci_hi": round(ci[1], 5),
                "mde_80pct": round(se * 2.80, 5),
                "covariates": "+".join(c for c in covars if c in z.columns)}

    base = [c for c in ("age_at_visit", "sex_encoded", "bmi_v1", "interval_years")
            if c in m.columns]
    out = []
    log("\n  -- does microbiome age at baseline predict later BMI change? --")
    out.append(fit(m, "delta_bmi", "accel_v1", base, "AgeAccel(t1) -> delta BMI"))
    log("\n  -- do the two changes move together? --")
    out.append(fit(m, "delta_bmi", "delta_accel", base, "delta AgeAccel <-> delta BMI"))
    log("\n  -- reverse: does BMI change predict later microbiome age? --")
    out.append(fit(m, "accel_v2", "delta_bmi",
                   [c for c in base if c != "bmi_v1"] + ["accel_v1"],
                   "delta BMI -> AgeAccel(t2), adj. AgeAccel(t1)"))
    log("\n  -- cross-sectional anchor at baseline --")
    out.append(fit(m, "bmi_v1", "accel_v1",
                   [c for c in base if c != "bmi_v1"], "AgeAccel(t1) <-> BMI(t1)"))

    out = [o for o in out if o]
    if out:
        res = pd.DataFrame(out)
        res["q_fdr"] = clp.bh_fdr(res["p"].values)
        res["significant"] = res["q_fdr"] < 0.05
        res.to_csv(OUT / "followup_delta_bmi.csv", index=False)
        banner("SECTION C RESULTS")
        log(res[["model", "n", "beta_std", "se", "p", "q_fdr",
                 "mde_80pct", "significant"]].to_string(index=False))
        log("\n  Compare |beta_std| against mde_80pct: a coefficient below the MDE")
        log("  means the test was underpowered, not that the effect is absent.")
        log(f"\n  wrote {OUT/'followup_delta_bmi.csv'}")


if __name__ == "__main__":
    main()
