#!/usr/bin/env python3
"""
nonlinear_interaction_effects.py
===============================================================================
point 7: go beyond the linear main-effect null.

Four sub-analyses, all on the 9 Path A tests (microbiome(t1) -> aging(t2)):

  1. Within-person CHANGE-to-CHANGE: does Delta(microbiome) correlate with
     Delta(outcome), independent of levels? A pure cross-lagged null can
     still coexist with a real within-person coupling.
  2. NONLINEARITY: add a squared term for microbiome(t1) to the standard
     model. Tests whether extreme values (not just the linear trend) predict
     change in aging.
  3. EXTREME AgeAccel: dummy-code the top and bottom decile of gut_ageaccel(t1)
     against the middle 80%, re-run the model. A linear null can hide an
     effect concentrated only in the tails.
  4. EFFECT MODIFICATION (interactions): microbiome(t1) x {age, sex,
     metabolic_health, on_medication, disease_burden}, testing whether any of
     these subgroups shows an effect the pooled sample does not.

Reuses cross_lagged_panel.py's loaders throughout -- no phenotype logic is
reimplemented. Frailty is NOT available in this dataset (no validated frailty
index table found in earlier data-inventory passes) and is explicitly skipped
below rather than approximated.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python nonlinear_interaction_effects.py 2>&1 | tee nonlinear_interaction_effects.log

Writes: results_cross_lagged/nonlinear_change.csv
        results_cross_lagged/nonlinear_quadratic.csv
        results_cross_lagged/nonlinear_extreme.csv
        results_cross_lagged/nonlinear_interactions.csv
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

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


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else s * 0.0


def build_wide(long_x: pd.DataFrame, value_col: str, out_df: pd.DataFrame,
              microbiome_var: str) -> pd.DataFrame | None:
    meas = next((c for c in out_df.columns if c not in ("study_id", "visit", "visit_n")), None)
    if meas is None:
        return None
    wide_x = clp.make_wide(long_x, value_col)
    wide_x = wide_x.rename(columns={f"{value_col}_t1": f"{microbiome_var}_t1",
                                    f"{value_col}_t2": f"{microbiome_var}_t2"})
    wide_y = clp.make_wide(out_df, meas)
    wide_y = wide_y.rename(columns={f"{meas}_t1": "outcome_t1", f"{meas}_t2": "outcome_t2"})
    return wide_x.join(wide_y, how="inner")


def main() -> None:
    banner("NONLINEAR / INTERACTION / CHANGE-TO-CHANGE ANALYSES (Adinath point 7)")

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
    outcomes = clp.load_functional_outcomes()
    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].drop_duplicates("study_id")

    # ---- effect-modifier covariates ----
    banner("Loading effect-modifier covariates")

    # Metabolic health: z-score of glucose + hba1c (higher = worse metabolic health)
    try:
        bt = pd.read_parquet(clp.DATA_ROOT / "blood_tests" / "blood_tests.parquet")
        if bt.index.names != [None]:
            bt = bt.reset_index()
        bt_pid = next(c for c in bt.columns if c in ("participant_id", "study_id"))
        bt_stg = next(c for c in bt.columns if "research_stage" in c.lower())
        bt = bt.rename(columns={bt_pid: "study_id", bt_stg: "visit"})
        bt = bt.sort_values(["study_id", "visit"])
        bt["visit_n"] = bt.groupby("study_id").cumcount() + 1
        bt1 = bt[bt["visit_n"] == 1].copy()
        gluc_col = next((c for c in bt1.columns if "glucose" in c.lower()), None)
        hba1c_col = next((c for c in bt1.columns if "hba1c" in c.lower() or "a1c" in c.lower()), None)
        parts = []
        for c in (gluc_col, hba1c_col):
            if c:
                parts.append(zscore(pd.to_numeric(bt1[c], errors="coerce")))
        if parts:
            bt1 = bt1[["study_id"]].copy()
            bt1["metabolic_health"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
            bt1 = bt1.drop_duplicates("study_id")
            log(f"  metabolic_health (glucose={gluc_col}, hba1c={hba1c_col}): "
                f"n={bt1['metabolic_health'].notna().sum():,}")
        else:
            bt1 = pd.DataFrame(columns=["study_id", "metabolic_health"])
            log("  no glucose/hba1c column found; metabolic_health unavailable")
    except Exception as exc:
        bt1 = pd.DataFrame(columns=["study_id", "metabolic_health"])
        log(f"  blood_tests load failed ({exc}); metabolic_health unavailable")

    # Medication use, visit 1
    try:
        meds = clp._load_parquet_flat(clp.DATA_ROOT / "medications" / "medications.parquet")
        m_pid = next((c for c in meds.columns if c in ("participant_id", "study_id")), None)
        if m_pid and m_pid != "study_id":
            meds = meds.rename(columns={m_pid: "study_id"})
        meds["on_medication"] = (~meds["medication"].astype(str).str.lower()
                                 .str.contains("no medications taken", na=False)).astype(float)
        med_flag = meds.groupby("study_id", as_index=False)["on_medication"].max()
        log(f"  on_medication: n={len(med_flag):,}")
    except Exception as exc:
        med_flag = pd.DataFrame(columns=["study_id", "on_medication"])
        log(f"  medication load failed ({exc}); on_medication unavailable")

    # Disease burden: distinct diagnosis count from medical_conditions
    try:
        mc = pd.read_parquet(clp.DATA_ROOT / "medical_conditions" / "medical_conditions.parquet")
        if mc.index.names != [None]:
            mc = mc.reset_index()
        mc_pid = next(c for c in mc.columns if c in ("participant_id", "study_id"))
        mc = mc.rename(columns={mc_pid: "study_id"})
        n_cond = mc.groupby("study_id").size().rename("disease_burden").reset_index()
        log(f"  disease_burden: n={len(n_cond):,}, mean={n_cond['disease_burden'].mean():.2f}")
    except Exception as exc:
        n_cond = pd.DataFrame(columns=["study_id", "disease_burden"])
        log(f"  medical_conditions load failed ({exc}); disease_burden unavailable")

    log("  NOTE: frailty index is not available in this dataset (no validated "
        "frailty table found in prior data-inventory passes) -- not tested here.")

    modifiers = (demo.merge(bt1, on="study_id", how="left")
                     .merge(med_flag, on="study_id", how="left")
                     .merge(n_cond, on="study_id", how="left")
                     .set_index("study_id"))
    modifiers["disease_burden"] = modifiers["disease_burden"].fillna(0)

    change_rows, quad_rows, extreme_rows, inter_rows = [], [], [], []

    for microbiome_var, dataset in PATH_A_TESTS:
        long_x = ageaccel_df if microbiome_var == "gut_ageaccel" else gut_div
        value_col = "gut_ageaccel" if microbiome_var == "gut_ageaccel" else microbiome_var
        out_df = outcomes.get(dataset)
        if out_df is None or out_df.empty:
            log(f"  {microbiome_var} x {dataset}: outcome missing, skipped")
            continue
        wide = build_wide(long_x, value_col, out_df, microbiome_var)
        if wide is None:
            continue
        wide = wide.join(modifiers, how="left")
        label = f"{microbiome_var} -> {dataset}"

        # ---------------------------------------------------------- 1. change-to-change
        needed = [f"{microbiome_var}_t1", f"{microbiome_var}_t2", "outcome_t1", "outcome_t2"]
        sub = wide[needed].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 100:
            d_x = sub[f"{microbiome_var}_t2"] - sub[f"{microbiome_var}_t1"]
            d_y = sub["outcome_t2"] - sub["outcome_t1"]
            d_x_z, d_y_z = zscore(d_x), zscore(d_y)
            X = sm.add_constant(d_x_z)
            res = sm.OLS(d_y_z, X).fit()
            b, se, p = float(res.params.iloc[1]), float(res.bse.iloc[1]), float(res.pvalues.iloc[1])
            log(f"  [change-to-change] {label:<32} n={len(sub):<5} beta={b:+.4f}  p={p:.4f}")
            change_rows.append({"test": label, "microbiome_var": microbiome_var,
                                "outcome": dataset, "n": len(sub), "beta_std": round(b, 5),
                                "se": round(se, 5), "p": round(p, 6)})

        # ---------------------------------------------------------- 2. quadratic
        need2 = [f"{microbiome_var}_t1", "outcome_t1", "outcome_t2", "age_at_visit", "sex_encoded"]
        sub2 = wide[need2].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub2) >= 200:
            z = sub2.copy()
            for c in [f"{microbiome_var}_t1", "outcome_t1", "outcome_t2", "age_at_visit"]:
                z[c] = zscore(z[c])
            z["x_sq"] = z[f"{microbiome_var}_t1"] ** 2
            X = sm.add_constant(z[["outcome_t1", f"{microbiome_var}_t1", "x_sq",
                                   "age_at_visit", "sex_encoded"]].astype(float))
            y = z["outcome_t2"].astype(float)
            res = sm.OLS(y, X).fit()
            b_sq, se_sq, p_sq = (float(res.params["x_sq"]), float(res.bse["x_sq"]),
                                float(res.pvalues["x_sq"]))
            log(f"  [quadratic]        {label:<32} n={len(sub2):<5} "
                f"beta_sq={b_sq:+.4f}  p={p_sq:.4f}")
            quad_rows.append({"test": label, "microbiome_var": microbiome_var,
                              "outcome": dataset, "n": len(sub2), "beta_sq": round(b_sq, 5),
                              "se_sq": round(se_sq, 5), "p_sq": round(p_sq, 6)})

        # ---------------------------------------------------------- 3. extreme deciles
        if len(sub2) >= 200:
            z2 = sub2.copy()
            lo, hi = z2[f"{microbiome_var}_t1"].quantile([0.10, 0.90])
            z2["extreme"] = ((z2[f"{microbiome_var}_t1"] <= lo) |
                             (z2[f"{microbiome_var}_t1"] >= hi)).astype(float)
            for c in ["outcome_t1", "outcome_t2", "age_at_visit"]:
                z2[c] = zscore(z2[c])
            X = sm.add_constant(z2[["outcome_t1", "extreme", "age_at_visit",
                                    "sex_encoded"]].astype(float))
            y = z2["outcome_t2"].astype(float)
            res = sm.OLS(y, X).fit()
            b_e, se_e, p_e = (float(res.params["extreme"]), float(res.bse["extreme"]),
                              float(res.pvalues["extreme"]))
            n_extreme = int(z2["extreme"].sum())
            log(f"  [extreme decile]   {label:<32} n={len(z2):<5} n_extreme={n_extreme:<5} "
                f"beta={b_e:+.4f}  p={p_e:.4f}")
            extreme_rows.append({"test": label, "microbiome_var": microbiome_var,
                                 "outcome": dataset, "n": len(z2), "n_extreme": n_extreme,
                                 "beta_extreme": round(b_e, 5), "se_extreme": round(se_e, 5),
                                 "p_extreme": round(p_e, 6)})

        # ---------------------------------------------------------- 4. interactions
        for mod in ("age_at_visit", "sex_encoded", "metabolic_health",
                   "on_medication", "disease_burden"):
            need4 = [f"{microbiome_var}_t1", "outcome_t1", "outcome_t2",
                    "age_at_visit", "sex_encoded", mod]
            sub4 = wide[list(dict.fromkeys(need4))].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub4) < 150:
                continue
            z4 = sub4.copy()
            for c in [f"{microbiome_var}_t1", "outcome_t1", "outcome_t2", "age_at_visit", mod]:
                if c != "sex_encoded":
                    z4[c] = zscore(z4[c])
            z4 = z4.rename(columns={f"{microbiome_var}_t1": "X", "outcome_t1": "Y1",
                                    "outcome_t2": "Y2", "age_at_visit": "AGE",
                                    "sex_encoded": "SEX", mod: "M"})
            other = [c for c in ("AGE", "SEX") if c != ("AGE" if mod == "age_at_visit" else
                                                          "SEX" if mod == "sex_encoded" else "__none__")]
            formula = "Y2 ~ Y1 + X * M" + "".join(f" + {c}" for c in other)
            try:
                res = smf.ols(formula, data=z4.astype(float)).fit()
                if "X:M" not in res.params.index:
                    continue
                b_i, p_i = float(res.params["X:M"]), float(res.pvalues["X:M"])
                inter_rows.append({"test": label, "microbiome_var": microbiome_var,
                                   "outcome": dataset, "modifier": mod, "n": len(z4),
                                   "interaction_beta": round(b_i, 5), "p": round(p_i, 6)})
            except Exception as exc:
                log(f"    [{label} x {mod}] interaction model failed: {exc}")

    banner("1. WITHIN-PERSON CHANGE-TO-CHANGE")
    if change_rows:
        chg = pd.DataFrame(change_rows)
        _, chg["q_fdr"], _, _ = multipletests(chg["p"], alpha=0.05, method="fdr_bh")
        chg.to_csv(OUT / "nonlinear_change.csv", index=False)
        log(chg.to_string(index=False))
        log(f"  FDR-significant: {int((chg['q_fdr'] < 0.05).sum())}/{len(chg)}")

    banner("2. QUADRATIC (NONLINEAR) TERM")
    if quad_rows:
        quad = pd.DataFrame(quad_rows)
        _, quad["q_fdr"], _, _ = multipletests(quad["p_sq"], alpha=0.05, method="fdr_bh")
        quad.to_csv(OUT / "nonlinear_quadratic.csv", index=False)
        log(quad.to_string(index=False))
        log(f"  FDR-significant: {int((quad['q_fdr'] < 0.05).sum())}/{len(quad)}")

    banner("3. EXTREME DECILE (TOP+BOTTOM 10% vs MIDDLE 80%)")
    if extreme_rows:
        ext = pd.DataFrame(extreme_rows)
        _, ext["q_fdr"], _, _ = multipletests(ext["p_extreme"], alpha=0.05, method="fdr_bh")
        ext.to_csv(OUT / "nonlinear_extreme.csv", index=False)
        log(ext.to_string(index=False))
        log(f"  FDR-significant: {int((ext['q_fdr'] < 0.05).sum())}/{len(ext)}")

    banner("4. EFFECT MODIFICATION (INTERACTIONS)")
    if inter_rows:
        inter = pd.DataFrame(inter_rows)
        _, inter["q_fdr"], _, _ = multipletests(inter["p"], alpha=0.05, method="fdr_bh")
        inter.to_csv(OUT / "nonlinear_interactions.csv", index=False)
        log(inter.sort_values("p").to_string(index=False))
        log(f"  FDR-significant: {int((inter['q_fdr'] < 0.05).sum())}/{len(inter)}")
        if (inter["q_fdr"] < 0.05).any():
            log("\n  FDR-significant interactions found -- these subgroups show an "
                "effect the pooled Path A null may be masking.")
        else:
            log("\n  No modifier significantly changes the microbiome-aging relationship "
                "after FDR correction -- the null generalises across age, sex, metabolic "
                "health, medication use, and disease burden.")


if __name__ == "__main__":
    main()
