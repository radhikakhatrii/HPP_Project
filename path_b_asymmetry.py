#!/usr/bin/env python3
"""
path_b_asymmetry.py
===============================================================================
Does host physiological aging predict subsequent gut microbiome change?

This is the well-powered version of Path B. In cross_lagged_panel.py, Path B
tests were limited to n = 509-522 because each required one specific phenotype,
measured at BOTH visits. The two smallest p-values in that analysis were both
Path B (grip -> AgeAccel, beta = +0.114, p = 0.083; diastolic BP -> richness,
beta = +0.054, p = 0.071), and both sat INSIDE the Path B minimum detectable
effect band of 0.083-0.184. They could not have been detected even if real.

Two changes recover power:

  1. EXPOSURE: a composite functional-aging score at visit 1, averaged over
     whichever components a participant has, rather than a single phenotype
     requiring complete data. Tolerating partial missingness is the point.

  2. OUTCOME: total compositional shift between visits, measured as Aitchison
     distance on the CLR-transformed species matrix. This is a direct
     measurement rather than a model residual, so it avoids the ICC = 0.483
     noise floor that AgeAccel carries.

Runs, in order:
  A. Primary   : shift ~ aging_t1 + age + sex + baseline diversity + interval
  B. Mirror    : the Path A equivalent, for the asymmetry contrast
  C. Held-out  : primary model restricted to participants who were NOT in the
                 original 509-522 discovery samples. The hypothesis came from
                 those participants, so this is the closest thing to internal
                 replication available and it must be reported.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python path_b_asymmetry.py

Writes to results_cross_lagged/:
    path_b_asymmetry.csv          all model coefficients
    path_b_composite_coverage.csv how many components each participant had
    path_b_asymmetry.png          figure
===============================================================================
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp   # reuse the exact loaders used by the paper

OUT_DIR = Path(clp.RESULTS_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = clp.log
rows: list[dict] = []


def banner(msg: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {msg}")
    log("=" * 78)


# --------------------------------------------------------------------------- #
# 1. Microbiome compositional shift between visits
# --------------------------------------------------------------------------- #
def compute_shift(gut: pd.DataFrame) -> pd.DataFrame:
    """Per-participant Aitchison and Bray-Curtis distance, visit 1 -> visit 2."""
    banner("1. Microbiome compositional shift (t1 -> t2)")

    sp = clp._load_linked_species(gut, clp.DATA_ROOT)
    if sp is None:
        log("  FATAL: could not load linked species parquet.")
        return pd.DataFrame()

    sp_meta = {"participant_id", "study_id", "cohort", "research_stage",
               "array_index", "sample_name", "collection_timestamp",
               "collection_date", "timezone"}
    abund_cols = [c for c in sp.columns
                  if c not in sp_meta and str(sp[c].dtype).startswith("float")]
    log(f"  species columns: {len(abund_cols)}")

    pid = next(c for c in ("participant_id", "study_id") if c in sp.columns)
    stage = next(c for c in sp.columns if "research_stage" in c.lower())
    sp = sp.rename(columns={pid: "study_id", stage: "visit"})

    # Prevalence filter, matching the clock in cross_lagged_panel.py
    A = sp[abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    prev = (A > 0).mean(axis=0)
    keep = prev[prev > 0.05].index.tolist()
    log(f"  species retained at >5% prevalence: {len(keep)}")
    A = A[keep]

    # Close to proportions, then CLR with a half-minimum pseudocount.
    tot = A.sum(axis=1).replace(0, np.nan)
    P = A.div(tot, axis=0)
    pos = P.values[P.values > 0]
    pseudo = (pos.min() / 2.0) if pos.size else 1e-6
    log(f"  CLR pseudocount (half smallest non-zero proportion): {pseudo:.3e}")
    Pf = P.fillna(0.0).values + pseudo
    L = np.log(Pf)
    CLR = L - L.mean(axis=1, keepdims=True)

    clr_df = pd.DataFrame(CLR, columns=keep)
    clr_df["study_id"] = sp["study_id"].values
    clr_df["visit"] = sp["visit"].values

    prop_df = pd.DataFrame(P.fillna(0.0).values, columns=keep)
    prop_df["study_id"] = sp["study_id"].values
    prop_df["visit"] = sp["visit"].values

    # Attach visit_n from the gut visit table (ordered by collection date there)
    vis = gut[["study_id", "visit", "visit_n"]].drop_duplicates()
    clr_df = clr_df.merge(vis, on=["study_id", "visit"], how="inner")
    prop_df = prop_df.merge(vis, on=["study_id", "visit"], how="inner")

    # Average duplicate samples within a visit
    clr_df = clr_df.groupby(["study_id", "visit_n"], as_index=False)[keep].mean()
    prop_df = prop_df.groupby(["study_id", "visit_n"], as_index=False)[keep].mean()

    out = []
    for frame, label in ((clr_df, "aitchison"), (prop_df, "bray_curtis")):
        t1 = frame[frame["visit_n"] == 1].set_index("study_id")[keep]
        t2 = frame[frame["visit_n"] == 2].set_index("study_id")[keep]
        common = t1.index.intersection(t2.index)
        X1, X2 = t1.loc[common].values, t2.loc[common].values
        if label == "aitchison":
            d = np.sqrt(((X1 - X2) ** 2).sum(axis=1))
        else:
            num = np.abs(X1 - X2).sum(axis=1)
            den = (X1 + X2).sum(axis=1)
            d = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        out.append(pd.DataFrame({"study_id": common, label: d}))
        log(f"  {label:<12}: n={len(common):,}  "
            f"mean={np.nanmean(d):.3f}  sd={np.nanstd(d):.3f}")

    shift = out[0].merge(out[1], on="study_id", how="outer")

    # Baseline diversity, needed as a covariate (a low-diversity gut has more
    # room to move, so shift and baseline diversity are mechanically linked)
    div = clp.compute_diversity(gut)
    base = (div[div["visit_n"] == 1][["study_id", "shannon", "richness"]]
            .rename(columns={"shannon": "shannon_t1", "richness": "richness_t1"}))
    shift = shift.merge(base, on="study_id", how="left")
    return shift


# --------------------------------------------------------------------------- #
# 2. Visit interval
# --------------------------------------------------------------------------- #
def compute_interval(gut: pd.DataFrame) -> pd.DataFrame:
    banner("2. Visit interval")
    date_col = next((c for c in gut.columns
                     if c in ("collection_date", "collection_timestamp")), None)
    if date_col is None:
        log("  WARNING: no collection date column; interval unavailable.")
        return pd.DataFrame(columns=["study_id", "interval_years"])
    g = gut.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    t1 = g[g["visit_n"] == 1][["study_id", date_col]].rename(columns={date_col: "d1"})
    t2 = g[g["visit_n"] == 2][["study_id", date_col]].rename(columns={date_col: "d2"})
    m = t1.drop_duplicates("study_id").merge(t2.drop_duplicates("study_id"), on="study_id")
    m["interval_years"] = (m["d2"] - m["d1"]).dt.days / 365.25
    log(f"  n={len(m):,}  median={m['interval_years'].median():.2f} yr")
    return m[["study_id", "interval_years"]]


# --------------------------------------------------------------------------- #
# 3. Composite functional-aging score at visit 1
# --------------------------------------------------------------------------- #
def build_composite(outcomes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardise each phenotype at visit 1, sign-align so that HIGHER always
    means MORE aged, then average whichever components a participant has.

    Sign convention: grip strength declines with age, so it is negated.
    Blood pressure and waist circumference rise with age and are kept as-is.
    """
    banner("3. Composite functional-aging score at visit 1")
    SIGN = {"hand_grip": -1.0, "blood_pressure": +1.0, "anthropometrics": +1.0}

    parts, coverage = [], []
    for name, df in outcomes.items():
        if df is None or df.empty:
            log(f"  {name}: empty, skipped")
            continue
        meas = next((c for c in df.columns
                     if c not in ("study_id", "visit", "visit_n")), None)
        if meas is None:
            log(f"  {name}: no measurement column, skipped")
            continue
        t1 = df[df["visit_n"] == 1][["study_id", meas]].dropna()
        t1 = t1.groupby("study_id", as_index=False)[meas].mean()
        mu, sd = t1[meas].mean(), t1[meas].std()
        if not sd or sd == 0:
            log(f"  {name}: zero variance, skipped")
            continue
        z = SIGN.get(name, 1.0) * (t1[meas] - mu) / sd
        parts.append(pd.DataFrame({"study_id": t1["study_id"], f"z_{name}": z}))
        coverage.append({"component": name, "column": meas, "n": len(t1),
                         "sign": SIGN.get(name, 1.0)})
        log(f"  {name:<16} col={meas:<34} n={len(t1):,}  sign={SIGN.get(name,1.0):+.0f}")

    if not parts:
        return pd.DataFrame(), pd.DataFrame(coverage)

    comp = parts[0]
    for p in parts[1:]:
        comp = comp.merge(p, on="study_id", how="outer")
    zcols = [c for c in comp.columns if c.startswith("z_")]
    comp["n_components"] = comp[zcols].notna().sum(axis=1)
    comp["aging_t1"] = comp[zcols].mean(axis=1, skipna=True)
    comp = comp[comp["n_components"] >= 1]

    log(f"\n  participants with >=1 component: {len(comp):,}")
    for k in range(1, len(zcols) + 1):
        log(f"    exactly {k} component(s): {(comp['n_components']==k).sum():,}")
    log("\n  This is the number that decides whether the analysis is better")
    log("  powered than the original Path B tests (n = 509-522). If nearly")
    log("  everyone has only one component, the phenotypes are missing in the")
    log("  SAME people and little power is gained.")
    return comp, pd.DataFrame(coverage)


# --------------------------------------------------------------------------- #
# 4. Model fitting
# --------------------------------------------------------------------------- #
def fit(df: pd.DataFrame, y: str, x: str, covars: list[str], label: str,
        note: str = "") -> dict | None:
    cols = [y, x] + covars
    sub = df[[c for c in cols if c in df.columns]].dropna()
    if len(sub) < 100:
        log(f"  {label}: n={len(sub)} after listwise deletion, skipped")
        return None
    z = sub.copy()
    for c in z.columns:
        if c == "sex_numeric":
            continue
        sd = z[c].std()
        if sd and sd > 0:
            z[c] = (z[c] - z[c].mean()) / sd
    X = sm.add_constant(z[[x] + [c for c in covars if c in z.columns]].astype(float))
    res = sm.OLS(z[y].astype(float), X).fit()
    beta = float(res.params[x])
    se = float(res.bse[x])
    p = float(res.pvalues[x])
    ci = [float(v) for v in res.conf_int().loc[x].tolist()]
    log(f"  {label:<46} beta={beta:+.4f}  SE={se:.4f}  "
        f"p={p:.4g}  n={len(sub):,}")
    return {"model": label, "outcome": y, "exposure": x, "n": len(sub),
            "beta_std": round(beta, 5), "se": round(se, 5),
            "t": round(float(res.tvalues[x]), 4), "p": p,
            "ci_lo": round(ci[0], 5), "ci_hi": round(ci[1], 5),
            "r2": round(float(res.rsquared), 4),
            "covariates": "+".join(c for c in covars if c in z.columns),
            "mde_80pct": round(se * 2.80, 5), "note": note}


# --------------------------------------------------------------------------- #
def main() -> None:
    banner("PATH B ASYMMETRY: does host aging precede microbiome change?")

    gut = clp.load_longitudinal_gut()
    pop = clp.load_population()
    outcomes = clp.load_functional_outcomes()

    shift = compute_shift(gut)
    if shift.empty:
        log("\nFATAL: no compositional shift computed. Stopping.")
        sys.exit(1)
    interval = compute_interval(gut)
    comp, coverage = build_composite(outcomes)
    if comp.empty:
        log("\nFATAL: composite aging score is empty. Stopping.")
        sys.exit(1)

    demo_cols = ["study_id"] + [c for c in ("age_at_visit", "age", "sex_numeric", "sex")
                                if c in pop.columns]
    demo = pop[demo_cols].drop_duplicates("study_id").copy()
    if "age_at_visit" in demo.columns and "age" not in demo.columns:
        demo = demo.rename(columns={"age_at_visit": "age"})
    if "sex" in demo.columns and "sex_numeric" not in demo.columns:
        demo["sex_numeric"] = pd.to_numeric(demo["sex"], errors="coerce")

    d = (shift.merge(comp[["study_id", "aging_t1", "n_components"]], on="study_id", how="inner")
              .merge(interval, on="study_id", how="left")
              .merge(demo, on="study_id", how="left"))
    log(f"\n  analysis frame: {len(d):,} participants")

    covars = [c for c in ("age", "sex_numeric", "shannon_t1", "interval_years")
              if c in d.columns]
    log(f"  covariates: {covars}")

    # ---- A. Primary --------------------------------------------------------
    banner("4A. PRIMARY: host aging (t1) -> microbiome shift (t1->t2)")
    for dist in ("aitchison", "bray_curtis"):
        if dist in d.columns:
            r = fit(d, dist, "aging_t1", covars,
                    f"aging_t1 -> {dist} shift", "primary")
            if r:
                rows.append(r)

    # ---- B. Mirror ---------------------------------------------------------
    banner("4B. MIRROR: baseline microbiome -> change in aging score")
    log("  Reported for the asymmetry contrast. Requires the aging score at")
    log("  BOTH visits, so n falls back toward the original Path A samples.")
    parts = []
    SIGN = {"hand_grip": -1.0, "blood_pressure": +1.0, "anthropometrics": +1.0}
    for name, df in outcomes.items():
        if df is None or df.empty:
            continue
        meas = next((c for c in df.columns
                     if c not in ("study_id", "visit", "visit_n")), None)
        if meas is None:
            continue
        t2 = df[df["visit_n"] == 2][["study_id", meas]].dropna()
        t2 = t2.groupby("study_id", as_index=False)[meas].mean()
        mu, sd = t2[meas].mean(), t2[meas].std()
        if not sd or sd == 0:
            continue
        parts.append(pd.DataFrame({"study_id": t2["study_id"],
                                   f"z2_{name}": SIGN.get(name, 1.0) * (t2[meas] - mu) / sd}))
    if parts:
        c2 = parts[0]
        for p_ in parts[1:]:
            c2 = c2.merge(p_, on="study_id", how="outer")
        z2 = [c for c in c2.columns if c.startswith("z2_")]
        c2["aging_t2"] = c2[z2].mean(axis=1, skipna=True)
        d2 = d.merge(c2[["study_id", "aging_t2"]], on="study_id", how="inner")
        for mb in ("shannon_t1", "richness_t1"):
            if mb in d2.columns:
                r = fit(d2, "aging_t2", mb,
                        [c for c in covars if c != mb] + ["aging_t1"],
                        f"{mb} -> aging_t2 (adj. aging_t1)", "mirror")
                if r:
                    rows.append(r)

    # ---- C. Held-out replication ------------------------------------------
    banner("4C. HELD-OUT: participants absent from the discovery samples")
    log("  The hypothesis came from the original Path B tests, which used only")
    log("  participants with a single phenotype at BOTH visits. Re-fitting on")
    log("  those NOT in that set is the closest available internal replication.")
    discovery = set()
    for name, df in outcomes.items():
        if df is None or df.empty:
            continue
        meas = next((c for c in df.columns
                     if c not in ("study_id", "visit", "visit_n")), None)
        if meas is None:
            continue
        w = df.dropna(subset=[meas])
        both = (w.groupby("study_id")["visit_n"].nunique() >= 2)
        discovery |= set(both[both].index)
    log(f"  discovery participants (complete phenotype at both visits): {len(discovery):,}")
    held = d[~d["study_id"].isin(discovery)]
    log(f"  held-out participants: {len(held):,}")
    if len(held) >= 150:
        for dist in ("aitchison", "bray_curtis"):
            if dist in held.columns:
                r = fit(held, dist, "aging_t1", covars,
                        f"HELD-OUT aging_t1 -> {dist}", "held_out")
                if r:
                    rows.append(r)
    else:
        log("  too few held-out participants for a stable estimate; skipped.")

    # ---- Save --------------------------------------------------------------
    if not rows:
        log("\nNo models fitted.")
        return
    res = pd.DataFrame(rows)
    res["q_fdr"] = clp.bh_fdr(res["p"].values)
    res["significant"] = res["q_fdr"] < 0.05
    res.to_csv(OUT_DIR / "path_b_asymmetry.csv", index=False)
    coverage.to_csv(OUT_DIR / "path_b_composite_coverage.csv", index=False)

    banner("RESULTS")
    log(res[["model", "n", "beta_std", "se", "p", "q_fdr",
             "mde_80pct", "significant"]].to_string(index=False))
    prim = res[res["note"] == "primary"]
    if not prim.empty and prim["significant"].any():
        log("\n  PRIMARY MODEL IS SIGNIFICANT AFTER FDR.")
        log("  Check the held-out row before treating this as a finding.")
    elif not prim.empty:
        log("\n  Primary model not significant. Compare |beta| against mde_80pct:")
        log("  if |beta| < MDE the test remains underpowered rather than null.")

    # ---- Figure ------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 0.62 * len(res) + 2.2))
        y = np.arange(len(res))
        colors = ["#C62828" if s else "#546E7A" for s in res["significant"]]
        for i, (_, r) in enumerate(res.iterrows()):
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=colors[i], lw=2)
            ax.scatter([r["beta_std"]], [i], color=colors[i], s=55, zorder=5)
            ax.text(r["ci_hi"] + 0.005, i, f"p={r['p']:.3g}  n={r['n']:,}",
                    va="center", fontsize=8)
        ax.axvline(0, color="black", lw=0.9, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(res["model"], fontsize=8.5)
        ax.set_xlabel("Standardised coefficient (95% CI)")
        ax.set_title("Does host aging precede gut microbiome change?\n"
                     "Path B asymmetry test, HPP repeat-sampled cohort",
                     fontsize=11)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(OUT_DIR / "path_b_asymmetry.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        log(f"  figure failed: {exc}")

    log(f"\nWrote {OUT_DIR/'path_b_asymmetry.csv'}")
    log(f"Wrote {OUT_DIR/'path_b_composite_coverage.csv'}")
    log(f"Wrote {OUT_DIR/'path_b_asymmetry.png'}")


if __name__ == "__main__":
    main()
