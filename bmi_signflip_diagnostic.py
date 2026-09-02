#!/usr/bin/env python3
"""
bmi_signflip_diagnostic.py
===============================================================================
Investigates the same-visit AgeAccel <-> BMI sign flip found in
review_followup.py's Section C: v16 reported beta=-0.138 (P=0.001, n=521,
"consistent with the known inverse relationship between adiposity and
microbial diversity"); the corrected pipeline gives beta=+0.087 (P=0.0002,
n=1,771) -- still significant, now positive.

The historical 523-participant leak-free AgeAccel ID list was never saved
before the B1/BUG4 fixes (build_gut_clock_ageaccel didn't write it out until
after the fix), so the exact old sample can't be reconstructed and re-tested
directly. Two things CAN be tested directly, both cheap and decisive:

  1. STABILITY ACROSS RANDOM SUBSAMPLES of size ~523 from the current,
     corrected 1,771-participant sample. If the sign flips in a non-trivial
     fraction of random 523-sized draws, that is strong evidence the v16
     result was ordinary small-sample noise on a modest true effect, not a
     contradiction requiring further explanation.

  2. DOB-SOURCED vs BACKFILL-SOURCED SUBGROUPS. The pre-fix pipeline could
     only assign an age (and therefore only ever include a participant in
     AgeAccel) using sociodemographics.DOB directly -- population.parquet's
     year_of_birth backfill did not exist yet. So restricting the corrected
     analysis to ONLY participants whose age came from DOB directly is a
     much closer (though not perfect) proxy for "the kind of participant the
     old 523-set was built from" than a uniform random subsample. If the
     DOB-only subgroup shows the OLD sign (negative) while the
     backfill-recovered subgroup shows the NEW sign (positive), that
     localises the flip to WHICH participants were added, not to a
     construction bug.

Both are diagnostic only -- neither changes which number is "correct" (the
full corrected 1,771-participant estimate remains the primary, best-powered
one); they explain WHY it differs from v16, so the direction change can be
written up rather than left as an unexplained anomaly.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 bmi_signflip_diagnostic.py 2>&1 | tee bmi_signflip_diagnostic.log

Writes: results_cross_lagged/bmi_signflip_diagnostic.csv
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
log = clp.log
SEED = clp.RANDOM_SEED
N_BOOTSTRAP = 500
SUBSAMPLE_N = 523


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def fit_same_visit(df: pd.DataFrame) -> dict:
    """OLS: BMI_t1 ~ AgeAccel_t1, both standardised. Returns beta/se/p/n."""
    sub = df[["age_accel_t1", "bmi_t1"]].dropna()
    n = len(sub)
    if n < 30:
        return {"n": n, "beta": np.nan, "se": np.nan, "p": np.nan}
    z = sub.copy()
    for c in z.columns:
        mu, sd = z[c].mean(), z[c].std()
        if sd > 0:
            z[c] = (z[c] - mu) / sd
    X = sm.add_constant(z[["age_accel_t1"]].astype(float))
    y = z["bmi_t1"].astype(float)
    res = sm.OLS(y, X).fit()
    return {"n": n, "beta": float(res.params["age_accel_t1"]),
           "se": float(res.bse["age_accel_t1"]), "p": float(res.pvalues["age_accel_t1"])}


def main() -> None:
    banner("BMI SIGN-FLIP DIAGNOSTIC: AgeAccel <-> BMI, same visit")

    # ---- Rebuild the same-visit AgeAccel/BMI frame the same way review_followup.py does ----
    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut)
    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid: "study_id", stg: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    ga, _ = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
    ga_w = ga.pivot_table(index="study_id", columns="visit_n", values="gut_ageaccel")
    ga_w = ga_w.rename(columns={1: "age_accel_t1", 2: "age_accel_t2"})

    anthro = pd.read_parquet(clp.DATA_ROOT / "anthropometrics" / "anthropometrics.parquet")
    if anthro.index.names != [None]:
        anthro = anthro.reset_index()
    a_pid = next(c for c in anthro.columns if c in ("participant_id", "study_id"))
    a_stg = next(c for c in anthro.columns if "research_stage" in c.lower())
    bmi_col = next((c for c in anthro.columns if c.lower() in ("bmi", "body_mass_index")), None)
    anthro = anthro.rename(columns={a_pid: "study_id", a_stg: "visit"})
    anthro["visit_n"] = anthro.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))
    bmi_w = anthro.pivot_table(index="study_id", columns="visit_n", values=bmi_col, aggfunc="mean")
    bmi_w = bmi_w.rename(columns={1: "bmi_t1", 2: "bmi_t2"})
    ga_w.columns.name = None
    bmi_w.columns.name = None

    frame = ga_w.join(bmi_w, how="inner")
    log(f"  full corrected frame: n={frame['age_accel_t1'].notna().sum():,} with "
        f"AgeAccel_t1, n={frame.dropna(subset=['age_accel_t1','bmi_t1']).shape[0]:,} "
        f"with both AgeAccel_t1 and BMI_t1")

    full_result = fit_same_visit(frame)
    log(f"  FULL corrected sample: beta={full_result['beta']:+.4f} "
        f"se={full_result['se']:.4f} p={full_result['p']:.4g} n={full_result['n']:,}")

    # ---- Diagnostic 1: stability across random subsamples of size ~523 ----
    banner(f"1. Stability across {N_BOOTSTRAP} random subsamples of n={SUBSAMPLE_N}")
    rng = np.random.default_rng(SEED)
    both = frame.dropna(subset=["age_accel_t1", "bmi_t1"])
    betas = []
    for i in range(N_BOOTSTRAP):
        if len(both) < SUBSAMPLE_N:
            log(f"  full sample ({len(both)}) smaller than {SUBSAMPLE_N}, skipping bootstrap")
            break
        sub_ids = rng.choice(both.index.values, size=SUBSAMPLE_N, replace=False)
        r = fit_same_visit(both.loc[sub_ids])
        if np.isfinite(r["beta"]):
            betas.append(r["beta"])
    if betas:
        betas = np.array(betas)
        frac_negative = float((betas < 0).mean())
        log(f"  {len(betas)} subsamples fit. beta range=[{betas.min():+.4f}, {betas.max():+.4f}], "
            f"mean={betas.mean():+.4f}, SD={betas.std():.4f}")
        log(f"  fraction of subsamples with a NEGATIVE beta (v16's sign): {frac_negative:.1%}")
        if frac_negative > 0.05:
            log("  -> a non-trivial fraction of random n=523 draws from the SAME corrected")
            log("     data flip sign. The v16 result is consistent with ordinary small-sample")
            log("     noise on a modest true effect, not a separate contradiction to resolve.")
        else:
            log("  -> random subsamples of this size essentially never flip sign; the v16")
            log("     result is NOT well explained by generic sampling noise alone -- the")
            log("     specific 523-participant composition (not just its size) likely matters.")

    # ---- Diagnostic 2: DOB-sourced vs backfill-sourced subgroup ----
    banner("2. DOB-sourced vs population.parquet-backfill-sourced subgroup")
    log("  Re-deriving which participants have DOB directly in sociodemographics")
    log("  (the only age source the pre-fix pipeline could use) vs only via the")
    log("  population.parquet backfill (recovered participants, did not exist in v16).")
    try:
        socio_dir = clp.DATA_ROOT / "sociodemographics"
        socio_files = list(socio_dir.glob("*.parquet")) if socio_dir.exists() else []
        dob_ids = set()
        for f in socio_files:
            sd = pd.read_parquet(f)
            if sd.index.names != [None]:
                sd = sd.reset_index()
            sd_pid = next((c for c in sd.columns if c in ("participant_id", "study_id")), None)
            if sd_pid and "DOB" in sd.columns:
                has_dob = sd[sd["DOB"].notna()][sd_pid].unique()
                dob_ids.update(has_dob)
        log(f"  participants with DOB directly available: {len(dob_ids):,}")

        both["_source"] = np.where(both.index.isin(dob_ids), "dob", "backfill_only")
        for src_label, sub in both.groupby("_source"):
            r = fit_same_visit(sub)
            log(f"  {src_label:<14} n={r['n']:,}  beta={r['beta']:+.4f}  "
                f"se={r['se']:.4f}  p={r['p']:.4g}")
        log("\n  If the 'dob' subgroup's sign matches v16 (negative) and the")
        log("  'backfill_only' subgroup's sign is positive, the flip is explained by WHICH")
        log("  participants the fix added, not by any change to how AgeAccel itself is built.")
    except Exception as exc:
        log(f"  Diagnostic 2 failed: {exc}")

    out_rows = [{"analysis": "full_corrected", **full_result}]
    pd.DataFrame(out_rows).to_csv(OUT / "bmi_signflip_diagnostic.csv", index=False)
    log(f"\n  wrote {OUT/'bmi_signflip_diagnostic.csv'}")
    banner("DONE")


if __name__ == "__main__":
    main()
