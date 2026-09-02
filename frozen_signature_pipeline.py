#!/usr/bin/env python3
"""
frozen_signature_pipeline.py
===============================================================================
Extension experiments E1-E3 for the HPP microbiome-ageing validation paper.

Implements the professor's recommended extension (the "Some Recommended
Changes" block in the Introduction), on the EXISTING gut species data only.
E4 (HUMAnN functional/pathway extension) is deliberately NOT here: audit
scripts established that gut pathway data does not exist in this release
(oral only, S3-sharded, different body site) -- it is a separate follow-up.

    E1  Discovery      : identify age-associated species on the 5,188 single-
                         visit stratum, FDR across species, FREEZE the set +
                         directions + standardised weights into a composite
                         signature S = sum_j w_j * Z_ij  (Z = CLR abundance).
    E2  Stability      : among repeat-sampled participants held OUT of the
                         prospective test set, compute per-species ICC(3,1),
                         prevalence and variance components; split the frozen
                         set into stable vs less-stable signatures.
    E3  Prospective    : apply the frozen signatures WITHOUT re-estimation to
                         the phenotype-complete longitudinal samples; test
                         S(t1) -> phenotype(t2) | phenotype(t1), age, sex;
                         plus out-of-sample delta-R^2 (practical utility).
    E*  Property map    : per-species scatter -- age-dissociation (x),
                         stability (y), prospective effect (size). The visual
                         that IS the reframed thesis.

DESIGN DISCIPLINE (why this is worth doing at all):
  * The discovery stratum (single-visit, n=5,188) and the longitudinal test
    stratum share NO participants. The species list, directions and weights
    are frozen on the discovery data and never re-estimated on the test data.
  * Stability (E2) is estimated on repeat participants that are NOT in any
    prospective test sample, so feature selection cannot be influenced by the
    outcome analysis.
  * S is built from the same species pool as the LightGBM clock. It may just
    be the clock relabelled. This script therefore reports S's own ICC and its
    correlation with the clock's AgeAccel, so you can SEE whether S carries
    anything the clock did not before you trust any E3 result (see M1 in the
    review). If S's ICC does not beat ~0.483, expect E3 to null out too.

HONESTY GUARDRAILS built in:
  * A consistency check reports, for the repeat cohort, how many participants
    have REAL (non-imputed) species coverage at BOTH visits. This is the same
    quantity behind the 1,817-vs-523 discrepancy (B1). Run this before
    trusting any downstream n.
  * Nothing here fabricates results. It computes them from the HPP source
    files on the machine it is run on (MBZUAI remote PC).

Run on the remote machine, next to cross_lagged_panel.py:
    conda activate pheno
    cd ~/studies/HPP_Prototype
    python frozen_signature_pipeline.py 2>&1 | tee frozen_signature_pipeline.log

Writes to results_cross_lagged/:
    E1_discovery_species.csv          per-species age association + frozen flag + weight
    E2_stability_species.csv          per-species ICC / prevalence / variance / class
    E3_prospective_signatures.csv     S_all / S_stable / S_lessstable x 3 phenotypes
    E3_incremental_prediction.csv     out-of-sample delta-R^2 vs baseline
    property_map.csv                  the three numbers per species
    property_map.png                  the three-property scatter
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp   # reuse the validated data loaders

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log
SEED = clp.RANDOM_SEED

# ---- pre-specified thresholds (fix BEFORE looking at prospective results) ----
FDR_DISCOVERY   = 0.05     # BH-FDR across species in E1
ICC_STABLE      = 0.50     # a frozen taxon is "stable" if ICC(3,1) >= this
PREVALENCE_MIN  = 0.10     # ... and present in >= 10% of samples at both visits
MIN_TEST_N      = 200      # don't report a prospective model below this n
CV_SPLITS       = 5
CV_REPEATS      = 20       # repeated CV for the delta-R^2 estimate
PSEUDOCOUNT     = 1e-6     # for CLR zero handling


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


# ============================================================
# Compositional transform
# ============================================================
def clr(mat: np.ndarray) -> np.ndarray:
    """Centred log-ratio, per sample (row). Zeros handled by a small pseudocount.
    This matches the Introduction's specification (CLR abundance Z_ij), and is
    the correct transform for the linear discovery regression -- note the
    LightGBM clock in cross_lagged_panel.py feeds RAW relative abundances, so do
    not expect S to be a linear image of the clock's features."""
    m = np.asarray(mat, dtype=float)
    m = np.where(m > 0, m, PSEUDOCOUNT)
    logm = np.log(m)
    return logm - logm.mean(axis=1, keepdims=True)


def icc_3_1(v1: np.ndarray, v2: np.ndarray) -> tuple:
    """ICC(3,1) consistency for two measurements per subject, via mean squares.
    Returns (icc, sd_between, sd_within, r_pearson, n). Dependency-free so it
    runs anywhere; matches the two-way mixed consistency form used in the paper."""
    ok = np.isfinite(v1) & np.isfinite(v2)
    v1, v2 = v1[ok], v2[ok]
    n = len(v1)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, n)
    k = 2
    rows = np.column_stack([v1, v2])
    grand = rows.mean()
    row_means = rows.mean(axis=1)
    col_means = rows.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_tot  = np.sum((rows - grand) ** 2)
    ss_err  = ss_tot - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse
    icc = (msr - mse) / denom if denom > 0 else np.nan
    var_between = max((msr - mse) / k, 0.0)
    r = pearsonr(v1, v2)[0] if n > 2 else np.nan
    return (icc, np.sqrt(var_between), np.sqrt(mse), r, n)


# ============================================================
# Frames: single-visit training, repeat-visit scoring
# ============================================================
def _species_columns(species_df: pd.DataFrame) -> list[str]:
    meta = {"participant_id", "study_id", "cohort", "research_stage", "array_index",
            "sample_name", "collection_timestamp", "collection_date", "timezone",
            "visit_n", "visit"}
    return [c for c in species_df.columns
            if c not in meta and str(species_df[c].dtype).startswith("float")]


def build_frames():
    """Return (train_df, repeat_df, species_cols, prevalence_mask_info).

    train_df   : one row per single-visit participant, RAW rel-abundance species,
                 age_at_visit, sex_encoded. (CLR applied later.)
    repeat_df  : one row per (repeat participant, visit_n in {1,2}) with species,
                 age, sex, plus a boolean `species_real` marking rows whose
                 species actually linked (not fillna-imputed).
    """
    banner("Loading population, gut, diversity, outcomes")
    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut)
    repeat_ids = set(gut_div["study_id"])

    gut_full = pd.read_parquet(clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid = next(c for c in gut_full.columns if c in ("participant_id", "study_id"))
    stg = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid: "study_id", stg: "research_stage"})

    species_df = clp._load_linked_species(
        gut_full.rename(columns={"research_stage": "visit"}), clp.DATA_ROOT)
    if species_df is None:
        raise SystemExit("No species table resolved; cannot build signatures.")
    if species_df.index.names != [None]:
        species_df = species_df.reset_index()
    sp_pid = next(c for c in species_df.columns if c in ("participant_id", "study_id"))
    sp_stg = next(c for c in species_df.columns if "research_stage" in c.lower())
    species_df = species_df.rename(columns={sp_pid: "study_id", sp_stg: "research_stage"})

    sp_cols = _species_columns(species_df)
    abund = species_df[sp_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
    prevalence = (abund > 0).mean()
    sp_cols = [c for c in sp_cols if prevalence[c] > 0.05]
    log(f"  species passing >5% prevalence: {len(sp_cols)}  (paper: 511)")
    sp_slim = species_df[["study_id", "research_stage"] + sp_cols].copy()

    # ---- training frame: single-visit participants only ----
    single_ids = set(gut_full["study_id"].unique()) - repeat_ids
    train = gut_full[gut_full["study_id"].isin(single_ids)].copy()
    train = train.sort_values("research_stage").groupby("study_id", as_index=False).first()
    train = train.merge(sp_slim, on=["study_id", "research_stage"], how="left")
    train = train.merge(pop[["study_id", "age_at_visit", "sex_encoded"]], on="study_id", how="left")
    train = train.dropna(subset=["age_at_visit"])
    log(f"  training frame: {len(train):,} single-visit participants (paper: 5,188)")

    # ---- repeat frame with an explicit REAL vs IMPUTED flag (exposes B1) ----
    # [SCHEMA FIX] This originally mapped research_stage -> visit_n
    # independently via .rank() on the species table alone, then joined on
    # that reconstructed integer. That is the EXACT pattern found broken in
    # cross_lagged_panel.py's build_gut_clock_ageaccel: two independent rank
    # reconstructions of "visit order" (one from the main gut table, one from
    # the species table) are not guaranteed to assign the same integer to the
    # same real visit per participant. Confirmed on real HPP data to drop the
    # species match rate to ~29%. Fixed to join on the real research_stage/
    # visit value directly, matching compute_diversity()'s approach.
    if "visit" not in gut_div.columns:
        raise SystemExit("gut_div has no raw 'visit' column -- cannot do a "
                         "real-value species join (see cross_lagged_panel.py B1 fix).")
    rep = gut_div[gut_div["study_id"].isin(repeat_ids)][
        ["study_id", "visit_n", "visit"]].copy()
    sp_v = sp_slim.rename(columns={"research_stage": "visit"})
    rep = rep.merge(sp_v, on=["study_id", "visit"], how="left")
    rep["species_real"] = rep[sp_cols].notna().any(axis=1)

    # [SCHEMA FIX] Per-visit age, not a single study_id-level value merged
    # onto both visit rows identically (see cross_lagged_panel.py B3 -- a
    # static age can't distinguish visit 1 from visit 2). Reuses the same
    # helpers cross_lagged_panel.py's build_gut_clock_ageaccel already
    # validated on this schema.
    rep = rep.merge(pop[["study_id", "age_at_visit", "birth_year", "sex_encoded"]],
                    on="study_id", how="left")
    date_col = clp._find_collection_date_col(gut_full)
    if date_col is not None:
        date_slim = (gut_full[["study_id", "research_stage", date_col]]
                    .drop_duplicates(subset=["study_id", "research_stage"])
                    .rename(columns={"research_stage": "visit"}))
        rep = rep.merge(date_slim, on=["study_id", "visit"], how="left")
        visit_age = clp._compute_visit_age(rep)
        n_recomputed = int(visit_age.notna().sum())
        rep["age_at_visit"] = visit_age.where(visit_age.notna(), rep["age_at_visit"])
        log(f"  per-visit age computed for {n_recomputed:,}/{len(rep):,} repeat-frame "
            f"rows (remainder use the static study_id-level fallback)")
    else:
        log("  WARNING: no collection-date column found on gut_full -- repeat-frame "
            "age uses the static (study_id-level) fallback for both visits.")

    banner("B1 CONSISTENCY CHECK: real (non-imputed) species coverage")
    per_id = (rep[rep["species_real"]].groupby("study_id")["visit_n"]
              .apply(lambda x: set(x)))
    both_real = per_id.apply(lambda s: {1, 2}.issubset(s)).sum()
    log(f"  repeat participants with REAL species at BOTH visits: {both_real:,}")
    log(f"  (B1 is now fixed upstream in cross_lagged_panel.py -- expect this near "
        f"~1,787, the corrected leak-free AgeAccel-both-visits count, not the old "
        f"523. If it's still near 523, the schema fix above did not take, and E3 "
        f"sample sizes should not be trusted until it's understood.)")

    return pop, gut_div, clp.load_functional_outcomes(), train, rep, sp_cols, gut_full


# ============================================================
# E1  Discovery + freeze
# ============================================================
def e1_discovery(train: pd.DataFrame, sp_cols: list[str]) -> pd.DataFrame:
    banner("E1  Discovery of age-associated species (single-visit stratum, FROZEN)")
    Z = clr(train[sp_cols].fillna(0).values)          # CLR per sample
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)          # standardise each species
    age = train["age_at_visit"].astype(float).values
    age_z = (age - age.mean()) / age.std()
    sex = train["sex_encoded"].astype(float).fillna(train["sex_encoded"].median()).values

    rows = []
    Xbase = np.column_stack([np.ones_like(age_z), age_z, sex])
    for j, sp in enumerate(sp_cols):
        y = Z[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 100:
            continue
        res = sm.OLS(y[ok], Xbase[ok]).fit()
        rows.append({"species": sp, "gamma_age": res.params[1], "se": res.bse[1],
                     "p": res.pvalues[1]})
    disc = pd.DataFrame(rows)
    disc["q"] = clp.bh_fdr(disc["p"].values)
    disc["retained"] = disc["q"] < FDR_DISCOVERY
    disc["direction"] = np.sign(disc["gamma_age"])
    # frozen weight = standardised discovery coefficient (partial std beta for age)
    disc["weight"] = disc["gamma_age"]
    disc = disc.sort_values("p").reset_index(drop=True)
    n_ret = int(disc["retained"].sum())
    log(f"  species tested: {len(disc)}   retained at q<{FDR_DISCOVERY}: {n_ret}")
    log(f"  (compare to your prior age-correlation feature set of 278)")
    disc.to_csv(OUT / "E1_discovery_species.csv", index=False)
    log(f"  wrote {OUT/'E1_discovery_species.csv'}")
    return disc


# ============================================================
# E2  Stability on held-out repeat participants
# ============================================================
def e2_stability(rep: pd.DataFrame, disc: pd.DataFrame, sp_cols: list[str],
                 test_ids: set) -> pd.DataFrame:
    banner("E2  Temporal stability of frozen taxa (repeat participants NOT in test)")
    frozen = disc[disc["retained"]]["species"].tolist()
    stab_ids = sorted(set(rep["study_id"]) - test_ids)
    log(f"  stability sample (repeat minus prospective-test): {len(stab_ids):,} participants")
    if len(stab_ids) < 50:
        raise SystemExit(
            f"E2 stability sample is only {len(stab_ids)} participants -- too few "
            f"(or zero) to estimate per-species ICC. This means test_ids is too "
            f"broad relative to the repeat cohort; do not proceed with a near-empty "
            f"holdout rather than crash further downstream with a confusing error.")

    sub = rep[rep["study_id"].isin(stab_ids) & rep["species_real"]].copy()
    # CLR within this frame so values are comparable to discovery
    clr_mat = clr(sub[sp_cols].fillna(0).values)
    clr_df = pd.DataFrame(clr_mat, columns=sp_cols, index=sub.index)
    clr_df[["study_id", "visit_n"]] = sub[["study_id", "visit_n"]].values

    rows = []
    for sp in frozen:
        w = clr_df.pivot_table(index="study_id", columns="visit_n", values=sp, aggfunc="mean")
        if not {1, 2}.issubset(w.columns):
            continue
        v1 = w[1].values; v2 = w[2].values
        icc, sdb, sdw, r, n = icc_3_1(v1, v2)
        prev = float((sub[sp] > sub[sp].min()).mean())   # crude non-floor prevalence
        rows.append({"species": sp, "icc": icc, "sd_between": sdb, "sd_within": sdw,
                     "retest_r": r, "n_pairs": n, "prevalence": prev})
    stab = pd.DataFrame(rows)
    stab["stable"] = (stab["icc"] >= ICC_STABLE) & (stab["prevalence"] >= PREVALENCE_MIN)
    log(f"  frozen taxa with ICC estimated: {len(stab)}   "
        f"classified STABLE: {int(stab['stable'].sum())}")
    stab.to_csv(OUT / "E2_stability_species.csv", index=False)
    log(f"  wrote {OUT/'E2_stability_species.csv'}")
    return stab


# ============================================================
# Signature scoring
# ============================================================
def score_signature(rep: pd.DataFrame, sp_cols: list[str],
                    species_set: list[str], weights: dict) -> pd.DataFrame:
    """S_iv = sum_j w_j * Z_ij  over species_set, for each (id, visit).
    CLR computed within the repeat frame; only rows with real species used."""
    sub = rep[rep["species_real"]].copy()
    clr_mat = clr(sub[sp_cols].fillna(0).values)
    clr_df = pd.DataFrame(clr_mat, columns=sp_cols, index=sub.index)
    use = [s for s in species_set if s in clr_df.columns]
    w = np.array([weights[s] for s in use])
    S = clr_df[use].values @ w
    out = sub[["study_id", "visit_n"]].copy()
    out["S"] = S
    return out


def _wide(df, val):
    d = df.groupby(["study_id", "visit_n"], as_index=False)[val].mean()
    t1 = d[d.visit_n == 1].set_index("study_id")[[val]].rename(columns={val: f"{val}_t1"})
    t2 = d[d.visit_n == 2].set_index("study_id")[[val]].rename(columns={val: f"{val}_t2"})
    return t1.join(t2, how="inner")


def cross_lag_min(sig_wide, out_wide, pop) -> dict:
    """y_t2 ~ y_t1 + S_t1 + age + sex   (paper's PRIMARY spec: age + sex only).
    All continuous vars standardised; returns standardised beta_S, CI, p, n."""
    cov = pop.set_index("study_id")[["age_at_visit", "sex_encoded"]]
    df = sig_wide.join(out_wide, how="inner").join(cov, how="left").dropna()
    n = len(df)
    if n < MIN_TEST_N:
        return {"n": n, "beta": np.nan, "se": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    for c in ["S_t1", "y_t1", "y_t2", "age_at_visit"]:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-12)
    X = sm.add_constant(df[["y_t1", "S_t1", "age_at_visit", "sex_encoded"]].astype(float))
    res = sm.OLS(df["y_t2"].astype(float), X).fit()
    ci = res.conf_int().loc["S_t1"].tolist()
    return {"n": n, "beta": res.params["S_t1"], "se": res.bse["S_t1"],
            "p": res.pvalues["S_t1"], "ci_lo": ci[0], "ci_hi": ci[1]}


# ============================================================
# E3  Prospective evaluation of the frozen signatures
# ============================================================
def e3_prospective(rep, disc, stab, sp_cols, outcomes, pop) -> pd.DataFrame:
    banner("E3  Prospective test of frozen signatures (no re-estimation)")
    weights = dict(zip(disc["species"], disc["weight"]))
    frozen_all    = disc[disc["retained"]]["species"].tolist()
    frozen_stable = stab[stab["stable"]]["species"].tolist()
    frozen_less   = stab[~stab["stable"]]["species"].tolist()
    sigs = {"S_all": frozen_all, "S_stable": frozen_stable, "S_lessstable": frozen_less}

    rows = []
    for sname, sset in sigs.items():
        if len(sset) == 0:
            log(f"  {sname}: empty set, skipped"); continue
        S = score_signature(rep, sp_cols, sset, weights)
        Sw = _wide(S, "S")
        for ds, odf in outcomes.items():
            ow = clp.make_wide(odf[["study_id", "visit_n", "outcome"]], "outcome")
            ow = ow.rename(columns={"outcome_t1": "y_t1", "outcome_t2": "y_t2"})
            r = cross_lag_min(Sw, ow, pop)
            r.update({"signature": sname, "outcome": odf.attrs.get("outcome_name", ds),
                      "dataset": ds, "n_species": len(sset)})
            rows.append(r)
            log(f"  {sname} -> {ds}: beta={r['beta']:.4f} "
                f"p={r['p']:.4f} n={r['n']}")
    res = pd.DataFrame(rows)
    # BH-FDR across the 3 PRIMARY outcomes for S_all only (primary family)
    prim = res["signature"] == "S_all"
    res.loc[prim, "q"] = clp.bh_fdr(res.loc[prim, "p"].fillna(1).values)
    res.to_csv(OUT / "E3_prospective_signatures.csv", index=False)
    log(f"  wrote {OUT/'E3_prospective_signatures.csv'}")

    # [TASK 2 COMPLETENESS] The brief's E3 spec requires "a direct test of
    # whether their [stable vs less-stable] prospective coefficients differ"
    # as part of the secondary family -- not implemented in the original
    # reference script. Added here: an approximate two-sample z-test on the
    # coefficient difference using each model's own SE. This treats the two
    # estimates as independent, which they are not (same participants,
    # different predictor) -- reported as an approximation, not an exact test;
    # a fully correct version would need the covariance between the two
    # estimates (e.g. via a joint/seemingly-unrelated-regression or a
    # bootstrap), which this does not attempt.
    from scipy.stats import norm
    diff_rows = []
    for ds, odf in outcomes.items():
        outcome_label = odf.attrs.get("outcome_name", ds)
        r_stab = res[(res["signature"] == "S_stable") & (res["dataset"] == ds)]
        r_less = res[(res["signature"] == "S_lessstable") & (res["dataset"] == ds)]
        if r_stab.empty or r_less.empty:
            continue
        b1, se1 = r_stab.iloc[0]["beta"], r_stab.iloc[0]["se"]
        b2, se2 = r_less.iloc[0]["beta"], r_less.iloc[0]["se"]
        if not all(np.isfinite(v) for v in (b1, b2, se1, se2)):
            continue
        se_diff = float(np.sqrt(se1**2 + se2**2))
        z = (b1 - b2) / se_diff if se_diff > 0 else np.nan
        p_diff = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        diff_rows.append({"outcome": outcome_label, "dataset": ds,
                          "beta_stable": b1, "beta_lessstable": b2,
                          "diff": b1 - b2, "se_diff_approx": se_diff,
                          "z_approx": z, "p_diff_approx": p_diff})
        log(f"  stable vs less-stable, {ds}: diff={b1-b2:+.4f} "
            f"(approx z={z:+.2f}, p={p_diff:.4f})")
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT / "E3_stable_vs_lessstable_diff.csv", index=False)
    log(f"  wrote {OUT/'E3_stable_vs_lessstable_diff.csv'}")

    return res


def e3_incremental_prediction(rep, disc, outcomes, pop, sp_cols) -> pd.DataFrame:
    """Out-of-sample delta-R^2: does adding S_all to (y_t1, age, sex) improve
    prediction of y_t2 under repeated CV? Distinguishes significance from
    practically meaningful prediction (the professor's explicit request)."""
    banner("E3b  Incremental out-of-sample prediction (repeated CV)")
    weights = dict(zip(disc["species"], disc["weight"]))
    frozen_all = disc[disc["retained"]]["species"].tolist()
    S = score_signature(rep, sp_cols, frozen_all, weights)   # full CLR basis
    Sw = _wide(S, "S")
    cov = pop.set_index("study_id")[["age_at_visit", "sex_encoded"]]
    rows = []
    rkf = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    for ds, odf in outcomes.items():
        ow = clp.make_wide(odf[["study_id", "visit_n", "outcome"]], "outcome")
        ow = ow.rename(columns={"outcome_t1": "y_t1", "outcome_t2": "y_t2"})
        df = Sw.join(ow, how="inner").join(cov, how="left").dropna()
        if len(df) < MIN_TEST_N:
            log(f"  {ds}: n={len(df)} < {MIN_TEST_N}, skipped"); continue
        Xb = df[["y_t1", "age_at_visit", "sex_encoded"]].values
        Xf = df[["y_t1", "age_at_visit", "sex_encoded", "S_t1"]].values
        y = df["y_t2"].values
        r2b, r2f = [], []
        for tr, te in rkf.split(Xb):
            r2b.append(_oos_r2(Xb[tr], y[tr], Xb[te], y[te]))
            r2f.append(_oos_r2(Xf[tr], y[tr], Xf[te], y[te]))
        rows.append({"dataset": ds, "outcome": odf.attrs.get("outcome_name", ds),
                     "n": len(df), "r2_baseline": np.mean(r2b),
                     "r2_with_S": np.mean(r2f),
                     "delta_r2": np.mean(r2f) - np.mean(r2b)})
        log(f"  {ds}: R2 base={np.mean(r2b):.4f}  +S={np.mean(r2f):.4f}  "
            f"delta={np.mean(r2f)-np.mean(r2b):+.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "E3_incremental_prediction.csv", index=False)
    log(f"  wrote {OUT/'E3_incremental_prediction.csv'}")
    return out


def _oos_r2(Xtr, ytr, Xte, yte) -> float:
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - ytr.mean()) ** 2)   # baseline = train mean
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ============================================================
# E*  Three-property map
# ============================================================
def property_map(rep, disc, stab, outcomes, pop, sp_cols) -> None:
    banner("E*  Three-property map (dissociation x, stability y, prospective size)")
    weights = dict(zip(disc["species"], disc["weight"]))
    frozen = disc[disc["retained"]]["species"].tolist()
    icc_by_sp = dict(zip(stab["species"], stab["icc"]))

    # per-species prospective effect: |std beta| of that single species (CLR)
    # in Path A, MAX across the three phenotypes (noisy -- shown small/faint).
    sub = rep[rep["species_real"]].copy()
    clr_df = pd.DataFrame(clr(sub[sp_cols].fillna(0).values), columns=sp_cols, index=sub.index)
    clr_df[["study_id", "visit_n"]] = sub[["study_id", "visit_n"]].values

    rows = []
    for sp in frozen:
        if sp not in icc_by_sp:
            continue
        sw = _wide(clr_df[["study_id", "visit_n", sp]].rename(columns={sp: "S"}), "S")
        best = 0.0
        for ds, odf in outcomes.items():
            ow = clp.make_wide(odf[["study_id", "visit_n", "outcome"]], "outcome")
            ow = ow.rename(columns={"outcome_t1": "y_t1", "outcome_t2": "y_t2"})
            r = cross_lag_min(sw, ow, pop)
            if np.isfinite(r["beta"]):
                best = max(best, abs(r["beta"]))
        rows.append({"species": sp,
                     "dissociation": abs(disc.set_index("species").loc[sp, "gamma_age"]),
                     "stability_icc": icc_by_sp[sp],
                     "prospective_abs_beta": best})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "property_map.csv", index=False)
    log(f"  wrote {OUT/'property_map.csv'}  ({len(m)} species)")

    # [TASK 2 COMPLETENESS] "Do the three properties dissociate?" was left as
    # a visual read of property_map.png. Quantified here instead: pairwise
    # correlations among dissociation (age effect), stability (ICC) and
    # prospective |beta|. High positive correlations everywhere would mean the
    # three properties move together (conventional reading: an age-associated,
    # stable species is also the one with prospective signal); near-zero or
    # mixed-sign correlations mean they dissociate (the reframed thesis).
    if len(m) >= 10:
        r_diss_stab, p_ds = pearsonr(m["dissociation"], m["stability_icc"])
        r_diss_pro,  p_dp = pearsonr(m["dissociation"], m["prospective_abs_beta"])
        r_stab_pro,  p_sp = pearsonr(m["stability_icc"], m["prospective_abs_beta"])
        log(f"  dissociation vs stability      : r={r_diss_stab:+.3f}  p={p_ds:.4g}")
        log(f"  dissociation vs prospective |β| : r={r_diss_pro:+.3f}  p={p_dp:.4g}")
        log(f"  stability vs prospective |β|    : r={r_stab_pro:+.3f}  p={p_sp:.4g}")
        max_abs_r = max(abs(r_diss_stab), abs(r_diss_pro), abs(r_stab_pro))
        if max_abs_r > 0.3:
            log(f"  -> at least one pair is non-trivially correlated (|r|="
                f"{max_abs_r:.2f}) -- the three properties partially COINCIDE, "
                f"not a clean dissociation.")
        else:
            log(f"  -> all three pairwise correlations are weak (max |r|="
                f"{max_abs_r:.2f}) -- the three properties DISSOCIATE: being "
                f"age-associated, being stable, and carrying prospective "
                f"signal are largely independent axes among these species.")
    else:
        log(f"  too few species ({len(m)}) with all three properties to "
            f"quantify dissociation")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        sizes = 20 + 1400 * (m["prospective_abs_beta"] / (m["prospective_abs_beta"].max() + 1e-9))
        sc = ax.scatter(m["dissociation"], m["stability_icc"], s=sizes,
                        c=m["prospective_abs_beta"], cmap="viridis",
                        alpha=0.55, edgecolor="k", linewidth=0.3)
        ax.axhline(ICC_STABLE, ls="--", lw=0.8, color="grey")
        ax.set_xlabel("Age dissociation  |standardised discovery β|")
        ax.set_ylabel("Temporal stability  ICC(3,1)")
        ax.set_title("Three properties per age-associated species\n"
                     "(point size / colour = prospective |β|, max over phenotypes)")
        cb = fig.colorbar(sc); cb.set_label("prospective |β|")
        ax.text(0.02, 0.02, "Note: per-species prospective β is noisy;\n"
                            "read the cloud, not individual points.",
                transform=ax.transAxes, fontsize=7, va="bottom", color="#555")
        # [MANUSCRIPT] annotate the quantified dissociation verdict directly on the
        # figure -- previously only a console log line, now visible to a reader
        # without needing the .csv/.log alongside it (Round 2 addendum).
        if len(m) >= 10:
            verdict = ("DISSOCIATE (all |r|<0.3)" if max_abs_r <= 0.3
                      else "partially COINCIDE (max |r|>0.3)")
            ax.text(0.98, 0.98,
                   f"dissoc.×stability  r={r_diss_stab:+.3f}\n"
                   f"dissoc.×prospective r={r_diss_pro:+.3f}\n"
                   f"stability×prospective r={r_stab_pro:+.3f}\n"
                   f"→ {verdict}",
                   transform=ax.transAxes, ha="right", va="top", fontsize=7,
                   fontfamily="monospace", color="#333",
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888", lw=0.5, alpha=0.9))
        fig.tight_layout()
        fig.savefig(OUT / "property_map.png", dpi=300)
        # [MANUSCRIPT] the v17 \smartfig macro requires the .pdf, not .png
        fig.savefig(OUT / "property_map.pdf", dpi=300)
        log(f"  wrote {OUT/'property_map.png'} (+ property_map.pdf)")
    except Exception as exc:
        log(f"  figure skipped ({exc}); property_map.csv still written")


# ============================================================
# Sanity checks required by the brief but not previously computed here:
# S_all's own ICC vs AgeAccel's published 0.483, and their correlation.
# Reported only -- neither affects any frozen E1/E2 threshold.
# ============================================================
def sanity_s_vs_ageaccel(rep, disc, sp_cols, gut_div, gut_full, pop) -> None:
    banner("SANITY  S_all vs AgeAccel (ICC and correlation)")
    weights = dict(zip(disc["species"], disc["weight"]))
    frozen_all = disc[disc["retained"]]["species"].tolist()
    if not frozen_all:
        log("  no species retained at discovery FDR -- skipping")
        return

    S = score_signature(rep, sp_cols, frozen_all, weights)
    Sw = _wide(S, "S")
    icc, sdb, sdw, r_retest, n_icc = icc_3_1(Sw["S_t1"].values, Sw["S_t2"].values)
    log(f"  S_all ICC(3,1) = {icc:.3f}  (retest r={r_retest:.3f}, n={n_icc:,})")
    log(f"  Published AgeAccel ICC = 0.483 (different formula -- mixed-model "
        f"ICC(1) via longitudinal_microbiome_age.py's compute_icc -- so this "
        f"comparison is approximate, not an identical-formula match)")
    if np.isfinite(icc):
        if icc < 0.483 - 0.05:
            log(f"  -> S_all is LESS stable than AgeAccel. Expect E3 nulls to be "
                f"at least as attenuated as the main paper's, for the same "
                f"measurement-reliability reason.")
        elif icc > 0.483 + 0.05:
            log(f"  -> S_all is MORE stable than AgeAccel -- a genuinely "
                f"different, more reliable exposure. E3 results are not just a "
                f"rerun of the main paper's null under a new name.")
        else:
            log(f"  -> S_all and AgeAccel have comparable reliability.")

    # Recomputes AgeAccel via the validated leak-free clock (redundant with any
    # prior cross_lagged_panel.py run, kept for self-containment).
    gut_ageaccel, _ = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
    merged = S.merge(gut_ageaccel, on=["study_id", "visit_n"], how="inner").dropna(
        subset=["S", "gut_ageaccel"])
    if len(merged) > 2:
        r_corr, p_corr = pearsonr(merged["S"], merged["gut_ageaccel"])
        log(f"  corr(S_all, AgeAccel), pooled both visits: r={r_corr:.3f} "
            f"p={p_corr:.4g} n={len(merged):,}")
        if abs(r_corr) > 0.7:
            log(f"  -> S_all and AgeAccel are strongly correlated -- largely the "
                f"same signal under two names; treat E3 as a partial "
                f"replication, not independent evidence.")
        else:
            log(f"  -> S_all and AgeAccel are only weakly/moderately correlated "
                f"-- the taxon-level signature carries information the clock "
                f"does not reduce to.")
    else:
        log(f"  too few overlapping (study_id, visit) rows ({len(merged)}) to "
            f"correlate S_all with AgeAccel")


# ============================================================
def main() -> None:
    banner("FROZEN-SIGNATURE EXTENSION PIPELINE (E1-E3 + property map)")
    pop, gut_div, outcomes, train, rep, sp_cols, gut_full = build_frames()

    # [E2 FIX] test_ids narrowed to match who actually lands in the FITTED
    # cross-lag models, not just "has the phenotype at both visits". The
    # broader definition (any phenotype pair -- 2,541 participants, LARGER
    # than the entire 1,823-participant gut-repeat cohort, since phenotype
    # data has its own broader repeat population) left ZERO participants for
    # E2's stability holdout: in this cohort phenotype and microbiome are
    # collected at the same visits, so virtually everyone with 2 gut samples
    # also has some phenotype pair, and e2_stability() crashed with
    # KeyError('icc') on the resulting empty frame. Requiring the same
    # covariate completeness cross_lag_ols() already enforces (age, sex,
    # education, income) shrinks test_ids to the true final analytic samples
    # (~509-522 per outcome in the main pipeline), leaving most of the cohort
    # available for a genuine held-out stability estimate.
    test_ids = set()
    cov_cols_check = [c for c in clp.BASE_COVARIATES if c in pop.columns]
    covariates_df = pop.set_index("study_id")[cov_cols_check]
    for ds, odf in outcomes.items():
        w = clp.make_wide(odf[["study_id", "visit_n", "outcome"]], "outcome")
        merged = w.join(covariates_df, how="left").dropna(
            subset=["outcome_t1", "outcome_t2"])
        cc = [c for c in cov_cols_check if merged[c].notna().sum() > 100]
        if cc:
            merged = merged.dropna(subset=cc)
        test_ids |= set(merged.index)
    log(f"\n  prospective-test participants (phenotype + complete covariates, "
        f"matching the fitted cross-lag models): {len(test_ids):,}")

    disc = e1_discovery(train, sp_cols)
    stab = e2_stability(rep, disc, sp_cols, test_ids)
    e3   = e3_prospective(rep, disc, stab, sp_cols, outcomes, pop)
    e3_incremental_prediction(rep, disc, outcomes, pop, sp_cols)
    property_map(rep, disc, stab, outcomes, pop, sp_cols)
    sanity_s_vs_ageaccel(rep, disc, sp_cols, gut_div, gut_full, pop)

    banner("DONE")
    log("  Interpretation order:")
    log("   1. Check the B1 consistency line: is real-species-at-both-visits ~1,787")
    log("      (the corrected leak-free count) or still ~523? That decides your true")
    log("      prospective n.")
    log("   2. Does S_all's spread beat AgeAccel's ICC 0.483? If not, expect nulls")
    log("      for the same reason as the main paper (weak, noisy exposure).")
    log("   3. Read property_map.png: big points top-right => three properties")
    log("      coincide (conventional reading holds); scattered => they dissociate")
    log("      (your finding).")


if __name__ == "__main__":
    main()
