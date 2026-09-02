#!/usr/bin/env python3
"""
Cross-lagged panel analysis: microbiome diversity ↔ functional aging (HPP).

RESEARCH QUESTION
-----------------
Are gut microbiome characteristics DRIVERS or CONSEQUENCES of the aging process?

WHY THIS QUESTION
-----------------
Li, Tang, ..., Segal & Razzak (arXiv:2510.12384) state explicitly in their
limitations: "our study establishes strong associations but cannot definitively
prove causation; mechanistic studies are needed to determine whether the observed
changes, especially in the microbiome, are drivers or consequences of the aging
process."

The cross-lagged panel model (CLPM) is the standard epidemiological tool for
exactly this. With two measurements of both X and Y per person:
    Path A: X(t1) → Y(t2) | Y(t1)   [X predicts future Y beyond Y's own baseline]
    Path B: Y(t1) → X(t2) | X(t1)   [Y predicts future X beyond X's own baseline]
Comparing A vs B tests causal direction without a randomised trial.

DATASET AVAILABILITY (verified empirically)
--------------------------------------------
    gut_microbiome      1,823 / 8,838  with 2+ research stages
    hand_grip           2,543 / 10,415 with 2+ research stages  → overlap: 1,823
    blood_pressure      2,543 / 10,415 with 2+ research stages  → overlap: 1,823
    anthropometrics     2,537 / 10,404 with 2+ research stages  → overlap: 1,817

This gives a clean, n=1,823 cross-lagged dataset -- unusually large for this
design (typical published CLPMs use n=200-500).

FUNCTIONAL AGING OUTCOMES
--------------------------
The paper's "aging axis" focuses on biological deterioration. We use three
objectively measured, monotonically age-declining phenotypes:
  1. Hand grip strength  -- widely validated sarcopenia / functional aging marker
  2. Systolic blood pressure  -- cardiovascular aging (rises with age)
  3. BMI / waist circumference  -- adiposity / metabolic aging

MICROBIOME EXPOSURES
--------------------
  1. Shannon diversity (alpha diversity) -- richness + evenness
  2. Species richness  -- number of detected species
  3. Dominant phylum ratios (Firmicutes/Bacteroidetes) -- where available
  4. Gut-microbiome-derived AgeAccel  -- the microbiome "aging score"
     (built on single-visit participants ONLY to avoid leakage)

STATISTICAL MODEL
-----------------
Random-effects panel regression (random intercept = person) with two lags:
    Y(t2) = β0 + β1·Y(t1) + β2·X(t1) + β3·Z + u_i + ε_{it}
where Z = covariates (age, sex, BMI [if Y is not BMI], education, income).

We report:
  - β2 (Path A cross-lag): X(t1) → Y(t2)
  - The symmetric β2* (Path B): Y(t1) → X(t2)
  - Standardised betas (Cohen's d interpretation)
  - 95% CI, p-value, FDR-corrected q-value across all cross-lag tests

NOTE ON CAUSAL LANGUAGE
-----------------------
CLPM establishes temporal precedence and incremental prediction, not
randomised causation. We use "predicts" language, not "causes", and note
residual confounding by unmeasured time-varying factors as a limitation.

USAGE
-----
    conda activate pheno
    python3 cross_lagged_panel.py

    export HPP_DATA_ROOT=/path/to/hpp_datasets   # default: ~/studies/hpp_datasets
    export HPP_RESULTS_DIR=/path/to/output        # default: ./results_cross_lagged
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as _pe
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- style guide ----------------------------------------------------------- #
_PATH_A  = "#2166AC"   # microbiome → aging (blue)
_PATH_B  = "#D6604D"   # aging → microbiome (red)
_BIDIR   = "#762A83"   # bidirectional (purple)
_NEUTRAL = "#878787"   # neutral / no effect
_INK     = "#1A1A1A"
_PANEL_BG= "#F8F8F6"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["IBM Plex Sans", "Arial", "DejaVu Sans"],
    "font.size":         7,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
    "axes.edgecolor":    _INK,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    _PANEL_BG,
    "figure.facecolor":  "white",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "savefig.dpi":       300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

def _lift(artist, dy: int = -2, alpha: float = 0.25) -> None:
    """Drop-shadow on data marks that carry a result."""
    artist.set_path_effects([
        _pe.withSimplePatchShadow(offset=(1, dy), shadow_rgbFace=_INK, alpha=alpha),
        _pe.Normal(),
    ])

# ============================================================
# CONFIGURATION  (all thresholds declared here, none buried)
# ============================================================

DATA_ROOT   = Path(os.environ.get("HPP_DATA_ROOT",  Path.home() / "studies" / "hpp_datasets"))
RESULTS_DIR = Path(os.environ.get("HPP_RESULTS_DIR", "results_cross_lagged"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
ALPHA_FDR     = 0.05       # BH-FDR threshold for cross-lag tests
MIN_OVERLAP   = 500        # minimum n for a cross-lag pair to be reported
CLOCK_CV_FOLDS = 5         # folds for clock r² estimate

# Covariates: not the outcomes, not the exposures.
#
# The paper's stated primary model (Methods, meth:crosslag) is
# Y_t2 ~ Y_t1 + X_t1 + age + sex ONLY -- matching PRIMARY_COVARIATES below and
# the E1-E3 spec in frozen_signature_pipeline.py's cross_lag_min(). Also
# requiring education_level and household_income (both sparse) via dropna
# throttles all 18 main cross-lag tests to n~509-522, well below what's
# actually available (~1,800+ for diversity, ~1,780+ for AgeAccel) -- use
# PRIMARY_COVARIATES for the primary analysis; BASE_COVARIATES (full set) is
# for the fully-adjusted sensitivity analysis described in the paper
# (subsec:covariate) only.
PRIMARY_COVARIATES = ["age_at_visit", "sex_encoded"]
BASE_COVARIATES = ["age_at_visit", "sex_encoded", "education_level", "household_income"]

# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values same shape as pvals."""
    pv = np.asarray(pvals, dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    rank  = np.empty(n); rank[order] = np.arange(1, n+1)
    q = np.minimum(1.0, pv * n / rank)
    # enforce monotonicity from right
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return q


def standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score cols in-place; returns df."""
    df = df.copy()
    for c in cols:
        mu, sd = df[c].mean(), df[c].std()
        if sd > 0:
            df[c] = (df[c] - mu) / sd
    return df


def _partial_r(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Partial Pearson r between y and x after regressing out controls."""
    def resid(v):
        cov = sm.add_constant(controls)
        return sm.OLS(v, cov).fit().resid
    ry, rx = resid(y), resid(x)
    r, p = pearsonr(ry, rx)
    return r, p


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def _load_parquet_flat(path: Path) -> pd.DataFrame:
    """Load a parquet, reset any MultiIndex, return flat DataFrame."""
    df = pd.read_parquet(path)
    if df.index.names != [None]:
        df = df.reset_index()
    return df


def load_population() -> pd.DataFrame:
    """
    HPP population.parquet is a cohort-level summary (study_id = cohort number).
    Participant-level demographics (age, sex) live in sociodemographics.
    We build one row per participant_id from sociodemographics.
    """
    log("\n[Step 1] Loading population & covariates...")

    # ── Try sociodemographics for participant-level data ──────
    socio_dir = DATA_ROOT / "sociodemographics"
    socio_files = list(socio_dir.glob("*.parquet")) if socio_dir.exists() else []
    log(f"  sociodemographics files: {[f.name for f in socio_files]}")

    frames = []
    for f in socio_files:
        try:
            df = _load_parquet_flat(f)
            pid = next((c for c in df.columns if c in ("participant_id","study_id")), None)
            if pid:
                frames.append(df)
        except Exception as e:
            log(f"  {f.name}: {e}")

    if frames:
        pop = pd.concat(frames, ignore_index=True)
        pid_col = next((c for c in pop.columns if c in ("participant_id","study_id")), None)
        if pid_col and pid_col != "study_id":
            if "study_id" in pop.columns:
                pop = pop.drop(columns=["study_id"])
            pop = pop.rename(columns={pid_col: "study_id"})
        log(f"  Loaded sociodemographics: {pop.shape}, cols: {list(pop.columns[:10])}")
    else:
        # Fall back to population parquet (cohort-level — limited use)
        log("  WARNING: no sociodemographics found — using population.parquet (cohort-level)")
        pop = _load_parquet_flat(DATA_ROOT / "population" / "population.parquet")
        pid_col = next((c for c in pop.columns if c in ("participant_id","study_id")), None)
        if pid_col and pid_col != "study_id":
            pop = pop.rename(columns={pid_col: "study_id"})

    # ── Sex — HPP uses 'gender' column ───────────────────────
    sex_col = next((c for c in pop.columns
                    if c.lower() in ("sex", "gender", "sex_numeric")), None)
    if sex_col is None:
        sex_col = next((c for c in pop.columns if "sex" in c.lower()), None)
    if sex_col:
        raw_sex = pop[sex_col].astype(str).str.strip()
        pop["sex_encoded"] = raw_sex.map(
            {"Male": 1, "Female": 0, "male": 1, "female": 0,
             "M": 1, "F": 0, "1": 1, "0": 0, "2": 0}
        ).astype(float)
        # fill any unmapped with numeric coerce
        mask = pop["sex_encoded"].isna()
        pop.loc[mask, "sex_encoded"] = pd.to_numeric(
            pop.loc[mask, sex_col], errors="coerce")
        log(f"  Sex from '{sex_col}': non-null={pop['sex_encoded'].notna().sum():,}")
    else:
        pop["sex_encoded"] = np.nan
        log("  WARNING: no sex/gender column found")

    # ── Age — HPP stores birth year as integer in 'DOB' column ──
    # [B3] age_at_visit computed here is a SINGLE static value per participant
    # (birth year subtracted from a fixed reference year), not a true per-visit
    # age. It is kept as a fallback for participants where no per-row
    # collection date can be found, but build_gut_clock_ageaccel() prefers a
    # genuine per-visit age computed from birth_year + each row's own
    # collection date, matching longitudinal_microbiome_age.py's approach.
    # birth_year itself IS a valid participant-level constant (it does not
    # change across visits), so collapsing it in the groupby().first() below
    # is correct; only age_at_visit's use as a stand-in for "age at this visit"
    # was the bug.
    pop["birth_year"] = np.nan
    # tracked separately so we can later check whether participants whose age
    # came from the population.parquet backfill differ systematically from
    # those with DOB already present in sociodemographics -- see [BUG4-CHECK]
    # below.
    if "DOB" in pop.columns:
        birth_year = pd.to_numeric(pop["DOB"], errors="coerce")
        pop["birth_year"] = birth_year
        pop["age_at_visit"] = 2020 - birth_year
        log(f"  Age from DOB (birth year): mean={pop['age_at_visit'].mean():.1f} "
            f"non-null={pop['age_at_visit'].notna().sum():,}")
    else:
        # generic fallback
        age_col = next((c for c in pop.columns
                        if "age" in c.lower() and "visit" not in c.lower()
                        and "education" not in c.lower()), None)
        dob_col = next((c for c in pop.columns
                        if "dob" in c.lower() or "birth_year" in c.lower()), None)
        if age_col:
            pop["age_at_visit"] = pd.to_numeric(pop[age_col], errors="coerce")
            log(f"  Age from '{age_col}': mean={pop['age_at_visit'].mean():.1f}")
        elif dob_col:
            birth_year = pd.to_numeric(pop[dob_col], errors="coerce")
            pop["birth_year"] = birth_year
            pop["age_at_visit"] = 2020 - birth_year
            log(f"  Age from '{dob_col}': mean={pop['age_at_visit'].mean():.1f}")
        else:
            pop["age_at_visit"] = np.nan
            log("  WARNING: no age or DOB column found — age_at_visit will be NaN")

    # [BUG4 FIX] sociodemographics.DOB is confirmed sparse on real data --
    # birth_year covered only ~1,050/3,655 repeat-visit rows (~523
    # participants), which is exactly where the leak-free AgeAccel
    # both-visits count bottlenecked, even after the B1/B2 species-merge fix
    # confirmed a 100% match rate. longitudinal_microbiome_age.py (the
    # pipeline that reaches ~1,817-1,823 participants) sources birth year from
    # population.parquet's year_of_birth column instead. Backfill birth_year
    # gaps from that same source rather than leaving them NaN, so both
    # pipelines draw on the same underlying birth-year data.
    try:
        pop_file = _load_parquet_flat(DATA_ROOT / "population" / "population.parquet")
        pid_pf = next((c for c in pop_file.columns
                       if c in ("participant_id", "study_id")), None)
        yob_pf = next((c for c in pop_file.columns
                       if "year_of_birth" in c.lower()), None)
        # [BUG4-CHECK] also try to backfill sex from the same file -- the
        # sociodemographics 'gender' non-null count matched DOB's non-null
        # count exactly (7,416 = 7,416) on the real data, meaning sex is
        # blank for precisely the same rows that are missing DOB. Without a
        # second source, the backfill-only group's sex balance is simply
        # unknowable, not merely unchecked.
        sex_pf = next((c for c in pop_file.columns
                       if c.lower() in ("sex", "gender", "sex_numeric")), None)
        if pid_pf and yob_pf:
            pf_cols = [pid_pf, yob_pf] + ([sex_pf] if sex_pf else [])
            yob_slim = (pop_file[pf_cols]
                       .rename(columns={pid_pf: "study_id", yob_pf: "_yob_pop",
                                        **({sex_pf: "_sex_pop"} if sex_pf else {})})
                       .drop_duplicates(subset=["study_id"]))
            pop = pop.merge(yob_slim, on="study_id", how="left")
            n_before = int(pop["birth_year"].notna().sum())
            # [BUG4-CHECK] tag rows filled ONLY by the backfill (DOB was
            # missing) so we can compare this group against the
            # DOB-already-present group and check the backfill isn't
            # introducing a systematic age/sex skew into who gets included.
            # Tagged pre-dedup so a participant with one row from each source
            # file is still tagged by which row actually supplies the value.
            pop["_age_source"] = np.where(pop["birth_year"].notna(), "dob",
                                          np.where(pd.to_numeric(pop["_yob_pop"], errors="coerce").notna(),
                                                   "population_backfill", "missing"))

            # [CHECK-3] Age-source authority: where BOTH sociodemographics.DOB
            # and population.parquet.year_of_birth exist for the same
            # participant, do they agree? This is the co-author-facing
            # question -- not just "which field is more complete" but "do the
            # two fields conflict where both are present".
            _pop_yob_numeric = pd.to_numeric(pop["_yob_pop"], errors="coerce")
            _both_present = pop["birth_year"].notna() & _pop_yob_numeric.notna()
            if _both_present.sum() > 0:
                _diff = pop.loc[_both_present, "birth_year"] - _pop_yob_numeric[_both_present]
                _n_both = int(_both_present.sum())
                _n_exact = int((_diff == 0).sum())
                _n_close = int((_diff.abs() <= 1).sum())
                _n_conflict = int((_diff.abs() > 1).sum())
                log(f"  [CHECK-3] Where both DOB and population.parquet "
                    f"year_of_birth exist (n={_n_both:,}): exact match="
                    f"{_n_exact:,} ({100*_n_exact/_n_both:.1f}%), within "
                    f"±1yr={_n_close:,} ({100*_n_close/_n_both:.1f}%), "
                    f"disagree >1yr={_n_conflict:,} "
                    f"({100*_n_conflict/_n_both:.1f}%), "
                    f"max |diff|={_diff.abs().max():.0f} yr")
                if _n_conflict > 0:
                    _conflict_ids = pop.loc[_both_present][_diff.abs() > 1][
                        "study_id"].tolist()
                    log(f"  [CHECK-3] {_n_conflict} conflicting study_id "
                        f"(sample): {_conflict_ids[:10]}")
            else:
                log(f"  [CHECK-3] No rows have both DOB and population.parquet "
                    f"year_of_birth -- cannot check agreement, only completeness.")

            pop["birth_year"] = pop["birth_year"].where(
                pop["birth_year"].notna(), pd.to_numeric(pop["_yob_pop"], errors="coerce"))
            if sex_pf:
                sex_pop_mapped = pop["_sex_pop"].astype(str).str.strip().map(
                    {"Male": 1, "Female": 0, "male": 1, "female": 0,
                     "M": 1, "F": 0, "1": 1, "0": 0, "2": 0}).astype(float)
                pop["sex_encoded"] = pop["sex_encoded"].where(
                    pop["sex_encoded"].notna(), sex_pop_mapped)
                pop = pop.drop(columns=["_sex_pop"])
            pop = pop.drop(columns=["_yob_pop"])
            n_after = int(pop["birth_year"].notna().sum())
            pop["age_at_visit"] = 2020 - pop["birth_year"]
            log(f"  [BUG4] Backfilled birth_year from population.parquet['{yob_pf}']: "
                f"{n_before:,} -> {n_after:,} non-null (+{n_after - n_before:,})")
            log(f"  [BUG4] Sex backfill from population.parquet: "
                + (f"column '{sex_pf}' found, {pop['sex_encoded'].notna().sum():,} "
                   f"non-null overall" if sex_pf else
                   "NO sex/gender column found in population.parquet -- sex balance "
                   "of the backfill-only group cannot be checked."))
        else:
            log(f"  [BUG4] population.parquet has no participant_id/year_of_birth "
                f"column to backfill birth_year from (found pid={pid_pf}, yob={yob_pf}).")
    except Exception as e:
        log(f"  [BUG4] WARNING: could not backfill birth_year from population.parquet: {e}")

    # ── Education / income ───────────────────────────────────
    for tgt, candidates in [
        ("education_level",  ["education_level","education","years_of_education"]),
        ("household_income", ["household_income","income","annual_income"]),
    ]:
        col = next((c for c in pop.columns if c.lower() in candidates), None)
        if col:
            pop[tgt] = pd.to_numeric(pop[col], errors="coerce")
        elif tgt not in pop.columns:
            pop[tgt] = np.nan

    keep = ["study_id","age_at_visit","birth_year","sex_encoded","education_level",
           "household_income","_age_source"]
    pop = pop[[c for c in keep if c in pop.columns]].copy()
    # One row per participant (take first if multiple rows due to repeat visits)
    pop = pop.groupby("study_id", as_index=False).first()
    log(f"  Population: {len(pop):,} participants")
    log(f"  age non-null: {pop['age_at_visit'].notna().sum():,}")
    log(f"  sex non-null: {pop['sex_encoded'].notna().sum():,}")
    log(f"  study_id dtype: {pop['study_id'].dtype}, sample: {pop['study_id'].head(3).tolist()}")

    # [BUG4-CHECK] Run on the FINAL, deduplicated, one-row-per-participant
    # population (not the pre-dedup row-level table, where a participant with
    # rows from both source files could be tagged inconsistently across rows).
    if "_age_source" in pop.columns:
        dob_grp  = pop[pop["_age_source"] == "dob"]
        fill_grp = pop[pop["_age_source"] == "population_backfill"]
        if len(dob_grp) and len(fill_grp):
            age_d, age_f = dob_grp["age_at_visit"], fill_grp["age_at_visit"]
            sex_d, sex_f = dob_grp["sex_encoded"], fill_grp["sex_encoded"]
            pooled_sd = np.sqrt((age_d.var() + age_f.var()) / 2)
            smd_age = (age_f.mean() - age_d.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            female_d = 100 * (1 - sex_d.mean()) if sex_d.notna().any() else np.nan
            female_f = 100 * (1 - sex_f.mean()) if sex_f.notna().any() else np.nan
            log(f"  [BUG4-CHECK] DOB-present group: n={len(dob_grp):,}, "
                f"age={age_d.mean():.1f}±{age_d.std():.1f}, "
                f"%female={female_d:.1f}, sex non-null={sex_d.notna().sum():,}")
            log(f"  [BUG4-CHECK] Backfill-only group: n={len(fill_grp):,}, "
                f"age={age_f.mean():.1f}±{age_f.std():.1f}, "
                f"%female={female_f:.1f}, sex non-null={sex_f.notna().sum():,}")
            log(f"  [BUG4-CHECK] Standardised age difference (backfill - DOB): "
                f"{smd_age:+.3f}"
                + ("  <-- WARNING: |SMD| > 0.1, backfilled participants may not "
                   "be missing-at-random with respect to age" if abs(smd_age) > 0.1
                   else "  (small, consistent with missing-at-random)"))
            if sex_d.notna().any() and sex_f.notna().any():
                pooled_sd_sex = np.sqrt((sex_d.var() + sex_f.var()) / 2)
                smd_sex = (sex_f.mean() - sex_d.mean()) / pooled_sd_sex if pooled_sd_sex > 0 else np.nan
                log(f"  [BUG4-CHECK] Standardised sex difference (backfill - DOB): "
                    f"{smd_sex:+.3f}"
                    + ("  <-- WARNING: |SMD| > 0.1" if abs(smd_sex) > 0.1 else "  (small)"))
        pop = pop.drop(columns=["_age_source"])
    return pop


def load_longitudinal_gut() -> pd.DataFrame:
    log("\n[Step 1b] Loading gut microbiome (all visits)...")
    gut = pd.read_parquet(DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut.index.names != [None]:
        gut = gut.reset_index()

    pid  = next(c for c in gut.columns if c in ("participant_id", "study_id"))
    stage = next(c for c in gut.columns if "research_stage" in c.lower())
    gut = gut.rename(columns={pid: "study_id", stage: "visit"})

    # Keep only participants with ≥2 visits
    counts = gut.groupby("study_id")["visit"].nunique()
    repeat_ids = counts[counts >= 2].index
    gut = gut[gut["study_id"].isin(repeat_ids)].copy()
    log(f"  Gut repeat participants: {gut['study_id'].nunique():,}, rows: {len(gut):,}")

    # Sort visits so visit 1 < visit 2
    gut["visit_n"] = gut.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    return gut


def load_functional_outcomes() -> dict[str, pd.DataFrame]:
    """Load grip, blood pressure, anthropometrics — wide format with visit label."""
    log("\n[Step 1c] Loading functional aging outcomes...")
    out = {}
    # Each entry: (path, primary_candidates, fallback_candidates).
    # primary_candidates are tried FIRST as a group (most-populated among them
    # wins); fallback_candidates are only used if NONE of the primary columns
    # are present, and trigger a loud warning -- this prevents a "most non-null
    # wins" selection from silently picking a physiologically different measure
    # (e.g. diastolic BP, which is non-monotonic with age and a poor aging
    # marker, instead of systolic BP, which rises with age and was intended).
    datasets = {
        "hand_grip":     (DATA_ROOT / "hand_grip"     / "hand_grip.parquet",
                          ["grip_strength","hand_grip_strength","dominant_hand_grip",
                           "finger_grip_strength_left","finger_grip_strength_right",
                           "hand_grip_strength_left","hand_grip_strength_right"],
                          []),
        # NOTE ON COLUMN NAMES: HPP orders these as
        # "<posture>_blood_pressure_<systolic|diastolic>", NOT
        # "<posture>_<systolic>_blood_pressure". An earlier version of this list
        # used the latter word order, so NO primary candidate ever matched and
        # every blood-pressure result silently fell back to DIASTOLIC. Systolic
        # was present the whole time (lying_blood_pressure_systolic: 12,944
        # non-null). Matching here is exact, so the names must be exact.
        # Systolic is the intended construct: it rises monotonically with age,
        # whereas diastolic peaks near age 60 and then declines, which makes it
        # a poor and partly non-monotonic aging marker.
        "blood_pressure":(DATA_ROOT / "blood_pressure"/ "blood_pressure.parquet",
                          ["lying_blood_pressure_systolic",
                           "sitting_blood_pressure_systolic",
                           "standing_one_min_blood_pressure_systolic",
                           "standing_three_min_blood_pressure_systolic",
                           "systolic","systolic_bp","sbp","systolic_blood_pressure"],
                          ["lying_blood_pressure_diastolic","diastolic_bp"]),
        "anthropometrics":(DATA_ROOT / "anthropometrics"/ "anthropometrics.parquet",
                           ["bmi","body_mass_index","waist_circumference","waist_cm"],
                           []),
    }
    for name, (path, primary_cols, fallback_cols) in datasets.items():
        if not path.exists():
            log(f"  {name}: NOT FOUND at {path}")
            continue
        df = pd.read_parquet(path)
        if df.index.names != [None]:
            df = df.reset_index()
        pid   = next((c for c in df.columns if c in ("participant_id","study_id")), None)
        stage = next((c for c in df.columns if "research_stage" in c.lower()), None)
        if not pid or not stage:
            log(f"  {name}: missing pid/stage columns")
            continue
        df = df.rename(columns={pid: "study_id", stage: "visit"})
        # find the measurement column — prefer PRIMARY candidates (most non-null
        # among them wins); only fall back to fallback_cols, then any numeric
        # column, if no primary candidate is present at all.
        _exclude = {"study_id","visit","array_index","index","participant_id",
                    "cohort","collection_timestamp","collection_date","timezone"}
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c.lower() not in _exclude]
        primary_present  = [c for c in primary_cols  if c in df.columns]
        fallback_present = [c for c in fallback_cols if c in df.columns]
        if primary_present:
            meas_col = max(primary_present, key=lambda c: df[c].notna().sum())
            candidate_present = primary_present
        elif fallback_present:
            meas_col = max(fallback_present, key=lambda c: df[c].notna().sum())
            candidate_present = fallback_present
            log(f"    WARNING: {name} has no primary (e.g. systolic) column present -- "
                f"falling back to '{meas_col}', which measures a different construct.")
        elif numeric_cols:
            meas_col = max(numeric_cols, key=lambda c: df[c].notna().sum())
            candidate_present = []
            log(f"    WARNING: {name} has no primary or fallback candidate present -- "
                f"guessing '{meas_col}' from all numeric columns; verify this is intended.")
        else:
            meas_col = None
            candidate_present = []
        # Log all candidate non-null counts to help diagnose
        if candidate_present:
            for c in candidate_present:
                log(f"    {name} col '{c}': {df[c].notna().sum():,} non-null")
        if meas_col is None:
            log(f"  {name}: no measurement column found. Columns: {list(df.columns[:15])}")
            continue
        df[meas_col] = pd.to_numeric(df[meas_col], errors="coerce")
        # For grip: if selected column is sparse, try mean of left+right
        if name == "hand_grip" and df[meas_col].notna().sum() < 5000:
            left  = next((c for c in df.columns if "left"  in c.lower() and "grip" in c.lower()), None)
            right = next((c for c in df.columns if "right" in c.lower() and "grip" in c.lower()), None)
            if left and right:
                l = pd.to_numeric(df[left],  errors="coerce")
                r = pd.to_numeric(df[right], errors="coerce")
                composite = l.combine_first(r)   # take left; fill missing with right
                if composite.notna().sum() > df[meas_col].notna().sum():
                    df["grip_composite"] = composite
                    meas_col = "grip_composite"
                    log(f"  hand_grip: using composite left+right, non-null={composite.notna().sum():,}")
        counts = df.groupby("study_id")["visit"].nunique()
        repeat_ids = counts[counts >= 2].index
        df = df[df["study_id"].isin(repeat_ids)].copy()
        df["visit_n"] = df.groupby("study_id")["visit"].transform(
            lambda x: x.rank(method="dense").astype(int)
        )
        out[name] = df[["study_id", "visit", "visit_n", meas_col]].rename(
            columns={meas_col: "outcome"}
        )
        out[name].attrs["outcome_name"] = meas_col
        out[name].attrs["dataset_name"] = name
        log(f"  {name} ({meas_col}): {df['study_id'].nunique():,} repeat participants, col='{meas_col}'")
    return out


# ============================================================
# STEP 2: EXTRACT MICROBIOME DIVERSITY AT EACH VISIT
# ============================================================

def _load_linked_species(gut: pd.DataFrame, data_root: Path) -> pd.DataFrame | None:
    """
    HPP stores species abundances as a 'linked parquet' pointer column.
    Each row of gut has a column like 'metaphlan_abundance_species_parquet'
    containing a relative path to the actual species parquet.
    We load ONE shared species parquet (all rows point to the same file).
    """
    ptr_col = next((c for c in gut.columns if "species" in c.lower() and "parquet" in c.lower()), None)
    if ptr_col is None:
        ptr_col = next((c for c in gut.columns if "abundance" in c.lower() and "parquet" in c.lower()), None)
    if ptr_col is None:
        return None

    # Get distinct paths
    paths = gut[ptr_col].dropna().unique()
    log(f"  Linked species parquet pointer col: '{ptr_col}', distinct paths: {len(paths)}")
    if len(paths) == 0:
        return None

    frames = []
    gut_dir = data_root / "gut_microbiome"
    for rel_path in paths[:5]:  # usually 1 unique path; guard against many
        # rel_path may be like './abundance/urs.parquet' or an absolute path
        rel_path = str(rel_path)
        candidate = gut_dir / rel_path.lstrip("./")
        if not candidate.exists():
            candidate2 = Path(rel_path)
            candidate = candidate2 if candidate2.exists() else None
        if candidate is None or not Path(candidate).exists():
            log(f"  WARNING: linked parquet not found: {rel_path}")
            continue
        log(f"  Loading species parquet: {candidate}")
        sp = pd.read_parquet(candidate)
        if sp.index.names != [None]:
            sp = sp.reset_index()
        frames.append(sp)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_diversity(gut: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Shannon diversity and species richness per visit.
    HPP gut_microbiome parquet uses a 'linked parquet' format — species
    abundances are in a separate file referenced by a pointer column.
    """
    log("\n[Step 2] Computing microbiome diversity per visit...")
    log(f"  Total columns in gut parquet: {len(gut.columns)}")
    log(f"  Columns: {list(gut.columns)}")

    # ---- Try to load linked species parquet ----
    species_df = _load_linked_species(gut, DATA_ROOT)

    if species_df is not None:
        log(f"  Species parquet shape: {species_df.shape}")
        log(f"  Species parquet columns[:8]: {list(species_df.columns[:8])}")

        # Identify metadata columns in the species parquet
        sp_meta = {"participant_id","study_id","cohort","research_stage",
                   "array_index","sample_name","collection_timestamp",
                   "collection_date","timezone"}
        sp_abund_cols = [c for c in species_df.columns
                         if c not in sp_meta
                         and str(species_df[c].dtype).startswith("float")]
        log(f"  Species abundance columns: {len(sp_abund_cols)}")

        if len(sp_abund_cols) > 5:
            # Compute diversity IN the species parquet (avoid wide merge)
            # MetaPhlAn abundances may be percentages (0–100) or fractions (0–1)
            # Values may also be negative (imputed) — take abs then normalise
            abund = species_df[sp_abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
            row_sums = abund.sum(axis=1)
            # For richness: count species with positive value (before normalisation)
            species_df = species_df.copy()
            species_df["richness"] = (abund > 0).sum(axis=1).astype(float)
            # Normalise to proportions for Shannon
            row_sums_safe = row_sums.replace(0, np.nan)
            p = abund.div(row_sums_safe, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lp = np.where(p > 0, np.log(p), 0.0)
            species_df["shannon"] = -(p.values * lp).sum(axis=1)
            # rows with sum=0 get shannon=0 (not NaN)
            species_df.loc[row_sums == 0, "shannon"] = np.nan
            log(f"  Shannon  mean={species_df['shannon'].mean():.3f}  sd={species_df['shannon'].std():.3f}")
            log(f"  Richness mean={species_df['richness'].mean():.1f}  sd={species_df['richness'].std():.1f}")

            # Join on participant_id + research_stage (the true shared keys in HPP)
            # gut has: study_id (=participant_id renamed), visit (=research_stage renamed)
            # species_df has: participant_id, research_stage
            sp_pid   = next((c for c in ("participant_id","study_id") if c in species_df.columns), None)
            sp_stage = next((c for c in species_df.columns if "research_stage" in c.lower()), None)

            if sp_pid and sp_stage:
                log(f"  Joining on (study_id+visit) ↔ ({sp_pid}+{sp_stage})")
                sp_small = species_df[[sp_pid, sp_stage, "shannon", "richness"]].copy()
                sp_small = sp_small.rename(columns={sp_pid: "study_id", sp_stage: "visit"})
                # gut has 'visit' column (original research_stage value before visit_n encoding)
                left_cols = list(dict.fromkeys(["study_id","visit","visit_n"]))
                left_cols = [c for c in left_cols if c in gut.columns]
                merged = gut[left_cols].merge(sp_small, on=["study_id","visit"], how="left")
            else:
                # last-resort: join on study_id only
                log(f"  WARNING: falling back to study_id-only join")
                sp_pid = sp_pid or "participant_id"
                sp_small = species_df[[sp_pid, "shannon", "richness"]].rename(
                    columns={sp_pid: "study_id"})
                left_cols_fb = [c for c in ("study_id","visit","visit_n") if c in gut.columns]
                merged = gut[left_cols_fb].merge(sp_small, on="study_id", how="left")

            log(f"  Merged shape (pre-dedup): {merged.shape}, "
                f"non-null Shannon: {merged['shannon'].notna().sum()}")
            if merged["shannon"].notna().sum() == 0:
                log("  ERROR: join produced 0 matches. Printing sample keys for diagnosis:")
                log(f"    gut study_id sample: {gut['study_id'].head(3).tolist()}")
                log(f"    gut visit sample: {gut['visit'].head(3).tolist() if 'visit' in gut.columns else 'N/A'}")
                log(f"    species study_id sample: {sp_small['study_id'].head(3).tolist()}")
                log(f"    species visit sample: {sp_small['visit'].head(3).tolist() if 'visit' in sp_small.columns else 'N/A'}")

            # Deduplicate to ONE row per (study_id, visit_n). "visit" (raw
            # research_stage) is 1:1 with visit_n per participant by construction
            # (visit_n is a dense rank of visit), so "first" is safe and keeps the
            # raw stage available downstream for visit-alignment checks.
            agg_dict = {"shannon": "mean", "richness": "mean"}
            if "visit" in merged.columns:
                agg_dict["visit"] = "first"
            merged = (merged.groupby(["study_id","visit_n"], as_index=False)
                            .agg(agg_dict))
            log(f"  After dedup: {merged.shape}, unique pairs: {len(merged)}")
            out_cols = ["study_id","visit_n","shannon","richness"]
            if "visit" in merged.columns:
                out_cols.insert(1, "visit")
            return merged[out_cols]

    # ---- Fallback: look for diversity columns already in main parquet ----
    div_col = next((c for c in gut.columns if "shannon" in c.lower()), None)
    rich_col = next((c for c in gut.columns if "richness" in c.lower() or "n_species" in c.lower()), None)
    if div_col:
        gut = gut.copy()
        gut["shannon"] = pd.to_numeric(gut[div_col], errors="coerce")
        gut["richness"] = pd.to_numeric(gut[rich_col], errors="coerce") if rich_col else np.nan
        log(f"  Using pre-computed diversity: '{div_col}'")
        log(f"  Shannon mean={gut['shannon'].mean():.3f} sd={gut['shannon'].std():.3f}")
        return gut[["study_id","visit","visit_n","shannon","richness"]]

    log("  WARNING: no species abundance data found. Cross-lag will use NaN diversity.")
    gut = gut.copy()
    gut["shannon"] = np.nan
    gut["richness"] = np.nan
    return gut[["study_id","visit","visit_n","shannon","richness"]]


# ============================================================
# STEP 3: BUILD GUT-CLOCK AGE ACCELERATION (leak-free)
# ============================================================

def _find_collection_date_col(df: pd.DataFrame) -> str | None:
    return next((c for c in df.columns
                 if "collection_date" in c.lower() or "collection_timestamp" in c.lower()),
                None)


def _compute_visit_age(df: pd.DataFrame, birth_year_col: str = "birth_year") -> pd.Series:
    """
    Genuine per-visit chronological age, computed from each row's own
    collection date and the participant's birth year. A single static age
    reused across both visits would mechanically force the two timepoints
    apart and corrupt any within-person change analysis (see
    longitudinal_microbiome_age.py for the same argument). Returns NaN for
    rows that don't have both a parseable date and a birth year; callers
    should fall back to the static age_at_visit for those rows only.
    """
    date_col = _find_collection_date_col(df)
    if date_col is None or birth_year_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return dates.dt.year - pd.to_numeric(df[birth_year_col], errors="coerce")


def build_gut_clock_ageaccel(gut_diversity: pd.DataFrame,
                              full_gut: pd.DataFrame,
                              pop: pd.DataFrame) -> tuple:
    """
    Train a gut-derived biological age clock on SINGLE-VISIT participants only.
    Predict AgeAccel for the repeat-visit participants (both visits held out).

    Returns (DataFrame[study_id, visit_n, gut_ageaccel], clock_data dict).
    """
    log("\n[Step 3] Building leak-free gut-clock AgeAccel...")
    repeat_ids = set(gut_diversity["study_id"])

    # ---- Load species parquet (same pointer mechanism as compute_diversity) ----
    species_df = _load_linked_species(full_gut, DATA_ROOT)

    sp_meta = {"participant_id", "study_id", "cohort", "research_stage",
               "array_index", "sample_name", "collection_timestamp",
               "collection_date", "timezone", "visit_n"}

    if species_df is not None:
        sp_abund_cols = [c for c in species_df.columns
                         if c not in sp_meta
                         and str(species_df[c].dtype).startswith("float")]
        # prevalence filter: keep species present in >5% of samples
        abund_mat = species_df[sp_abund_cols].apply(
            pd.to_numeric, errors="coerce").fillna(0).abs()
        prevalence = (abund_mat > 0).mean()
        sp_abund_cols = [c for c in sp_abund_cols if prevalence[c] > 0.05]
        log(f"  {len(sp_abund_cols)} species pass prevalence >5%")
        feature_mode = "species"
    else:
        sp_abund_cols = []
        feature_mode = "diversity_only"
        log("  Species parquet not found — falling back to diversity only.")

    # ---- Flatten species_df to (study_id, research_stage, species...) ----
    if species_df is not None:
        if species_df.index.names != [None]:
            species_df = species_df.reset_index()
        pid_sp = next((c for c in species_df.columns
                       if c in ("participant_id", "study_id")), None)
        stg_sp = next((c for c in species_df.columns
                       if "research_stage" in c.lower()), None)
        if pid_sp and stg_sp:
            sp_slim = species_df[[pid_sp, stg_sp] + sp_abund_cols].copy()
            sp_slim = sp_slim.rename(columns={pid_sp: "study_id",
                                              stg_sp: "research_stage"})
        else:
            sp_slim = None
            sp_abund_cols = []
            feature_mode = "diversity_only"
    else:
        sp_slim = None

    # ---- TRAINING frame: single-visit participants from full_gut ----
    # full_gut has all participants; repeat_ids are those with 2+ visits in gut_diversity
    pid_fg = next((c for c in full_gut.columns
                   if c in ("participant_id", "study_id")), "study_id")
    stg_fg = next((c for c in full_gut.columns
                   if "research_stage" in c.lower()), None)
    fg = full_gut.copy()
    if pid_fg != "study_id":
        fg = fg.rename(columns={pid_fg: "study_id"})
    if stg_fg and stg_fg != "research_stage":
        fg = fg.rename(columns={stg_fg: "research_stage"})

    # keep only single-visit participants (baseline visit only)
    single_ids = set(fg["study_id"].unique()) - repeat_ids
    if stg_fg:
        train_base = fg[fg["study_id"].isin(single_ids)].copy()
        # take one row per participant (lowest research_stage)
        train_base = (train_base.sort_values("research_stage")
                                .groupby("study_id", as_index=False).first())
    else:
        train_base = fg[fg["study_id"].isin(single_ids)].copy()

    train_base = train_base.merge(pop[["study_id", "age_at_visit", "birth_year", "sex_encoded"]],
                                  on="study_id", how="left")
    # Prefer true age at this participant's own (single) visit over the
    # static "2020 - birth_year" placeholder; single-visit participants
    # enrolled in a different year would otherwise train the clock on a
    # mislabelled age. Falls back to the static value where no collection
    # date is available.
    visit_age_train = _compute_visit_age(train_base)
    n_train_recomputed = int(visit_age_train.notna().sum())
    train_base["age_at_visit"] = visit_age_train.where(
        visit_age_train.notna(), train_base["age_at_visit"])
    log(f"  Training frame: per-visit age computed for "
        f"{n_train_recomputed:,}/{len(train_base):,} rows "
        f"(remainder use the static age_at_visit fallback).")

    # attach species abundances to training frame
    if sp_slim is not None and "research_stage" in train_base.columns:
        train_base = train_base.merge(sp_slim, on=["study_id", "research_stage"],
                                      how="left")
    elif sp_slim is not None:
        # no research_stage in full_gut — merge on study_id only (single-visit: 1 row each)
        sp_sv = sp_slim.groupby("study_id", as_index=False).first()
        train_base = train_base.merge(sp_sv.drop(columns=["research_stage"]),
                                      on="study_id", how="left")

    # attach shannon/richness from gut_diversity for those that overlap
    div_sv = gut_diversity[gut_diversity["study_id"].isin(single_ids)][
        ["study_id", "shannon", "richness"]].groupby("study_id", as_index=False).first()
    train_base = train_base.merge(div_sv, on="study_id", how="left")

    train_df = train_base.dropna(subset=["age_at_visit"])

    # [CHECK-1] Leak-free disjointness. The paper's core methodological claim
    # is that no repeat-visit participant ever enters clock training. This
    # holds by construction (train_df is built from single_ids = all_ids -
    # repeat_ids), but the training population changed size (5,188 -> 6,807)
    # with the birth-year backfill, so verify explicitly rather than trust the
    # construction silently.
    _overlap = set(train_df["study_id"]) & repeat_ids
    if len(_overlap) != 0:
        log(f"  [CHECK-1] FAIL: {len(_overlap)} participants leaked into "
            f"training! Sample IDs: {sorted(_overlap)[:10]}")
    else:
        log(f"  [CHECK-1] PASS: training ({len(train_df):,}) and longitudinal "
            f"stratum ({len(repeat_ids):,}) share 0 participants. Leak-free "
            f"discipline holds after the birth-year backfill.")

    # build feature matrix
    if sp_abund_cols and all(c in train_df.columns for c in sp_abund_cols):
        X_train = train_df[sp_abund_cols].fillna(0).astype(float).values
        if "shannon" in train_df.columns:
            X_train = np.hstack([X_train,
                                 train_df[["shannon", "richness"]].fillna(0).values])
    else:
        if "shannon" in train_df.columns:
            X_train = train_df[["shannon", "richness"]].fillna(0).values
        else:
            X_train = np.zeros((len(train_df), 2))
        sp_abund_cols = []
        feature_mode = "diversity_only"

    y_train = train_df["age_at_visit"].astype(float).values
    # [FIG2] carried through purely for the 4-panel clock-validation figure's
    # sex-stratified MAE panel -- not used in any model fit, so it cannot leak
    # into or otherwise affect the clock itself.
    sex_train = (train_df["sex_encoded"].values if "sex_encoded" in train_df.columns
                else np.full(len(train_df), np.nan))
    log(f"  Training clock on n={len(y_train):,} single-visit participants (mode={feature_mode})")

    _empty_clock = {"y_train": np.array([]), "y_pred_cv": np.array([]), "r_cv": np.nan,
                    "sex_train": np.array([])}
    if len(y_train) < CLOCK_CV_FOLDS * 2:
        log(f"  WARNING: only {len(y_train)} training samples — skipping clock, returning NaN AgeAccel.")
        gut_div_copy = gut_diversity[["study_id", "visit_n"]].copy()
        gut_div_copy["gut_ageaccel"] = np.nan
        return gut_div_copy, _empty_clock

    try:
        from lightgbm import LGBMRegressor
        clock = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=RANDOM_SEED, verbose=-1)
    except ImportError:
        clock = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_SEED)

    y_pred_cv = cross_val_predict(clock, X_train, y_train, cv=CLOCK_CV_FOLDS)
    r_cv = pearsonr(y_train, y_pred_cv)[0]
    mae_cv = float(np.mean(np.abs(y_train - y_pred_cv)))
    log(f"  Clock CV r = {r_cv:.3f} (R² = {r2_score(y_train, y_pred_cv):.3f}, "
        f"MAE = {mae_cv:.2f} yr)")
    clock.fit(X_train, y_train)

    # ---- PREDICTION frame: repeat-visit participants from gut_diversity ----
    repeat_df = gut_diversity[gut_diversity["study_id"].isin(repeat_ids)].copy()
    repeat_df = repeat_df.merge(pop[["study_id", "age_at_visit", "birth_year"]],
                                on="study_id", how="left")

    # gut_diversity (the source of repeat_df) does not carry a raw
    # collection-date column, so attach it from the full raw frame (fg), keyed
    # on the real visit value -- the same key used for the species merge below
    # -- then compute a genuine per-visit age instead of reusing the static
    # age_at_visit for both visits.
    #
    # fg's stage column can be named several things depending on the caller
    # (main() already renames it to "visit" before passing fg in as
    # full_gut), so look it up defensively rather than assuming a literal
    # "research_stage" -- an incorrect assumption here silently falls back to
    # the static age for every repeat-visit row with no warning.
    date_col_fg = _find_collection_date_col(fg)
    fg_stage_col = next((c for c in fg.columns
                         if c == "visit" or "research_stage" in c.lower()), None)
    if date_col_fg is not None and fg_stage_col is not None and "visit" in repeat_df.columns:
        date_slim = (fg[["study_id", fg_stage_col, date_col_fg]]
                    .drop_duplicates(subset=["study_id", fg_stage_col])
                    .rename(columns={fg_stage_col: "visit"}))
        repeat_df = repeat_df.merge(date_slim, on=["study_id", "visit"], how="left")
        visit_age_rep = _compute_visit_age(repeat_df)
        n_rep_recomputed = int(visit_age_rep.notna().sum())
        repeat_df["age_at_visit"] = visit_age_rep.where(
            visit_age_rep.notna(), repeat_df["age_at_visit"])
        log(f"  Repeat-visit frame: per-visit age computed for "
            f"{n_rep_recomputed:,}/{len(repeat_df):,} rows "
            f"(remainder use the static age_at_visit fallback).")
    else:
        log(f"  WARNING: date_col={date_col_fg}, stage_col={fg_stage_col} on the "
            f"raw gut frame -- repeat-visit AgeAccel will use the static age_at_visit "
            f"(same value at both visits) throughout.")

    # Join species on the REAL visit/research_stage value, exactly as
    # compute_diversity() already does, rather than matching on a literal
    # "research_stage" column -- repeat_df never has one, only "visit" (the
    # same raw stage value under a different name). Matching on the wrong
    # column name falls through to a rank-based fallback that independently
    # re-derives visit_n from the species table's own research_stage values;
    # two independent rank reconstructions are not guaranteed to agree per
    # participant, which is how species data fails to link for repeat-visit
    # rows.
    if sp_slim is not None and "visit" in repeat_df.columns:
        sp_join = sp_slim.rename(columns={"research_stage": "visit"})
        n_before = len(repeat_df)
        repeat_df = repeat_df.merge(sp_join, on=["study_id", "visit"], how="left",
                                    indicator="_merge_flag")
        n_matched = int((repeat_df["_merge_flag"] == "both").sum())
        log(f"  species join on (study_id, visit): {n_matched:,}/{n_before:,} rows matched")
        repeat_df = repeat_df.drop(columns=["_merge_flag"])
    elif sp_slim is not None:
        log("  WARNING: repeat_df has no raw 'visit' (research_stage) value -- "
            "falling back to rank-reconstructed visit_n, which is NOT guaranteed "
            "to align with the species table's own visit ordering per participant.")
        sp_vn = sp_slim.copy()
        sp_vn["visit_n"] = (sp_vn.groupby("study_id")["research_stage"]
                            .transform(lambda x: x.rank(method="dense").astype(int)))
        sp_vn = sp_vn.drop(columns=["research_stage"])
        sp_vn["visit_n"] = sp_vn["visit_n"].astype(int)
        repeat_df["visit_n"] = repeat_df["visit_n"].astype(int)
        repeat_df = repeat_df.merge(sp_vn, on=["study_id", "visit_n"], how="left")

    # deduplicate any columns that doubled up after merge (e.g. shannon_x/shannon_y)
    repeat_df = repeat_df.loc[:, ~repeat_df.columns.duplicated(keep="first")]

    if sp_abund_cols and all(c in repeat_df.columns for c in sp_abund_cols):
        # A row only carries a genuine microbiome sample if the merge
        # above actually matched -- i.e. at least one abundance column is
        # non-null before filling. Rows where the merge failed get an all-NaN
        # abundance block; predicting on a zero-filled vector for those rows
        # would fabricate an AgeAccel value that does not correspond to any
        # real sample (Rule 4: never fabricate a number). Those rows are
        # excluded from AgeAccel below instead of silently zero-filled.
        species_matched = repeat_df[sp_abund_cols].notna().any(axis=1).values
        X_rep = repeat_df[sp_abund_cols].fillna(0).astype(float).values
        X_rep = np.hstack([X_rep,
                           repeat_df[["shannon", "richness"]].fillna(0).values])
        n_unmatched = int((~species_matched).sum())
        if n_unmatched:
            log(f"  WARNING: {n_unmatched} repeat-visit rows have no matching "
                f"species record for their visit and will be EXCLUDED from "
                f"AgeAccel (not zero-filled into a fake prediction).")
    else:
        X_rep = repeat_df[["shannon", "richness"]].fillna(0).values
        species_matched = np.ones(len(repeat_df), dtype=bool)  # diversity-only mode: nothing to fail

    bio_age_pred = clock.predict(X_rep)
    bio_age_pred[~species_matched] = np.nan

    repeat_df = repeat_df.reset_index(drop=True)
    repeat_df["bio_age_pred"] = bio_age_pred

    # AgeAccel = residuals of bio_age ~ chron_age on the held-out repeat population
    valid = repeat_df.dropna(subset=["age_at_visit", "bio_age_pred"])
    y_ols = valid["bio_age_pred"].astype(float).values
    X_ols = valid["age_at_visit"].astype(float).values
    ols = sm.OLS(y_ols, sm.add_constant(X_ols)).fit()
    repeat_df["gut_ageaccel"] = np.nan
    # use .values to avoid RangeIndex vs pandas-index misalignment
    resid = np.asarray(ols.resid)
    repeat_df.loc[valid.index, "gut_ageaccel"] = resid

    result = repeat_df[["study_id", "visit_n", "gut_ageaccel"]].drop_duplicates(
        subset=["study_id", "visit_n"])
    n_valid = result["gut_ageaccel"].notna().sum()
    both_visits_mask = (result.dropna(subset=["gut_ageaccel"])
                        .groupby("study_id")["visit_n"]
                        .apply(lambda v: {1, 2}.issubset(set(v))))
    both_visits_ids = sorted(both_visits_mask[both_visits_mask].index)
    both_visits = len(both_visits_ids)
    log(f"  AgeAccel predictions: {n_valid:,} / {len(result):,} rows")
    log(f"  Participants with AgeAccel at BOTH visits: {both_visits:,}")

    # [CHECK-2a] Save the leak-free both-visits participant list so it can be
    # diffed against results_longitudinal/paired_visits.csv (the reliability
    # pipeline's own both-visits list) to explain any remaining gap between
    # the two pipelines' counts.
    pd.DataFrame({"study_id": both_visits_ids}).to_csv(
        RESULTS_DIR / "leak_free_ageaccel_both_visits.csv", index=False)
    log(f"  [CHECK-2a] wrote {RESULTS_DIR/'leak_free_ageaccel_both_visits.csv'} "
        f"({both_visits:,} study_id)")

    # [CHECK-2b] ICC of the corrected leak-free AgeAccel, using the SAME
    # method (random-intercept mixed model, ICC(1) = var_between /
    # (var_between + var_within)) as longitudinal_microbiome_age.py's
    # compute_icc(), which produced the published 0.483 -- so this is a direct,
    # apples-to-apples comparison, not a different formula that happens to
    # look similar.
    icc_input = result[result["study_id"].isin(both_visits_ids)][
        ["study_id", "gut_ageaccel"]].dropna()
    try:
        import statsmodels.formula.api as smf
        _md = smf.mixedlm("gut_ageaccel ~ 1", icc_input,
                          groups=icc_input["study_id"]).fit()
        _var_between = float(_md.cov_re.iloc[0, 0])
        _var_within = float(_md.scale)
        icc_leak_free = _var_between / (_var_between + _var_within)
        log(f"  [CHECK-2b] Leak-free AgeAccel ICC (same method as the "
            f"published 0.483): {icc_leak_free:.3f}  "
            f"(between-SD={np.sqrt(_var_between):.3f}, "
            f"within-SD={np.sqrt(_var_within):.3f}, n={both_visits:,})")
        if abs(icc_leak_free - 0.483) < 0.02:
            log(f"  [CHECK-2b] Leak-free and reliability-pipeline AgeAccel "
                f"ICCs agree closely -- the two pipelines now measure "
                f"essentially the same quantity.")
        else:
            log(f"  [CHECK-2b] ICCs differ by {abs(icc_leak_free-0.483):.3f} "
                f"-- treat the two AgeAccel measures as distinct until this "
                f"is understood.")
    except Exception as e:
        log(f"  [CHECK-2b] ICC computation failed: {e}")

    clock_data = {"y_train": y_train, "y_pred_cv": y_pred_cv, "r_cv": r_cv,
                  "mae_cv": mae_cv, "sex_train": sex_train}
    return result, clock_data


# ============================================================
# STEP 4: CROSS-LAGGED PANEL MODEL
# ============================================================

def make_wide(df_long: pd.DataFrame, value_col: str,
              id_col="study_id", visit_col="visit_n") -> pd.DataFrame:
    """
    Pivot a long dataframe (one row per person×visit) to wide:
      value_t1, value_t2  (visits 1 and 2 only).
    Deduplicates by taking the mean per (id, visit) before pivoting.
    """
    # Guard: ensure one row per (id, visit)
    df_long = (df_long.groupby([id_col, visit_col], as_index=False)[[value_col]].mean())
    t1 = df_long[df_long[visit_col] == 1].set_index(id_col)[[value_col]].rename(
        columns={value_col: f"{value_col}_t1"})
    t2 = df_long[df_long[visit_col] == 2].set_index(id_col)[[value_col]].rename(
        columns={value_col: f"{value_col}_t2"})
    return t1.join(t2, how="inner")


def cross_lag_ols(wide: pd.DataFrame, x_col: str, y_col: str,
                  covariates_df: pd.DataFrame,
                  covariate_set: list[str] = PRIMARY_COVARIATES) -> dict:
    """
    OLS cross-lagged regression:
        y_t2 ~ y_t1 + x_t1 + covariates

    covariate_set defaults to PRIMARY_COVARIATES (age + sex, matching Methods
    meth:crosslag). Pass BASE_COVARIATES explicitly for the fully-adjusted
    sensitivity analysis.

    Returns dict with beta, se, t, p, n, r2, r2_full, std_beta.
    """
    # Use left join so participants without covariate data aren't silently dropped
    merged = wide.join(covariates_df, how="left").dropna(
        subset=[f"{x_col}_t1", f"{x_col}_t2", f"{y_col}_t1", f"{y_col}_t2"]
    )
    # Only use covariates that actually exist and have non-trivial data
    cov_cols = [c for c in covariate_set
                if c in merged.columns and merged[c].notna().sum() > 100]
    if cov_cols:
        merged = merged.dropna(subset=cov_cols)
    n = len(merged)
    if n < MIN_OVERLAP:
        return {"n": n, "p": np.nan, "beta": np.nan, "se": np.nan,
                "t": np.nan, "r2": np.nan, "std_beta": np.nan, "ci95": (np.nan, np.nan)}

    # Force all model columns to float (sociodemographics may contain object dtypes)
    model_cols = [f"{x_col}_t1", f"{x_col}_t2", f"{y_col}_t1", f"{y_col}_t2"] + cov_cols
    std = merged.copy()
    for c in model_cols:
        std[c] = pd.to_numeric(std[c], errors="coerce")
    std = std.dropna(subset=model_cols)
    n = len(std)
    if n < MIN_OVERLAP:
        return {"n": n, "p": np.nan, "beta": np.nan, "se": np.nan,
                "t": np.nan, "r2": np.nan, "std_beta": np.nan, "ci95": (np.nan, np.nan)}

    # Standardise all variables for comparable effect sizes
    for c in model_cols:
        mu, sd_val = std[c].mean(), std[c].std()
        if sd_val > 0:
            std[c] = (std[c] - mu) / sd_val

    X = sm.add_constant(std[[f"{y_col}_t1", f"{x_col}_t1"] + cov_cols].astype(float))
    y = std[f"{y_col}_t2"].astype(float)
    ols = sm.OLS(y, X).fit()

    key = f"{x_col}_t1"
    beta = ols.params[key]
    se   = ols.bse[key]
    t    = ols.tvalues[key]
    p    = ols.pvalues[key]
    ci   = ols.conf_int().loc[key].tolist()

    # Positive control: the autoregressive coefficient (y_t1 -> y_t2) is
    # expected to be large and highly significant for any real, measured trait
    # -- it is the strongest effect the model should be able to detect. Logging
    # it lets us confirm a cross-lag null reflects an absence of cross-lag
    # signal specifically, not a model/pipeline that fails to detect anything.
    auto_key = f"{y_col}_t1"
    beta_auto = ols.params[auto_key]
    p_auto    = ols.pvalues[auto_key]

    return {"beta": beta, "se": se, "t": t, "p": p, "n": n,
            "beta_auto": beta_auto, "p_auto": p_auto,
            "r2": ols.rsquared, "std_beta": beta,
            "ci95": tuple(ci)}


def run_cross_lagged_panel(gut_div: pd.DataFrame,
                           gut_ageaccel: pd.DataFrame,
                           outcomes: dict[str, pd.DataFrame],
                           pop: pd.DataFrame,
                           covariate_set: list[str] = PRIMARY_COVARIATES,
                           spec_label: str = "primary") -> pd.DataFrame:
    """
    For every (microbiome_exposure × functional_outcome) pair, run both directions:
        A: exposure(t1) → outcome(t2) | outcome(t1)
        B: outcome(t1) → exposure(t2) | exposure(t1)

    covariate_set/spec_label let this run twice with different adjustment sets
    -- PRIMARY_COVARIATES (age+sex, matching Methods) as the main analysis, and
    BASE_COVARIATES (+education+income) as a labeled sensitivity analysis. Both
    calls are tagged via the "covariate_spec" column in the returned frame.
    """
    log(f"\n[Step 4] Cross-lagged panel models... (covariate_spec={spec_label}: "
        f"{covariate_set})")

    exposures = {
        "shannon":      gut_div[["study_id","visit_n","shannon"]],
        "richness":     gut_div[["study_id","visit_n","richness"]],
        "gut_ageaccel": gut_ageaccel[["study_id","visit_n","gut_ageaccel"]],
    }

    covariates_df = pop.set_index("study_id")[
        [c for c in covariate_set if c in pop.columns]
    ]

    rows = []
    for exp_name, exp_df in exposures.items():
        exp_wide = make_wide(exp_df, exp_name)

        for ds_name, out_df in outcomes.items():
            out_wide = make_wide(out_df, "outcome")
            out_wide = out_wide.rename(columns={"outcome_t1": "y_t1", "outcome_t2": "y_t2"})
            outcome_label = out_df.attrs.get("outcome_name", ds_name)

            combined = exp_wide.join(out_wide, how="inner")
            # combined has columns: {exp}_t1, {exp}_t2, y_t1, y_t2

            # Path A: exposure(t1) → outcome(t2) | outcome(t1)
            # Rename so cross_lag_ols sees x_col=exp_name, y_col="outcome"
            wide_A = combined.rename(columns={"y_t1": "outcome_t1", "y_t2": "outcome_t2"})
            rA = cross_lag_ols(wide_A, x_col=exp_name, y_col="outcome",
                               covariates_df=covariates_df, covariate_set=covariate_set)
            rA.update({"direction": "A", "path": f"{exp_name}(t1) → {outcome_label}(t2)",
                       "exposure": exp_name, "outcome": outcome_label, "dataset": ds_name})

            # Path B: outcome(t1) → exposure(t2) | exposure(t1)
            wide_B = combined.rename(columns={"y_t1": "outcome_t1", "y_t2": "outcome_t2"})
            rB = cross_lag_ols(wide_B, x_col="outcome", y_col=exp_name,
                               covariates_df=covariates_df, covariate_set=covariate_set)
            rB.update({"direction": "B", "path": f"{outcome_label}(t1) → {exp_name}(t2)",
                       "exposure": outcome_label, "outcome": exp_name, "dataset": ds_name})

            pair_key = f"{exp_name}__x__{ds_name}"
            rA["pair_key"] = pair_key
            rB["pair_key"] = pair_key
            rA["microbiome_var"] = exp_name   # always the microbiome metric
            rB["microbiome_var"] = exp_name
            rA["covariate_spec"] = spec_label
            rB["covariate_spec"] = spec_label
            rows.extend([rA, rB])
            # Positive control alongside the cross-lag result: beta_auto/p_auto
            # (Path A) is the outcome's own y_t1->y_t2 autoregressive effect --
            # expect it large & p<<0.05. If it is NOT, the model isn't detecting
            # real signal at all, and a cross-lag null there is uninformative.
            auto_b = rA.get("beta_auto", float("nan"))
            auto_p = rA.get("p_auto", float("nan"))
            log(f"  {exp_name} × {outcome_label}: A p={rA['p']:.4f}, B p={rB['p']:.4f}  "
                f"(n={rA['n']})  [positive control: outcome autoregression "
                f"beta={auto_b:.3f}, p={auto_p:.2e}]")

    results = pd.DataFrame(rows)
    valid_p = results["p"].notna()
    if valid_p.sum() > 0:
        results.loc[valid_p, "q_fdr"] = bh_fdr(results.loc[valid_p, "p"].values)
    else:
        results["q_fdr"] = np.nan
    results["significant"] = results["q_fdr"] < ALPHA_FDR

    return results


# ============================================================
# STEP 5: MATCHED-PAIR CAUSAL DIRECTION TEST
# ============================================================

def causal_direction_test(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each (exposure × outcome) pair, compare Path A vs Path B:
      - Dominant direction = whichever has lower p-value
      - Report difference in standardised betas and whether either/both are significant
    """
    log("\n[Step 5] Causal direction summary...")
    pairs = results.groupby("pair_key")

    direction_rows = []
    for pair_key, grp in pairs:
        exp, ds = pair_key.split("__x__")
        out = grp[grp["direction"] == "A"]["outcome"].iloc[0] if (grp["direction"] == "A").any() else ""
        if len(grp) < 2:
            continue
        pathA = grp[grp["direction"] == "A"].iloc[0] if (grp["direction"] == "A").any() else None
        pathB = grp[grp["direction"] == "B"].iloc[0] if (grp["direction"] == "B").any() else None
        if pathA is None or pathB is None:
            continue
        sigA = pathA["significant"]
        sigB = pathB["significant"]
        if sigA and not sigB:
            verdict = "MICROBIOME → AGING"
        elif sigB and not sigA:
            verdict = "AGING → MICROBIOME"
        elif sigA and sigB:
            verdict = "BIDIRECTIONAL"
        else:
            verdict = "no significant effect"

        direction_rows.append({
            "exposure": exp, "outcome": out, "dataset": ds,
            "beta_A": pathA["beta"], "p_A": pathA["p"], "q_A": pathA["q_fdr"],
            "beta_B": pathB["beta"], "p_B": pathB["p"], "q_B": pathB["q_fdr"],
            "n": pathA["n"],
            "verdict": verdict,
        })
        bA = pathA['beta']; pA = pathA['p']
        bB = pathB['beta']; pB = pathB['p']
        log(f"  {exp} × {out}:  {verdict}  "
            f"(βA={bA:.3f} p={pA:.4f}, βB={bB:.3f} p={pB:.4f})"
            if not (np.isnan(bA) or np.isnan(bB)) else
            f"  {exp} × {out}:  {verdict}  (insufficient n)")

    return pd.DataFrame(direction_rows)


# ============================================================
# STEP 6: INCIDENT DISEASE ANALYSIS
# ============================================================

def load_incident_diseases(pop: pd.DataFrame,
                            gut_div_t1: pd.DataFrame,
                            covariate_set: list[str] = PRIMARY_COVARIATES
                            ) -> pd.DataFrame | None:
    """
    Using curated_phenotypes that change across visits:
    - Disease at t1 = prevalent (exclude)
    - Disease at t2 but NOT t1 = INCIDENT
    Test: gut microbiome at t1 predicts incident disease at t2.

    Defaults to PRIMARY_COVARIATES (age+sex), matching Methods, rather than
    the full BASE_COVARIATES set (age+sex+education+income) -- requiring the
    sparser education/income columns via dropna over-adjusts and shrinks the
    incident-disease sample unnecessarily.

    Requires a genuine events-per-variable (EPV) of >=10 and excludes any
    phenotype that doesn't meet it, rather than tabulating and flagging it:
    with shannon+richness+covariates+const as the parameter count, EPV can
    run well under 10 (the paper's own fragility floor used elsewhere,
    Table E8) for phenotypes with few incident events, and reporting a
    coefficient anyway would understate how unstable it is.
    """
    log("\n[Step 6] Incident disease analysis...")
    pheno_dir = DATA_ROOT / "curated_phenotypes"
    if not pheno_dir.exists():
        log("  curated_phenotypes directory not found.")
        return None

    # Find parquets with confirmed value-change
    changed_phenotypes = []
    for f in sorted(pheno_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            if df.index.names != [None]:
                df = df.reset_index()
            pid   = next((c for c in df.columns if c in ("participant_id","study_id")), None)
            stage = next((c for c in df.columns if "research_stage" in c.lower()), None)
            if not pid or not stage:
                continue
            df = df.rename(columns={pid: "study_id", stage: "visit"})
            df["visit_n"] = df.groupby("study_id")["visit"].transform(
                lambda x: x.rank(method="dense").astype(int)
            )
            pheno_col = next((c for c in df.columns
                              if c not in ("study_id","visit","visit_n")
                              and "curated" in c.lower()), None)
            if pheno_col is None:
                pheno_col = next((c for c in df.columns
                                  if c not in ("study_id","visit","visit_n")), None)
            if pheno_col is None:
                continue
            # Check for value change
            repeat = df[df.groupby("study_id")["visit_n"].transform("max") >= 2]
            if repeat.empty:
                continue
            changed = repeat.groupby("study_id")[pheno_col].nunique()
            n_changed = (changed >= 2).sum()
            if n_changed < 50:   # too few incident cases
                continue
            changed_phenotypes.append((f.name, df, pheno_col, n_changed))
            log(f"  {f.name}: {n_changed} changed (potential incident cases)")
        except Exception as e:
            log(f"  {f.name}: ERROR {e}")

    if not changed_phenotypes:
        log("  No phenotypes with sufficient incident cases found.")
        return None

    # For each phenotype: build incident case flag and test gut microbiome(t1) → incident(t2)
    incident_rows = []
    cov_cols = [c for c in covariate_set if c in pop.columns]
    n_params = 2 + len(cov_cols) + 1  # shannon + richness + covariates + const

    for fname, df, pheno_col, n_changed in changed_phenotypes[:10]:  # cap at 10 phenotypes
        try:
            t1 = df[df["visit_n"] == 1][["study_id", pheno_col]].rename(
                columns={pheno_col: "pheno_t1"})
            t2 = df[df["visit_n"] == 2][["study_id", pheno_col]].rename(
                columns={pheno_col: "pheno_t2"})
            wide = t1.merge(t2, on="study_id", how="inner")

            # Incident = disease at t2 but not t1
            # Try binary: positive value = disease present
            try:
                wide["pheno_t1_bin"] = (pd.to_numeric(wide["pheno_t1"], errors="coerce") > 0)
                wide["pheno_t2_bin"] = (pd.to_numeric(wide["pheno_t2"], errors="coerce") > 0)
            except Exception:
                continue
            wide["incident"] = (~wide["pheno_t1_bin"]) & wide["pheno_t2_bin"]
            n_incident = wide["incident"].sum()
            if n_incident < 30:
                continue

            # Merge with gut diversity at t1
            gut_t1 = gut_div_t1[["study_id","shannon","richness"]]
            merged = wide.merge(gut_t1, on="study_id", how="inner")
            merged = merged.merge(pop[["study_id"] + cov_cols], on="study_id", how="inner")
            merged = merged.dropna(subset=["shannon","richness"] + cov_cols + ["incident"])

            n = len(merged)
            if n < MIN_OVERLAP:
                continue

            n_incident_final = int(merged["incident"].sum())
            epv = n_incident_final / n_params
            if epv < 10:
                log(f"    {fname}: n_incident={n_incident_final} EPV={epv:.1f} "
                    f"(<10, {n_params} params) -- EXCLUDED as fragile, not tabulated")
                continue

            # Logistic regression: incident ~ shannon + covariates
            import statsmodels.discrete.discrete_model as disc
            X = sm.add_constant(merged[["shannon","richness"] + cov_cols])
            logit = sm.Logit(merged["incident"].astype(int), X)
            try:
                fit = logit.fit(disp=0, method="bfgs")
                or_shannon = np.exp(fit.params["shannon"])
                p_shannon  = fit.pvalues["shannon"]
                ci = np.exp(fit.conf_int().loc["shannon"].tolist())
                incident_rows.append({
                    "phenotype": fname.replace(".parquet",""),
                    "n_total": n,
                    "n_incident": n_incident_final,
                    "epv": round(epv, 1),
                    "OR_shannon": or_shannon,
                    "CI_lo": ci[0], "CI_hi": ci[1],
                    "p_shannon": p_shannon,
                })
                log(f"    {fname}: n_incident={n_incident_final}, EPV={epv:.1f}, "
                    f"OR={or_shannon:.3f}, p={p_shannon:.4f}")
            except Exception as e2:
                log(f"    {fname}: logit failed: {e2}")
        except Exception as e:
            log(f"  {fname}: {e}")

    if not incident_rows:
        return None

    incident_df = pd.DataFrame(incident_rows)
    incident_df["q_fdr"] = bh_fdr(incident_df["p_shannon"].values)
    incident_df["significant"] = incident_df["q_fdr"] < ALPHA_FDR
    return incident_df


# ============================================================
# STEP 7: VISUALISATION (aesthetic rewrite)
# ============================================================

import colorsys
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Polygon as MPoly


def _hsv_shift(hex_color: str, ds: float = 0.0, dv: float = 0.0) -> tuple:
    r, g, b = to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, np.clip(s + ds, 0, 1), np.clip(v + dv, 0, 1))


def _draw_cell_3d(ax, j: int, i: int, w: float = 1.0, h: float = 1.0) -> None:
    """Pseudo-3D bevel on a heatmap cell: bright top/left edge, dark bottom/right edge."""
    hw, hh = w * 0.5, h * 0.5
    # top-left bright bevel (L-shape)
    ax.plot([j - hw, j - hw, j + hw],
            [i + hh, i - hh, i - hh],
            color="white", lw=2.2, alpha=0.60, zorder=4,
            solid_capstyle="butt", solid_joinstyle="miter")
    # bottom-right dark bevel (L-shape, inverted)
    ax.plot([j + hw, j + hw, j - hw],
            [i - hh, i + hh, i + hh],
            color=_INK, lw=2.2, alpha=0.30, zorder=4,
            solid_capstyle="butt", solid_joinstyle="miter")


def _bar3d(ax, x: float, y0: float, height: float,
           w: float = 0.55, depth: float = 0.18, color: str = _PATH_A) -> None:
    """Draw one pseudo-3D bar: front face + lighter top + darker right side."""
    top_c  = _hsv_shift(color, ds=-0.30, dv=+0.35)   # lighter top face
    side_c = _hsv_shift(color, ds=+0.10, dv=-0.32)   # darker right face
    x0, x1, y1 = x - w / 2, x + w / 2, y0 + height
    dx, dy = depth, depth * 0.60                       # perspective offset

    # draw in back-to-front order
    side  = MPoly([[x1, y0], [x1+dx, y0+dy], [x1+dx, y1+dy], [x1, y1]],
                  closed=True, facecolor=side_c, edgecolor=_INK, lw=0.7, zorder=3)
    top   = MPoly([[x0, y1], [x1, y1], [x1+dx, y1+dy], [x0+dx, y1+dy]],
                  closed=True, facecolor=top_c, edgecolor=_INK, lw=0.7, zorder=4)
    front = MPoly([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                  closed=True, facecolor=color, edgecolor=_INK, lw=0.7, zorder=5)
    for p in (side, top, front):
        ax.add_patch(p)


# ── short display labels (covers dataset names AND raw column names) ──────
_SHORT = {
    # microbiome metrics
    "shannon":                        "Shannon",
    "richness":                       "Richness",
    "gut_ageaccel":                   "Gut AgeAccel",
    # outcome dataset names
    "hand_grip":                      "Hand grip",
    "blood_pressure":                 "Blood pressure",
    "anthropometrics":                "Anthropometrics",
    # raw column names that appear as 'outcome' in direction_df / heatmap Path B
    "hand_grip_strength_right":       "Hand grip",
    "hand_grip_strength_left":        "Hand grip",
    "grip_composite":                 "Hand grip",
    "lying_blood_pressure_diastolic": "Diastolic BP",
    "systolic_blood_pressure":        "Systolic BP",
    "waist_circumference":            "Waist circ.",
    "bmi":                            "BMI",
    "body_mass_index":                "BMI",
}

def _sl(key: str) -> str:
    """Return a short display label for any key (case/strip-insensitive fallback)."""
    if key in _SHORT:
        return _SHORT[key]
    # try stripped + lowercased
    k2 = key.strip().lower()
    for sk, sv in _SHORT.items():
        if sk.lower() == k2:
            return sv
    # partial-match: return first dict value whose key is a substring of key
    for sk, sv in _SHORT.items():
        if sk in key or key in sk:
            return sv
    return key.replace("_", " ")

_EXPOSURE_LABELS = {k: _sl(k) for k in ["shannon", "richness", "gut_ageaccel"]}
_OUTCOME_LABELS  = {k: _sl(k) for k in ["hand_grip", "blood_pressure", "anthropometrics"]}


def plot_cross_lag_heatmap(results: pd.DataFrame, out_dir: Path) -> None:
    """Beta-coefficient heatmap with pseudo-3D cell emboss and all 3 microbiome rows."""
    log("\n[Step 7] Plotting cross-lag heatmap...")
    try:
        from matplotlib.colors import LinearSegmentedColormap
        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5.2),
            gridspec_kw={"wspace": 0.50, "left": 0.13, "right": 0.96,
                         "top": 0.82, "bottom": 0.24})

        # Row / col order.
        # BOTH panels are pivoted on "microbiome_var" (the microbiome metric,
        # recorded on every row regardless of direction) x "dataset" (the aging
        # outcome). This makes the two panels cell-by-cell comparable: cell
        # (i, j) is the SAME variable pair in both, just tested in the opposite
        # direction.
        #
        # NOTE: an earlier version pivoted on "exposure" for both directions.
        # For Path B, "exposure" holds the OUTCOME label, and each Path-B
        # exposure only ever pairs with its own dataset -- so the pivot produced
        # a matrix with only the diagonal populated, and aggfunc="mean" silently
        # averaged the three microbiome variables into each diagonal cell. That
        # hid 6 of the 9 Path B effects (including the largest, hand grip ->
        # gut AgeAccel, beta=+0.114). Pivoting on microbiome_var fixes this.
        row_order = ["shannon", "richness", "gut_ageaccel"]
        col_order = ["hand_grip", "blood_pressure", "anthropometrics"]

        for ax, direction, title, accent in zip(
            axes,
            ["A", "B"],
            ["Path A  |  Microbiome(t₁) → Outcome(t₂)",
             "Path B  |  Outcome(t₁) → Microbiome(t₂)"],
            [_PATH_A, _PATH_B],
        ):
            sub = results[results["direction"] == direction]

            # pivot and force row / col order (fills missing with NaN)
            matrix = sub.pivot_table(index="microbiome_var", columns="dataset",
                                     values="std_beta", aggfunc="mean")
            sig_mat = sub.pivot_table(index="microbiome_var", columns="dataset",
                                      values="q_fdr", aggfunc="mean")
            nom_mat = sub.pivot_table(index="microbiome_var", columns="dataset",
                                      values="p", aggfunc="mean")
            matrix  = matrix.reindex(index=row_order, columns=col_order)
            sig_mat = sig_mat.reindex(index=row_order, columns=col_order)
            nom_mat = nom_mat.reindex(index=row_order, columns=col_order)

            cmap = LinearSegmentedColormap.from_list(
                "cl", [_PATH_B, "#F7F7F7", _PATH_A] if direction == "A"
                      else [_PATH_A, "#F7F7F7", _PATH_B], N=256)
            cmap.set_bad("#EBEBEB")

            nrows, ncols = matrix.shape
            im = ax.imshow(matrix.values, cmap=cmap,
                           vmin=-0.3, vmax=0.3, aspect="auto",
                           extent=[-0.5, ncols - 0.5, nrows - 0.5, -0.5])

            # pseudo-3D bevel on every non-NaN cell
            for i in range(nrows):
                for j in range(ncols):
                    if not np.isnan(matrix.iloc[i, j]):
                        _draw_cell_3d(ax, j, i)

            col_labels = [_sl(c) for c in col_order]
            row_labels = [_sl(r) for r in row_order]
            ax.set_xticks(range(ncols))
            ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=8)
            ax.set_yticks(range(nrows))
            ax.set_yticklabels(row_labels, fontsize=8)

            # Axis labels make the direction explicit: in Path A the microbiome
            # metric is the predictor, in Path B it is the predicted variable.
            if direction == "A":
                ax.set_ylabel("Microbiome metric at t₁ (predictor)", fontsize=7.5)
                ax.set_xlabel("Aging outcome at t₂ (predicted)", fontsize=7.5)
            else:
                ax.set_ylabel("Microbiome metric at t₂ (predicted)", fontsize=7.5)
                ax.set_xlabel("Aging outcome at t₁ (predictor)", fontsize=7.5)

            ax.set_title(title, fontsize=8.5, color=accent, fontweight="bold", pad=10)

            # significance markers + value annotations
            for i in range(nrows):
                for j in range(ncols):
                    q   = sig_mat.iloc[i, j]
                    p_n = nom_mat.iloc[i, j]
                    v   = matrix.iloc[i, j]
                    if not np.isnan(q) and q < ALPHA_FDR:
                        ax.text(j, i - 0.22, "◆", ha="center", va="center",
                                fontsize=9, color=_INK, fontweight="bold", zorder=5)
                    elif not np.isnan(p_n) and p_n < 0.05:
                        ax.text(j, i - 0.22, "·", ha="center", va="center",
                                fontsize=13, color=_INK, alpha=0.5, zorder=5)
                    if not np.isnan(v):
                        txt_c = "white" if abs(v) > 0.18 else _INK
                        ax.text(j, i + 0.30, f"{v:+.3f}", ha="center", va="center",
                                fontsize=6, fontfamily="monospace",
                                color=txt_c, alpha=0.88, zorder=5)

            cbar = plt.colorbar(im, ax=ax, pad=0.03, fraction=0.028, shrink=0.75)
            cbar.set_label("Std. β", fontsize=6.5, labelpad=4)
            cbar.ax.tick_params(labelsize=6)
            for k in range(ncols + 1):
                ax.axvline(k - 0.5, color=_INK, lw=0.4, alpha=0.25, zorder=2)
            for k in range(nrows + 1):
                ax.axhline(k - 0.5, color=_INK, lw=0.4, alpha=0.25, zorder=2)

        fig.suptitle(
            "Cross-Lagged Panel: Gut Microbiome ↔ Functional Aging  (HPP, n≈1,823)",
            fontsize=9.5, fontweight="bold", fontfamily="serif", y=0.96)
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"cross_lag_heatmap.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/cross_lag_heatmap.pdf/.png")
    except Exception as e:
        import traceback; log(f"  Heatmap failed: {e}\n{traceback.format_exc()}")


def plot_direction_verdict(direction_df: pd.DataFrame, out_dir: Path) -> None:
    """−log₁₀(p) lollipop chart with glow halos on each dot."""
    log("\n[Step 7b] Plotting direction verdicts...")
    if direction_df.empty or "verdict" not in direction_df.columns:
        log("  No direction data to plot.")
        return
    try:
        df = direction_df.copy()
        df["pair"] = df["exposure"].apply(_sl) + " × " + df["outcome"].apply(_sl)
        df = df.dropna(subset=["p_A", "p_B"]).sort_values("p_A")

        fig, ax = plt.subplots(figsize=(9.0, max(4.0, len(df) * 0.72)),
                               gridspec_kw={"left": 0.28, "right": 0.84,
                                            "top": 0.88, "bottom": 0.11})
        y_pos = np.arange(len(df))

        # alternating row bands
        for i in range(len(df)):
            ax.axhspan(i - 0.5, i + 0.5, color=(_PANEL_BG if i % 2 == 0 else "white"),
                       zorder=0, lw=0)

        for i, (_, row) in enumerate(df.iterrows()):
            for p, color, off in [(row["p_A"], _PATH_A, +0.20),
                                  (row["p_B"], _PATH_B, -0.20)]:
                if np.isnan(p):
                    continue
                lp = -np.log10(max(p, 1e-10))
                y  = y_pos[i] + off
                sig = p < ALPHA_FDR

                # stem line
                ax.plot([0, lp], [y, y], color=color, lw=0.9, alpha=0.4, zorder=2)

                # glow halo (large translucent dot behind)
                ax.scatter(lp, y, s=200 if sig else 110, color=color,
                           alpha=0.18, zorder=3, linewidths=0)
                # main dot
                marker = "D" if sig else "o"
                size   = 60  if sig else 38
                sc = ax.scatter(lp, y, s=size, color=color, marker=marker,
                                zorder=5, edgecolors=_INK, linewidths=0.6,
                                alpha=1.0 if sig else 0.72)
                if sig:
                    _lift(sc, dy=-1, alpha=0.25)
                # p label — stagger by offset so Path A and B don't collide
                ax.text(lp + 0.06, y, f"p={p:.2f}", va="center", fontsize=5.5,
                        fontfamily="monospace", color=color, zorder=6,
                        alpha=0.85)

        fdr_x = -np.log10(ALPHA_FDR)
        ax.axvline(fdr_x, ls="--", lw=1.0, color=_INK, alpha=0.45, zorder=1)
        ax.text(fdr_x + 0.04, -0.7, "FDR 0.05", fontsize=5.5,
                color=_INK, alpha=0.65, va="top")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["pair"].values, fontsize=7.5)
        ax.set_xlabel("−log₁₀(p)", labelpad=6)
        ax.set_xlim(left=0)
        ax.invert_yaxis()

        leg = [mpatches.Patch(color=_PATH_A, label="Path A: Microbiome→Outcome"),
               mpatches.Patch(color=_PATH_B, label="Path B: Outcome→Microbiome")]
        ax.legend(handles=leg, frameon=False, fontsize=6.5, loc="lower right")

        fig.suptitle("Causal Direction: −log₁₀(p) per Pair  (HPP)",
                     fontsize=9.5, fontweight="bold", fontfamily="serif")
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"direction_verdicts.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/direction_verdicts.pdf/.png")
    except Exception as e:
        import traceback; log(f"  Direction verdict failed: {e}\n{traceback.format_exc()}")


def plot_path_comparison(results: pd.DataFrame, out_dir: Path) -> None:
    """Forest plot: section headers as dedicated rows, outcome rows indented below."""
    log("\n[Step 7c] Forest plot: Path A vs B...")
    try:
        EXP_ORDER = ["shannon", "richness", "gut_ageaccel"]
        DS_ORDER  = ["hand_grip", "blood_pressure", "anthropometrics"]

        # Count total rows (1 header + N data rows per group)
        groups = []
        for exp in EXP_ORDER:
            data_rows = [ds for ds in DS_ORDER
                         if not results[(results["microbiome_var"] == exp) &
                                        (results["dataset"] == ds)].empty]
            if data_rows:
                groups.append((exp, data_rows))

        total_rows = sum(1 + len(dr) for _, dr in groups) + len(groups) * 0.3
        fig, ax = plt.subplots(
            figsize=(9.5, max(5.5, total_rows * 0.72)),
            gridspec_kw={"left": 0.20, "right": 0.86,
                         "top": 0.94, "bottom": 0.08})

        y_pos  = 0
        yticks: list[float] = []
        ylabels: list[str]  = []

        for exp, data_rows in groups:
            # ── Section header row ────────────────────────────────────
            ax.axhspan(y_pos - 0.46, y_pos + 0.46,
                       color="#E5EEF8", zorder=0, lw=0)
            ax.axhline(y_pos - 0.46, color=_PATH_A, lw=1.0, alpha=0.35, zorder=1)
            yticks.append(y_pos)
            ylabels.append(f"▶  {_sl(exp)}")
            y_pos += 1

            # ── Data rows ────────────────────────────────────────────
            for ds in data_rows:
                grp   = results[(results["microbiome_var"] == exp) &
                                 (results["dataset"] == ds)]
                pathA = grp[grp["direction"] == "A"]
                pathB = grp[grp["direction"] == "B"]

                # alternating light/white bands (only on data rows)
                shade = "#F6F6F4" if (y_pos % 2 == 0) else "white"
                ax.axhspan(y_pos - 0.46, y_pos + 0.46,
                           color=shade, zorder=0, lw=0)

                for sub, color, off in [(pathA, _PATH_A, +0.17),
                                        (pathB, _PATH_B, -0.17)]:
                    if sub.empty:
                        continue
                    row = sub.iloc[0]
                    if np.isnan(row["beta"]):
                        continue
                    b, lo, hi = row["beta"], row["ci95"][0], row["ci95"][1]
                    y   = y_pos + off
                    sig = bool(row["significant"])

                    # CI whisker with shadow
                    ci_line, = ax.plot([lo, hi], [y, y], color=color,
                                       lw=2.2, solid_capstyle="round", zorder=3)
                    _lift(ci_line, dy=-1, alpha=0.18)

                    # glow + main marker
                    ax.scatter([b], [y], s=110 if sig else 65,
                               color=color, alpha=0.18, zorder=3, linewidths=0)
                    mk, sz, al = ("D", 52, 1.0) if sig else ("o", 32, 0.72)
                    sc = ax.scatter([b], [y], color=color, marker=mk,
                                    s=sz, zorder=5, alpha=al,
                                    edgecolors=_INK, linewidths=0.5)
                    if sig:
                        _lift(sc, dy=-1, alpha=0.22)

                    # p-value: right-aligned, staggered vertically to avoid collision
                    p_val = row["p"]
                    q_val = row.get("q_fdr", np.nan)
                    lbl   = f"q={q_val:.2f}" if sig else f"p={p_val:.2f}"
                    ax.text(0.988, y, lbl,
                            transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=5.8,
                            fontfamily="monospace",
                            color=color if sig else _NEUTRAL)

                yticks.append(y_pos)
                ylabels.append(f"    {_sl(ds)}")   # indent outcome label
                y_pos += 1

            y_pos += 0.4   # gap between groups

        # shaded +/-0.10 equivalence band (SESOI, TOST analysis in
        # cross_lag_equivalence_bounds.py). Drawn from the same pre-specified
        # constant, not re-derived from data -- kept in sync by referencing the
        # literal 0.10 the equivalence script also uses, rather than importing that
        # module here (avoids a circular import; the value is a fixed design choice,
        # not a fitted quantity that could drift).
        _SESOI = 0.10
        ax.axvspan(-_SESOI, _SESOI, color="#E8E8E8", alpha=0.6, zorder=0,
                  label=f"Equivalence band (|β|<{_SESOI})")
        ax.axvline(0, color=_INK, lw=0.9, ls="--", alpha=0.40, zorder=1)
        ax.set_yticks(yticks)
        yticklabels = ax.set_yticklabels(ylabels, fontsize=7.8)
        # bold the section headers
        for lbl, label_text in zip(yticklabels, ylabels):
            if label_text.startswith("▶"):
                lbl.set_fontweight("bold")
                lbl.set_fontstyle("italic")
                lbl.set_color(_PATH_A)

        ax.set_xlabel("Standardised β", labelpad=6, fontsize=8)
        ax.invert_yaxis()

        leg = [
            mpatches.Patch(color=_PATH_A, label="Path A  Microbiome(t₁)→Outcome(t₂)"),
            mpatches.Patch(color=_PATH_B, label="Path B  Outcome(t₁)→Microbiome(t₂)"),
            ax.scatter([], [], marker="D", color=_NEUTRAL, s=42,
                       edgecolors=_INK, lw=0.5, label="FDR q<0.05"),
            ax.scatter([], [], marker="o", color=_NEUTRAL, s=26,
                       alpha=0.65, edgecolors=_INK, lw=0.5, label="Nominal"),
            mpatches.Patch(color="#E8E8E8", alpha=0.6, label=f"Equivalence band (|β|<{_SESOI})"),
        ]
        ax.legend(handles=leg, frameon=False, fontsize=6.5, loc="lower right",
                  handletextpad=0.4, labelspacing=0.35)

        fig.suptitle("Cross-Lagged Coefficients by Microbiome Metric  (HPP)",
                     fontsize=9.5, fontweight="bold", fontfamily="serif")
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"forest_path_comparison.{ext}",
                        dpi=300, bbox_inches="tight")
        # [MANUSCRIPT] also save under the v17 \smartfig name (Fig4_crosslag_forest.pdf)
        plt.savefig(out_dir / "Fig4_crosslag_forest.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/forest_path_comparison.pdf/.png (+ Fig4_crosslag_forest.pdf)")
    except Exception as e:
        import traceback
        log(f"  Forest plot failed: {e}\n{traceback.format_exc()}")


def plot_gut_clock(clock_data: dict, out_dir: Path) -> None:
    """[MANUSCRIPT] Fig2_clock.pdf -- 4-panel clock validation from REAL per-
    participant training predictions (not summary statistics): (a) predicted
    vs chronological age scatter, (b) calibration (fitted slope/intercept vs
    the identity line), (c) mean residual by chronological-age decile, (d) MAE
    by sex and age tertile. All four panels derive from the exact same
    y_train/y_pred_cv/sex_train arrays clock_data carries -- no separate re-fit,
    no risk of a panel silently drifting from the others."""
    log("\n[Step 7d] Gut clock 4-panel validation figure...")
    try:
        y_train   = np.asarray(clock_data["y_train"])
        y_pred_cv = np.asarray(clock_data["y_pred_cv"])
        r_cv      = clock_data["r_cv"]
        sex_train = np.asarray(clock_data.get("sex_train", []))
        if len(y_train) < 10:
            log("  Insufficient clock data — skipping.")
            return
        resid = y_pred_cv - y_train

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5),
                                 gridspec_kw={"hspace": 0.32, "wspace": 0.28,
                                              "left": 0.08, "right": 0.96,
                                              "top": 0.92, "bottom": 0.07})
        ax_scatter, ax_calib, ax_decile, ax_strat = axes.flatten()

        # ---- (a) scatter: predicted vs chronological age ----
        hb = ax_scatter.hexbin(y_train, y_pred_cv, gridsize=35, cmap="Blues",
                               mincnt=1, linewidths=0.2, edgecolors=_PANEL_BG, alpha=0.92)
        cbar = plt.colorbar(hb, ax=ax_scatter, pad=0.02, fraction=0.04, shrink=0.75)
        cbar.set_label("count", fontsize=6, labelpad=3)
        cbar.ax.tick_params(labelsize=5.5)
        m, c = np.polyfit(y_train, y_pred_cv, 1)
        xs = np.linspace(y_train.min(), y_train.max(), 200)
        ax_scatter.plot(xs, m * xs + c, color=_INK, lw=2.5, alpha=0.15, zorder=3)
        line, = ax_scatter.plot(xs, m * xs + c, color=_PATH_A, lw=1.8, zorder=4)
        _lift(line, dy=-1, alpha=0.22)
        mn, mx = min(y_train.min(), y_pred_cv.min()), max(y_train.max(), y_pred_cv.max())
        ax_scatter.plot([mn, mx], [mn, mx], color=_INK, lw=0.8, ls="--", alpha=0.35, zorder=2)
        ax_scatter.set_xlabel("Chronological age (years)", labelpad=5, fontsize=8)
        ax_scatter.set_ylabel("Gut clock predicted age (years)", labelpad=5, fontsize=8)
        r2 = r_cv ** 2
        ax_scatter.text(0.04, 0.95, f"r = {r_cv:.3f}\nR² = {r2:.3f}\nn = {len(y_train):,}",
                        transform=ax_scatter.transAxes, va="top", fontsize=7.5,
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_INK, lw=0.5, alpha=0.88))
        ax_scatter.set_title("(a) Predicted vs chronological age", fontsize=8.5, loc="left")

        # [FIG1 panel-d dependency] Persist the headline clock metrics that back
        # THIS panel so Fig1_flow.pdf's age-encoding tile can cite the exact same
        # number instead of a second, possibly-drifting hardcode (Rule 4).
        pd.DataFrame([{
            "r_cv": float(r_cv), "r2_cv": float(r2), "mae_cv": float(np.mean(np.abs(resid))),
            "n": int(len(y_train)),
        }]).to_csv(out_dir / "clock_headline_metrics.csv", index=False)
        log(f"  Saved: {out_dir}/clock_headline_metrics.csv (r={r_cv:.4f}, R²={r2:.4f}, "
            f"n={len(y_train):,}) -- source for Fig1 panel-d age-encoding tile")

        # ---- (b) calibration: fitted line vs identity, annotated slope/intercept ----
        ax_calib.plot(xs, m * xs + c, color=_PATH_A, lw=2.0, label=f"fit: y={m:.3f}x+{c:.2f}")
        ax_calib.plot([mn, mx], [mn, mx], color=_INK, lw=1.0, ls="--", alpha=0.5, label="identity (y=x)")
        ax_calib.set_xlabel("Chronological age (years)", labelpad=5, fontsize=8)
        ax_calib.set_ylabel("Predicted age (years)", labelpad=5, fontsize=8)
        ax_calib.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax_calib.text(0.97, 0.05,
                     f"slope={m:.3f}\nintercept={c:.2f}\n(predictions compress\ntoward the cohort mean)",
                     transform=ax_calib.transAxes, ha="right", va="bottom", fontsize=6.5,
                     fontfamily="monospace", color=_NEUTRAL)
        ax_calib.set_title("(b) Calibration", fontsize=8.5, loc="left")

        # ---- (c) mean residual by chronological-age decile ----
        deciles = pd.qcut(y_train, 10, duplicates="drop")
        dec_df = pd.DataFrame({"decile": deciles, "resid": resid, "age": y_train})
        dec_stats = dec_df.groupby("decile", observed=True).agg(
            mean_resid=("resid", "mean"), age_lo=("age", "min"), age_hi=("age", "max"),
            n=("resid", "size")).reset_index()
        colors = [_PATH_A if v >= 0 else _PATH_B for v in dec_stats["mean_resid"]]
        ax_decile.bar(range(len(dec_stats)), dec_stats["mean_resid"], color=colors, alpha=0.85,
                     edgecolor=_INK, linewidth=0.4)
        ax_decile.axhline(0, color=_INK, lw=0.8)
        ax_decile.set_xticks(range(len(dec_stats)))
        ax_decile.set_xticklabels([f"{int(r.age_lo)}–{int(r.age_hi)}" for r in dec_stats.itertuples()],
                                  fontsize=6, rotation=45, ha="right")
        ax_decile.set_xlabel("Chronological-age decile (years)", labelpad=5, fontsize=8)
        ax_decile.set_ylabel("Mean residual (predicted − actual, years)", labelpad=5, fontsize=8)
        ax_decile.set_title("(c) Residual by age decile (compression pattern)", fontsize=8.5, loc="left")

        # ---- (d) MAE by sex and age tertile ----
        strat_labels, strat_maes, strat_ns = [], [], []
        if len(sex_train) == len(y_train) and np.isfinite(sex_train).any():
            for sex_val, sex_lbl in [(1, "Male"), (0, "Female")]:
                mask = sex_train == sex_val
                if mask.sum() >= 10:
                    strat_labels.append(sex_lbl)
                    strat_maes.append(float(np.mean(np.abs(resid[mask]))))
                    strat_ns.append(int(mask.sum()))
        tertile_edges = np.quantile(y_train, [0, 1 / 3, 2 / 3, 1.0])
        tertile_names = ["Youngest tertile", "Middle tertile", "Oldest tertile"]
        for i in range(3):
            lo, hi = tertile_edges[i], tertile_edges[i + 1]
            mask = (y_train >= lo) & (y_train <= hi if i == 2 else y_train < hi)
            if mask.sum() >= 10:
                strat_labels.append(f"{tertile_names[i]}\n[{lo:.0f}–{hi:.0f}]")
                strat_maes.append(float(np.mean(np.abs(resid[mask]))))
                strat_ns.append(int(mask.sum()))
        if strat_labels:
            bar_colors = [_PATH_A if "tertile" not in lbl.lower() else _NEUTRAL for lbl in strat_labels]
            ax_strat.bar(range(len(strat_labels)), strat_maes, color=bar_colors, alpha=0.85,
                        edgecolor=_INK, linewidth=0.4)
            for i, (mae_v, n_v) in enumerate(zip(strat_maes, strat_ns)):
                ax_strat.text(i, mae_v + 0.15, f"n={n_v:,}", ha="center", fontsize=6, color=_NEUTRAL)
            ax_strat.set_xticks(range(len(strat_labels)))
            ax_strat.set_xticklabels(strat_labels, fontsize=6.5)
            ax_strat.set_ylabel("MAE (years)", labelpad=5, fontsize=8)
        else:
            ax_strat.text(0.5, 0.5, "sex_train unavailable for this run", ha="center",
                         va="center", transform=ax_strat.transAxes, fontsize=8, color=_NEUTRAL)
        ax_strat.set_title("(d) MAE by sex and age tertile", fontsize=8.5, loc="left")

        fig.suptitle("Gut Microbiome Clock Validation  (HPP, corrected leak-free pipeline)",
                     fontsize=10.5, fontweight="bold", fontfamily="serif")
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"gut_clock_performance.{ext}", dpi=300, bbox_inches="tight")
        # [MANUSCRIPT] the v17 \smartfig name
        plt.savefig(out_dir / "Fig2_clock.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/gut_clock_performance.pdf/.png (+ Fig2_clock.pdf, 4-panel)")
    except Exception as e:
        import traceback; log(f"  Clock plot failed: {e}\n{traceback.format_exc()}")


def plot_richness_3d(gut_div: pd.DataFrame, out_dir: Path) -> None:
    """Pseudo-3D bar chart: species richness at visit 1 vs visit 2 (side-by-side)."""
    log("\n[Step 7e] Richness 3D bar chart (visit 1 vs 2)...")
    try:
        v1 = gut_div[gut_div["visit_n"] == 1]["richness"].dropna()
        v2 = gut_div[gut_div["visit_n"] == 2]["richness"].dropna()
        if len(v1) < 10 or len(v2) < 10:
            log("  Insufficient richness data — skipping.")
            return

        all_vals = np.concatenate([v1.values, v2.values])
        n_bins   = 18
        bins     = np.linspace(np.percentile(all_vals, 1),
                               np.percentile(all_vals, 99), n_bins + 1)
        centers  = 0.5 * (bins[:-1] + bins[1:])
        h1, _   = np.histogram(v1, bins=bins)
        h2, _   = np.histogram(v2, bins=bins)

        bw     = bins[1] - bins[0]
        half_w = bw * 0.36          # half-width of each bar (side-by-side, no gap)
        depth  = bw * 0.30          # how deep the 3-D extrusion looks

        fig, ax = plt.subplots(figsize=(8.5, 5.0),
                               gridspec_kw={"left": 0.11, "right": 0.97,
                                            "top": 0.87, "bottom": 0.14})
        ax.set_facecolor("white")

        max_y = max(h1.max(), h2.max())

        for cx, cnt1, cnt2 in zip(centers, h1, h2):
            # Visit-1 bar: centred left of bin centre
            if cnt1 > 0:
                _bar3d(ax, cx - half_w/2, y0=0, height=cnt1,
                       w=half_w * 1.85, depth=depth, color=_PATH_A)
            # Visit-2 bar: centred right of bin centre
            if cnt2 > 0:
                _bar3d(ax, cx + half_w/2, y0=0, height=cnt2,
                       w=half_w * 1.85, depth=depth, color=_PATH_B)

        # Axis limits must be set manually — add_patch() doesn't trigger autoscale
        ax.set_xlim(bins[0] - bw * 0.5, bins[-1] + bw * 0.5 + depth)
        ax.set_ylim(0, max_y * 1.30 + depth * 0.55)

        from scipy.stats import mannwhitneyu
        _, p_mw = mannwhitneyu(v1, v2, alternative="two-sided")
        ax.text(0.97, 0.97,
                f"Visit 1  median = {np.median(v1):.0f}\n"
                f"Visit 2  median = {np.median(v2):.0f}\n"
                f"Mann-Whitney p = {p_mw:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=_INK, lw=0.6, alpha=0.90))

        ax.set_xlabel("Species richness  (# species detected)", labelpad=5)
        ax.set_ylabel("Participants", labelpad=5)

        leg = [mpatches.Patch(color=_PATH_A, label="Visit 1"),
               mpatches.Patch(color=_PATH_B, label="Visit 2")]
        ax.legend(handles=leg, frameon=False, fontsize=7.5, loc="upper left")

        fig.suptitle(
            "Gut Species Richness: Visit 1 vs Visit 2  (HPP repeat participants)",
            fontsize=9, fontweight="bold", fontfamily="serif")
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"richness_3d_visit_comparison.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/richness_3d_visit_comparison.pdf/.png")
    except Exception as e:
        import traceback
        log(f"  Richness 3D plot failed: {e}\n{traceback.format_exc()}")


def plot_effect_wheel(direction_df: pd.DataFrame, out_dir: Path) -> None:
    """Radar / wheel chart — |β_A| vs |β_B| for every pair, coloured by direction."""
    log("\n[Step 7f] Effect-size summary wheel...")
    if direction_df.empty:
        log("  No data for wheel.")
        return
    try:
        df = direction_df.dropna(subset=["beta_A", "beta_B"]).copy()
        if len(df) == 0:
            log("  All betas NaN — skipping wheel.")
            return

        log(f"  Wheel exposure values: {df['exposure'].unique().tolist()}")
        log(f"  Wheel outcome values:  {df['outcome'].unique().tolist()}")

        n    = len(df)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        labels = df["exposure"].apply(_sl) + "\n×\n" + df["outcome"].apply(_sl)

        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        # radial reference rings
        for r in [0.05, 0.10, 0.15, 0.20]:
            ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200,
                    color=_INK, lw=0.4, alpha=0.18, ls="--")
        ax.text(np.pi/4, 0.21, "0.20", fontsize=5, color=_INK, alpha=0.5)
        ax.text(np.pi/4, 0.11, "0.10", fontsize=5, color=_INK, alpha=0.5)

        for ang, (_, row) in zip(angles, df.iterrows()):
            bA = abs(row["beta_A"]) if not np.isnan(row["beta_A"]) else 0
            bB = abs(row["beta_B"]) if not np.isnan(row["beta_B"]) else 0

            verdict = row["verdict"]
            if "MICROBIOME" in verdict:
                col = _PATH_A
            elif "AGING" in verdict:
                col = _PATH_B
            elif "BIDIR" in verdict:
                col = _BIDIR
            else:
                col = _NEUTRAL

            # spoke for Path A (outer)
            ax.annotate("", xy=(ang, bA), xytext=(ang, 0),
                        arrowprops=dict(arrowstyle="-|>", color=_PATH_A,
                                        lw=2.5, mutation_scale=8))
            # spoke for Path B (slightly offset angle)
            dang = 0.08
            ax.annotate("", xy=(ang+dang, bB), xytext=(ang+dang, 0),
                        arrowprops=dict(arrowstyle="-|>", color=_PATH_B,
                                        lw=2.5, mutation_scale=8))
            # dot at tip coloured by verdict
            ax.scatter([ang], [max(bA, bB) + 0.01], s=50,
                       color=col, zorder=5, edgecolors=_INK, lw=0.4)

        ax.set_thetagrids(np.degrees(angles), labels=labels,
                          fontsize=6.0)
        ax.set_ylim(0, 0.28)
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)

        leg = [mpatches.Patch(color=_PATH_A,  label="Path A (|β|)"),
               mpatches.Patch(color=_PATH_B,  label="Path B (|β|)"),
               mpatches.Patch(color=_NEUTRAL, label="No sig. effect"),
               mpatches.Patch(color=_BIDIR,   label="Bidirectional")]
        ax.legend(handles=leg, frameon=False, fontsize=6,
                  loc="upper right", bbox_to_anchor=(1.38, 1.12))

        fig.suptitle("Effect Magnitudes: |β_A| vs |β_B| per Pair",
                     fontsize=9.5, fontweight="bold", fontfamily="serif", y=1.01)
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"effect_wheel.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/effect_wheel.pdf/.png")
    except Exception as e:
        import traceback; log(f"  Effect wheel failed: {e}\n{traceback.format_exc()}")


def plot_incident_disease(incident_df: pd.DataFrame, out_dir: Path) -> None:
    """OR forest plot for incident disease analysis (styled)."""
    if incident_df is None or len(incident_df) == 0:
        return
    log("\n[Step 7g] Incident disease OR plot...")
    try:
        df = incident_df.sort_values("p_shannon")
        fig, ax = plt.subplots(figsize=(10, max(4, len(df)*0.52)),
                               gridspec_kw={"left": 0.28, "right": 0.94,
                                            "top": 0.90, "bottom": 0.12})
        for i, (_, row) in enumerate(df.iterrows()):
            color = _PATH_B if row["significant"] else _NEUTRAL
            ax.axhspan(i-0.5, i+0.5, color=_PANEL_BG if i%2==0 else "white", lw=0)
            ci_line, = ax.plot([row["CI_lo"], row["CI_hi"]], [i, i],
                                color=color, lw=2.2, solid_capstyle="round", zorder=3)
            _lift(ci_line, dy=-1, alpha=0.20)
            mk = "D" if row["significant"] else "o"
            sc = ax.scatter([row["OR_shannon"]], [i], color=color,
                            marker=mk, s=55, zorder=5,
                            edgecolors=_INK, linewidths=0.5)
            if row["significant"]:
                _lift(sc, dy=-1, alpha=0.22)
            ax.text(0.98, i, f"p={row['p_shannon']:.3f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=5.5, fontfamily="monospace",
                    color=color if row["significant"] else _NEUTRAL)
        ax.axvline(1.0, color=_INK, lw=0.9, ls="--", alpha=0.5)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["phenotype"], fontsize=7)
        ax.set_xlabel("OR (Shannon diversity per SD) for incident disease", labelpad=5)
        ax.invert_yaxis()
        fig.suptitle("Gut Microbiome Diversity → Incident Disease  (HPP)",
                     fontsize=9, fontweight="bold", fontfamily="serif")
        for ext in ("pdf", "png"):
            plt.savefig(out_dir / f"incident_disease_OR.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        log(f"  Saved: {out_dir}/incident_disease_OR.pdf/.png")
        # NOTE: this in-module plot is illustrative only (fed by the
        # diagnostic in-module load_incident_diseases()) and does NOT also
        # save Fig6_disease.pdf. The manuscript's Fig6_disease.pdf is
        # produced exclusively by fig6_disease_combined.py, which reads the
        # CANONICAL standalone incident_disease.py output and adds the
        # disease-burden panel. Two scripts writing the same manuscript
        # filename is exactly how the incident_disease.csv collision
        # (below) happens -- don't repeat it for this figure.
    except Exception as e:
        log(f"  Incident disease plot failed: {e}")


# ============================================================
# STEP 8: SAVE RESULTS
# ============================================================

def build_exclusion_funnel(gut_full: pd.DataFrame, gut_div: pd.DataFrame,
                           gut_ageaccel: pd.DataFrame,
                           outcomes: dict[str, pd.DataFrame],
                           results: pd.DataFrame) -> pd.DataFrame:
    """
    [Task 1.3] Per-stage participant counts for every (exposure x outcome)
    cross-lagged test, so the sample-size gap documented in the paper's
    Table A9 -- 1,817 with AgeAccel at both visits under the reliability
    pipeline vs. 523 under this leak-free pipeline -- and query (iii), whether
    Shannon/richness are genuinely capped by the same participants as the
    leak-free AgeAccel clock or only coincide with it because both are
    phenotype-limited, are auditable from saved numbers rather than asserted.
    """
    log("\n[Funnel] Building per-stage exclusion funnel...")
    repeat_cohort_n = gut_full["study_id"].nunique()

    exposures = {"shannon": gut_div, "richness": gut_div, "gut_ageaccel": gut_ageaccel}
    rows = []
    for exp_name, exp_df in exposures.items():
        exp_valid = exp_df.dropna(subset=[exp_name])
        exp_both = (exp_valid.groupby("study_id")["visit_n"]
                   .apply(lambda v: {1, 2}.issubset(set(v))))
        exp_both_ids = set(exp_both[exp_both].index)

        for ds_name, out_df in outcomes.items():
            out_both = (out_df.dropna(subset=["outcome"])
                       .groupby("study_id")["visit_n"]
                       .apply(lambda v: {1, 2}.issubset(set(v))))
            out_both_ids = set(out_both[out_both].index)

            both_sides_ids = exp_both_ids & out_both_ids

            final_n = np.nan
            if results is not None and not results.empty:
                match = results[(results["exposure"] == exp_name) &
                                (results["dataset"] == ds_name) &
                                (results["direction"] == "A")]
                if not match.empty:
                    final_n = match.iloc[0]["n"]

            rows.append({
                "exposure": exp_name,
                "outcome": ds_name,
                "stage1_repeat_cohort_n": repeat_cohort_n,
                "stage2_exposure_both_visits_n": len(exp_both_ids),
                "stage3_outcome_both_visits_n": len(out_both_ids),
                "stage4_exposure_and_outcome_both_visits_n": len(both_sides_ids),
                "stage5_final_model_n": final_n,
            })

    funnel = pd.DataFrame(rows)
    log("\n" + funnel.to_string(index=False))

    # Explicit settlement of Table A9 query (iii).
    ga_n = {row["outcome"]: row["stage4_exposure_and_outcome_both_visits_n"]
           for _, row in funnel[funnel["exposure"] == "gut_ageaccel"].iterrows()}
    for exp_name in ("shannon", "richness"):
        for ds_name, n_ga in ga_n.items():
            match = funnel[(funnel["exposure"] == exp_name) & (funnel["outcome"] == ds_name)]
            if match.empty:
                continue
            n_exp = match.iloc[0]["stage4_exposure_and_outcome_both_visits_n"]
            verdict = ("SAME size -- diversity IS effectively capped to the "
                      "clock population for this pair" if n_exp == n_ga else
                      "DIFFERENT size -- diversity is NOT capped by the clock; "
                      "any similarity in the paper's reported n is "
                      "phenotype-driven coincidence, not a shared dependency")
            log(f"  [A9-iii] {exp_name} x {ds_name}: n={n_exp:,}  vs.  "
                f"gut_ageaccel x {ds_name}: n={n_ga:,}  ->  {verdict}")

    return funnel


def report_covariate_spec_comparison(results_primary: pd.DataFrame,
                                     results_sensitivity: pd.DataFrame,
                                     gut_div: pd.DataFrame, gut_ageaccel: pd.DataFrame,
                                     outcomes: dict, pop: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the 9 (exposure x outcome) base pairs: n under the
    PRIMARY spec (age+sex, matching Methods) vs the SENSITIVITY spec
    (+education+income), plus education/income availability within the base
    (exposure+outcome-both-visits) cohort. Quantifies the covariate
    over-adjustment before comparing coefficients under the corrected spec.
    n is identical for Path A and Path B of the same pair by construction
    (both use the same four required columns, just relabeled), so this is
    reported once per pair, not once per direction.
    """
    log("\nCovariate-spec comparison (primary age+sex vs sensitivity "
        "full covariates)...")
    exposures = {"shannon": gut_div, "richness": gut_div, "gut_ageaccel": gut_ageaccel}
    pop_idx = pop.set_index("study_id")
    rows = []
    for exp_name, exp_df in exposures.items():
        exp_valid = exp_df.dropna(subset=[exp_name])
        exp_both = (exp_valid.groupby("study_id")["visit_n"]
                   .apply(lambda v: {1, 2}.issubset(set(v))))
        exp_both_ids = set(exp_both[exp_both].index)

        for ds_name, out_df in outcomes.items():
            out_both = (out_df.dropna(subset=["outcome"])
                       .groupby("study_id")["visit_n"]
                       .apply(lambda v: {1, 2}.issubset(set(v))))
            out_both_ids = set(out_both[out_both].index)
            base_ids = exp_both_ids & out_both_ids
            base_pop = pop_idx.reindex(list(base_ids))
            edu_avail = (int(base_pop["education_level"].notna().sum())
                        if "education_level" in base_pop.columns else np.nan)
            inc_avail = (int(base_pop["household_income"].notna().sum())
                        if "household_income" in base_pop.columns else np.nan)
            n_base = len(base_ids)

            n_primary = np.nan
            n_sensitivity = np.nan
            if results_primary is not None and not results_primary.empty:
                m = results_primary[(results_primary["exposure"] == exp_name) &
                                    (results_primary["dataset"] == ds_name) &
                                    (results_primary["direction"] == "A")]
                if not m.empty:
                    n_primary = m.iloc[0]["n"]
            if results_sensitivity is not None and not results_sensitivity.empty:
                m = results_sensitivity[(results_sensitivity["exposure"] == exp_name) &
                                        (results_sensitivity["dataset"] == ds_name) &
                                        (results_sensitivity["direction"] == "A")]
                if not m.empty:
                    n_sensitivity = m.iloc[0]["n"]

            rows.append({
                "exposure": exp_name, "outcome": ds_name,
                "base_cohort_n": n_base,
                "education_available_n": edu_avail,
                "education_available_pct": round(100 * edu_avail / n_base, 1) if n_base else np.nan,
                "income_available_n": inc_avail,
                "income_available_pct": round(100 * inc_avail / n_base, 1) if n_base else np.nan,
                "n_primary_age_sex": n_primary,
                "n_sensitivity_full_covariates": n_sensitivity,
                "n_recovered_by_primary_spec": (
                    n_primary - n_sensitivity
                    if (pd.notna(n_primary) and pd.notna(n_sensitivity)) else np.nan),
            })
    cmp_df = pd.DataFrame(rows)
    log("\n" + cmp_df.to_string(index=False))
    cmp_df.to_csv(RESULTS_DIR / "covariate_spec_comparison.csv", index=False)
    log(f"  wrote {RESULTS_DIR/'covariate_spec_comparison.csv'}")
    return cmp_df


def save_results(results: pd.DataFrame, direction_df: pd.DataFrame,
                 incident_df, out_dir: Path) -> None:
    log("\n[Step 8] Saving results...")
    results.to_csv(out_dir / "cross_lag_all_paths.csv", index=False)
    direction_df.to_csv(out_dir / "causal_direction_summary.csv", index=False)
    if incident_df is not None:
        # Writes to a filename DISTINCT from "incident_disease.csv" on
        # purpose -- that is the standalone incident_disease.py script's
        # output, the CANONICAL source for Table E7/E8 (it has a proper
        # covariate fallback ladder and reports shannon AND richness as
        # separate tests; this in-module version fits richness into the
        # model but never extracts its own coefficient). This in-module
        # version exists only to feed plot_incident_disease() within this
        # script's own Step 7 -- an illustrative plot, not a manuscript
        # number source -- so the two must never share a filename.
        incident_df.to_csv(out_dir / "incident_disease_inmodule_diagnostic.csv",
                           index=False)

    sig = results[results["significant"]]
    n_sig = len(sig)
    log(f"\n  Total cross-lag tests: {len(results)}")
    log(f"  FDR-significant (q<{ALPHA_FDR}): {n_sig}")
    if n_sig:
        log("\n  SIGNIFICANT CROSS-LAGGED EFFECTS:")
        for _, r in sig.iterrows():
            log(f"    {r['path']}:  β={r['beta']:.3f} (95% CI {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}), "
                f"p={r['p']:.4f}, q={r['q_fdr']:.4f}, n={r['n']}")

    if "verdict" in direction_df.columns and not direction_df.empty:
        verdicts = direction_df["verdict"].value_counts()
        log("\n  CAUSAL DIRECTION VERDICTS:")
        for v, c in verdicts.items():
            log(f"    {v}: {c} pair(s)")
    else:
        log("\n  CAUSAL DIRECTION VERDICTS: no pairs with sufficient n")

    if incident_df is not None and not incident_df.empty:
        sig_inc = incident_df[incident_df["significant"]]
        log(f"\n  Incident disease: {len(sig_inc)}/{len(incident_df)} phenotypes FDR-significant")
        if not sig_inc.empty:
            for _, r in sig_inc.iterrows():
                log(f"    {r['phenotype']}: OR={r['OR_shannon']:.3f} "
                    f"[{r['CI_lo']:.3f}–{r['CI_hi']:.3f}] p={r['p_shannon']:.4f}")

    log(f"\n  All outputs saved to: {out_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 60)
    log("CROSS-LAGGED PANEL ANALYSIS: Gut Microbiome ↔ Functional Aging")
    log(f"Data root : {DATA_ROOT}")
    log(f"Results   : {RESULTS_DIR}")
    log("=" * 60)

    # Step 1: Load
    pop          = load_population()
    gut_full     = pd.read_parquet(DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    log(f"  gut_full raw shape: {gut_full.shape}")
    if gut_full.index.names != [None]: gut_full = gut_full.reset_index()
    pid_col  = next(c for c in gut_full.columns if c in ("participant_id","study_id"))
    stg_col  = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int)
    )
    counts = gut_full.groupby("study_id")["visit"].nunique()
    repeat_ids = counts[counts >= 2].index
    gut_long = gut_full[gut_full["study_id"].isin(repeat_ids)].copy()

    outcomes = load_functional_outcomes()

    # Step 2: Diversity
    gut_div = compute_diversity(gut_long)

    # [CHECK-0] visit_n alignment: gut_div and each outcome frame each rank
    # their OWN "visit" (raw research stage) into visit_n independently. If a
    # participant's stage set differs between the microbiome data and an
    # outcome (e.g. gut has stages {10,15} but grip has {10,12}), visit_n=2
    # would silently pair different calendar times across the two variables --
    # injecting noise and biasing cross-lag estimates toward null. Confirm
    # that doesn't happen rather than assume it.
    log("\n[CHECK-0] visit_n alignment (gut microbiome vs. each outcome)")
    for ds_name, out_df in outcomes.items():
        if "visit" not in gut_div.columns or "visit" not in out_df.columns:
            log(f"  {ds_name}: SKIPPED, missing 'visit' column on one side")
            continue
        merged_chk = gut_div[["study_id", "visit_n", "visit"]].merge(
            out_df[["study_id", "visit_n", "visit"]],
            on=["study_id", "visit_n"], suffixes=("_gut", "_out"))
        if len(merged_chk) == 0:
            log(f"  {ds_name}: no overlapping (study_id, visit_n) rows to check")
            continue
        matched = (merged_chk["visit_gut"] == merged_chk["visit_out"]).sum()
        total   = len(merged_chk)
        pct     = 100 * matched / total
        log(f"  {ds_name}: {matched:,}/{total:,} ({pct:.1f}%) rows have the SAME "
            f"raw research_stage on both sides at matching visit_n"
            + ("" if pct > 95 else "  <-- WARNING: substantial misalignment, "
               "cross-lag results for this pair may be attenuated by noise"))

    # Step 3: Clock AgeAccel
    gut_ageaccel, clock_data = build_gut_clock_ageaccel(gut_div, gut_full, pop)

    # [CHECK-2c] gut_ageaccel coverage: how many repeat-visit participants have
    # a valid clock prediction at BOTH visit 1 and visit 2, and how much of
    # that overlaps with each outcome dataset's own both-visit participants
    # (the actual analytic n each cross-lag test will run on).
    valid_ga = gut_ageaccel.dropna(subset=["gut_ageaccel"])
    per_id = valid_ga.groupby("study_id")["visit_n"].apply(lambda x: sorted(x.unique()))
    both_visits = per_id.apply(lambda v: 1 in v and 2 in v).sum()
    log(f"\n[CHECK-2c] gut_ageaccel: {valid_ga['study_id'].nunique():,} participants with "
        f"a valid value, {both_visits:,} at BOTH visit 1 and 2")
    for ds_name, out_df in outcomes.items():
        out_ids_t1 = set(out_df[out_df["visit_n"] == 1]["study_id"])
        out_ids_t2 = set(out_df[out_df["visit_n"] == 2]["study_id"])
        ga_ids_t1 = set(valid_ga[valid_ga["visit_n"] == 1]["study_id"])
        ga_ids_t2 = set(valid_ga[valid_ga["visit_n"] == 2]["study_id"])
        overlap = (ga_ids_t1 & out_ids_t1) & (ga_ids_t2 & out_ids_t2)
        log(f"  overlap with '{ds_name}' (both visits, both sides): {len(overlap):,}")

    # Step 4: Cross-lagged models -- PRIMARY spec (age+sex, matches Methods
    # meth:crosslag and the E1-E3 spec). This is now the main analysis.
    results = run_cross_lagged_panel(gut_div, gut_ageaccel, outcomes, pop,
                                     covariate_set=PRIMARY_COVARIATES,
                                     spec_label="primary")

    # SENSITIVITY spec (+education+income), matching what the paper already
    # describes as its fully-adjusted sensitivity analysis (subsec:covariate)
    # -- kept as a labeled secondary analysis only, not the primary result.
    results_sensitivity = run_cross_lagged_panel(
        gut_div, gut_ageaccel, outcomes, pop,
        covariate_set=BASE_COVARIATES, spec_label="sensitivity_full_covariates")
    results_sensitivity.to_csv(
        RESULTS_DIR / "cross_lag_all_paths_sensitivity_full_covariates.csv", index=False)
    log(f"  wrote {RESULTS_DIR/'cross_lag_all_paths_sensitivity_full_covariates.csv'}")

    report_covariate_spec_comparison(results, results_sensitivity, gut_div,
                                     gut_ageaccel, outcomes, pop)

    # Step 4b: Exclusion funnel (Task 1.3 -- settles Table A9 queries (i) and (iii))
    funnel_df = build_exclusion_funnel(gut_full, gut_div, gut_ageaccel, outcomes, results)
    funnel_df.to_csv(RESULTS_DIR / "exclusion_funnel.csv", index=False)

    # Step 5: Causal direction verdict
    direction_df = causal_direction_test(results)

    # Step 6: Incident disease
    gut_div_t1 = gut_div[gut_div["visit_n"] == 1]
    incident_df = load_incident_diseases(pop, gut_div_t1)

    # Step 7: Plots
    plot_cross_lag_heatmap(results, RESULTS_DIR)
    plot_direction_verdict(direction_df, RESULTS_DIR)
    plot_path_comparison(results, RESULTS_DIR)
    plot_gut_clock(clock_data, RESULTS_DIR)
    plot_richness_3d(gut_div, RESULTS_DIR)
    plot_effect_wheel(direction_df, RESULTS_DIR)
    plot_incident_disease(incident_df, RESULTS_DIR)

    # Step 8: Save
    save_results(results, direction_df, incident_df, RESULTS_DIR)

    log("\n[DONE]")


if __name__ == "__main__":
    main()
