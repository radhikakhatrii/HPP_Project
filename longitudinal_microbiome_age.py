#!/usr/bin/env python3
"""
Longitudinal stability of gut-microbiome-derived age acceleration (HPP).

RESEARCH QUESTION
-----------------
Is gut-microbiome-derived age acceleration a stable individual trait, or does
it move within a person over time -- and how much room is there to move it?

WHY THIS QUESTION
-----------------
Li, Tang, ..., Segal & Razzak (arXiv:2510.12384) identify accelerated vs.
decelerated aging archetypes on a microbiome-centric axis, and state in their
limitations that the analyses are cross-sectional and "cannot definitively
prove causation", calling for "leveraging longitudinal data". Their archetypes
are defined at a single timepoint.

Before asking "how can we improve the microbiome to improve age", one has to
know whether the metric is stable enough to measure and mobile enough to
target. Those are opposite failure modes and both are fatal:
  - all between-person variance  -> fixed trait, no intervention can move it
  - all within-person variance   -> measurement noise, nothing real to target
The variance decomposition below distinguishes them, and the within-person SD
is a quantitative ceiling on how much any intervention could ever shift the
score.

WHY GUT ONLY
------------
Of the five omics in the published clock, only the gut microbiome has
meaningful repeat sampling in this dataset (verified against
knowledgebase.pheno.ai/datasets.html and directly in the parquets):
    gut microbiome     1,823 / 8,838 with 2+ research stages
    oral microbiome        0 / 8,816
    rna_seq                3 / 3,863
    serum_lipidomics     177 / 6,144
    nightingale           92 / 1,552
    blood_tests            0 / 10,252   <- so PhenoAge can never be longitudinal
Scope claims accordingly: this tests the gut component specifically, which is
the modality carrying most of the weight in the paper's microbiome-centric
axis -- not the full five-omics score.

METHOD NOTES
------------
1. AgeAccel is defined as RESIDUALS FROM REGRESSING BIOLOGICAL AGE ON
   CHRONOLOGICAL AGE, matching arXiv:2510.12384 verbatim. This is not the same
   as (predicted - chronological), which carries regression-to-the-mean bias.
2. The clock is trained on BASELINE SAMPLES ONLY, one per participant. Training
   on all visits would leak the same person into train and test and inflate
   apparent test-retest reliability -- the exact artifact this script exists to
   measure.
3. Model is HistGradientBoostingRegressor. The published clock uses LightGBM;
   gradient boosting was independently confirmed to beat ElasticNet on this
   data (r=0.356 vs 0.300 on species-level features). LightGBM is used instead
   if it is installed.

USAGE
-----
    conda activate pheno
    python3 longitudinal_microbiome_age.py

Paths default to ~/studies/hpp_datasets and ./results_longitudinal, and can be
overridden without editing the file:

    export HPP_DATA_ROOT=/path/to/hpp_datasets
    export HPP_RESULTS_DIR=/path/to/output

Every threshold used anywhere in the script is declared in the CONFIGURATION
block below -- nothing is buried in the analysis code.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless server, no display
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================== #
# CONFIGURATION                                                               #
# Every path and threshold used anywhere in this script is declared here.     #
# Paths and the results directory can be overridden with environment          #
# variables so the script runs unchanged on a different machine or account.   #
# =========================================================================== #

# Data root. Override with:  export HPP_DATA_ROOT=/path/to/hpp_datasets
DEFAULT_HPP_ROOT = Path.home() / "studies" / "hpp_datasets"
HPP = Path(os.environ.get("HPP_DATA_ROOT", DEFAULT_HPP_ROOT)).expanduser()

# Output directory. Override with:  export HPP_RESULTS_DIR=/path/to/results
RESULTS = Path(os.environ.get(
    "HPP_RESULTS_DIR", Path(__file__).parent / "results_longitudinal")).expanduser()
RESULTS.mkdir(parents=True, exist_ok=True)

# Dataset locations relative to the data root.
REL_GUT_META    = "gut_microbiome/gut_microbiome.parquet"
REL_GUT_SPECIES = ("gut_microbiome/abundance/metaphlan4/aggregated/"
                   "metaphlan_abundance_species.parquet")
REL_POPULATION      = "population/population.parquet"
REL_MEDICAL_COND    = "medical_conditions/medical_conditions.parquet"
REL_ANTHROPOMETRICS = "anthropometrics/anthropometrics.parquet"
REL_CURATED_DIR     = "curated_phenotypes"

GUT_META        = HPP / REL_GUT_META
GUT_SPECIES     = HPP / REL_GUT_SPECIES
POPULATION      = HPP / REL_POPULATION
MEDICAL_COND    = HPP / REL_MEDICAL_COND
ANTHROPOMETRICS = HPP / REL_ANTHROPOMETRICS
CURATED_DIR     = HPP / REL_CURATED_DIR

# Curated phenotypes tested as secondary disease outcomes. File stem ->
# column name follows the HPP convention "<stem>__curated_phenotype".
# NOTE: "bmi" is deliberately excluded. BMI is used as a CONFOUNDER in the
# models below, so testing it as an outcome would be circular.
CURATED_PHENOTYPES = [
    "abdominal_adiposity", "diabetes", "hypertension", "hyperlipidemia",
    "ischemic_heart_disease", "mafld", "ckd", "osteoporosis", "depression",
    "anxiety", "osa", "sleep_quality", "migraine",
]

# Phenotype columns are loaded under this prefix so they can never collide with
# a covariate or an existing column during the merge.
PHENO_PREFIX = "pheno_"

RANDOM_STATE = 42

# ---- cohort filters ------------------------------------------------------- #
MIN_PREVALENCE   = 0.05    # species must be present in >5% of samples
AGE_MIN, AGE_MAX = 20, 90  # plausible adult age range; guards against bad dates
MIN_INTERVAL_YR  = 0.25    # visits closer than this are not a real re-test

# ---- minimum sample sizes -------------------------------------------------- #
MIN_REPEAT_PARTICIPANTS = 100  # below this, a stability estimate is meaningless
MIN_PAIRS_FOR_CORR      = 30   # below this, a correlation is not interpretable
MIN_FEATURES_FOR_CLOCK  = 10   # below this, do not attempt to fit a clock

# ---- analysis parameters --------------------------------------------------- #
ACCEL_THRESHOLD    = 0.0            # AgeAccel > this = "accelerated" archetype.
                                    # 0 is the natural cut: AgeAccel is a
                                    # regression residual and so is mean-zero.
RTM_QUANTILE       = 0.80           # top quintile, for regression-to-the-mean
SPEARMAN_BROWN_TARGETS = (0.50, 0.70, 0.80)   # ICC levels to report k for
FDR_ALPHA          = 0.05           # BH-FDR level for age-correlation filter
CV_FOLDS           = 5

# ---- interpretation thresholds (reporting only, never alter a computation) - #
ICC_LOW            = 0.30   # below: dominated by within-person variance
ICC_HIGH           = 0.75   # above: behaves as a fixed trait
ICC_GAP_MEANINGFUL = 0.15   # diversity-vs-AgeAccel gap worth calling out
KAPPA_WEAK         = 0.20   # below: classification barely beats chance
ICC_GAIN_MEANINGFUL = 0.03  # smallest ICC improvement worth claiming

# ---- figure palette (style guide) ----------------------------------------- #
PATH_A  = "#2166AC"   # microbiome → aging (blue)
PATH_B  = "#D6604D"   # aging → microbiome (red)
BIDIR   = "#762A83"   # bidirectional (purple)
NEUTRAL = "#878787"   # neutral / stability
COL_INK = "#1A1A1A"
# backward-compat aliases
COL_TEAL = PATH_A
COL_RUST = PATH_B
COL_GRAY = NEUTRAL
COL_LGRAY= "#CCCCCC"

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patheffects as _pe
from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde as _gaussian_kde

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["IBM Plex Sans", "Arial", "DejaVu Sans"],
    "font.size":          7,
    "axes.labelsize":     8,
    "axes.titlesize":     9,
    "axes.edgecolor":     "#1A1A1A",
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     "#F8F8F6",
    "figure.facecolor":   "white",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.7,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.dpi":        300,
})

_MONO = {"fontfamily": "monospace"}   # IBM Plex Mono fallback for stat callouts


def _lift(patch, ax, dy: int = -2, alpha: float = 0.28) -> None:
    """Drop-shadow elevation used on all data marks that carry a result."""
    patch.set_path_effects([
        _pe.withSimplePatchShadow(offset=(1, dy), shadow_rgbFace="#1A1A1A",
                                  alpha=alpha),
        _pe.Normal(),
    ])


def _gradient_fill(ax, bar, top_color: str, bottom_color: str) -> None:
    """Vertical 2-stop gradient clipped to a bar's path — the elevated-bar look."""
    x, y = bar.get_x(), bar.get_y()
    w, h = bar.get_width(), bar.get_height()
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = _LSC.from_list("bar", [top_color, bottom_color])
    im = ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto",
                   cmap=cmap, zorder=bar.get_zorder())
    im.set_clip_path(bar)
    bar.set_facecolor("none")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Loading -- preserves research_stage, unlike the main pipeline's _load()      #
# --------------------------------------------------------------------------- #
def load_preserving_visits(path: Path, label: str) -> pd.DataFrame:
    """Read an HPP parquet WITHOUT collapsing to one row per participant.

    The main pipeline's _load() aggregates repeat visits away (mean for numeric,
    first for categorical) because every other analysis is cross-sectional.
    That aggregation is exactly what destroys the signal this script needs, so
    this reads the MultiIndex out into columns and leaves every row intact.
    """
    if not path.exists():
        sys.exit(f"FATAL: {label} not found at {path}")

    df = pd.read_parquet(path)
    print(f"  [{label}] raw shape = {df.shape}")

    if isinstance(df.index, pd.MultiIndex):
        print(f"  [{label}] index levels = {list(df.index.names)}")
        df = df.reset_index()
    else:
        df = df.reset_index()

    if "participant_id" in df.columns:
        # Some HPP tables (population in particular) already carry a column
        # literally named "study_id" holding the COHORT number, not a
        # participant identifier. Renaming participant_id on top of it would
        # produce two columns with the same name, after which df["study_id"]
        # returns a DataFrame and every downstream Series op fails. Drop the
        # cohort column first -- the participant id is the join key we want.
        df = df.drop(columns=["study_id"], errors="ignore")
        df = df.rename(columns={"participant_id": "study_id"})
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    if "study_id" not in df.columns:
        sys.exit(f"FATAL: no participant identifier in {label}. "
                 f"Columns seen: {list(df.columns)[:15]}")

    df["study_id"] = pd.to_numeric(df["study_id"], errors="coerce")
    df = df.dropna(subset=["study_id"])
    df["study_id"] = df["study_id"].astype("int64")

    if "research_stage" in df.columns:
        n_multi = (df.groupby("study_id")["research_stage"].nunique() >= 2).sum()
        print(f"  [{label}] {df['study_id'].nunique():,} participants, "
              f"{len(df):,} rows, {n_multi:,} with 2+ research stages")
    else:
        print(f"  [{label}] WARNING: no research_stage column "
              f"({df['study_id'].nunique():,} participants, {len(df):,} rows)")

    return df


def build_longitudinal_frame() -> tuple[pd.DataFrame, list[str]]:
    """Join species abundances to visit metadata and chronological age."""
    banner("STEP 1  Load gut microbiome preserving repeat visits")

    meta    = load_preserving_visits(GUT_META, "gut_microbiome meta")
    species = load_preserving_visits(GUT_SPECIES, "gut species")
    pop     = load_preserving_visits(POPULATION, "population")

    scols = [c for c in species.columns if "|s__" in c] or \
            [c for c in species.columns if "s__" in c]
    if not scols:
        sys.exit("FATAL: no species columns found (expected '|s__' in names)")
    print(f"  species feature columns: {len(scols):,}")

    # Join species to metadata. Prefer sample_name when both carry it, since
    # that is unambiguous; fall back to (study_id, research_stage).
    meta_cols = [c for c in ["study_id", "research_stage", "collection_date",
                             "sample_name"] if c in meta.columns]
    if "sample_name" in species.columns and "sample_name" in meta.columns:
        key = ["sample_name"]
        print("  joining species <-> meta on sample_name")
    elif "research_stage" in species.columns:
        key = ["study_id", "research_stage"]
        print("  joining species <-> meta on (study_id, research_stage)")
    else:
        sys.exit("FATAL: cannot align species to visits -- species table has "
                 f"neither sample_name nor research_stage. Columns: {list(species.columns)[:15]}")

    keep_species = [c for c in (["study_id"] + key + scols) if c in species.columns]
    df = meta[meta_cols].merge(
        species[list(dict.fromkeys(keep_species))],
        on=key, how="inner", suffixes=("", "_sp"))
    if "study_id_sp" in df.columns:
        df = df.drop(columns=["study_id_sp"])
    print(f"  joined shape = {df.shape}")

    # Chronological age at each visit, from that visit's collection date.
    # Using a single baseline age for both visits would mechanically force the
    # two timepoints apart and corrupt the whole analysis.
    # Sex encoding confirmed against the HPP coding dictionary
    # (metadata/coding_mapping.csv, field code 9): raw "0" = Female, "1" = Male.
    pop_cols = ["study_id", "year_of_birth"] + (["sex"] if "sex" in pop.columns else [])
    pop_small = pop[pop_cols].drop_duplicates("study_id")
    df = df.merge(pop_small, on="study_id", how="inner")
    if "sex" in df.columns:
        df["sex_numeric"] = (df["sex"]
                             .map({"Female": 0, "Male": 1, "0": 0, "1": 1,
                                   0: 0, 1: 1})
                             .astype("float64"))
        print(f"  sex mapped: {df['sex_numeric'].notna().sum():,} of {len(df):,} "
              f"(0=Female, 1=Male)")

    if "collection_date" not in df.columns:
        sys.exit("FATAL: no collection_date -- cannot compute age at each visit")

    df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
    df = df.dropna(subset=["collection_date"])
    df["age"] = (df["collection_date"].dt.year
                 - pd.to_numeric(df["year_of_birth"], errors="coerce"))
    df = df.dropna(subset=["age"])
    df = df[df["age"].between(AGE_MIN, AGE_MAX)]

    df[scols] = df[scols].fillna(0.0)

    # Shannon diversity, carried through as a SPECIFICITY CONTROL. If diversity
    # turns out to be stable within-person while the age-acceleration score is
    # not, the instability is a property of the CLOCK, not of the microbiome
    # itself -- two very different conclusions. Without this comparison the
    # stability result is ambiguous.
    X = df[scols].values / 100.0                    # relative abundance -> fraction
    X_safe = np.where(X > 0, X, 1e-10)
    df["shannon"]  = -np.sum(X_safe * np.log(X_safe), axis=1)
    df["richness"] = (X > 0).sum(axis=1)
    print(f"  Shannon diversity: mean={df['shannon'].mean():.3f}, "
          f"SD={df['shannon'].std():.3f}")

    n_repeat = (df.groupby("study_id").size() >= 2).sum()
    print(f"\n  final: {len(df):,} samples, {df['study_id'].nunique():,} participants")
    print(f"  participants with 2+ usable samples: {n_repeat:,}")
    if n_repeat < MIN_REPEAT_PARTICIPANTS:
        sys.exit(f"FATAL: only {n_repeat} participants have 2+ usable samples "
                 f"(need >= {MIN_REPEAT_PARTICIPANTS}) -- not enough for a "
                 "stability analysis. Check the join above.")

    return df, scols


# --------------------------------------------------------------------------- #
# Clock                                                                        #
# --------------------------------------------------------------------------- #
def fit_clock(df: pd.DataFrame, scols: list[str]) -> pd.DataFrame:
    """Train on baseline samples only, then predict microbiome age everywhere.

    Training on every visit would put the same participant in both train and
    test. The model would learn person-specific quirks and reproduce them at
    both timepoints, inflating test-retest reliability -- which is precisely
    the quantity being measured. Baseline-only training keeps the reliability
    estimate honest.
    """
    banner("STEP 2  Fit microbiome age clock on baseline samples only")

    df = df.sort_values(["study_id", "collection_date"]).copy()
    df["visit_index"] = df.groupby("study_id").cumcount()

    # CRITICAL: train ONLY on participants who have a single visit, so that
    # every repeat-visit participant is entirely held out and BOTH of their
    # visits are out-of-sample.
    #
    # An earlier version trained on all baseline samples, which put the visit-1
    # sample of every repeat participant INTO the training set while visit 2
    # stayed out. Test-retest then compared an overfit in-sample prediction
    # against an honest out-of-sample one, which DEFLATES the correlation and
    # made reliability look far worse than it is (ICC 0.184 vs 0.489 once the
    # participants are properly held out). Leakage usually inflates a result;
    # here the asymmetry between the two visits suppressed it instead.
    counts     = df.groupby("study_id").size()
    repeat_ids = set(counts[counts >= 2].index)

    baseline = df[(df["visit_index"] == 0) & (~df["study_id"].isin(repeat_ids))]
    print(f"  training samples (single-visit participants only): {len(baseline):,}")
    print(f"  fully held-out repeat participants               : {len(repeat_ids):,}")
    print(f"  their samples (both visits out-of-sample)        : "
          f"{df['study_id'].isin(repeat_ids).sum():,}")

    prev = (baseline[scols] > 0).mean()
    keep = prev[prev > MIN_PREVALENCE].index.tolist()
    print(f"  {len(keep):,}/{len(scols):,} species pass prevalence >{MIN_PREVALENCE:.0%}")
    if len(keep) < MIN_FEATURES_FOR_CLOCK:
        sys.exit(f"FATAL: only {len(keep)} species pass the prevalence filter "
                 f"(need >= {MIN_FEATURES_FOR_CLOCK})")

    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
        model_name = "LightGBM (matches published clock)"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        model_name = "HistGradientBoosting (LightGBM not installed)"
    print(f"  model: {model_name}")

    from scipy.stats import pearsonr
    from sklearn.model_selection import KFold, cross_val_predict

    X_base = baseline[keep].values
    y_base = baseline["age"].values

    cv_pred = cross_val_predict(
        model, X_base, y_base,
        cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
    r_cv   = float(pearsonr(cv_pred, y_base)[0])
    mae_cv = float(np.mean(np.abs(cv_pred - y_base)))
    print(f"  cross-validated clock on baseline: r={r_cv:.3f}, MAE={mae_cv:.1f} yr")

    model.fit(X_base, y_base)
    df["microbiome_age"] = model.predict(df[keep].values)

    # AgeAccel = residuals of biological age regressed on chronological age,
    # per arXiv:2510.12384. NOT (predicted - chronological), which carries
    # regression-to-the-mean bias: shrunk predictions make younger people look
    # accelerated and older people decelerated purely as an artifact.
    import statsmodels.api as sm
    # Fit BA ~ CA on the analysis population (the fully held-out repeat-visit
    # participants). Held-out predictions shrink toward the mean more than
    # in-sample ones, so a regression fitted across both groups would leave
    # systematic age-related structure in the held-out residuals.
    rep = df[df["study_id"].isin(repeat_ids)]
    ols = sm.OLS(rep["microbiome_age"].astype(float),
                 sm.add_constant(rep["age"].astype(float))).fit()
    df["age_accel"] = (df["microbiome_age"].astype(float)
                       - (ols.params.iloc[0]
                          + ols.params.iloc[1] * df["age"].astype(float)))
    print(f"  AgeAccel = residuals of BA ~ CA  (slope={ols.params.iloc[1]:.3f})")
    print(f"  AgeAccel: mean={df['age_accel'].mean():+.3f}, "
          f"SD={df['age_accel'].std():.3f} yr")

    # Sanity check: residuals must be uncorrelated with chronological age by
    # construction. If not, something upstream is wrong.
    # Checked on the analysis population, since that is where the regression
    # was fitted and therefore where the residuals are orthogonal to age.
    rep2 = df[df["study_id"].isin(repeat_ids)]
    r_check = float(pearsonr(rep2["age_accel"], rep2["age"])[0])
    print(f"  sanity check, corr(AgeAccel, chronological age) = {r_check:+.4f} "
          f"(should be ~0 on the analysis population)")

    return df


# --------------------------------------------------------------------------- #
# Stability                                                                    #
# --------------------------------------------------------------------------- #
def compute_icc(data: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """ICC(1) via a random-intercept model: var_between / (var_between + var_within).

    Returns (icc, sd_between, sd_within). NaNs if the model fails to converge.
    """
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm(f"{col} ~ 1", data, groups=data["study_id"]).fit()
        var_between = float(md.cov_re.iloc[0, 0])
        var_within  = float(md.scale)
        icc = var_between / (var_between + var_within)
        return icc, np.sqrt(var_between), np.sqrt(var_within)
    except Exception as exc:
        print(f"    mixed model failed for {col} ({type(exc).__name__}: {exc})")
        return np.nan, np.nan, np.nan


def analyse_stability(df: pd.DataFrame) -> float:
    """Returns the ICC of AgeAccel, which Step 5 needs for Spearman-Brown."""
    banner("STEP 3  Within-person stability of microbiome age acceleration")

    from scipy.stats import pearsonr

    counts  = df.groupby("study_id").size()
    repeat  = df[df["study_id"].isin(counts[counts >= 2].index)].copy()
    n_rep   = repeat["study_id"].nunique()
    print(f"  participants with 2+ samples: {n_rep:,}  ({len(repeat):,} samples)")

    # ---- variance decomposition via random-intercept model ----------------- #
    print(f"\n  VARIANCE DECOMPOSITION (random-intercept model)")
    icc, sd_between, sd_within = compute_icc(repeat, "age_accel")
    if np.isfinite(icc):
        print(f"    between-person SD : {sd_between:.3f} yr")
        print(f"    within-person  SD : {sd_within:.3f} yr")
        print(f"    ICC               : {icc:.3f}")

    # ---- SPECIFICITY CONTROL: is the microbiome unstable, or the clock? ---- #
    icc_sh, sdb_sh, sdw_sh = compute_icc(repeat, "shannon")
    icc_ri, _, _           = compute_icc(repeat, "richness")
    print(f"\n  SPECIFICITY CONTROL (same participants, same samples)")
    print(f"    ICC, Shannon diversity : {icc_sh:.3f}")
    print(f"    ICC, species richness  : {icc_ri:.3f}")
    print(f"    ICC, AgeAccel          : {icc:.3f}")
    if np.isfinite(icc_sh) and np.isfinite(icc):
        if icc_sh > icc + ICC_GAP_MEANINGFUL:
            print("    -> Diversity is MORE stable than AgeAccel. The instability is")
            print("       a property of the CLOCK, not of the microbiome itself.")
        elif abs(icc_sh - icc) <= ICC_GAP_MEANINGFUL:
            print("    -> Diversity and AgeAccel are comparably unstable. The gut")
            print("       microbiome itself fluctuates at this timescale.")
        else:
            print("    -> AgeAccel is more stable than diversity (unexpected).")

    # ---- direct test-retest on first vs second visit ----------------------- #
    # visit_index (from cumcount on date-sorted rows) rather than .nth(),
    # whose return shape differs across pandas versions.
    pair_cols = ["study_id", "age", "microbiome_age", "age_accel", "shannon",
                 "collection_date"]
    first  = repeat[repeat["visit_index"] == 0][pair_cols]
    second = repeat[repeat["visit_index"] == 1][pair_cols]
    paired = first.merge(second, on="study_id", suffixes=("_v1", "_v2"))
    paired["interval_yr"] = ((paired["collection_date_v2"]
                              - paired["collection_date_v1"]).dt.days / 365.25)
    paired = paired[paired["interval_yr"] > MIN_INTERVAL_YR]
    print(f"\n  TEST-RETEST  (n={len(paired):,} pairs, "
          f"median interval {paired['interval_yr'].median():.2f} yr)")

    r_age = float(pearsonr(paired["microbiome_age_v1"],
                           paired["microbiome_age_v2"])[0])
    r_acc = float(pearsonr(paired["age_accel_v1"], paired["age_accel_v2"])[0])
    print(f"    microbiome age  visit1 vs visit2 : r = {r_age:.3f}")
    print(f"    AgeAccel        visit1 vs visit2 : r = {r_acc:.3f}")

    # ---- longitudinal rate of aging --------------------------------------- #
    # A LONGITUDINAL definition of accelerated aging: did the microbiome age
    # by more than the calendar time that actually elapsed? The published
    # archetypes are cross-sectional; this is measured rather than inferred.
    paired["delta_mb_age"] = (paired["microbiome_age_v2"]
                              - paired["microbiome_age_v1"])
    paired["aging_rate"]   = paired["delta_mb_age"] / paired["interval_yr"]
    pct_accel = 100 * (paired["aging_rate"] > 1.0).mean()

    print(f"\n  LONGITUDINAL AGING RATE  (microbiome yr per calendar yr)")
    print(f"    mean   : {paired['aging_rate'].mean():+.3f}")
    print(f"    median : {paired['aging_rate'].median():+.3f}")
    print(f"    SD     : {paired['aging_rate'].std():.3f}")
    print(f"    aging faster than real time (rate > 1.0): {pct_accel:.1f}%")

    # ---- ARCHETYPE REPRODUCIBILITY ----------------------------------------- #
    # arXiv:2510.12384 classifies participants into aging-accelerated vs
    # aging-decelerated groups from a single timepoint. If that label does not
    # reproduce when the same person is re-measured, the archetype is not a
    # stable property of the person. AgeAccel is mean-zero by construction
    # (it is a regression residual), so sign is the natural cut.
    acc_v1 = paired["age_accel_v1"] > ACCEL_THRESHOLD
    acc_v2 = paired["age_accel_v2"] > ACCEL_THRESHOLD
    agree  = float((acc_v1 == acc_v2).mean())

    p1, p2 = acc_v1.mean(), acc_v2.mean()
    p_chance = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (agree - p_chance) / (1 - p_chance) if p_chance < 1 else np.nan

    stay_accel = float(acc_v2[acc_v1].mean()) if acc_v1.any() else np.nan

    print(f"\n  ARCHETYPE REPRODUCIBILITY  "
          f"(accelerated = AgeAccel > {ACCEL_THRESHOLD:g})")
    print(f"    same class at both visits : {100*agree:.1f}%")
    print(f"    expected by chance        : {100*p_chance:.1f}%")
    print(f"    Cohen's kappa             : {kappa:.3f}")
    print(f"    of those accelerated at v1, still accelerated at v2: "
          f"{100*stay_accel:.1f}%")
    if kappa < KAPPA_WEAK:
        print("    -> Classification is barely better than chance. The accelerated/")
        print("       decelerated label is not a reproducible property of a person")
        print("       when measured from a single sample.")

    # ---- ATTENUATION FROM UNRELIABILITY ------------------------------------ #
    # Classical psychometrics: an unreliable measure attenuates every
    # association it participates in. observed_r = true_r * sqrt(ICC).
    # Required N scales as 1/r^2, so N inflates by 1/ICC.
    if np.isfinite(icc) and icc > 0:
        atten   = np.sqrt(icc)
        n_infl  = 1.0 / icc
        print(f"\n  ATTENUATION DUE TO UNRELIABILITY")
        print(f"    attenuation factor sqrt(ICC)     : {atten:.3f}")
        print(f"    -> a true association of r is observed as {atten:.2f}r")
        print(f"    sample-size inflation (1/ICC)    : {n_infl:.1f}x")
        print(f"    -> detecting a true effect needs {n_infl:.1f}x more participants")
        print("       than an equivalent effect on a perfectly reliable measure.")

        # ---- SPEARMAN-BROWN: repeat samples needed for usable reliability -- #
        print(f"\n  SAMPLES PER PERSON NEEDED (Spearman-Brown)")
        for target in SPEARMAN_BROWN_TARGETS:
            k = target * (1 - icc) / (icc * (1 - target))
            print(f"    to reach ICC={target:.2f}: k = {k:.1f} samples per person")

    # ---- REGRESSION TO THE MEAN -------------------------------------------- #
    # Practical warning for trial design: if participants are enrolled BECAUSE
    # they are accelerated at baseline, an unreliable measure guarantees they
    # will appear to improve at follow-up with no intervention at all.
    q_cut = paired["age_accel_v1"].quantile(RTM_QUANTILE)
    top = paired[paired["age_accel_v1"] >= q_cut]
    grand = paired["age_accel_v1"].mean()
    obs_rev = float(top["age_accel_v1"].mean() - top["age_accel_v2"].mean())
    exp_rev = float((1 - r_acc) * (top["age_accel_v1"].mean() - grand))

    print(f"\n  REGRESSION TO THE MEAN  "
          f"(top {1 - RTM_QUANTILE:.0%} at visit 1, n={len(top):,})")
    print(f"    mean AgeAccel at v1 : {top['age_accel_v1'].mean():+.3f} yr")
    print(f"    mean AgeAccel at v2 : {top['age_accel_v2'].mean():+.3f} yr")
    print(f"    observed reversion  : {obs_rev:+.3f} yr")
    print(f"    predicted by RTM    : {exp_rev:+.3f} yr")
    print("    -> A single-arm trial enrolling 'accelerated agers' would show")
    print(f"       an apparent {obs_rev:.2f} yr improvement from this artifact alone,")
    print("       with no intervention. Controls are mandatory.")

    # ---- interpretation ---------------------------------------------------- #
    print("\n  INTERPRETATION")
    if np.isfinite(icc):
        if icc < ICC_LOW:
            print(f"    ICC={icc:.2f} is LOW. Most variance is within-person, i.e.")
            print("    the score is dominated by noise/fluctuation rather than a")
            print("    stable individual characteristic. As an intervention target")
            print("    it is unreliable at the individual level -- and this would")
            print("    explain null lifestyle effects independently of exposure quality.")
        elif icc > ICC_HIGH:
            print(f"    ICC={icc:.2f} is HIGH. The score behaves largely as a fixed")
            print("    trait over this interval, leaving little within-person room")
            print("    for any intervention to move it.")
        else:
            print(f"    ICC={icc:.2f} is MODERATE: reliable enough to measure, with")
            print("    real within-person movement. Viable as a target.")
        print(f"\n    CEILING ON INTERVENTION EFFECT: within-person SD = "
              f"{sd_within:.2f} yr.")
        print("    No intervention can plausibly shift this score by more than a")
        print("    fraction of that. Any claimed effect larger than it should be")
        print("    treated as implausible.")

    # ---- outputs ----------------------------------------------------------- #
    summary = pd.DataFrame([dict(
        n_participants_repeat=n_rep, n_pairs=len(paired),
        median_interval_yr=paired["interval_yr"].median(),
        icc_age_accel=icc, sd_between_yr=sd_between, sd_within_yr=sd_within,
        icc_shannon=icc_sh, icc_richness=icc_ri,
        retest_r_microbiome_age=r_age, retest_r_age_accel=r_acc,
        aging_rate_mean=paired["aging_rate"].mean(),
        aging_rate_sd=paired["aging_rate"].std(),
        pct_rate_above_1=pct_accel,
        archetype_agreement_pct=100 * agree,
        archetype_chance_pct=100 * p_chance,
        archetype_kappa=kappa,
        attenuation_factor=np.sqrt(icc) if np.isfinite(icc) and icc > 0 else np.nan,
        n_inflation=1.0 / icc if np.isfinite(icc) and icc > 0 else np.nan,
        rtm_observed_reversion_yr=obs_rev,
        rtm_predicted_reversion_yr=exp_rev,
    )])
    summary.to_csv(RESULTS / "stability_summary.csv", index=False)
    paired.to_csv(RESULTS / "paired_visits.csv", index=False)
    print(f"\n  wrote stability_summary.csv and paired_visits.csv")

    _figures(paired, r_acc, icc)
    return icc


def _figures(paired: pd.DataFrame, r_acc: float, icc: float) -> None:
    fig = plt.figure(figsize=(12.0, 4.2))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.55,
                            left=0.07, right=0.97, top=0.82, bottom=0.18)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── Panel A: AgeAccel test-retest scatter ─────────────────────────────────
    ax = axes[0]
    all_vals = pd.concat([paired["age_accel_v1"], paired["age_accel_v2"]]).dropna()
    lim = float(np.nanpercentile(np.abs(all_vals), 99)) * 1.05

    # Colour by deviation from diagonal (shows regressor-to-mean)
    delta = (paired["age_accel_v2"] - paired["age_accel_v1"]).abs()
    vmax  = delta.quantile(0.95)
    # bead treatment: 3 px points with 1 px darker-edge halo
    sc = ax.scatter(paired["age_accel_v1"], paired["age_accel_v2"],
                    s=9, alpha=0.55, c=delta, cmap="coolwarm",
                    vmin=0, vmax=vmax, linewidths=0.8,
                    edgecolors=COL_INK, zorder=3)
    # zero-reference: flat, dashed, shadow-free
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.9, color=COL_INK,
            alpha=0.5, zorder=1)
    # regression line with elevation shadow
    m, b = np.polyfit(paired["age_accel_v1"].dropna(),
                      paired["age_accel_v2"].dropna(), 1)
    xs = np.linspace(-lim, lim, 100)
    reg_line, = ax.plot(xs, m * xs + b, color=PATH_A, lw=1.4, zorder=4)
    _lift(reg_line, ax, dy=-1, alpha=0.22)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("AgeAccel, Visit 1 (yr)", labelpad=6)
    ax.set_ylabel("AgeAccel, Visit 2 (yr)", labelpad=6)
    # stat callout in IBM Plex Mono
    ax.text(0.05, 0.95,
            f"r = {r_acc:.3f}\nICC = {icc:.3f}\nn = {len(paired):,}",
            transform=ax.transAxes, va="top", fontsize=6.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COL_LGRAY, alpha=0.9))
    cbar = plt.colorbar(sc, ax=ax, pad=0.03, fraction=0.03, shrink=0.80)
    cbar.set_label("|Δ AgeAccel| (yr)", fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5.5, pad=2)
    ax.text(-0.18, 1.05, "a", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", fontfamily="sans-serif")

    # ── Panel B: Longitudinal aging-rate distribution (KDE alpha-fade) ────────
    ax = axes[1]
    rates = paired["aging_rate"].dropna()
    bins  = np.linspace(rates.quantile(0.01), rates.quantile(0.99), 40)
    # soft histogram bars with gradient fill
    n_hist, bin_edges, patches = ax.hist(
        rates, bins=bins, color=PATH_A, alpha=0.0, linewidth=0, density=True)
    for patch in patches:
        _gradient_fill(ax, patch,
                       top_color=PATH_A,
                       bottom_color=PATH_A + "55")   # 33% alpha at bottom
        patch.set_alpha(0.75)

    # KDE curve with elevation shadow
    kde  = _gaussian_kde(rates, bw_method=0.3)
    xs   = np.linspace(bins[0], bins[-1], 200)
    kde_line, = ax.plot(xs, kde(xs), color=PATH_A, lw=1.4, zorder=4)
    _lift(kde_line, ax, dy=-1, alpha=0.22)

    # KDE fill: vertical alpha-fade from curve down to baseline
    ys = kde(xs)
    ax.fill_between(xs, ys, alpha=0.18, color=PATH_A, zorder=2)

    # reference lines — flat, dashed, shadow-free
    ax.axvline(1.0, ls="--", lw=1.2, color=PATH_B, zorder=1)
    ax.axvline(rates.mean(), ls=":", lw=1.0, color=COL_INK, zorder=1)
    pct_accel = float((rates > 1).mean() * 100)

    # label lines directly above the plot, outside axes — no legend needed
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.35
    ax.annotate("Rate = 1", xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(1.0, 1.06), textcoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=6, color=PATH_B,
                arrowprops=dict(arrowstyle="-", color=PATH_B, lw=0.7))
    ax.annotate(f"Mean\n{rates.mean():.2f}", xy=(rates.mean(), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(rates.mean(), 1.06), textcoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=6, color=COL_INK,
                arrowprops=dict(arrowstyle="-", color=COL_INK, lw=0.7))

    ax.set_xlabel("Microbiome aging rate\n(mb-yr per calendar yr)", labelpad=6)
    ax.set_ylabel("Density")
    ax.text(0.98, 0.97,
            f"{pct_accel:.0f}% rate > 1",
            transform=ax.transAxes, va="top", ha="right", fontsize=6.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COL_LGRAY, alpha=0.9))
    ax.text(-0.18, 1.05, "b", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", fontfamily="sans-serif")

    # ── Panel C: Aging rate vs chronological age ──────────────────────────────
    ax = axes[2]
    valid = paired.dropna(subset=["age_v1", "aging_rate"])
    hb = ax.hexbin(valid["age_v1"], valid["aging_rate"],
                   gridsize=25, cmap="Blues", mincnt=1, linewidths=0.2)
    # zero-reference: flat, dashed, shadow-free
    ax.axhline(1.0, ls="--", lw=1.0, color=PATH_B, alpha=0.8, zorder=1)

    valid_s = valid.sort_values("age_v1")
    roll_med = valid_s["aging_rate"].rolling(
        window=50, center=True, min_periods=20).median()
    med_line, = ax.plot(valid_s["age_v1"], roll_med,
                        color=COL_INK, lw=1.2, label="Rolling median", zorder=4)
    _lift(med_line, ax, dy=-1, alpha=0.20)

    cbar2 = plt.colorbar(hb, ax=ax, pad=0.03, fraction=0.03, shrink=0.80)
    cbar2.set_label("Count", fontsize=6, labelpad=4)
    cbar2.ax.tick_params(labelsize=5.5, pad=2)
    ax.set_xlabel("Chronological age at Visit 1 (yr)", labelpad=6)
    ax.set_ylabel("Microbiome aging rate", labelpad=6)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.text(-0.18, 1.05, "c", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", fontfamily="sans-serif")

    # ── Shared formatting ─────────────────────────────────────────────────────
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # suptitle in serif per style guide
    fig.suptitle(
        "Longitudinal Stability of Gut-Microbiome-Derived Age Acceleration (HPP)",
        fontsize=9, fontweight="bold", y=1.04,
        fontfamily="serif",
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_longitudinal_stability.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved figure_longitudinal_stability.pdf / .png")


# --------------------------------------------------------------------------- #
# STEP 4  Stability-weighted clock                                             #
# --------------------------------------------------------------------------- #
def build_stability_weighted_clock(df: pd.DataFrame, scols: list[str]) -> None:
    """Constructive fix: select features on age-correlation AND within-person
    stability, rather than age-correlation alone.

    The published pipeline (arXiv:2510.12384) selects features purely by
    Spearman correlation with chronological age (BH-FDR < 0.05) then SHAP.
    Nothing in that procedure asks whether a taxon is *reproducible within a
    person*. A species can correlate with age across the population while
    bouncing around wildly within an individual -- such a feature adds
    cross-sectional accuracy but injects within-person noise. Step 3 tests for
    exactly this by comparing AgeAccel reliability against the reliability of
    the raw diversity measures computed on the same samples.

    ANTI-LEAKAGE DESIGN. Participants with repeat visits are split in half:
      SEL  -- used ONLY to estimate per-species within-person stability
      EVAL -- used ONLY to evaluate the resulting clock's ICC
    The clock itself is trained on baseline samples of non-EVAL participants.
    Selecting features for stability and then measuring stability on the same
    samples would guarantee an improvement whether or not one exists.
    """
    banner("STEP 4  Stability-weighted clock (constructive fix)")

    from scipy.stats import pearsonr, spearmanr
    from sklearn.model_selection import KFold, cross_val_predict
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm

    counts     = df.groupby("study_id").size()
    repeat_ids = np.array(sorted(counts[counts >= 2].index))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(repeat_ids)
    half      = len(repeat_ids) // 2
    sel_ids   = set(repeat_ids[:half])
    eval_ids  = set(repeat_ids[half:])
    print(f"  repeat-visit participants: {len(repeat_ids):,}")
    print(f"    SEL  (stability estimation): {len(sel_ids):,}")
    print(f"    EVAL (ICC evaluation)      : {len(eval_ids):,}")

    # ---- per-species within-person stability, estimated on SEL only -------- #
    sel = df[df["study_id"].isin(sel_ids)]
    v1 = sel[sel["visit_index"] == 0].set_index("study_id")
    v2 = sel[sel["visit_index"] == 1].set_index("study_id")
    common = v1.index.intersection(v2.index)
    v1, v2 = v1.loc[common], v2.loc[common]
    print(f"  estimating species stability on {len(common):,} paired SEL participants")

    # Only species with enough non-zero observations can have a meaningful
    # within-person correlation.
    train_pool = df[(df["visit_index"] == 0) & (~df["study_id"].isin(eval_ids))]
    prev = (train_pool[scols] > 0).mean()
    candidates = prev[prev > MIN_PREVALENCE].index.tolist()
    print(f"  {len(candidates):,}/{len(scols):,} species pass prevalence >{MIN_PREVALENCE:.0%}")

    stab = {}
    a = v1[candidates].values
    b = v2[candidates].values
    for j, sp in enumerate(candidates):
        x, y = a[:, j], b[:, j]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            stab[sp] = 0.0
        else:
            stab[sp] = float(np.corrcoef(x, y)[0, 1])
    stability = pd.Series(stab).fillna(0.0)
    print(f"  species stability: median r={stability.median():.3f}, "
          f"IQR {stability.quantile(.25):.3f}-{stability.quantile(.75):.3f}")

    # ---- age-correlation filter, matching the published procedure ---------- #
    ages = train_pool["age"].values
    pvals, rhos = [], []
    for sp in candidates:
        rho, p = spearmanr(train_pool[sp].values, ages)
        rhos.append(0.0 if not np.isfinite(rho) else rho)
        pvals.append(1.0 if not np.isfinite(p) else p)
    rej, _, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method="fdr_bh")
    age_sig = [sp for sp, keep in zip(candidates, rej) if keep]
    print(f"  {len(age_sig):,} species pass age-correlation BH-FDR < {FDR_ALPHA}")

    # Method A = published criterion. Method B additionally requires the taxon
    # to be reproducible within a person.
    stab_cut = stability[age_sig].median()
    stable_sig = [sp for sp in age_sig if stability[sp] >= stab_cut]
    print(f"  stability cut (median of age-significant): r >= {stab_cut:.3f}")
    print(f"  Method A (age only)            : {len(age_sig):,} features")
    print(f"  Method B (age + stability)     : {len(stable_sig):,} features")

    try:
        from lightgbm import LGBMRegressor
        make_model = lambda: LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        make_model = lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    eval_df = df[df["study_id"].isin(eval_ids)].copy()
    rows = []
    for label, feats in [("A: age-correlation only (published)", age_sig),
                         ("B: age-correlation + stability", stable_sig)]:
        if len(feats) < MIN_FEATURES_FOR_CLOCK:
            print(f"  {label}: only {len(feats)} features "
                  f"(need >= {MIN_FEATURES_FOR_CLOCK}) -- skipped")
            continue

        model = make_model()
        Xtr, ytr = train_pool[feats].values, train_pool["age"].values
        cv_pred = cross_val_predict(
            model, Xtr, ytr,
            cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))
        r_cv   = float(pearsonr(cv_pred, ytr)[0])
        mae_cv = float(np.mean(np.abs(cv_pred - ytr)))

        model.fit(Xtr, ytr)
        ev = eval_df.copy()
        ev["mb_age"] = model.predict(ev[feats].values)
        resid = sm.OLS(ev["mb_age"].astype(float),
                       sm.add_constant(ev["age"].astype(float))).fit().resid
        ev["acc"] = resid

        icc_b, sdb, sdw = compute_icc(ev, "acc")
        p1 = ev[ev["visit_index"] == 0].set_index("study_id")["acc"]
        p2 = ev[ev["visit_index"] == 1].set_index("study_id")["acc"]
        idx = p1.index.intersection(p2.index)
        r_retest = (float(pearsonr(p1.loc[idx], p2.loc[idx])[0])
                    if len(idx) > MIN_PAIRS_FOR_CORR else np.nan)

        print(f"\n  {label}")
        print(f"    features            : {len(feats):,}")
        print(f"    clock accuracy      : r={r_cv:.3f}, MAE={mae_cv:.1f} yr")
        print(f"    AgeAccel ICC (EVAL) : {icc_b:.3f}")
        print(f"    AgeAccel retest r   : {r_retest:.3f}")

        rows.append(dict(method=label, n_features=len(feats),
                         clock_r=r_cv, clock_mae=mae_cv,
                         icc=icc_b, retest_r=r_retest,
                         sd_between=sdb, sd_within=sdw))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "stability_weighted_clock.csv", index=False)
    stability.rename("within_person_stability").to_frame().to_csv(
        RESULTS / "species_stability.csv")
    print("\n  wrote stability_weighted_clock.csv, species_stability.csv")

    if len(out) == 2:
        d_icc = out.iloc[1]["icc"] - out.iloc[0]["icc"]
        d_r   = out.iloc[1]["clock_r"] - out.iloc[0]["clock_r"]
        print(f"\n  TRADE-OFF")
        print(f"    ICC change          : {d_icc:+.3f}")
        print(f"    clock accuracy (r)  : {d_r:+.3f}")
        if d_icc > ICC_GAIN_MEANINGFUL:
            print("    -> Stability-aware selection improves reliability. Reporting the")
            print("       accuracy cost alongside it is essential: a clock optimised")
            print("       purely for cross-sectional r can be unusable per-individual.")
        else:
            print("    -> No meaningful reliability gain. The instability is not driven")
            print("       by a separable subset of unstable taxa, which points instead")
            print("       to the modelling step itself amplifying sample-level noise.")


# --------------------------------------------------------------------------- #
# STEP 5  Trait vs. instability, and whether repeat sampling strengthens        #
#         the disease associations reported in arXiv:2510.12384                 #
# --------------------------------------------------------------------------- #
def _partial_corr(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """Partial correlation of x with y, adjusting for covariates Z.

    Both x and y are residualised on Z (with intercept) and the residuals
    correlated. Used instead of raw regression coefficients because the
    predictors being compared have DIFFERENT VARIANCES by construction --
    Var(mean of k) = s2_between + s2_within/k -- so raw betas are not
    comparable across them. A correlation is scale-free and therefore is.
    """
    from scipy.stats import pearsonr
    import statsmodels.api as sm
    Zc = sm.add_constant(Z, has_constant="add")
    ry = sm.OLS(y, Zc).fit().resid
    rx = sm.OLS(x, Zc).fit().resid
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def _load_disease_outcomes(ids: np.ndarray) -> pd.DataFrame:
    """Comorbidity count + BMI, one row per participant, restricted to `ids`.

    medical_conditions is long format (one row per diagnosis), so the count is
    the quantity of interest. Every table is explicitly deduplicated to one row
    per participant and asserted, because an earlier analysis silently produced
    n=24,005 rows from a 12,014-person cohort via a many-to-many merge.
    """
    out = pd.DataFrame({"study_id": ids})

    if MEDICAL_COND.exists():
        mc = load_preserving_visits(MEDICAL_COND, "medical_conditions")
        n_cond = mc.groupby("study_id").size().rename("n_conditions").reset_index()
        out = out.merge(n_cond, on="study_id", how="left")
        out["n_conditions"] = out["n_conditions"].fillna(0)
    else:
        print(f"  WARNING: {MEDICAL_COND} missing -- n_conditions unavailable")
        out["n_conditions"] = np.nan

    if ANTHROPOMETRICS.exists():
        an = load_preserving_visits(ANTHROPOMETRICS, "anthropometrics")
        if "bmi" in an.columns:
            bmi = (an.groupby("study_id")["bmi"].mean().rename("bmi").reset_index())
            out = out.merge(bmi, on="study_id", how="left")
        else:
            out["bmi"] = np.nan
    else:
        out["bmi"] = np.nan

    assert out["study_id"].is_unique, "disease frame has duplicate participants"
    return out


def _load_curated(stem: str) -> pd.DataFrame | None:
    """One curated phenotype, deduplicated to one row per participant."""
    path = CURATED_DIR / f"{stem}.parquet"
    if not path.exists():
        return None
    col = f"{stem}__curated_phenotype"
    try:
        ph = pd.read_parquet(path)
    except Exception:
        return None
    if isinstance(ph.index, pd.MultiIndex) or ph.index.name:
        ph = ph.reset_index()
    if "participant_id" in ph.columns:
        ph = ph.drop(columns=["study_id"], errors="ignore")
        ph = ph.rename(columns={"participant_id": "study_id"})
    if col not in ph.columns or "study_id" not in ph.columns:
        return None

    ph["study_id"] = pd.to_numeric(ph["study_id"], errors="coerce")
    ph = ph.dropna(subset=["study_id"])
    ph["study_id"] = ph["study_id"].astype("int64")

    v = ph[col]
    if v.dtype == object or str(v.dtype) == "category":
        v = pd.Series(pd.Categorical(v).codes, index=ph.index).where(lambda s: s >= 0)

    # Prefixed so a phenotype named like a covariate (e.g. "bmi") can never
    # collide during the merge and silently become bmi_x / bmi_y.
    out_col = f"{PHENO_PREFIX}{stem}"
    ph = pd.DataFrame({"study_id": ph["study_id"],
                       out_col: pd.to_numeric(v, errors="coerce")})

    # Deduplicate: max() = "ever diagnosed". Without this the merge fans out.
    ph = ph.groupby("study_id", as_index=False)[out_col].max()
    assert ph["study_id"].is_unique
    return ph


def trait_state_disease(df: pd.DataFrame, icc_single: float) -> None:
    banner("STEP 5  Trait vs. instability, and value of repeat sampling")

    from statsmodels.stats.multitest import multipletests

    # ---- exactly two visits, so person-mean reliability is constant -------- #
    counts  = df.groupby("study_id").size()
    two_ids = counts[counts == 2].index
    d2 = df[df["study_id"].isin(two_ids)].sort_values(["study_id", "visit_index"])
    print(f"  participants with exactly 2 visits: {len(two_ids):,}")
    print("  (restricted to exactly 2 so every person-mean averages the same")
    print("   number of samples -- otherwise reliability varies person to person)")

    v1 = d2[d2["visit_index"] == 0].set_index("study_id")
    v2 = d2[d2["visit_index"] == 1].set_index("study_id")
    ids = v1.index.intersection(v2.index)
    if len(ids) < MIN_REPEAT_PARTICIPANTS:
        print(f"  only {len(ids)} usable pairs -- skipping")
        return

    per = pd.DataFrame({
        "study_id":    ids,
        "accel_v1":    v1.loc[ids, "age_accel"].values,
        "accel_v2":    v2.loc[ids, "age_accel"].values,
        "age":         v1.loc[ids, "age"].values,
        "sex_numeric": v1.loc[ids, "sex_numeric"].values
                       if "sex_numeric" in v1.columns else np.nan,
    })
    per["accel_mean"]  = per[["accel_v1", "accel_v2"]].mean(axis=1)
    per["instability"] = (per["accel_v1"] - per["accel_v2"]).abs()

    dis = _load_disease_outcomes(per["study_id"].values)
    per = per.merge(dis, on="study_id", how="left")
    assert per["study_id"].is_unique, "participant frame duplicated after merge"

    # Reliability of a 2-visit mean, from Spearman-Brown.
    icc_mean = 2 * icc_single / (1 + icc_single)
    predicted_ratio = np.sqrt(icc_mean / icc_single) if icc_single > 0 else np.nan
    print(f"\n  reliability of a single visit  : ICC = {icc_single:.3f}")
    print(f"  reliability of a 2-visit mean  : ICC = {icc_mean:.3f} (Spearman-Brown)")
    print(f"  => association with the mean should be stronger by a factor of "
          f"{predicted_ratio:.3f}")
    print("     (this is a falsifiable prediction; a large mismatch below would")
    print("      indicate an error rather than a finding)")

    # Keep only covariates that are usable: present, not all-missing, and not
    # constant. A constant or empty column makes the design matrix singular and
    # the partial-correlation regressions fail.
    covar_cols = []
    for c in ["age", "sex_numeric", "bmi"]:
        if c not in per.columns:
            continue
        vals = pd.to_numeric(per[c], errors="coerce")
        if vals.notna().sum() < MIN_REPEAT_PARTICIPANTS:
            print(f"  covariate '{c}' dropped: only {vals.notna().sum()} non-missing")
        elif vals.dropna().nunique() < 2:
            print(f"  covariate '{c}' dropped: constant")
        else:
            covar_cols.append(c)
    if not covar_cols:
        print("  no usable covariates -- cannot adjust; skipping Step 5")
        return
    print(f"  adjusting for: {', '.join(covar_cols)}")
    rows = []

    def test_outcome(name: str, yvals: pd.Series, frame: pd.DataFrame) -> dict | None:
        sub = frame[[*covar_cols, "accel_v1", "accel_v2", "accel_mean",
                     "instability"]].copy()
        sub["_y"] = yvals
        sub = sub.apply(pd.to_numeric, errors="coerce").astype("float64").dropna()
        if len(sub) < MIN_REPEAT_PARTICIPANTS or sub["_y"].nunique() < 2:
            return None
        Z = sub[covar_cols].values
        y = sub["_y"].values
        r1,  p1  = _partial_corr(y, sub["accel_v1"].values,    Z)
        r2,  p2  = _partial_corr(y, sub["accel_v2"].values,    Z)
        rm,  pm  = _partial_corr(y, sub["accel_mean"].values,  Z)
        ri,  pi  = _partial_corr(y, sub["instability"].values, Z)
        r_single = float(np.mean([abs(r1), abs(r2)]))
        ratio = abs(rm) / r_single if r_single > 1e-9 else np.nan
        return dict(outcome=name, n=len(sub),
                    r_visit1=r1, p_visit1=p1, r_visit2=r2, p_visit2=p2,
                    r_mean=rm, p_mean=pm,
                    r_instability=ri, p_instability=pi,
                    r_single_avg=r_single, observed_ratio=ratio)

    # ---- primary outcome: comorbidity count -------------------------------- #
    print(f"\n  PRIMARY OUTCOME: comorbidity count (n_conditions)")
    prim = test_outcome("n_conditions", per["n_conditions"], per)
    if prim:
        print(f"    n = {prim['n']:,}   covariates: {', '.join(covar_cols)}")
        print(f"    partial r, visit 1 only  : {prim['r_visit1']:+.4f}  "
              f"(p={prim['p_visit1']:.3g})")
        print(f"    partial r, visit 2 only  : {prim['r_visit2']:+.4f}  "
              f"(p={prim['p_visit2']:.3g})")
        print(f"    partial r, 2-visit mean  : {prim['r_mean']:+.4f}  "
              f"(p={prim['p_mean']:.3g})")
        print(f"    partial r, instability   : {prim['r_instability']:+.4f}  "
              f"(p={prim['p_instability']:.3g})")
        print(f"    observed ratio mean/single: {prim['observed_ratio']:.3f}  "
              f"(predicted {predicted_ratio:.3f})")
        rows.append(prim)

        # Negative binomial check: n_conditions is a count, and partial
        # correlation assumes neither a link function nor a variance structure.
        try:
            import statsmodels.api as sm
            sub = per[[*covar_cols, "accel_mean", "n_conditions"]].apply(
                pd.to_numeric, errors="coerce").astype("float64").dropna()
            X = sm.add_constant(sub[[*covar_cols, "accel_mean"]])
            nb = sm.GLM(sub["n_conditions"], X,
                        family=sm.families.NegativeBinomial()).fit()
            print(f"    [check] negative-binomial GLM, accel_mean: "
                  f"beta={nb.params['accel_mean']:+.4f}, "
                  f"p={nb.pvalues['accel_mean']:.3g}")
        except Exception as exc:
            print(f"    [check] NB GLM unavailable ({type(exc).__name__})")

    # ---- secondary outcomes: curated phenotypes ---------------------------- #
    print(f"\n  SECONDARY OUTCOMES: curated phenotypes")
    sec = []
    for stem in CURATED_PHENOTYPES:
        ph = _load_curated(stem)
        if ph is None:
            continue
        out_col = f"{PHENO_PREFIX}{stem}"
        assert out_col not in per.columns, f"{out_col} would collide with an existing column"
        merged = per.merge(ph, on="study_id", how="left")
        assert len(merged) == len(per), f"{stem} merge changed row count"
        res = test_outcome(stem, merged[out_col], merged)
        if res:
            sec.append(res)
    if sec:
        sdf = pd.DataFrame(sec)
        for col, tag in [("p_mean", "mean"), ("p_instability", "instab")]:
            rej, padj, _, _ = multipletests(sdf[col], alpha=FDR_ALPHA,
                                            method="fdr_bh")
            sdf[f"p_fdr_{tag}"]   = padj
            sdf[f"sig_fdr_{tag}"] = rej
        print(f"    {len(sdf)} phenotypes tested, BH-FDR across them")
        print(f"    significant (2-visit mean) : {int(sdf['sig_fdr_mean'].sum())}")
        print(f"    significant (instability)  : {int(sdf['sig_fdr_instab'].sum())}")
        show = sdf.sort_values("p_fdr_mean")[
            ["outcome", "n", "r_visit1", "r_mean", "p_fdr_mean",
             "r_instability", "p_fdr_instab"]]
        print(show.to_string(index=False,
                             float_format=lambda v: f"{v:+.4f}"))
        rows.extend(sec)

        valid = sdf["observed_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid):
            print(f"\n    median observed ratio (mean vs single): "
                  f"{valid.median():.3f}   predicted {predicted_ratio:.3f}")
            if abs(valid.median() - predicted_ratio) < 0.25:
                print("    -> matches theory: averaging repeat samples recovers")
                print("       association strength lost to measurement error.")
            else:
                print("    -> departs from theory; interpret with caution.")

    if rows:
        out = pd.DataFrame(rows)
        out["icc_single"]      = icc_single
        out["icc_2visit_mean"] = icc_mean
        out["predicted_ratio"] = predicted_ratio
        out.to_csv(RESULTS / "trait_state_disease.csv", index=False)
        print(f"\n  wrote trait_state_disease.csv")


def main() -> None:
    print("Longitudinal stability of gut microbiome age acceleration | HPP")
    print(f"Data    : {HPP}")
    print(f"Results : {RESULTS}")
    print(f"Config  : prevalence>{MIN_PREVALENCE:.0%}, age {AGE_MIN}-{AGE_MAX}, "
          f"min interval {MIN_INTERVAL_YR} yr, seed {RANDOM_STATE}")
    if not HPP.exists():
        sys.exit(f"\nFATAL: data root does not exist: {HPP}\n"
                 "Set it explicitly, e.g.:\n"
                 "  export HPP_DATA_ROOT=/path/to/hpp_datasets")

    df, scols = build_longitudinal_frame()
    df = fit_clock(df, scols)
    icc = analyse_stability(df)
    build_stability_weighted_clock(df, scols)
    if np.isfinite(icc):
        trait_state_disease(df, icc)
    else:
        print("\nSkipping Step 5: ICC could not be estimated.")

    banner("DONE")


if __name__ == "__main__":
    main()
