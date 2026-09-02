"""
hpp_prototype.py
================================================================================
HPP adaptation of the counterfactual causal-inference pipeline for biological-
age reversal -- full-phenotype version for the MBZUAI GenMI lab.

All field names are confirmed
from the HPP Knowledgebase data dictionaries except the blood-test biomarker
block (see BLOOD_TEST_FIELDS below), which requires confirmation from someone
with access to the dataset:

    import pandas as pd
    bt = pd.read_parquet(HPP / "blood_tests/blood_tests.parquet")
    print(bt.columns.tolist())

Pipeline:
  1.  Load population (demographics + age)
  2.  Load and merge lifestyle, diet, blood tests, blood pressure
  3.  Compute PhenoAge / AgeAccel (Levine et al. 2018)
  4.  Filter to study population (accelerated agers aged 40-50)
  5.  CausalForestDML per lifestyle intervention -> AgeAccel
  6.  Rank interventions + covariate-balance diagnostics
  7.  Figures 1-5 + CSVs
  8.  Load gut microbiome (MetaPhlAn 4 genus-level)
  9.  Gut microbiome age clock (ElasticNet on genus abundances)
  10. CausalForestDML: lifestyle -> Shannon diversity
  11. Mediation: lifestyle -> microbiome -> PhenoAge
  12. Sweep all curated phenotypes as additional causal outcomes

Run inside the HPP environment:
    python hpp_prototype.py

Python 3.11 recommended. Requires: pandas, numpy, scipy, scikit-learn,
econml>=0.15, matplotlib, statsmodels.
================================================================================
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed, silences tkinter errors

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================================== #
# CONFIGURATION                                                                #
# =========================================================================== #
HPP          = Path("~/studies/hpp_datasets").expanduser()
RESULTS      = Path(__file__).parent / "results"
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Performance controls.
#
# The dominant cost in this pipeline is CausalForestDML: Step 12 alone fits one
# forest per (intervention x phenotype) pair, which is 5 x 18 = 90 forests, each
# cross-fitting two nuisance models 5 times. The original nuisance model was
# sklearn's GradientBoostingRegressor, which is single-threaded and uses exact
# split search; HistGradientBoostingRegressor is histogram-based and threaded,
# and is typically 10-30x faster at identical sample sizes.
#
#   HPP_FAST=1        fewer trees + 3-fold cross-fitting (screening runs)
#   HPP_SKIP_SWEEP=1  skip Step 12 entirely (it is ~90% of total runtime)
#   HPP_N_JOBS=-1     worker count handed to the causal forest
#
# NOTE: switching the nuisance learner changes point estimates in the 3rd-4th
# decimal place. Numbers will NOT be bit-identical to pre-patch runs.
# --------------------------------------------------------------------------- #
FAST_MODE  = os.environ.get("HPP_FAST", "0") == "1"
SKIP_SWEEP = os.environ.get("HPP_SKIP_SWEEP", "0") == "1"
N_JOBS     = int(os.environ.get("HPP_N_JOBS", "-1"))
CF_TREES   = 300 if FAST_MODE else 500
CF_CV      = 3   if FAST_MODE else 5


def _nuisance():
    """Fast, threaded nuisance learner for double machine learning."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, early_stopping=False,
        random_state=RANDOM_STATE)


def _make_cf(n_estimators=None, cv=None):
    """CausalForestDML with fast nuisances; tolerates econml versions
    whose signature does not expose n_jobs."""
    from econml.dml import CausalForestDML
    kw = dict(model_y=_nuisance(), model_t=_nuisance(),
              n_estimators=n_estimators or CF_TREES,
              cv=cv or CF_CV, random_state=RANDOM_STATE)
    try:
        return CausalForestDML(n_jobs=N_JOBS, **kw)
    except TypeError:
        return CausalForestDML(**kw)
STUDY_YEAR   = 2020    # HPP visit year (confirmed in cross_lagged_panel.py)
MEDIATION_MIN_N = 150  # minimum n to attempt mediation analysis

RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Blood-test field names for PhenoAge (Levine et al. 2018)                    #
# Field names below are best-guess from HPP naming conventions.               #
# Verify each against the 016-blood_tests data dictionary before running.     #
# Needs confirmation from someone with access to the dataset.                 #
# --------------------------------------------------------------------------- #
BLOOD_TEST_FIELDS = {
    # Confirmed from blood_tests_data_dictionary.csv (bt_<field>_float_value pattern).
    # HPP column name                    : (internal role,  required unit)
    # Column naming confirmed 2026-07-18: pattern is bt__<field>_float_value (double underscore).
    # lymphocytes, rdw, wbc confirmed. albumin/creatinine/glucose/crp/alp/mcv need confirmation.
    "bt__albumin_float_value":              ("albumin",    "g/dL"),
    "bt__creatinine_float_value":           ("creatinine", "mg/dL"),
    "bt__glucose_float_value":              ("glucose",    "mg/dL"),
    # CRP confirmed absent from blood_tests and every other HPP table searched (2026-07-26);
    # PhenoAge below uses the validated CRP-free formula, so no CRP field mapping is needed.
    "bt__lymphocytes_%_float_value":        ("lymph_pct",  "%"),
    "bt__mcv_float_value":                  ("mcv",        "fL"),
    "bt__rdw_float_value":                  ("rdw",        "%"),
    "bt__alkaline_phosphatase_float_value": ("alp",        "U/L"),
    "bt__wbc_float_value":                  ("wbc",        "10^9/L"),
}

# Conversion factors to reach the units the Levine 2018 PhenoAge coefficients
# require. If HPP already stores a biomarker in the target unit the factor is 1.0.
UNIT_CONVERSIONS = {
    # The published Levine 2018 coefficients are calibrated for SI units:
    #   albumin in g/L, creatinine in µmol/L, glucose in mmol/L.
    # HPP stores albumin in g/dL and creatinine in mg/dL (confirmed from the
    # Step 3 unit-detection log: "albumin: g/dL", "creatinine: mg/dL"), so BOTH
    # must be scaled up to SI units or their formula terms come out ~10x and
    # ~88x too small -- which silently compresses PhenoAge variance (the SD-3.4
    # AgeAccel symptom). Glucose likewise needs mg/dL → mmol/L (÷18.018): using
    # mg/dL saturates the Gompertz mortality function (median xb ≈ +9 vs ≈ -8 for
    # mmol/L, where only the latter yields PhenoAge ≈ chronological age).
    "albumin":    {"g/dL": 10.0,       "g/L": 1.0},       # g/dL → g/L    (×10)
    "creatinine": {"mg/dL": 88.42,     "umol/L": 1.0},    # mg/dL → µmol/L (×88.42)
    "glucose":    {"mg/dL": 1 / 18.018, "mmol/L": 1.0},   # mg/dL → mmol/L (÷18.018)
}

# Expected median ranges for each biomarker AFTER unit conversion (i.e. already
# in the units the Levine 2018 coefficients require). Used only as a
# post-conversion sanity check: if a converted median falls outside its range the
# pipeline prints a warning -- the fastest way to catch a unit mistake like the
# albumin/creatinine one that previously deflated PhenoAge.
EXPECTED_RANGES = {
    "albumin":    (35, 55),    # g/L
    "creatinine": (40, 120),   # µmol/L
    "glucose":    (3.5, 9.0),  # mmol/L
    "lymph_pct":  (10, 55),    # %
    "mcv":        (75, 105),   # fL
    "rdw":        (11, 16),    # %
    "alp":        (25, 150),   # U/L
    "wbc":        (3, 12),     # 10^9/L
}

# --------------------------------------------------------------------------- #
# Dataset paths (confirmed from HPP Knowledgebase data dictionaries)          #
# --------------------------------------------------------------------------- #
DS = {
    "population":         HPP / "population/population.parquet",
    "lifestyle":          HPP / "lifestyle_and_environment/lifestyle_and_environment.parquet",
    "diet_events":        HPP / "diet_logging/diet_logging_events.parquet",
    "blood_tests":        HPP / "blood_tests/blood_tests.parquet",
    "blood_pressure":     HPP / "blood_pressure/blood_pressure.parquet",
    # gut_microbiome: confirmed 2026-07-18 — abundance/metaphlan4/ subfolder.
    # Exact genus filename needs confirmation: run os.listdir(HPP / "gut_microbiome/abundance/metaphlan4/")
    "gut_microbiome":     HPP / "gut_microbiome/gut_microbiome.parquet",
    "gut_genus":          HPP / "gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_genus.parquet",
    # Species-level gut abundances (confirmed present 2026-07-26). Finer
    # taxonomic resolution than genus -- used to build a more accurate age
    # clock, since clock accuracy directly determines how noisy the derived
    # gut_age_accel outcome is in Step 9b.
    "gut_species":        HPP / "gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_species.parquet",
    # oral_microbiome: confirmed from VS Code explorer (2026-07-18).
    "oral_microbiome":    HPP / "oral_microbiome/oral_microbiome.parquet",
    "oral_genus":         HPP / "oral_microbiome/aggregated/metaphlan_abundances_aggregated/genus.parquet",
    "oral_pathways":      HPP / "oral_microbiome/aggregated/humann_aggregated/pathway_abundance_pathway_level.arrow",
    "curated_phenotypes": HPP / "curated_phenotypes",
    # Expanded covariates (confirmed via schema scan, 2026-07-26):
    "anthropometrics":              HPP / "anthropometrics/anthropometrics.parquet",
    "sociodemographics_initial_medical": HPP / "sociodemographics/initial_medical.parquet",
    "sociodemographics_ukbb":       HPP / "sociodemographics/ukbb.parquet",
    "medical_conditions":           HPP / "medical_conditions/medical_conditions.parquet",
}

# Lifestyle interventions: display name -> column in merged frame
INTERVENTIONS = {
    "Diet quality":               "diet_score",
    "Sleep duration (h)":         "sleep_hours",
    "Physical activity (min/wk)": "pa_weekly_min",
    "Smoking (current)":          "smoking_current_status",
    "Alcohol frequency":          "alcohol_current_frequency",
}

# Covariates used in all causal models
COVARIATES = ["age", "sex_numeric", "sbp", "dbp",
              "bmi", "household_income", "education_level", "n_conditions"]

# Curated phenotypes to sweep in Step 12
CURATED_PHENOTYPES = [
    ("abdominal_adiposity",    "abdominal_adiposity__curated_phenotype"),
    ("bmi",                    "bmi__curated_phenotype"),
    ("diabetes",               "diabetes__curated_phenotype"),
    ("hypertension",           "hypertension__curated_phenotype"),
    ("hyperlipidemia",         "hyperlipidemia__curated_phenotype"),
    ("ischemic_heart_disease", "ischemic_heart_disease__curated_phenotype"),
    ("mafld",                  "mafld__curated_phenotype"),
    ("ckd",                    "ckd__curated_phenotype"),
    ("osteoporosis",           "osteoporosis__curated_phenotype"),
    ("depression",             "depression__curated_phenotype"),
    ("anxiety",                "anxiety__curated_phenotype"),
    ("adhd",                   "adhd__curated_phenotype"),
    ("osa",                    "osa__curated_phenotype"),
    ("sleep_quality",          "sleep_quality__curated_phenotype"),
    ("migraine",               "migraine__curated_phenotype"),
    ("endometriosis",          "endometriosis__curated_phenotype"),
    ("menopause",              "menopause__curated_phenotype"),
    ("nmsc",                   "nmsc__curated_phenotype"),
]


# =========================================================================== #
# UTILITIES                                                                    #
# =========================================================================== #
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _load(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DS[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{key}' not found at expected path:\n  {path}")
    try:
        df = pd.read_parquet(path) if columns is None else pd.read_parquet(path, columns=columns)
    except Exception:
        # Some HPP parquets are Hive-partitioned directories (study_id=10/part.parquet).
        # Read with pyarrow dataset API using hive partitioning to avoid FieldRef errors.
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
        # Read only the physical columns (exclude partition keys and pyarrow internals).
        phys_cols = [c for c in dataset.schema.names
                     if not c.startswith("__") and c not in ("filename",)]
        tbl = dataset.to_table(columns=phys_cols if phys_cols else None)
        df = tbl.to_pandas()
        # In Hive-partitioned datasets, participant_id is a data column and
        # study_id is the partition key (cohort number = 10). Use participant_id as the join key.
        if "participant_id" in df.columns:
            df = df.drop(columns=["study_id"], errors="ignore")
            df = df.rename(columns={"participant_id": "study_id"})
        # Apply column filter AFTER rename so callers using "study_id" still match.
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            if keep:
                df = df[keep]

    # HPP stores data in long format with a MultiIndex (participant_id, study, visit, row).
    # Flatten to one row per participant keyed on study_id.
    # HPP parquets use a MultiIndex (participant_id, cohort[, visit, row]).
    # The real participant identifier is index level 0 ("participant_id").
    # The "study_id" column that may exist in the data is the cohort number (e.g. 10), not a participant ID.
    if isinstance(df.index, pd.MultiIndex):
        index_names = list(df.index.names)
        df = df.reset_index()
        # Rename index level 0 (participant_id) → study_id.
        # Use 'participant_id' if present (named index), else fall back to 'level_0' (unnamed index).
        pid_col = ("participant_id" if "participant_id" in df.columns
                   else index_names[0] if (index_names[0] is not None and index_names[0] in df.columns)
                   else "level_0" if "level_0" in df.columns
                   else None)
        if pid_col:
            df = df.rename(columns={pid_col: "study_id"})
        drop_cols = index_names[1:]                      # cohort, visit, row — metadata, not needed
        if "study_id" in drop_cols:
            drop_cols = [c for c in drop_cols if c != "study_id"]
        # Also drop any pre-existing "study_id" data column (cohort number) now shadowed by participant ID.
        # After rename, the old "study_id" column has been bumped — find it by position if duplicated.
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df = df.drop(columns=drop_cols, errors="ignore")
        # Aggregate: multiple rows per participant (visits) → one row per participant.
        if df["study_id"].duplicated().any():
            num_cols = [c for c in df.select_dtypes(include="number").columns if c != "study_id"]
            cat_cols = [c for c in df.columns if c not in num_cols and c != "study_id"]
            agg = {c: "mean" for c in num_cols}
            agg.update({c: "first" for c in cat_cols})
            df = df.groupby("study_id", as_index=False).agg(agg)
    elif "study_id" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "study_id"})

    # Ensure study_id is always int64 for consistent merging across all datasets.
    if "study_id" in df.columns:
        df["study_id"] = pd.to_numeric(df["study_id"], errors="coerce")
        df = df.dropna(subset=["study_id"])
        df["study_id"] = df["study_id"].astype("int64")

    print(f"  [{key:<22}] shape={df.shape}")
    return df


def _ensure_econml() -> None:
    try:
        import econml  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "econml>=0.15", "--quiet"])


# Publication-quality colour palette
C = dict(teal="#3a7d7b", rust="#a0522d", grey="#9aa0a6",
         green="#4a8c5f", red="#b0483a", ink="#33373b")


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size":          11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.8,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
    import matplotlib.pyplot as plt
    return plt


# =========================================================================== #
# STEP 1  Load population                                                      #
# Confirmed fields: study_id, year_of_birth (int), sex (category)             #
# =========================================================================== #
def step1_population() -> pd.DataFrame:
    banner("STEP 1  Load population")
    pop = _load("population")

    pop["age"] = STUDY_YEAR - pop["year_of_birth"].astype(float)
    # Sex encoding confirmed 2026-07-26 via HPP metadata/coding_mapping.csv (field code 9,
    # rows 773-774): raw "0" = Female, raw "1" = Male. sex_numeric follows the same
    # convention (0 = Female, 1 = Male) so it matches the HPP data dictionary directly.
    sex_col = "sex" if "sex" in pop.columns else "gender"
    pop["sex_numeric"] = (pop[sex_col]
                          .map({"Female": 0, "Male": 1, "0": 0, "1": 1, 0: 0, 1: 1})
                          .astype("Int64"))

    print(f"  participants : {pop['study_id'].nunique():,}")
    print(f"  age range    : {pop['age'].min():.0f}–{pop['age'].max():.0f} yr  "
          f"(mean {pop['age'].mean():.1f})")
    print(f"  sex          : {pop['sex'].value_counts().to_dict()}")
    return pop[["study_id", "year_of_birth", "age", "sex", "sex_numeric"]].copy()


# =========================================================================== #
# STEP 2  Load and merge lifestyle / diet / blood tests / blood pressure       #
# =========================================================================== #
def step2_load_merge(pop: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2  Load and merge datasets")
    d = pop.copy()

    # Lifestyle and environment (confirmed field names from data dictionary)
    lif = _load("lifestyle")
    _warn_missing(lif, [
        "activity_vigorous_days_weekly", "activity_vigorous_minutes_daily",
        "activity_moderate_days_weekly", "activity_moderate_minutes_daily",
        "activity_walking_10min_days_weekly", "activity_walking_minutes_daily",
        "sleep_hours_daily", "sleep_chronotype",
        "smoking_current_status", "alcohol_current_frequency",
    ])
    print(f"  lifestyle study_id sample : {lif['study_id'].dropna().iloc[:3].tolist() if 'study_id' in lif.columns else 'MISSING'}")
    print(f"  population study_id sample: {d['study_id'].dropna().iloc[:3].tolist()}")
    d = d.merge(lif.rename(columns={"sleep_hours_daily": "sleep_hours"}),
                on="study_id", how="left")
    print(f"  lifestyle merge matched   : {d['sleep_hours'].notna().sum():,} / {len(d):,}")
    d["pa_weekly_min"] = _build_pa(d)

    # Diet logging: aggregate event-level to person-level
    d = d.merge(_build_diet_score(), on="study_id", how="left")

    # Blood tests: PhenoAge biomarkers
    bt = _load("blood_tests")
    rename_map = {"study_id": "study_id"}
    for field, (role, unit) in BLOOD_TEST_FIELDS.items():
        if field in bt.columns:
            rename_map[field] = f"bt_{role}"
        else:
            print(f"  WARNING: blood_tests field '{field}' not found -- "
                  f"needs confirmation from someone with the dataset")
    bt_cols = [c for c in rename_map.values() if c in bt.rename(columns=rename_map)]
    d = d.merge(bt.rename(columns=rename_map)[bt_cols], on="study_id", how="left")

    # Blood pressure: covariate only
    bp = _load("blood_pressure",
               columns=["study_id", "sitting_blood_pressure_systolic",
                        "sitting_blood_pressure_diastolic"])
    d = d.merge(bp.rename(columns={
        "sitting_blood_pressure_systolic":  "sbp",
        "sitting_blood_pressure_diastolic": "dbp",
    }), on="study_id", how="left")

    # --- Expanded covariates (2026-07-26): the original covariate set (age,
    # sex, SBP, DBP) is narrow. BMI, socioeconomic status, and baseline
    # comorbidity burden are all plausible confounders of the lifestyle ->
    # aging relationship that were previously unmeasured -- adding them tests
    # whether the null direct/mediation results are due to omitted-variable
    # bias rather than a genuinely absent effect. ---

    # BMI: direct column, no derivation needed.
    anthro = _load("anthropometrics", columns=["study_id", "bmi"])
    d = d.merge(anthro, on="study_id", how="left")
    print(f"  BMI matched               : {d['bmi'].notna().sum():,} / {len(d):,}")

    # Socioeconomic status: education (initial_medical) + household income (ukbb).
    edu = _load("sociodemographics_initial_medical", columns=["study_id", "education"])
    d = d.merge(edu.rename(columns={"education": "education_level"}),
                on="study_id", how="left")
    inc = _load("sociodemographics_ukbb", columns=["study_id", "domestic_household_income"])
    d = d.merge(inc.rename(columns={"domestic_household_income": "household_income"}),
                on="study_id", how="left")
    print(f"  education matched         : {d['education_level'].notna().sum():,} / {len(d):,}")
    print(f"  household income matched  : {d['household_income'].notna().sum():,} / {len(d):,}")

    # Baseline comorbidity burden: medical_conditions is long-format (one row
    # per diagnosis per participant). _load() would collapse this to one row
    # via its generic "first"/"mean" aggregation, destroying the per-diagnosis
    # rows we need to count -- so this reads the raw file directly instead.
    d = d.merge(_build_comorbidity_count(), on="study_id", how="left")
    d["n_conditions"] = d["n_conditions"].fillna(0)
    print(f"  n_conditions computed     : mean={d['n_conditions'].mean():.2f}")

    print(f"\n  merged shape = {d.shape}")
    return d


def _warn_missing(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            print(f"  WARNING: expected column '{c}' not found")


def _build_pa(d: pd.DataFrame) -> pd.Series:
    """Weekly MVPA minutes (vigorous x2 + moderate + walking). MET convention.
    HPP stores days/week and minutes/day separately; weekly = days * min_per_day.

    [TASK 3 FIX] The per-day minute cap was 960 (16 hours) -- so generous that
    the formula's own theoretical maximum, 7*960*2 (vigorous) + 7*960
    (moderate) + 7*960 (walking) = 26,880 min/week, is 2.67x the total minutes
    in a week (10,080). That's not implausible, it's mathematically
    impossible: this is the same shape of defect flagged in the NHANES POC (an
    SD of ~1,366 min/wk traced to an analogous unit/cap issue). Tightened to
    180 min/day, the standard IPAQ truncation convention for self-reported
    vigorous/moderate/walking activity -- still generous (3 hours/day) but no
    longer able to produce a weekly total exceeding the week itself. Verify
    against HPP's actual survey codebook if a different convention applies.
    """
    def safe(col):
        return d[col].astype(float) if col in d.columns else pd.Series(
            np.nan, index=d.index)

    PA_DAILY_MIN_CAP = 180   # IPAQ-standard truncation; was 960 (16 hr/day)

    vig_d = safe("activity_vigorous_days_weekly").clip(0, 7)
    vig_m = safe("activity_vigorous_minutes_daily").clip(1, PA_DAILY_MIN_CAP)
    mod_d = safe("activity_moderate_days_weekly").clip(0, 7)
    mod_m = safe("activity_moderate_minutes_daily").clip(1, PA_DAILY_MIN_CAP)
    wal_d = safe("activity_walking_10min_days_weekly").clip(0, 7)
    wal_m = safe("activity_walking_minutes_daily").clip(1, PA_DAILY_MIN_CAP)

    weekly = (
        (vig_d * vig_m).fillna(0) * 2 +
        (mod_d * mod_m).fillna(0) +
        (wal_d * wal_m).fillna(0)
    )
    weekly[vig_d.isna() & mod_d.isna() & wal_d.isna()] = np.nan

    # [TASK 3] Report the full distribution, not just the median -- the
    # NHANES bug was only caught by noticing an implausible SD, and the
    # median alone would not have shown it. WEEK_MINUTES=10,080 is the hard
    # ceiling; anything above it is proof of a remaining coding error, not
    # just an outlier.
    WEEK_MINUTES = 7 * 24 * 60
    n_over_week = int((weekly > WEEK_MINUTES).sum())
    print(f"  PA weekly: n={weekly.notna().sum():,}, "
          f"mean={weekly.mean():.0f}, SD={weekly.std():.0f}, "
          f"median={weekly.median():.0f}, max={weekly.max():.0f} min/wk "
          f"(hard ceiling={WEEK_MINUTES})")
    if n_over_week > 0:
        print(f"  WARNING: {n_over_week} participants exceed the physical "
              f"maximum of {WEEK_MINUTES} min/wk -- a coding error remains; "
              f"do not use this covariate in adjusted models until resolved.")
    return weekly


def _build_diet_score() -> pd.DataFrame:
    """Aggregate diet_logging_events to person-level diet-quality score.
    Score = z(fibre/1000 kcal) - z(fat/1000 kcal) - z(sodium/1000 kcal).
    """
    print("  loading diet events...")
    ev = _load("diet_events")

    if "study_id" not in ev.columns:
        print("  WARNING: 'study_id' not found in diet events -- "
              "needs confirmation from someone with the dataset")
        return pd.DataFrame(columns=["study_id"])

    nutrient_cols = [c for c in ["calories_kcal", "dietary_fiber_g", "lipid_g",
                                  "sodium_mg", "alcohol_g", "carbohydrate_g",
                                  "protein_g"] if c in ev.columns]
    date_col   = "collection_date" if "collection_date" in ev.columns else None
    group_cols = ["study_id"] + ([date_col] if date_col else [])

    daily  = ev.groupby(group_cols)[nutrient_cols].sum().reset_index()
    person = daily.groupby("study_id")[nutrient_cols].mean().reset_index()

    if "calories_kcal" in person.columns and person["calories_kcal"].gt(0).any():
        per1k = 1000.0 / person["calories_kcal"].where(person["calories_kcal"] > 0)
        parts, labels = [], []
        for col, sign in [("dietary_fiber_g", +1), ("lipid_g", -1), ("sodium_mg", -1)]:
            if col in person.columns:
                x = person[col] * per1k
                parts.append(sign * (x - x.mean()) / x.std())
                labels.append(f"{'+'if sign>0 else '-'}{col}")
        if parts:
            person["diet_score"] = sum(parts)
            print(f"  diet score: {' '.join(labels)} per 1000 kcal (z-scored)")

    person["study_id"] = pd.to_numeric(person["study_id"], errors="coerce")
    person = person.dropna(subset=["study_id"])
    person["study_id"] = person["study_id"].astype("int64")
    print(f"  diet participants: {len(person):,}")
    return person


def _build_comorbidity_count() -> pd.DataFrame:
    """Count of distinct diagnoses per participant from medical_conditions
    (long-format, one row per diagnosis). Read directly rather than via
    _load(), which would collapse the long format to one row per participant
    ("first" per column) before we could count rows -- exactly the data we
    need here."""
    path = DS["medical_conditions"]
    if not path.exists():
        print("  WARNING: medical_conditions not found -- n_conditions not added")
        return pd.DataFrame(columns=["study_id", "n_conditions"])

    raw = pd.read_parquet(path)
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
    pid_col = next((c for c in ["participant_id", "study_id"] if c in raw.columns), None)
    if pid_col is None:
        print("  WARNING: no participant id column in medical_conditions -- "
              "n_conditions not added")
        return pd.DataFrame(columns=["study_id", "n_conditions"])

    raw = raw.rename(columns={pid_col: "study_id"})
    raw["study_id"] = pd.to_numeric(raw["study_id"], errors="coerce")
    raw = raw.dropna(subset=["study_id"])
    raw["study_id"] = raw["study_id"].astype("int64")

    n_cond = (raw.groupby("study_id").size()
                 .rename("n_conditions").reset_index())
    print(f"  medical_conditions rows   : {len(raw):,}  "
          f"({raw['study_id'].nunique():,} participants with >=1 diagnosis)")
    return n_cond


# =========================================================================== #
# STEP 3  PhenoAge (Levine et al. 2018)                                        #
# =========================================================================== #
def step3_phenoage(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3  PhenoAge (Levine et al. 2018)")

    def get(role, unit_key=None):
        col = f"bt_{role}"
        if col not in d.columns or d[col].isna().all():
            print(f"  WARNING: '{col}' missing or all-null -- "
                  f"needs confirmation from someone with the dataset")
            return pd.Series(np.nan, index=d.index)
        s = d[col].astype(float).copy()
        if unit_key and unit_key in UNIT_CONVERSIONS:
            detected = _detect_unit(s, unit_key)
            factor   = UNIT_CONVERSIONS[unit_key].get(detected, 1.0)
            if factor != 1.0:
                s = s * factor
            med = float(s.dropna().median()) if s.notna().any() else float("nan")
            print(f"  {role:<12}: {detected} (x{factor:g}), median={med:.3g}, "
                  f"n={s.notna().sum():,}")
        else:
            med = float(s.dropna().median()) if s.notna().any() else float("nan")
            print(f"  {role:<12}: median={med:.3g}, n={s.notna().sum():,}")
        # Post-conversion sanity check against expected physiological range --
        # a converted median outside its range almost always means a unit slip.
        lo, hi = EXPECTED_RANGES.get(role, (None, None))
        if lo is not None and s.notna().any() and not (lo <= med <= hi):
            print(f"    WARNING: {role} median {med:.3g} outside expected "
                  f"[{lo}, {hi}] (Levine units) -- verify unit conversion.")
        return s

    albumin = get("albumin",    "albumin")
    creat   = get("creatinine", "creatinine")
    glucose = get("glucose",    "glucose")
    # CRP: confirmed absent from every HPP table searched (blood_tests, serum_lipidomics,
    # nightingale_metabolomics, vascular_health, health_and_medical_history, medical_conditions;
    # confirmed 2026-07-26). Rather than impute a constant (which adds no information and is
    # not a validated approach), we use the published CRP-free PhenoAge formula: the CRP term
    # is dropped from xb entirely, keeping the original Levine 2018 coefficients for the
    # remaining 8 biomarkers unchanged. This mirrors the approach used for NHANES 2011-2018,
    # where CRP is also unavailable; validation studies report PhenoAge computed with vs.
    # without CRP correlates at r=0.967-0.99 (Pearson), confirming the exclusion is safe
    # rather than a improvised workaround.
    lymph   = get("lymph_pct")
    mcv     = get("mcv")
    rdw     = get("rdw")
    alp     = get("alp")
    wbc     = get("wbc")
    age     = d["age"]

    xb = (-19.9067
          - 0.0336  * albumin
          + 0.0095  * creat
          + 0.1953  * glucose
          - 0.0120  * lymph
          + 0.0268  * mcv
          + 0.3306  * rdw
          + 0.00188 * alp
          + 0.0554  * wbc
          + 0.0804  * age)

    g    = 0.0076927
    mort = (1.0 - np.exp(-np.exp(xb) * (np.exp(120.0 * g) - 1.0) / g)
            ).clip(1e-8, 1 - 1e-8)
    d["PhenoAge"] = 141.50 + np.log(-0.00553 * np.log(1.0 - mort)) / 0.090165
    d["AgeAccel"] = d["PhenoAge"] - age

    valid = d[["PhenoAge", "age", "AgeAccel"]].dropna()
    print(f"\n  complete PhenoAge : n={len(valid):,}")
    print(f"  mean chron. age   : {valid['age'].mean():.2f} "
          f"(SD {valid['age'].std():.2f})")
    print(f"  mean PhenoAge     : {valid['PhenoAge'].mean():.2f} "
          f"(SD {valid['PhenoAge'].std():.2f})")
    print(f"  mean AgeAccel     : {valid['AgeAccel'].mean():.2f} "
          f"(SD {valid['AgeAccel'].std():.2f})")
    return d


def _detect_unit(s: pd.Series, key: str) -> str:
    m = float(s.dropna().median()) if s.notna().any() else 0.0
    if key == "albumin":    return "g/dL"    if m < 10  else "g/L"
    if key == "creatinine": return "mg/dL"   if m < 10  else "umol/L"
    if key == "glucose":    return "mg/dL"   if m > 10  else "mmol/L"
    return "unknown"


# =========================================================================== #
# STEP 4  Study population (accelerated agers, 40-50 y)                        #
# =========================================================================== #
def step4_population(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 4  Study population (accelerated agers, 40-50 y)")
    mask  = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    study = d[mask].copy()
    print(f"  full cohort with AgeAccel : {d['AgeAccel'].notna().sum():,}")
    print(f"  aged 40-50                : {d['age'].between(40, 50).sum():,}")
    print(f"  accelerated (AgeAccel>0)  : {(d['AgeAccel'] > 0).sum():,}")
    print(f"  STUDY POPULATION          : {len(study):,}")
    print(f"  mean AgeAccel             : {study['AgeAccel'].mean():.2f} yr")
    return study


# =========================================================================== #
# STEP 5  CausalForestDML per intervention                                     #
# =========================================================================== #
def _winsorize(s: pd.Series, pct: float) -> pd.Series:
    """Clip a series to its [pct, 1-pct] quantile range. Used only as an optional
    sensitivity check to test whether extreme outliers (e.g. self-reported physical
    activity >2,000 min/wk, or AgeAccel >30 yr) are driving CI width -- not applied
    to the primary/headline analysis unless explicitly requested."""
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lower=lo, upper=hi)


def step5_causal(pop: pd.DataFrame, outcome: str = "AgeAccel",
                 interventions: dict | None = None,
                 winsorize_pct: float | None = None) -> dict:
    banner(f"STEP 5  CausalForestDML  (outcome = {outcome})"
           + (f"  [winsorized at {winsorize_pct:.0%}/{1-winsorize_pct:.0%}]"
              if winsorize_pct else ""))
    _ensure_econml()

    if interventions is None:
        interventions = INTERVENTIONS
    covars  = [c for c in COVARIATES if c in pop.columns]
    results = {}

    for name, tcol in interventions.items():
        if tcol not in pop.columns:
            print(f"\n  [{name}] SKIPPED: column '{tcol}' not in dataset")
            continue
        print(f"\n  [{name}]  T = {tcol}")
        sub = pop[[outcome, tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        if winsorize_pct:
            sub = sub.copy()
            sub[outcome] = _winsorize(sub[outcome], winsorize_pct)
            sub[tcol]    = _winsorize(sub[tcol], winsorize_pct)
        print(f"    n = {len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue

        Y   = sub[outcome].values
        T   = sub[tcol].values.astype(float)
        X   = sub[covars].values
        est = _make_cf()
        est.fit(Y, T, X=X)

        ate     = float(est.ate(X))
        lb, ub  = [float(v) for v in est.ate_interval(X, alpha=0.05)]
        ite     = est.effect(X)
        sd_t    = float(np.std(T))
        ns      = "  (n.s.)" if lb < 0 < ub else ""

        print(f"    ATE         = {ate:+.4f} per unit  "
              f"[95% CI {lb:+.4f}, {ub:+.4f}]{ns}")
        print(f"    std. effect = {ate*sd_t:+.4f} per +1 SD")
        print(f"    % benefit   = {100*np.mean(ite < 0):.1f}%")

        results[name] = dict(
            tcol=tcol, ate=ate, ci=(lb, ub), sd_t=sd_t,
            std_effect=ate * sd_t, ite=ite,
            sub=sub.reset_index(drop=True),
            balance=_covariate_balance(sub, tcol, covars),
        )
    return results


def _covariate_balance(sub: pd.DataFrame, tcol: str,
                       covars: list[str]) -> pd.DataFrame:
    from scipy.stats import pearsonr
    from sklearn.model_selection import cross_val_predict

    T     = sub[tcol].values.astype(float)
    X     = sub[covars].values
    t_hat = cross_val_predict(_nuisance(), X, T, cv=CF_CV, n_jobs=N_JOBS)
    t_res = T - t_hat

    rows = [dict(covariate=c,
                 corr_raw=float(pearsonr(sub[c], T)[0]),
                 corr_adj=float(pearsonr(sub[c], t_res)[0]))
            for c in covars]
    bal = pd.DataFrame(rows)
    print("    balance |corr| raw -> ortho: " +
          ", ".join(f"{r.covariate} {abs(r.corr_raw):.2f}->{abs(r.corr_adj):.2f}"
                    for _, r in bal.iterrows()))
    return bal


# =========================================================================== #
# STEP 6  Rank interventions by standardised effect size                       #
# =========================================================================== #
def step6_rank(results: dict) -> pd.DataFrame:
    banner("STEP 6  Intervention ranking")
    tbl = _rank_results(results)
    if tbl.empty:
        print("  No interventions had sufficient data (all skipped).")
        return tbl
    print(tbl[["Rank", "Intervention", "ATE", "Std_effect", "Pct_benefit"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return tbl


def _rank_results(results: dict) -> pd.DataFrame:
    rows = [dict(
        Intervention=name, ATE=r["ate"],
        CI_low=r["ci"][0], CI_high=r["ci"][1],
        SD_treatment=r["sd_t"], Std_effect=r["std_effect"],
        Reversal=-r["std_effect"],
        Pct_benefit=100 * np.mean(r["ite"] < 0),
    ) for name, r in results.items()]
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    tbl = tbl.sort_values("Reversal", ascending=False).reset_index(drop=True)
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))
    return tbl


# =========================================================================== #
# STEP 6b  Combined healthy-lifestyle score                                    #
#                                                                               #
# Motivation: a power-sensitivity check (design-sensitivity / minimum-        #
# detectable-effect analysis) showed every individual lifestyle lever's       #
# observed effect is well below what this sample size could reliably detect  #
# at 80% power (observed/MDE ratios of 0.07-0.25). Testing 5 separate noisy   #
# proxies also costs power outright vs. testing one combined construct. This  #
# builds a single z-scored composite (as in "Life's Essential 8"-style        #
# indices) so the signal from all 5 levers is pooled into one test instead    #
# of split five ways -- a genuine power improvement, not another p-hacking    #
# attempt at the same question.                                               #
# =========================================================================== #
def _build_healthy_lifestyle_score(d: pd.DataFrame) -> pd.Series:
    """z-score each lifestyle variable, sign-aligned so higher = healthier,
    then average into one composite. Missing components are simply excluded
    from that participant's average (not imputed), so the score is only
    NaN for someone missing every component."""
    # (column, sign): +1 if higher values are protective, -1 if harmful.
    components = [
        ("pa_weekly_min",             +1),
        ("diet_score",                +1),
        ("sleep_hours",               +1),
        ("smoking_current_status",    -1),
        ("alcohol_current_frequency", -1),
    ]
    z_parts = []
    for col, sign in components:
        if col not in d.columns:
            continue
        x = pd.to_numeric(d[col], errors="coerce")
        z = sign * (x - x.mean()) / x.std()
        z_parts.append(z)
    if not z_parts:
        return pd.Series(np.nan, index=d.index)
    return pd.concat(z_parts, axis=1).mean(axis=1, skipna=True)


def step6b_combined_score(study: pd.DataFrame) -> dict | None:
    banner("STEP 6b  Combined healthy-lifestyle score (pooled power check)")
    covars = [c for c in COVARIATES if c in study.columns]
    pop = study.copy()
    pop["healthy_lifestyle_score"] = _build_healthy_lifestyle_score(pop)

    sub = pop[["AgeAccel", "healthy_lifestyle_score"] + covars].dropna()
    print(f"  n = {len(sub):,}  (all 5 components combined into one z-scored index)")
    if len(sub) < 80:
        print("  SKIPPED (n < 80)")
        return None

    res = _fit_cf(sub, "healthy_lifestyle_score", "AgeAccel", covars)
    ns  = res["ci"][0] < 0 < res["ci"][1]
    print(f"    ATE         = {res['ate']:+.4f} per +1 SD of composite score  "
          f"[95% CI {res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]"
          + ("  (n.s.)" if ns else "  ***"))
    print(f"    % benefit   = {100*np.mean(res['ite'] < 0):.1f}%  "
          "(higher composite score -> lower AgeAccel)")

    out = pd.DataFrame([dict(
        n=len(sub), ate=res["ate"], ci_low=res["ci"][0], ci_high=res["ci"][1],
        pct_benefit=100 * np.mean(res["ite"] < 0), significant=not ns,
    )])
    out.to_csv(RESULTS / "combined_lifestyle_score_result.csv", index=False)
    print("  wrote combined_lifestyle_score_result.csv")
    return res


# =========================================================================== #
# STEP 7  Figures and CSVs                                                     #
# =========================================================================== #
def _ranked_figure(ranked: pd.DataFrame, path: Path, title: str, xlabel: str):
    if ranked is None or ranked.empty:
        print(f"  skipping {path.name} (no results)")
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(ranked) + 1.5)))
    r       = ranked.sort_values("Reversal").reset_index(drop=True)
    crosses = (r["CI_low"] < 0) & (r["CI_high"] > 0)
    colors  = [C["green"] if v < 0 else C["red"] for v in r["Std_effect"]]
    err     = np.abs((r["CI_high"] - r["CI_low"]) / 2.0 * r["SD_treatment"])
    maxabs  = float(r["Std_effect"].abs().max()) or 1.0

    for yi in range(len(r)):
        ns = bool(crosses.iloc[yi])
        ax.barh(yi, r["Std_effect"].iloc[yi],
                color=colors[yi], alpha=0.35 if ns else 0.85,
                hatch="///" if ns else "", edgecolor=C["ink"], linewidth=0.8,
                xerr=err.iloc[yi],
                error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
        if not ns:
            v = r["Std_effect"].iloc[yi]
            ax.text(v + np.sign(v) * 0.02 * maxabs, yi, "*",
                    va="center", fontsize=14, color=C["ink"])

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r["Intervention"], fontsize=9)
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.text(0.98, 0.02,
            "hatched/faded = n.s. (CI crosses 0)    * = CI excludes 0",
            transform=ax.transAxes, ha="right", fontsize=7, color=C["grey"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")


# =========================================================================== #
# STEP 6c  Treatment-effect heterogeneity: who actually benefits?              #
#                                                                               #
# Every average treatment effect in this pipeline is null. But CausalForestDML #
# estimates an individual treatment effect (ITE) for every participant, not    #
# just a population average -- and the fraction with a protective ITE has     #
# never been 0% or 100% for any lever (e.g. physical activity: roughly        #
# 55-72% depending on the run). A null average can still hide a real effect   #
# concentrated in an identifiable subgroup. This tests, for every             #
# (intervention x baseline covariate) pair, whether the covariate predicts    #
# the size of the individual effect -- i.e. who benefits -- with BH-FDR       #
# correction across all pairs, since this is again a multiple-testing        #
# problem, not a single test.                                                 #
# =========================================================================== #
def step6c_heterogeneity(results: dict, covars: list[str],
                         csv_name: str = "heterogeneity_results.csv") -> pd.DataFrame | None:
    """Tests whether baseline covariates modify each intervention's effect
    (treatment-effect heterogeneity / effect modification).

    IMPORTANT -- two invalid approaches were tried and rejected before this one:
    (1) Correlating a covariate directly against the ITE from a CausalForestDML
    that used that same covariate as an input. CausalForestDML estimates
    ITE(X) as an explicit function of X, so this is close to tautological and
    produced p~1e-90 for several pairs -- not a real signal.
    (2) Splitting the sample and predicting ITE only on a held-out half the
    model never saw. This still failed (p~1e-46): tau_hat(X) is deterministic
    once the forest is fit, and with a few hundred training rows feeding both
    nuisance models and a 500-tree forest across 8 covariates, the model can
    overfit to noise and learn a spurious X-based splitting rule -- which
    still shows up as a strong correlation on held-out X, because held-out X
    is drawn from the same distribution as training X. Sample splitting
    prevents literal circularity but does not prevent this.

    The valid, standard approach used here instead: a classical interaction-
    term regression on the RAW data (not a fitted model's own output).
    For each (intervention, covariate) pair, fit
        outcome ~ treatment + covariate + treatment:covariate + other_covars
    via OLS and test the treatment:covariate interaction coefficient. This
    tests whether the covariate modifies the treatment-outcome relationship
    directly in the observed data, with no circularity.
    """
    banner("STEP 6c  Treatment-effect heterogeneity (interaction-term regression)")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    rows = []
    n_errors_shown = 0
    for name, r in results.items():
        sub  = r["sub"]
        tcol = r["tcol"]
        outcome_candidates = [c for c in sub.columns if c not in covars and c != tcol]
        if len(outcome_candidates) != 1:
            print(f"  [{name}] could not identify outcome column "
                  f"(candidates: {outcome_candidates}) -- skipping")
            continue
        outcome_col = outcome_candidates[0]

        tested = 0
        for cov in covars:
            if cov not in sub.columns or cov == tcol:
                continue
            other_covars = [c for c in covars if c in sub.columns and c not in (tcol, cov)]
            cols = [outcome_col, tcol, cov] + other_covars
            # .astype("float64") strips pandas nullable extension dtypes (e.g.
            # Float64Dtype from sex_numeric's Int64 origin) down to plain numpy
            # float64 -- patsy/statsmodels' OLS formula API cannot interpret
            # pandas' nullable dtypes and raises TypeError otherwise.
            data = (sub[cols].apply(pd.to_numeric, errors="coerce")
                    .astype("float64").dropna())
            if len(data) < 100:
                if n_errors_shown < 3:
                    print(f"    [{name} x {cov}] SKIPPED: n={len(data)} < 100 after dropna")
                    n_errors_shown += 1
                continue

            data = data.rename(columns={outcome_col: "Y", tcol: "T", cov: "M"})
            other_renamed = [f"c{i}" for i in range(len(other_covars))]
            data = data.rename(columns=dict(zip(other_covars, other_renamed)))

            formula = "Y ~ T * M" + ("".join(f" + {c}" for c in other_renamed))
            try:
                model = smf.ols(formula, data=data).fit()
                inter_term = "T:M"
                if inter_term not in model.params.index:
                    if n_errors_shown < 3:
                        print(f"    [{name} x {cov}] SKIPPED: 'T:M' not in "
                              f"model params: {list(model.params.index)}")
                        n_errors_shown += 1
                    continue
                coef = model.params[inter_term]
                p    = model.pvalues[inter_term]
            except Exception as exc:
                if n_errors_shown < 3:
                    print(f"    [{name} x {cov}] ERROR: {type(exc).__name__}: {exc}")
                    n_errors_shown += 1
                continue

            rows.append(dict(Intervention=name, Covariate=cov, n=len(data),
                              interaction_coef=coef, p_raw=p))
            tested += 1
        print(f"  [{name}]  n={len(sub):,}, {tested}/{len(covars)-1} "
              f"covariate interactions successfully tested")

    if not rows:
        print("  No heterogeneity tests could be run.")
        return None

    het = pd.DataFrame(rows)
    rej, p_adj, _, _ = multipletests(het["p_raw"], alpha=0.05, method="fdr_bh")
    het["p_fdr"]   = p_adj
    het["sig_fdr"] = rej
    het = het.sort_values("p_fdr")

    print(f"  {len(het)} (intervention x covariate) pairs tested")
    print(f"  significant before FDR : {(het['p_raw'] < 0.05).sum()}")
    print(f"  significant after  FDR : {het['sig_fdr'].sum()}")
    if het["sig_fdr"].sum():
        print("\n  FDR-significant effect modification (interaction term):")
        print(het[het["sig_fdr"]]
              [["Intervention", "Covariate", "interaction_coef", "p_fdr"]]
              .to_string(index=False))
    else:
        print("  No covariate significantly predicts effect heterogeneity "
              "after FDR correction.")

    het.to_csv(RESULTS / csv_name, index=False)
    print(f"  wrote {csv_name}")
    return het


def step7_figures(d: pd.DataFrame, study: pd.DataFrame,
                  results: dict, ranked: pd.DataFrame):
    banner("STEP 7  Figures (300 DPI) and CSVs")
    plt = _style()
    import matplotlib.pyplot as plt  # noqa

    # Figure 1 -- cohort overview
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    full = d["AgeAccel"].dropna()
    axes[0].hist(full, bins=60, color=C["grey"], alpha=0.55, density=True,
                 label=f"Full cohort (n={len(full):,})")
    axes[0].hist(study["AgeAccel"], bins=30, color=C["teal"], alpha=0.75, density=True,
                 label=f"Study pop 40-50 (n={len(study):,})")
    axes[0].axvline(0, color=C["ink"], lw=0.8, ls="--")
    axes[0].set_xlabel("Age acceleration (yr)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A  Biological age acceleration")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlim(-25, 25)

    sc = axes[1].scatter(d["age"], d["PhenoAge"], c=d["AgeAccel"],
                         cmap="RdBu_r", vmin=-15, vmax=15, s=4, alpha=0.35)
    lims = [d["age"].min(), d["age"].max()]
    axes[1].plot(lims, lims, color=C["ink"], lw=0.8, ls="--")
    axes[1].set_xlabel("Chronological age (yr)")
    axes[1].set_ylabel("PhenoAge (yr)")
    axes[1].set_title("B  PhenoAge vs chronological age")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "AgeAccel (yr)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure1_ageaccel_overview.png")
    plt.close(fig)
    print("  saved figure1_ageaccel_overview.png")

    # Figure 2 -- ranked effects
    _ranked_figure(ranked, RESULTS / "figure2_ranked_effects.png",
                   "Ranked causal impact on biological-age reversal",
                   "Standardised effect on AgeAccel (yr per +1 SD)")

    if not results:
        return

    # Figure 3 -- top-intervention ITEs
    top_name    = ranked.iloc[0]["Intervention"]
    top         = results[top_name]
    sub, ite    = top["sub"], top["ite"]
    fig, axes   = plt.subplots(1, 2, figsize=(11, 4.4))
    pct         = 100 * np.mean(ite < 0)
    axes[0].hist(ite, bins=40, color=C["teal"], alpha=0.8)
    axes[0].axvline(0, color=C["ink"], lw=0.9, ls="--")
    axes[0].set_xlabel("Individual treatment effect (yr)")
    axes[0].set_ylabel("Participants")
    axes[0].set_title(f"A  ITE distribution  —  {top_name}")
    axes[0].text(0.03, 0.93, f"{pct:.0f}% benefit (ITE < 0)",
                 transform=axes[0].transAxes, fontsize=9, color=C["green"])
    axes[1].scatter(sub["age"], ite, s=8, alpha=0.5, color=C["teal"])
    axes[1].axhline(0, color=C["ink"], lw=0.8, ls="--")
    axes[1].set_xlabel("Chronological age (yr)")
    axes[1].set_ylabel("ITE (yr)")
    axes[1].set_title("B  Effect heterogeneity by age")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure3_top_intervention_ite.png")
    plt.close(fig)
    print("  saved figure3_top_intervention_ite.png")

    # Figure 4 -- dose-response
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t     = sub[top["tcol"]].values.astype(float)
    y_obs = sub["AgeAccel"].values
    y_cf  = y_obs - ite * (t - np.median(t))
    grp   = pd.DataFrame({"q": pd.qcut(t, 5, duplicates="drop"),
                           "obs": y_obs, "cf": y_cf, "t": t})
    agg   = grp.groupby("q", observed=True).mean(numeric_only=True)
    xp    = np.arange(len(agg))
    ax.plot(xp, agg["obs"], "-o",  color=C["rust"], label="Observed AgeAccel")
    ax.plot(xp, agg["cf"],  "--s", color=C["teal"], label="Counterfactual (median dose)")
    ax.set_xticks(xp)
    ax.set_xticklabels([f"Q{i+1}\n({v:.1f})" for i, v in enumerate(agg["t"])],
                       fontsize=8)
    ax.set_xlabel(f"Quintile of {top_name}")
    ax.set_ylabel("Mean AgeAccel (yr)")
    ax.set_title(f"Dose-response  —  {top_name}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure4_dose_response.png")
    plt.close(fig)
    print("  saved figure4_dose_response.png")

    # Figure 5 -- covariate balance (Love plots)
    names  = list(results.keys())
    ncols  = min(len(names), 4)
    nrows  = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 3.8 * nrows),
                              sharex=True)
    axes  = np.atleast_1d(np.array(axes)).flatten()
    gmax  = max(results[n]["balance"]["corr_raw"].abs().max() for n in names)
    gmax  = max(gmax, 0.12) * 1.12

    for ax, name in zip(axes, names):
        bal = results[name]["balance"]
        yb  = np.arange(len(bal))
        ax.hlines(yb, bal["corr_raw"].abs(), bal["corr_adj"].abs(),
                  color=C["grey"], lw=1, zorder=1)
        ax.scatter(bal["corr_raw"].abs(), yb, s=50, facecolors="none",
                   edgecolors=C["rust"], label="raw", zorder=2)
        ax.scatter(bal["corr_adj"].abs(), yb, s=50, color=C["teal"],
                   label="orthogonalised", zorder=3)
        ax.axvline(0.1, color=C["ink"], lw=0.8, ls=":")
        ax.set_yticks(yb)
        ax.set_yticklabels(bal["covariate"])
        ax.set_xlabel("|corr| with treatment")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-0.01, gmax)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Covariate balance: confounding removed by DML orthogonalisation",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005,
             "dotted = 0.1 threshold   open = raw correlation   "
             "filled = after orthogonalisation",
             ha="center", fontsize=8, color=C["grey"])
    fig.savefig(RESULTS / "figure5_covariate_balance.png")
    plt.close(fig)
    print("  saved figure5_covariate_balance.png")

    # CSVs
    ranked.to_csv(RESULTS / "intervention_ranking.csv", index=False)
    pd.concat([r["balance"].assign(intervention=n)
               for n, r in results.items()]).to_csv(
        RESULTS / "covariate_balance.csv", index=False)
    out = results[top_name]["sub"].copy()
    out["ITE_AgeAccel"] = results[top_name]["ite"]
    out.to_csv(RESULTS / "top_intervention_ITE.csv", index=False)
    print("  wrote intervention_ranking.csv, covariate_balance.csv, "
          "top_intervention_ITE.csv")


# =========================================================================== #
# STEP 8  Load gut microbiome (MetaPhlAn 4 genus-level)                        #
# Column format: k__Bacteria|p__...|c__...|o__...|f__...|g__GENUS (float64, %) #
# =========================================================================== #
def step8_load_microbiome(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    banner("STEP 8  Load gut microbiome (MetaPhlAn 4, genus-level)")
    meta  = _load("gut_microbiome",
                  columns=["study_id", "collection_date", "sample_name"])
    genus = _load("gut_genus")

    # Genus columns: full taxonomic path ending at g__ with no species suffix
    gcols = [c for c in genus.columns if "|g__" in c and "|s__" not in c]
    if not gcols:
        gcols = [c for c in genus.columns if "g__" in c]
    print(f"  genus columns: {len(gcols)}")

    if "study_id" in genus.columns:
        genus_wide = genus[["study_id"] + gcols].copy()
    else:
        genus_wide = (meta.merge(genus, on="sample_name", how="inner")
                          [["study_id"] + gcols])

    # Shannon diversity from percent relative abundances
    X      = genus_wide[gcols].fillna(0.0).values / 100.0
    X_safe = np.where(X > 0, X, 1e-10)
    genus_wide["gut_shannon"]  = -np.sum(X_safe * np.log(X_safe), axis=1)
    genus_wide["gut_richness"] = (X > 0).sum(axis=1)

    # For longitudinal data keep the most recent sample per participant
    if "collection_date" in meta.columns and "sample_name" in genus_wide.columns:
        latest = (meta.sort_values("collection_date", ascending=False)
                      .drop_duplicates("study_id", keep="first"))
        genus_wide = (latest[["study_id", "sample_name"]]
                      .merge(genus_wide.drop(columns=["study_id"], errors="ignore"),
                             on="sample_name", how="inner"))

    genus_per_person = genus_wide.drop_duplicates("study_id")
    print(f"  participants with gut microbiome: {len(genus_per_person):,}")
    print(f"  Shannon: mean={genus_per_person['gut_shannon'].mean():.2f}, "
          f"SD={genus_per_person['gut_shannon'].std():.2f}")

    merge_cols = ["study_id", "gut_shannon", "gut_richness"] + gcols
    d = d.merge(genus_per_person[merge_cols], on="study_id", how="left")

    # Additionally load SPECIES-level abundances for the age clock (Step 9).
    # Genus stays the basis for Shannon diversity and all mediation steps so
    # nothing downstream changes; species is finer-grained and is only used to
    # try to build a more accurate (= less noisy) clock. Failure here is
    # non-fatal: the clock falls back to genus features.
    scols: list[str] = []
    try:
        species = _load("gut_species")
        scols = [c for c in species.columns if "|s__" in c]
        if not scols:
            scols = [c for c in species.columns if "s__" in c]
        print(f"  species columns: {len(scols)}")
        if scols:
            if "study_id" in species.columns:
                sp_wide = species[["study_id"] + scols].copy()
            else:
                sp_wide = (meta.merge(species, on="sample_name", how="inner")
                               [["study_id"] + scols])
            sp_per_person = sp_wide.drop_duplicates("study_id")
            # Suffix avoids any chance of a species column name colliding with
            # an existing genus column already merged into d.
            sp_per_person = sp_per_person.rename(
                columns={c: f"{c}__sp" for c in scols})
            scols = [f"{c}__sp" for c in scols]
            d = d.merge(sp_per_person[["study_id"] + scols],
                        on="study_id", how="left")
            print(f"  merged species-level features for {len(sp_per_person):,} participants")
    except Exception as exc:
        print(f"  species-level load failed ({type(exc).__name__}: {exc}) -- "
              "clock will use genus features only")
        scols = []

    return d, gcols, scols


# =========================================================================== #
# STEP 9  Gut microbiome age clock                                             #
# =========================================================================== #
def step9_microbiome_clock(d: pd.DataFrame,
                           gcols: list[str],
                           scols: list[str] | None = None) -> tuple[pd.DataFrame, float, float]:
    """Gut microbiome age clock. The derived gut_age_accel (predicted minus
    chronological age) is used as an alternative outcome in Step 9b, so clock
    ACCURACY directly determines how noisy that outcome is: a weak clock means
    a noisy age-acceleration score, which costs statistical power in every
    downstream causal test. The original ElasticNet-only version scored
    r=0.25 (self-flagged "weak"), so this now also fits a gradient-boosting
    model -- which can capture nonlinear taxon-age relationships that a linear
    ElasticNet cannot -- and keeps whichever cross-validates better.
    """
    banner("STEP 9  Gut microbiome age clock (ElasticNet vs. gradient boosting)")
    from scipy.stats import pearsonr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Prefer species-level features (finer resolution) when available, else
    # fall back to genus. Both are just abundance columns to the model.
    if scols:
        feature_cols = [c for c in scols if c in d.columns]
        level = "species"
    else:
        feature_cols = []
        level = "genus"
    if not feature_cols:
        feature_cols = [c for c in gcols if c in d.columns]
        level = "genus"

    avail    = feature_cols
    clock_df = d[["study_id", "age"] + avail].dropna(subset=["age"]).copy()
    clock_df[avail] = clock_df[avail].fillna(0.0)

    # Lower prevalence threshold (was 0.10): rarer taxa can still carry age
    # signal, and the models below regularise/select internally anyway.
    prev = (clock_df[avail] > 0).mean()
    keep = prev[prev > 0.05].index.tolist()
    print(f"  using {level}-level features: "
          f"{len(keep)}/{len(avail)} pass prevalence >5%")
    if len(keep) < 5:
        print("  Insufficient genera for clock -- skipping")
        return d, 0.0, 0.0

    X    = clock_df[keep].values
    y    = clock_df["age"].values
    cv   = KFold(5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "ElasticNet": make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, cv=5,
                         random_state=RANDOM_STATE, max_iter=5000)),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE),
    }

    best_name, best_pred, best_r, best_mae = None, None, -np.inf, np.inf
    for mname, model in candidates.items():
        try:
            p = cross_val_predict(model, X, y, cv=cv)
        except Exception as exc:
            print(f"  {mname:<18}: FAILED ({type(exc).__name__}: {exc})")
            continue
        r_m   = float(pearsonr(p, y)[0])
        mae_m = float(np.mean(np.abs(p - y)))
        print(f"  {mname:<18}: r={r_m:.3f}, MAE={mae_m:.1f} yr")
        if r_m > best_r:
            best_name, best_pred, best_r, best_mae = mname, p, r_m, mae_m

    if best_pred is None:
        print("  All clock models failed -- skipping")
        return d, 0.0, 0.0

    pred, r, mae = best_pred, best_r, best_mae
    print(f"  --> selected {best_name}: r={r:.3f}, MAE={mae:.1f} yr")
    print(f"  quality: {'strong' if r >= 0.50 else 'moderate' if r >= 0.30 else 'weak'}")

    clock_df["gut_pred_age"]  = pred
    clock_df["gut_age_accel"] = pred - y
    d = d.merge(clock_df[["study_id", "gut_pred_age", "gut_age_accel"]],
                on="study_id", how="left")

    # Figure 6
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo, hi  = y.min() - 5, y.max() + 5
    ax.scatter(y, pred, s=8, alpha=0.3, color=C["teal"])
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.9, ls="--")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Microbiome-predicted age (yr)")
    ax.set_title("Gut microbiome age clock (MetaPhlAn 4, ElasticNet)")
    ax.text(0.04, 0.90, f"r = {r:.2f}\nMAE = {mae:.1f} yr",
            transform=ax.transAxes, fontsize=10, color=C["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure6_microbiome_clock.png")
    plt.close(fig)
    print("  saved figure6_microbiome_clock.png")
    return d, r, mae


# =========================================================================== #
# STEP 9b  Lifestyle -> GUT-MICROBIOME-DERIVED age acceleration                #
#                                                                               #
# Every causal test so far used AgeAccel from PhenoAge (blood biomarkers) as   #
# the outcome. But the reason HPP was chosen over a public dataset like        #
# NHANES is that prior work argues the MICROBIOME, not blood biomarkers, is    #
# the dominant driver of accelerated aging in this cohort -- and we have not   #
# been able to confirm a published, reproducible microbiome age-acceleration  #
# score exists to test that directly (pending confirmation from the dataset   #
# lead). Rather than wait, this step reuses our own already-validated gut     #
# microbiome age clock (Step 9, r=0.25, MAE=6.3yr) -- gut_age_accel is        #
# defined the same way PhenoAge's AgeAccel is (predicted age minus            #
# chronological age), just derived from microbiome composition instead of     #
# blood biomarkers. Repeating the exact same causal analysis with THIS as the #
# outcome is a legitimate, defensible way to test the actual microbiome-aging #
# hypothesis using only what we have already built and validated.             #
#                                                                               #
# Population note: unlike the PhenoAge analysis, this step does NOT require   #
# AgeAccel (PhenoAge) > 0 -- that filter defines "accelerated" with respect to #
# blood-biomarker aging, which is a different biological measure and should   #
# not gate who is included when the outcome itself has changed. The age       #
# 40-50 window is retained, since that restriction reflects the proposed      #
# critical intervention window itself, independent of which aging metric is   #
# used to measure it.                                                         #
# =========================================================================== #
def step9b_microbiome_outcome_causal(d: pd.DataFrame) -> dict | None:
    banner("STEP 9b  Lifestyle -> gut-microbiome-derived age acceleration")
    _ensure_econml()

    if "gut_age_accel" not in d.columns:
        print("  gut_age_accel not available (clock did not run) -- skipping")
        return None

    covars = [c for c in COVARIATES if c in d.columns]
    pop = d[d["age"].between(40, 50) & d["gut_age_accel"].notna()].copy()
    print(f"  population: age 40-50 with valid gut microbiome clock, n={len(pop):,}")
    print("  (not filtered on PhenoAge AgeAccel>0 -- see comment above)")

    results = step5_causal(pop, outcome="gut_age_accel", interventions=INTERVENTIONS)
    if not results:
        print("  No models fit.")
        return None

    ranked = step6_rank(results)
    ranked.to_csv(RESULTS / "intervention_ranking_microbiome_outcome.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure_microbiome_outcome_ranked_effects.png",
                   "Lifestyle causal effect on GUT-MICROBIOME-derived age acceleration",
                   "Standardised effect on gut_age_accel (per +1 SD)")
    print("  wrote intervention_ranking_microbiome_outcome.csv")

    step6c_heterogeneity(results, covars,
                        csv_name="heterogeneity_results_microbiome_outcome.csv")

    return results


# =========================================================================== #
# STEP 10  Lifestyle -> gut microbiome Shannon diversity                        #
# =========================================================================== #
def step10_causal_microbiome(d: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    banner("STEP 10  Lifestyle -> gut microbiome Shannon diversity")
    _ensure_econml()
    covars  = [c for c in COVARIATES if c in d.columns]
    mask    = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb   = d[mask & d["gut_shannon"].notna()].copy()
    print(f"  n = {len(popmb):,}")

    results = {}
    for name, tcol in INTERVENTIONS.items():
        if tcol not in popmb.columns:
            continue
        sub = popmb[["gut_shannon", tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"\n  [{name}]  n={len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue
        res = _fit_cf(sub, tcol, "gut_shannon", covars)
        ns  = " (n.s.)" if res["ci"][0] < 0 < res["ci"][1] else ""
        print(f"    ATE = {res['ate']:+.5f} Shannon/unit  "
              f"[95% CI {res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]{ns}")
        results[name] = res

    if not results:
        print("  no models fit -- skipping Figure 7")
        return results, None

    ranked = _rank_results(results)
    ranked.to_csv(RESULTS / "microbiome_diversity_ranking.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure7_microbiome_diversity_effects.png",
                   "Lifestyle causal effect on gut microbiome diversity",
                   "Standardised effect on Shannon diversity (per +1 SD)")
    print("  wrote microbiome_diversity_ranking.csv")
    return results, ranked


def _fit_cf(sub: pd.DataFrame, tcol: str, ycol: str,
            covars: list[str]) -> dict:
    Y   = sub[ycol].values
    T   = sub[tcol].values.astype(float)
    X   = sub[covars].values
    est = _make_cf()
    est.fit(Y, T, X=X)
    ate    = float(est.ate(X))
    lb, ub = [float(v) for v in est.ate_interval(X, alpha=0.05)]
    ite    = est.effect(X)
    sd_t   = float(np.std(T))
    return dict(tcol=tcol, ate=ate, ci=(lb, ub), sd_t=sd_t,
                std_effect=ate * sd_t, ite=ite,
                sub=sub.reset_index(drop=True))


# =========================================================================== #
# STEP 11  Mediation: lifestyle -> gut microbiome -> PhenoAge                  #
# =========================================================================== #
def step11_mediation(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 11  Mediation (lifestyle -> gut diversity -> PhenoAge accel)")
    import statsmodels.api as sm

    covars  = [c for c in COVARIATES if c in d.columns]
    mask    = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb   = d[mask & d["gut_shannon"].notna()].copy()
    M, Y    = "gut_shannon", "AgeAccel"

    def ols(y_vec, x_df):
        x_num = x_df.apply(pd.to_numeric, errors="coerce").astype(float)
        return sm.OLS(y_vec.astype(float), sm.add_constant(x_num)).fit()

    def path_coefficients(s: pd.DataFrame, T: str):
        a      = ols(s[M], s[[T] + covars]).params[T]
        full   = ols(s[Y], s[[M, T] + covars])
        b, cp  = full.params[M], full.params[T]
        c      = ols(s[Y], s[[T] + covars]).params[T]
        return a, b, cp, c

    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [(k, v) for k, v in INTERVENTIONS.items()
                    if v in popmb.columns]:
        s = popmb[[M, Y, T] + covars].copy()
        s[T] = pd.to_numeric(s[T], errors="coerce")
        s = s.dropna()
        if T == "sleep_hours":
            s = s[s[T].between(2, 14)]
        n = len(s)
        print(f"\n  [{name}]  n={n:,}")
        if n < MEDIATION_MIN_N:
            print(f"    SKIPPED (n < {MEDIATION_MIN_N})")
            continue

        a, b, cp, c = path_coefficients(s, T)
        indirect = a * b
        prop     = indirect / c if abs(c) > 1e-9 else np.nan

        boot_ind, boot_prop = [], []
        idx = np.arange(n)
        for _ in range(200):
            bs = s.iloc[rng.choice(idx, n, replace=True)]
            try:
                aa, bb, _, cc = path_coefficients(bs, T)
                boot_ind.append(aa * bb)
                boot_prop.append(aa * bb / cc if abs(cc) > 1e-9 else np.nan)
            except Exception:
                continue

        ci_ind  = np.nanpercentile(boot_ind,  [2.5, 97.5])
        ci_prop = np.nanpercentile(boot_prop, [2.5, 97.5])
        sig     = "" if ci_ind[0] < 0 < ci_ind[1] else "  (CI excludes 0)"

        print(f"    a  (T -> M)           = {a:+.5f}")
        print(f"    b  (M -> Y | T)       = {b:+.4f}")
        print(f"    c  total (T -> Y)     = {c:+.4f}")
        print(f"    c' direct (T -> Y|M)  = {cp:+.4f}")
        print(f"    indirect (a*b)        = {indirect:+.4f} "
              f"[95% CI {ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]{sig}")
        print(f"    proportion mediated   = {100*prop:5.1f}% "
              f"[{100*ci_prop[0]:.1f}%, {100*ci_prop[1]:.1f}%]")

        rows.append(dict(
            Intervention=name, n=n, a=a, b=b, total_c=c, direct_cprime=cp,
            indirect=indirect, indirect_lo=ci_ind[0], indirect_hi=ci_ind[1],
            prop_mediated=prop, prop_lo=ci_prop[0], prop_hi=ci_prop[1],
        ))

    if not rows:
        print("  No mediation models could be estimated.")
        return None

    med = pd.DataFrame(rows)
    med.to_csv(RESULTS / "mediation_results.csv", index=False)

    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.7 * len(med) + 1.5)))
    yb   = np.arange(len(med))
    prop = 100 * med["prop_mediated"].values
    lo   = 100 * med["prop_lo"].values
    hi   = 100 * med["prop_hi"].values
    ns   = [(med["indirect_lo"].iloc[i] < 0 < med["indirect_hi"].iloc[i])
            for i in range(len(med))]
    # "Proportion mediated" is a ratio (indirect/total effect) and can be
    # extremely unstable when the total effect is near zero -- the point
    # estimate occasionally falls outside its own bootstrap CI in that case.
    # Clip error-bar lengths to >=0 so matplotlib doesn't reject negative xerr;
    # the CI values themselves (lo/hi, printed and saved to CSV) are untouched.
    err_lo = np.clip(prop - lo, 0, None)
    err_hi = np.clip(hi - prop, 0, None)
    ax.barh(yb, prop,
            xerr=np.vstack([err_lo, err_hi]),
            color=[C["grey"] if n else C["teal"] for n in ns],
            alpha=0.85, error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(med["Intervention"])
    ax.set_xlabel("% of lifestyle -> PhenoAge effect mediated by gut microbiome")
    ax.set_title("Gut microbiome mediation of lifestyle effects")
    ax.text(0.98, 0.04, "grey = n.s. (indirect CI crosses 0)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure8_mediation.png")
    plt.close(fig)
    print("  saved figure8_mediation.png")
    return med


# =========================================================================== #
# STEP 11b  Species-level mediation: lifestyle -> individual taxa -> AgeAccel  #
#                                                                               #
# Motivation (Dr. Razzak, 2026-07-26): the research question is not only      #
# "which lifestyle lever reduces biological age" (direct effect -- tested in  #
# Step 5/10, found non-significant after CRP/outlier correction) but also     #
# "how do we improve the microbiome in a way that, in turn, improves the age  #
# factor" -- i.e. the INDIRECT/mediated pathway lifestyle -> specific taxon   #
# -> AgeAccel. Step 11's Shannon-diversity mediator is a single coarse number #
# that collapses the whole community and returned null for every lever; this #
# step repeats the same product-of-coefficients mediation test but for each   #
# individual genus, so a taxon-specific mechanism can be identified even when #
# the overall-diversity summary shows nothing.                                #
# =========================================================================== #
def _species_level_mediation(d: pd.DataFrame, gcols: list[str], tag: str,
                             outcome: str = "AgeAccel",
                             min_prevalence: float = 0.10,
                             max_taxa: int = 40,
                             n_boot: int = 200) -> pd.DataFrame | None:
    banner(f"STEP 11b  Species-level mediation ({tag}: lifestyle -> taxon -> {outcome})")
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    avail  = [c for c in gcols if c in d.columns]
    pop    = d[mask].copy()
    if not avail:
        print("  no taxon columns available -- skipping")
        return None

    # Same prevalence filter as the microbiome clock, so mediators are taxa
    # that are actually present in a meaningful fraction of the cohort.
    pop[avail] = pop[avail].fillna(0.0)
    prev = (pop[avail] > 0).mean()
    keep = prev[prev > min_prevalence].index.tolist()
    if not keep:
        print("  no taxa pass prevalence filter -- skipping")
        return None

    # Cap the number of taxa tested (highest-variance first) to keep the
    # (intervention x taxon) test count, and thus the FDR correction, sane.
    variances = pop[keep].var().sort_values(ascending=False)
    keep = variances.head(max_taxa).index.tolist()
    print(f"  testing {len(keep)} taxa (prevalence >{min_prevalence:.0%}, "
          f"top {max_taxa} by variance) x {len(INTERVENTIONS)} interventions")

    def ols(y_vec, x_df):
        x_num = x_df.apply(pd.to_numeric, errors="coerce").astype(float)
        return sm.OLS(y_vec.astype(float), sm.add_constant(x_num)).fit()

    def path_coefficients(s: pd.DataFrame, T: str, M: str):
        a     = ols(s[M], s[[T] + covars]).params[T]
        full  = ols(s[outcome], s[[M, T] + covars])
        b, cp = full.params[M], full.params[T]
        c     = ols(s[outcome], s[[T] + covars]).params[T]
        return a, b, cp, c

    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [(k, v) for k, v in INTERVENTIONS.items() if v in pop.columns]:
        for M in keep:
            s = pop[[M, outcome, T] + covars].copy()
            s[T] = pd.to_numeric(s[T], errors="coerce")
            s = s.dropna()
            if T == "sleep_hours":
                s = s[s[T].between(2, 14)]
            n = len(s)
            if n < MEDIATION_MIN_N:
                continue
            try:
                a, b, cp, c = path_coefficients(s, T, M)
            except Exception:
                continue
            indirect = a * b

            boot_ind = []
            idx = np.arange(n)
            for _ in range(n_boot):
                bs = s.iloc[rng.choice(idx, n, replace=True)]
                try:
                    aa, bb, _, _ = path_coefficients(bs, T, M)
                    boot_ind.append(aa * bb)
                except Exception:
                    continue
            if len(boot_ind) < n_boot // 2:
                continue
            ci_lo, ci_hi = np.nanpercentile(boot_ind, [2.5, 97.5])

            rows.append(dict(
                Intervention=name, Taxon=M, n=n, a=a, b=b,
                total_c=c, direct_cprime=cp, indirect=indirect,
                indirect_lo=ci_lo, indirect_hi=ci_hi,
            ))

    if not rows:
        print("  No species-level mediation models could be estimated.")
        return None

    med = pd.DataFrame(rows)
    # Raw (per-pair) significance, plus BH-FDR correction across every
    # (intervention x taxon) test -- this is a much larger multiple-comparison
    # problem than the single-mediator Step 11/16 analyses, so correction is
    # mandatory here, not optional.
    med["sig_raw"] = ~((med["indirect_lo"] < 0) & (med["indirect_hi"] > 0))
    # Approximate two-sided p-value from the bootstrap CI width via a normal
    # approximation (CI does not directly give a p-value; this is a coarse
    # ranking aid, not a substitute for a full bootstrap p-value).
    se = (med["indirect_hi"] - med["indirect_lo"]) / (2 * 1.96)
    z  = med["indirect"] / se.replace(0, np.nan)
    from scipy.stats import norm
    med["p_approx"] = 2 * (1 - norm.cdf(np.abs(z)))
    med["p_approx"] = med["p_approx"].fillna(1.0)
    rej, p_adj, _, _ = multipletests(med["p_approx"], alpha=0.05, method="fdr_bh")
    med["p_fdr"]      = p_adj
    med["sig_fdr"]    = rej

    med = med.sort_values("p_fdr")
    n_sig_raw = int(med["sig_raw"].sum())
    n_sig_fdr = int(med["sig_fdr"].sum())
    print(f"\n  {len(med)} (intervention x taxon) pairs tested")
    print(f"  significant before FDR correction : {n_sig_raw}")
    print(f"  significant after  FDR correction  : {n_sig_fdr}")
    if n_sig_fdr:
        print("\n  Top FDR-significant taxon mediators:")
        print(med[med["sig_fdr"]][["Intervention", "Taxon", "indirect", "p_fdr"]]
              .head(15).to_string(index=False))

    med.to_csv(RESULTS / f"{tag}_species_mediation.csv", index=False)
    print(f"  wrote {tag}_species_mediation.csv")

    # Figure: top 15 taxon-mediators by |indirect effect|, coloured by FDR sig.
    top = med.reindex(med["indirect"].abs().sort_values(ascending=False).index).head(15)
    if len(top):
        plt = _style()
        fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.45 * len(top) + 1.5)))
        yb = np.arange(len(top))
        labels = [f"{r.Taxon.split('|')[-1].replace('g__','')}  ({r.Intervention})"
                  for r in top.itertuples()]
        ax.barh(yb, top["indirect"],
                color=[C["teal"] if s else C["grey"] for s in top["sig_fdr"]],
                alpha=0.85)
        ax.axvline(0, color=C["ink"], lw=0.8)
        ax.set_yticks(yb)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"Indirect effect (a x b) on {outcome}")
        ax.set_title(f"Top taxon-level mediators, {tag} microbiome "
                     "(teal = FDR-significant)")
        fig.tight_layout()
        fig.savefig(RESULTS / f"figure_{tag}_species_mediation.png")
        plt.close(fig)
        print(f"  saved figure_{tag}_species_mediation.png")

    return med


# =========================================================================== #
# STEP 12  Curated-phenotype sweep                                             #
# Runs CausalForestDML for every (intervention, phenotype) pair.               #
# Output: phenotype_sweep.csv + figure9_phenotype_heatmap.png                 #
# =========================================================================== #
def step12_phenotype_sweep(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 12  Curated-phenotype sweep (lifestyle -> each outcome)")
    if SKIP_SWEEP:
        print("  SKIPPED (HPP_SKIP_SWEEP=1). This step fits one causal forest")
        print("  per intervention-phenotype pair and dominates total runtime.")
        return None
    _ensure_econml()
    curated_dir = DS["curated_phenotypes"]
    if not curated_dir.exists():
        print(f"  Directory not found: {curated_dir}")
        return None

    covars   = [c for c in COVARIATES if c in d.columns]
    all_rows = []

    for stem, outcome_col in CURATED_PHENOTYPES:
        parquet = curated_dir / f"{stem}.parquet"
        if not parquet.exists():
            print(f"  [{stem}] parquet not found -- skipping")
            continue
        pheno = pd.read_parquet(parquet)
        if isinstance(pheno.index, pd.MultiIndex):
            pheno = pheno.reset_index()
            pid_col = next((c for c in ["participant_id", pheno.columns[0]] if c in pheno.columns), None)
            if pid_col and pid_col != "study_id":
                pheno = pheno.rename(columns={pid_col: "study_id"})
        elif "study_id" not in pheno.columns:
            pheno = pheno.reset_index().rename(columns={pheno.index.name or "index": "study_id"})
        if "study_id" in pheno.columns:
            pheno["study_id"] = pd.to_numeric(pheno["study_id"], errors="coerce")
            pheno = pheno.dropna(subset=["study_id"])
            pheno["study_id"] = pheno["study_id"].astype("int64")
        if outcome_col not in pheno.columns:
            print(f"  [{stem}] column '{outcome_col}' not found -- skipping")
            continue

        df = d.merge(pheno[["study_id", outcome_col]], on="study_id", how="inner")
        if df[outcome_col].dtype == object or str(df[outcome_col].dtype) == "category":
            codes = pd.Categorical(df[outcome_col]).codes.astype(float)
            df[outcome_col] = pd.Series(codes, index=df.index).where(lambda x: x >= 0)

        n_total = df[outcome_col].notna().sum()
        if n_total < 100:
            print(f"  [{stem}] n={n_total} -- skipping")
            continue

        print(f"\n  [{stem}]  n={n_total:,}")
        for intv_name, tcol in INTERVENTIONS.items():
            if tcol not in df.columns:
                continue
            sub = df[[outcome_col, tcol] + covars].dropna()
            if len(sub) < 100:
                continue
            try:
                res = _fit_cf(sub, tcol, outcome_col, covars)
                ns  = res["ci"][0] < 0 < res["ci"][1]
                print(f"    {intv_name:<30} ATE={res['ate']:+.4f} "
                      f"[{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]"
                      + (" n.s." if ns else " ***"))
                all_rows.append(dict(
                    phenotype=stem, intervention=intv_name, n=len(sub),
                    ate=res["ate"], ci_low=res["ci"][0], ci_high=res["ci"][1],
                    std_effect=res["std_effect"], ns=ns))
            except Exception as exc:
                print(f"    {intv_name}: ERROR -- {exc}")

    if not all_rows:
        print("  No results produced.")
        return None

    sweep = pd.DataFrame(all_rows)

    # BH-FDR correction across all (intervention x phenotype) pairs. With 90+
    # tests at alpha=0.05, ~4-5 "significant" hits are expected by chance
    # alone -- the raw count printed per-phenotype above is not a result on
    # its own until corrected for multiple comparisons.
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    se = (sweep["ci_high"] - sweep["ci_low"]) / (2 * 1.96)
    z  = sweep["ate"] / se.replace(0, np.nan)
    sweep["p_approx"] = (2 * (1 - norm.cdf(np.abs(z)))).fillna(1.0)
    rej, p_adj, _, _  = multipletests(sweep["p_approx"], alpha=0.05, method="fdr_bh")
    sweep["p_fdr"]    = p_adj
    sweep["sig_fdr"]  = rej

    sweep.to_csv(RESULTS / "phenotype_sweep.csv", index=False)
    print(f"\n  wrote phenotype_sweep.csv  "
          f"({len(sweep)} intervention-phenotype pairs)")
    print(f"  significant before FDR correction : {(~sweep['ns']).sum()}")
    print(f"  significant after  FDR correction  : {sweep['sig_fdr'].sum()}")
    if sweep["sig_fdr"].sum():
        print("\n  FDR-significant pairs:")
        print(sweep[sweep["sig_fdr"]].sort_values("p_fdr")
              [["phenotype", "intervention", "ate", "p_fdr"]]
              .to_string(index=False))

    # Figure 9 -- effect heatmap
    try:
        pivot = sweep.pivot(index="phenotype", columns="intervention",
                            values="std_effect")
        sig   = sweep.pivot(index="phenotype", columns="intervention", values="ns")
        plt   = _style()
        import matplotlib.pyplot as plt  # noqa
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)),
                                         max(5, 0.5 * len(pivot))))
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                       vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i, pheno in enumerate(pivot.index):
            for j, intv_n in enumerate(pivot.columns):
                if not sig.loc[pheno, intv_n]:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=12, color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
            "Standardised effect (per +1 SD)", fontsize=9)
        ax.set_title("Lifestyle causal effects across curated phenotypes\n"
                     "* = 95% CI excludes 0", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS / "figure9_phenotype_heatmap.png")
        plt.close(fig)
        print("  saved figure9_phenotype_heatmap.png")
    except Exception as exc:
        print(f"  Heatmap could not be generated: {exc}")

    return sweep


# =========================================================================== #
# STEP 13  Load oral microbiome (MetaPhlAn 4 genus-level)                     #
# Same column format as gut: k__...|p__...|c__...|o__...|f__...|g__GENUS     #
# =========================================================================== #
def step13_load_oral_microbiome(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    banner("STEP 13  Load oral microbiome (MetaPhlAn 4, genus-level)")
    meta  = _load("oral_microbiome",
                  columns=["study_id", "collection_date", "sample_name"])
    genus = _load("oral_genus")

    gcols = [c for c in genus.columns if "|g__" in c and "|s__" not in c]
    if not gcols:
        gcols = [c for c in genus.columns if "g__" in c]
    print(f"  genus columns: {len(gcols)}")

    if "study_id" in genus.columns:
        genus_wide = genus[["study_id"] + gcols].copy()
    else:
        genus_wide = (meta.merge(genus, on="sample_name", how="inner")
                          [["study_id"] + gcols])

    X      = genus_wide[gcols].fillna(0.0).values / 100.0
    X_safe = np.where(X > 0, X, 1e-10)
    genus_wide["oral_shannon"]  = -np.sum(X_safe * np.log(X_safe), axis=1)
    genus_wide["oral_richness"] = (X > 0).sum(axis=1)

    if "collection_date" in meta.columns and "sample_name" in genus_wide.columns:
        latest = (meta.sort_values("collection_date", ascending=False)
                      .drop_duplicates("study_id", keep="first"))
        genus_wide = (latest[["study_id", "sample_name"]]
                      .merge(genus_wide.drop(columns=["study_id"], errors="ignore"),
                             on="sample_name", how="inner"))

    genus_per_person = genus_wide.drop_duplicates("study_id")
    print(f"  participants with oral microbiome: {len(genus_per_person):,}")
    print(f"  Shannon: mean={genus_per_person['oral_shannon'].mean():.2f}, "
          f"SD={genus_per_person['oral_shannon'].std():.2f}")

    oral_cols = ["study_id", "oral_shannon", "oral_richness"] + gcols
    d = d.merge(genus_per_person[oral_cols], on="study_id", how="left",
                suffixes=("", "_oral"))
    return d, gcols


# =========================================================================== #
# STEP 14  Oral microbiome age clock                                           #
# =========================================================================== #
def step14_oral_clock(d: pd.DataFrame,
                      oral_gcols: list[str]) -> tuple[pd.DataFrame, float, float]:
    banner("STEP 14  Oral microbiome age clock (ElasticNet, genus abundances)")
    from scipy.stats import pearsonr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    avail    = [c for c in oral_gcols if c in d.columns]
    clock_df = d[["study_id", "age"] + avail].dropna(subset=["age"]).copy()
    clock_df[avail] = clock_df[avail].fillna(0.0)

    prev = (clock_df[avail] > 0).mean()
    keep = prev[prev > 0.10].index.tolist()
    print(f"  {len(keep)}/{len(avail)} genera pass prevalence >10%")
    if len(keep) < 5:
        print("  Insufficient genera for clock -- skipping")
        return d, 0.0, 0.0

    X    = clock_df[keep].values
    y    = clock_df["age"].values
    cv   = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(
        make_pipeline(StandardScaler(),
                      ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50,
                                   cv=5, random_state=RANDOM_STATE,
                                   max_iter=5000)),
        X, y, cv=cv)
    r   = float(pearsonr(pred, y)[0])
    mae = float(np.mean(np.abs(pred - y)))
    print(f"  cross-validated clock: r={r:.3f}, MAE={mae:.1f} yr")
    print(f"  quality: {'strong' if r >= 0.50 else 'moderate' if r >= 0.30 else 'weak'}")

    clock_df["oral_pred_age"]  = pred
    clock_df["oral_age_accel"] = pred - y
    d = d.merge(clock_df[["study_id", "oral_pred_age", "oral_age_accel"]],
                on="study_id", how="left")

    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo, hi  = y.min() - 5, y.max() + 5
    ax.scatter(y, pred, s=8, alpha=0.3, color=C["rust"])
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.9, ls="--")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Microbiome-predicted age (yr)")
    ax.set_title("Oral microbiome age clock (MetaPhlAn 4, ElasticNet)")
    ax.text(0.04, 0.90, f"r = {r:.2f}\nMAE = {mae:.1f} yr",
            transform=ax.transAxes, fontsize=10, color=C["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure10_oral_clock.png")
    plt.close(fig)
    print("  saved figure10_oral_clock.png")
    return d, r, mae


# =========================================================================== #
# STEP 15  Lifestyle -> oral microbiome Shannon diversity                       #
# =========================================================================== #
def step15_causal_oral_microbiome(d: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    banner("STEP 15  Lifestyle -> oral microbiome Shannon diversity")
    _ensure_econml()
    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb  = d[mask & d["oral_shannon"].notna()].copy()
    print(f"  n = {len(popmb):,}")

    results = {}
    for name, tcol in INTERVENTIONS.items():
        if tcol not in popmb.columns:
            continue
        sub = popmb[["oral_shannon", tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"\n  [{name}]  n={len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue
        res = _fit_cf(sub, tcol, "oral_shannon", covars)
        ns  = " (n.s.)" if res["ci"][0] < 0 < res["ci"][1] else ""
        print(f"    ATE = {res['ate']:+.5f} Shannon/unit  "
              f"[95% CI {res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]{ns}")
        results[name] = res

    if not results:
        print("  no models fit -- skipping Figure 11")
        return results, None

    ranked = _rank_results(results)
    ranked.to_csv(RESULTS / "oral_diversity_ranking.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure11_oral_diversity_effects.png",
                   "Lifestyle causal effect on oral microbiome diversity",
                   "Standardised effect on Shannon diversity (per +1 SD)")
    print("  wrote oral_diversity_ranking.csv")
    return results, ranked


# =========================================================================== #
# STEP 16  Mediation: lifestyle -> oral microbiome -> PhenoAge                 #
# =========================================================================== #
def step16_oral_mediation(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 16  Mediation (lifestyle -> oral diversity -> PhenoAge accel)")
    import statsmodels.api as sm

    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb  = d[mask & d["oral_shannon"].notna()].copy()
    M, Y   = "oral_shannon", "AgeAccel"

    def ols(y_vec, x_df):
        x_num = x_df.apply(pd.to_numeric, errors="coerce").astype(float)
        return sm.OLS(y_vec.astype(float), sm.add_constant(x_num)).fit()

    def path_coefficients(s: pd.DataFrame, T: str):
        a      = ols(s[M], s[[T] + covars]).params[T]
        full   = ols(s[Y], s[[M, T] + covars])
        b, cp  = full.params[M], full.params[T]
        c      = ols(s[Y], s[[T] + covars]).params[T]
        return a, b, cp, c

    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [(k, v) for k, v in INTERVENTIONS.items()
                    if v in popmb.columns]:
        s = popmb[[M, Y, T] + covars].copy()
        s[T] = pd.to_numeric(s[T], errors="coerce")
        s = s.dropna()
        if T == "sleep_hours":
            s = s[s[T].between(2, 14)]
        n = len(s)
        print(f"\n  [{name}]  n={n:,}")
        if n < MEDIATION_MIN_N:
            print(f"    SKIPPED (n < {MEDIATION_MIN_N})")
            continue

        a, b, cp, c = path_coefficients(s, T)
        indirect = a * b
        prop     = indirect / c if abs(c) > 1e-9 else np.nan

        boot_ind, boot_prop = [], []
        idx = np.arange(n)
        for _ in range(200):
            bs = s.iloc[rng.choice(idx, n, replace=True)]
            try:
                aa, bb, _, cc = path_coefficients(bs, T)
                boot_ind.append(aa * bb)
                boot_prop.append(aa * bb / cc if abs(cc) > 1e-9 else np.nan)
            except Exception:
                continue

        ci_ind  = np.nanpercentile(boot_ind,  [2.5, 97.5])
        ci_prop = np.nanpercentile(boot_prop, [2.5, 97.5])
        sig     = "" if ci_ind[0] < 0 < ci_ind[1] else "  (CI excludes 0)"

        print(f"    a  (T -> M)           = {a:+.5f}")
        print(f"    b  (M -> Y | T)       = {b:+.4f}")
        print(f"    c  total (T -> Y)     = {c:+.4f}")
        print(f"    c' direct (T -> Y|M)  = {cp:+.4f}")
        print(f"    indirect (a*b)        = {indirect:+.4f} "
              f"[95% CI {ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]{sig}")
        print(f"    proportion mediated   = {100*prop:5.1f}% "
              f"[{100*ci_prop[0]:.1f}%, {100*ci_prop[1]:.1f}%]")

        rows.append(dict(
            Intervention=name, n=n, a=a, b=b, total_c=c, direct_cprime=cp,
            indirect=indirect, indirect_lo=ci_ind[0], indirect_hi=ci_ind[1],
            prop_mediated=prop, prop_lo=ci_prop[0], prop_hi=ci_prop[1],
        ))

    if not rows:
        print("  No mediation models could be estimated.")
        return None

    med = pd.DataFrame(rows)
    med.to_csv(RESULTS / "oral_mediation_results.csv", index=False)

    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.7 * len(med) + 1.5)))
    yb   = np.arange(len(med))
    prop = 100 * med["prop_mediated"].values
    lo   = 100 * med["prop_lo"].values
    hi   = 100 * med["prop_hi"].values
    ns   = [(med["indirect_lo"].iloc[i] < 0 < med["indirect_hi"].iloc[i])
            for i in range(len(med))]
    # See gut mediation figure (Step 11) for why xerr is clipped: the
    # "proportion mediated" ratio can put the point estimate outside its own
    # bootstrap CI when the total effect is near zero.
    err_lo = np.clip(prop - lo, 0, None)
    err_hi = np.clip(hi - prop, 0, None)
    ax.barh(yb, prop,
            xerr=np.vstack([err_lo, err_hi]),
            color=[C["grey"] if n else C["rust"] for n in ns],
            alpha=0.85, error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(med["Intervention"])
    ax.set_xlabel("% of lifestyle -> PhenoAge effect mediated by oral microbiome")
    ax.set_title("Oral microbiome mediation of lifestyle effects")
    ax.text(0.98, 0.04, "grey = n.s. (indirect CI crosses 0)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure12_oral_mediation.png")
    plt.close(fig)
    print("\n  saved figure12_oral_mediation.png")
    print("  wrote oral_mediation_results.csv")
    return med


# =========================================================================== #
# STEP 17  HumanN pathway analysis + gut vs oral concordance figure            #
# Compares lifestyle -> pathway abundance across the top HumanN metabolic      #
# pathways in the oral microbiome, and plots gut vs oral effect concordance.   #
# =========================================================================== #
def step17_pathway_concordance(d: pd.DataFrame,
                               gut_mb_results: dict,
                               oral_mb_results: dict) -> pd.DataFrame | None:
    banner("STEP 17  HumanN pathway analysis + gut vs oral concordance")
    _ensure_econml()
    plt    = _style()
    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    pop    = d[mask].copy()

    # ------------------------------------------------------------------ #
    # Part A  HumanN pathway abundance: top variable pathways             #
    # ------------------------------------------------------------------ #
    pathway_rows = []
    pw_path = DS["oral_pathways"]
    if pw_path.exists():
        print("  loading HumanN pathway abundance...")
        pw = pd.read_feather(pw_path) if str(pw_path).endswith(".arrow") else pd.read_parquet(pw_path)
        print(f"  shape={pw.shape}")

        # pathway columns: string names like "PWY-123: glycolysis"
        pw_cols = [c for c in pw.columns if c != "study_id" and
                   not c.startswith(("UNMAPPED", "UNINTEGRATED"))]
        if pw_cols and "study_id" in pw.columns:
            pw_merged = pop.merge(pw[["study_id"] + pw_cols],
                                  on="study_id", how="inner")
            # select top 20 most variable pathways
            var      = pw_merged[pw_cols].var().nlargest(20)
            top_pw   = var.index.tolist()
            print(f"  {len(top_pw)} most variable pathways selected for causal analysis")

            for pw_name in top_pw:
                for intv_name, tcol in INTERVENTIONS.items():
                    if tcol not in pw_merged.columns:
                        continue
                    sub = pw_merged[[pw_name, tcol] + covars].dropna()
                    if len(sub) < 100:
                        continue
                    try:
                        res = _fit_cf(sub, tcol, pw_name, covars)
                        ns  = res["ci"][0] < 0 < res["ci"][1]
                        pathway_rows.append(dict(
                            pathway=pw_name[:60], intervention=intv_name,
                            n=len(sub), ate=res["ate"],
                            ci_low=res["ci"][0], ci_high=res["ci"][1],
                            std_effect=res["std_effect"], ns=ns))
                    except Exception:
                        pass

        if pathway_rows:
            pw_df = pd.DataFrame(pathway_rows)
            pw_df.to_csv(RESULTS / "oral_pathway_effects.csv", index=False)
            print(f"  wrote oral_pathway_effects.csv "
                  f"({len(pw_df)} intervention-pathway pairs)")

            # Pathway heatmap
            try:
                piv = pw_df.pivot(index="pathway", columns="intervention",
                                  values="std_effect")
                sig = pw_df.pivot(index="pathway", columns="intervention",
                                  values="ns")
                fig, ax = plt.subplots(
                    figsize=(max(8, 1.5 * len(piv.columns)),
                             max(5, 0.4 * len(piv))))
                im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto",
                               vmin=-0.3, vmax=0.3)
                ax.set_xticks(range(len(piv.columns)))
                ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index, fontsize=8)
                for i, pw_n in enumerate(piv.index):
                    for j, iv_n in enumerate(piv.columns):
                        if not sig.loc[pw_n, iv_n]:
                            ax.text(j, i, "*", ha="center", va="center",
                                    fontsize=10, color="black")
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
                    "Standardised effect (per +1 SD)", fontsize=9)
                ax.set_title("Lifestyle causal effects on oral HumanN metabolic pathways\n"
                             "* = 95% CI excludes 0", fontsize=12, fontweight="bold")
                fig.tight_layout()
                fig.savefig(RESULTS / "figure13_oral_pathway_heatmap.png")
                plt.close(fig)
                print("  saved figure13_oral_pathway_heatmap.png")
            except Exception as exc:
                print(f"  Pathway heatmap could not be generated: {exc}")
    else:
        print(f"  HumanN pathway parquet not found at expected path -- skipping")

    # ------------------------------------------------------------------ #
    # Part B  Gut vs oral concordance (Figure 14)                         #
    # Interventions with consistent direction in both compartments are     #
    # the strongest candidates for host-wide lifestyle-microbiome effects. #
    # ------------------------------------------------------------------ #
    if gut_mb_results and oral_mb_results:
        shared = [n for n in gut_mb_results if n in oral_mb_results]
        if shared:
            gut_eff  = [gut_mb_results[n]["std_effect"]  for n in shared]
            oral_eff = [oral_mb_results[n]["std_effect"] for n in shared]
            gut_ns   = [gut_mb_results[n]["ci"][0]  < 0 < gut_mb_results[n]["ci"][1]
                        for n in shared]
            oral_ns  = [oral_mb_results[n]["ci"][0] < 0 < oral_mb_results[n]["ci"][1]
                        for n in shared]
            concordant = [g * o > 0 for g, o in zip(gut_eff, oral_eff)]

            fig, ax = plt.subplots(figsize=(6.5, 6.0))
            for i, name in enumerate(shared):
                color  = C["teal"]  if concordant[i] else C["rust"]
                marker = "o" if not (gut_ns[i] or oral_ns[i]) else "^"
                ax.scatter(gut_eff[i], oral_eff[i], s=90,
                           color=color, marker=marker, zorder=3,
                           alpha=0.55 if (gut_ns[i] or oral_ns[i]) else 0.9)
                ax.annotate(name, (gut_eff[i], oral_eff[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=C["ink"])

            lim = max(abs(v) for v in gut_eff + oral_eff) * 1.3 or 0.3
            ax.axhline(0, color=C["ink"], lw=0.7, ls="--")
            ax.axvline(0, color=C["ink"], lw=0.7, ls="--")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Standardised effect on gut Shannon diversity")
            ax.set_ylabel("Standardised effect on oral Shannon diversity")
            ax.set_title("Gut vs oral microbiome concordance\n"
                         "interventions in the same quadrant act on both compartments",
                         fontsize=11, fontweight="bold")
            n_conc = sum(concordant)
            ax.text(0.03, 0.97,
                    f"{n_conc}/{len(shared)} interventions concordant\n"
                    f"teal = concordant   rust = discordant\n"
                    f"circle = both sig.   triangle = one or both n.s.",
                    transform=ax.transAxes, va="top", fontsize=8, color=C["ink"])
            fig.tight_layout()
            fig.savefig(RESULTS / "figure14_gut_oral_concordance.png")
            plt.close(fig)
            print("  saved figure14_gut_oral_concordance.png")

            conc_df = pd.DataFrame(dict(
                intervention=shared, gut_std_effect=gut_eff,
                oral_std_effect=oral_eff, concordant=concordant,
                gut_ns=gut_ns, oral_ns=oral_ns))
            conc_df.to_csv(RESULTS / "gut_oral_concordance.csv", index=False)
            print("  wrote gut_oral_concordance.csv")
            return conc_df

    print("  Insufficient results for concordance plot.")
    return None


# =========================================================================== #
# MAIN                                                                         #
# =========================================================================== #
def main():
    print("HPP Counterfactual Causal-Inference Pipeline  |  GenMI lab, MBZUAI")
    print(f"Data   : {HPP}")
    print(f"Results: {RESULTS}\n")

    pop     = step1_population()
    d       = step2_load_merge(pop)
    d       = step3_phenoage(d)
    study   = step4_population(d)
    results = step5_causal(study)
    ranked  = step6_rank(results)
    step7_figures(d, study, results, ranked)

    combined_score_res = step6b_combined_score(study)

    covars_for_het = [c for c in COVARIATES if c in study.columns]
    heterogeneity  = step6c_heterogeneity(results, covars_for_het)

    # Full-cohort sensitivity analysis: the 40-50 age window and the
    # AgeAccel>0 ("accelerated agers only") filter each cost real sample size
    # (n=1,359 vs. n=5,687 with valid PhenoAge). The age window reflects a
    # specific scientific hypothesis (Li et al.'s proposed critical
    # intervention window) and dropping it outright would risk diluting a
    # real, age-concentrated effect into a null average. Rather than assume
    # either way, this reruns the same causal analysis on the unrestricted
    # cohort (age still included as a covariate) and explicitly tests
    # age x treatment interaction using the same validated interaction-term
    # method from Step 6c -- this directly tests whether the 40-50 window
    # matters, rather than assuming it does or discarding it blindly for
    # more power.
    banner("STEP 5c  Full-cohort sensitivity analysis (no age or AgeAccel>0 restriction)")
    pop_full = d[d["AgeAccel"].notna()].copy()
    print(f"  full cohort with valid AgeAccel, no age/accelerated-ager filter: "
          f"n={len(pop_full):,}  (vs. study population n={len(study):,})")
    results_full = step5_causal(pop_full, outcome="AgeAccel", interventions=INTERVENTIONS)
    ranked_full  = step6_rank(results_full)
    ranked_full.to_csv(RESULTS / "intervention_ranking_full_cohort.csv", index=False)
    print("  wrote intervention_ranking_full_cohort.csv")

    covars_full = [c for c in COVARIATES if c in pop_full.columns]
    step6c_heterogeneity(results_full, covars_full,
                        csv_name="heterogeneity_results_full_cohort.csv")
    print("  (heterogeneity_results_full_cohort.csv includes the age x treatment "
          "interaction test -- this is the direct test of whether the 40-50 "
          "window hypothesis holds)")

    # Sensitivity check: does the ranking survive outlier control? Winsorizes the
    # outcome (AgeAccel) and each intervention at the 1st/99th percentile before
    # re-fitting. This is a robustness check, not a replacement for the primary
    # (unwinsorized) analysis above -- both are reported so outlier sensitivity
    # is transparent rather than silently resolved one way or the other.
    banner("STEP 5b  Sensitivity check: winsorized at 1%/99%")
    results_wz = step5_causal(study, winsorize_pct=0.01)
    ranked_wz  = step6_rank(results_wz)
    ranked_wz.to_csv(RESULTS / "intervention_ranking_winsorized.csv", index=False)
    print("  wrote intervention_ranking_winsorized.csv")

    # Gut microbiome track
    d, gut_gcols, gut_scols   = step8_load_microbiome(d)
    d, gut_clock_r, gut_mae   = step9_microbiome_clock(d, gut_gcols, gut_scols)
    microbiome_outcome_res    = step9b_microbiome_outcome_causal(d)
    gut_mb_res, _             = step10_causal_microbiome(d)
    gut_med                   = step11_mediation(d)
    gut_species_med           = _species_level_mediation(d, gut_scols or gut_gcols, "gut")

    # Oral microbiome track
    d, oral_gcols             = step13_load_oral_microbiome(d)
    d, oral_clock_r, oral_mae = step14_oral_clock(d, oral_gcols)
    oral_mb_res, _            = step15_causal_oral_microbiome(d)
    oral_med                  = step16_oral_mediation(d)
    oral_species_med          = _species_level_mediation(d, oral_gcols, "oral")

    # Cross-compartment analysis + HumanN pathways
    concordance = step17_pathway_concordance(d, gut_mb_res, oral_mb_res)

    # Curated phenotype sweep
    sweep = step12_phenotype_sweep(d)

    banner("ANALYSIS COMPLETE")
    print(f"  Study population (accelerated agers 40-50) : n={len(study):,}")
    print(f"  Gut  microbiome clock : r={gut_clock_r:.2f}, MAE={gut_mae:.1f} yr")
    print(f"  Oral microbiome clock : r={oral_clock_r:.2f}, MAE={oral_mae:.1f} yr")
    if ranked is not None and len(ranked):
        top = ranked.iloc[0]
        print(f"  Top intervention      : {top['Intervention']} "
              f"(std. effect {top['Std_effect']:+.3f} yr per +1 SD)")
    if concordance is not None:
        n_conc = concordance["concordant"].sum()
        print(f"  Gut-oral concordance  : {n_conc}/{len(concordance)} "
              f"interventions act consistently on both microbiomes")
    if sweep is not None:
        sig = (~sweep["ns"]).sum()
        print(f"  Phenotype sweep       : {sig}/{len(sweep)} pairs reach significance")
    print(f"\n  All outputs written to: {RESULTS}/")
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
