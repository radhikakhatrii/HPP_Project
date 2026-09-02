#!/usr/bin/env python3
"""
Sanity check: inspect all key variables before any modelling.
Run this BEFORE cross_lagged_panel.py to verify joins will work.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. Population ────────────────────────────────────────────
sep("1. POPULATION")
pop = pd.read_parquet(DATA_ROOT / "population" / "population.parquet")
if pop.index.names != [None]: pop = pop.reset_index()
if "participant_id" in pop.columns and "study_id" not in pop.columns:
    pop = pop.rename(columns={"participant_id": "study_id"})
print(f"  Shape: {pop.shape}")
print(f"  study_id dtype: {pop['study_id'].dtype}")
print(f"  study_id sample: {pop['study_id'].head(5).tolist()}")
age_col = next((c for c in pop.columns if "age" in c.lower()), None)
print(f"  age column: {age_col}, non-null: {pop[age_col].notna().sum() if age_col else 'N/A'}")
sex_col = next((c for c in pop.columns if "sex" in c.lower()), None)
print(f"  sex column: {sex_col}, values: {pop[sex_col].value_counts().to_dict() if sex_col else 'N/A'}")

# ── 2. Gut microbiome pointer table ──────────────────────────
sep("2. GUT MICROBIOME POINTER TABLE")
gut = pd.read_parquet(DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
if gut.index.names != [None]: gut = gut.reset_index()
pid = next(c for c in gut.columns if c in ("participant_id","study_id"))
stage = next(c for c in gut.columns if "research_stage" in c.lower())
gut = gut.rename(columns={pid: "study_id", stage: "visit"})
print(f"  Shape: {gut.shape}")
print(f"  study_id dtype: {gut['study_id'].dtype}")
print(f"  study_id sample: {gut['study_id'].head(5).tolist()}")
print(f"  visit (research_stage) sample: {gut['visit'].head(5).tolist()}")
print(f"  visit dtype: {gut['visit'].dtype}")
counts = gut.groupby("study_id")["visit"].nunique()
repeat_ids = counts[counts >= 2].index
print(f"  Unique participants: {gut['study_id'].nunique():,}")
print(f"  Repeat participants (≥2 visits): {len(repeat_ids):,}")

# ── 3. Species parquet ───────────────────────────────────────
sep("3. SPECIES PARQUET")
sp_ptr = next((c for c in gut.columns if "species" in c.lower() and "parquet" in c.lower()), None)
if sp_ptr:
    sp_path_val = gut[sp_ptr].dropna().iloc[0]
    gut_dir = DATA_ROOT / "gut_microbiome"
    sp_path = gut_dir / str(sp_path_val).lstrip("./")
    if sp_path.exists():
        sp = pd.read_parquet(sp_path)
        if sp.index.names != [None]: sp = sp.reset_index()
        print(f"  Path: {sp_path}")
        print(f"  Shape: {sp.shape}")
        sp_pid = next((c for c in sp.columns if c in ("participant_id","study_id")), None)
        sp_stage = next((c for c in sp.columns if "research_stage" in c.lower()), None)
        print(f"  Join cols: pid='{sp_pid}', stage='{sp_stage}'")
        if sp_pid:
            print(f"  {sp_pid} dtype: {sp[sp_pid].dtype}, sample: {sp[sp_pid].head(5).tolist()}")
        if sp_stage:
            print(f"  {sp_stage} dtype: {sp[sp_stage].dtype}, sample: {sp[sp_stage].head(5).tolist()}")
        # Test join
        if sp_pid and sp_stage:
            gut_sub = gut[gut["study_id"].isin(repeat_ids)][["study_id","visit"]].copy()
            sp_sub  = sp[[sp_pid, sp_stage]].rename(columns={sp_pid:"study_id", sp_stage:"visit"})
            merged = gut_sub.merge(sp_sub, on=["study_id","visit"], how="inner")
            print(f"\n  JOIN TEST (gut repeat × species on study_id+visit):")
            print(f"    gut_sub rows: {len(gut_sub)}, sp_sub rows: {len(sp_sub)}")
            print(f"    matched rows: {len(merged)}")
            if len(merged) == 0:
                print("  *** JOIN FAILED — checking key values:")
                print(f"    gut study_id[:5]: {gut_sub['study_id'].head(5).tolist()}")
                print(f"    gut visit[:5]:    {gut_sub['visit'].head(5).tolist()}")
                print(f"    sp  study_id[:5]: {sp_sub['study_id'].head(5).tolist()}")
                print(f"    sp  visit[:5]:    {sp_sub['visit'].head(5).tolist()}")

# ── 4. Functional outcomes ───────────────────────────────────
sep("4. FUNCTIONAL OUTCOMES")
for name, fname, candidates in [
    ("hand_grip",      "hand_grip/hand_grip.parquet",
     ["grip_strength","hand_grip_strength","finger_grip_strength_left","dominant_hand_grip"]),
    ("blood_pressure", "blood_pressure/blood_pressure.parquet",
     ["systolic","systolic_bp","sbp","lying_blood_pressure_diastolic"]),
    ("anthropometrics","anthropometrics/anthropometrics.parquet",
     ["bmi","body_mass_index","waist_circumference"]),
]:
    path = DATA_ROOT / fname
    if not path.exists():
        print(f"  {name}: NOT FOUND"); continue
    df = pd.read_parquet(path)
    if df.index.names != [None]: df = df.reset_index()
    pid2   = next((c for c in df.columns if c in ("participant_id","study_id")), None)
    stage2 = next((c for c in df.columns if "research_stage" in c.lower()), None)
    if not pid2 or not stage2:
        print(f"  {name}: missing pid/stage — cols: {list(df.columns[:10])}"); continue
    df = df.rename(columns={pid2: "study_id", stage2: "visit"})
    meas = next((c for c in candidates if c in df.columns), None)
    if meas is None:
        excl = {"study_id","visit","array_index","index","participant_id","cohort",
                "collection_timestamp","collection_date","timezone"}
        meas = next((c for c in df.select_dtypes(include=np.number).columns
                     if c.lower() not in excl), None)
    counts2 = df.groupby("study_id")["visit"].nunique()
    repeat2 = counts2[counts2 >= 2].index
    print(f"\n  {name}:")
    print(f"    Shape: {df.shape}, meas col: '{meas}'")
    print(f"    study_id dtype: {df['study_id'].dtype}, sample: {df['study_id'].head(3).tolist()}")
    print(f"    visit dtype: {df['visit'].dtype}, sample: {df['visit'].head(3).tolist()}")
    print(f"    Repeat participants: {len(repeat2):,}")
    # Overlap with gut repeat
    overlap = set(repeat_ids) & set(repeat2)
    print(f"    Overlap with gut repeat: {len(overlap):,}")
    if meas and len(df[meas].dropna()) > 0:
        print(f"    {meas} stats: mean={df[meas].mean():.2f}, non-null={df[meas].notna().sum():,}")

# ── 5. Covariates availability ───────────────────────────────
sep("5. COVARIATE AVAILABILITY IN POPULATION")
for cov in ["age_at_visit","sex_encoded","education_level","household_income"]:
    col = age_col if cov == "age_at_visit" else (sex_col if cov == "sex_encoded" else
          next((c for c in pop.columns if cov in c.lower()), None))
    if col:
        nn = pop[col].notna().sum()
        print(f"  {cov} ('{col}'): {nn:,}/{len(pop):,} non-null")
    else:
        print(f"  {cov}: NOT FOUND in population")

print("\n[Sanity check complete]")
