#!/usr/bin/env python3
"""
audit_data_and_numbers.py
===============================================================================
Independent verification of the HPP microbiome-aging pipeline.

DESIGN PRINCIPLE: this script deliberately does NOT import cross_lagged_panel
or reuse its loaders. It re-reads the raw parquet files and re-derives every
headline number from scratch. If the pipeline has a bug, reusing its functions
would reproduce the bug and hide it. Agreement between two independently
written implementations is real evidence; agreement with itself is not.

Sections:
  1. Data inventory        -- every parquet: shape, index, dtypes, key columns
  2. Visit structure       -- duplicates, visit_n consistency, date vs stage order
  3. Independent recompute -- ICC, intervals, diversity, age, Spearman-Brown
  4. Cross-check           -- compare against the saved result CSVs
  5. Red flags             -- consolidated list of anything suspicious

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python audit_data_and_numbers.py > audit_report.txt 2>&1
    tail -60 audit_report.txt

Writes nothing except stdout. Safe to run at any time.
===============================================================================
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))
RESULTS = Path(os.environ.get("HPP_RESULTS_DIR", "results_cross_lagged"))
RESULTS_LONG = Path("results_longitudinal")

FLAGS: list[str] = []


def flag(msg: str) -> None:
    FLAGS.append(msg)
    print(f"  [FLAG] {msg}")


def section(t: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def sub(t: str) -> None:
    print(f"\n--- {t} ---")


def flat(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.names != [None]:
        df = df.reset_index()
    return df


# =========================================================================== #
section("1. DATA INVENTORY")
# =========================================================================== #
TABLES = {
    "gut_microbiome":   DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet",
    "population":       DATA_ROOT / "population" / "population.parquet",
    "hand_grip":        DATA_ROOT / "hand_grip" / "hand_grip.parquet",
    "blood_pressure":   DATA_ROOT / "blood_pressure" / "blood_pressure.parquet",
    "anthropometrics":  DATA_ROOT / "anthropometrics" / "anthropometrics.parquet",
}
for f in sorted((DATA_ROOT / "sociodemographics").glob("*.parquet")):
    TABLES[f"sociodem/{f.name}"] = f

loaded: dict[str, pd.DataFrame] = {}
for name, path in TABLES.items():
    sub(name)
    if not path.exists():
        flag(f"{name}: FILE MISSING at {path}")
        continue
    try:
        raw = pd.read_parquet(path)
        idx_names = list(raw.index.names)
        df = flat(path)
        loaded[name] = df
        print(f"  path        : {path}")
        print(f"  index levels: {idx_names}")
        print(f"  shape       : {df.shape}")
        dup = list(df.columns[df.columns.duplicated()])
        if dup:
            flag(f"{name}: DUPLICATE COLUMN NAMES after reset_index: {dup}")
        pid = next((c for c in df.columns if c in ("participant_id", "study_id")), None)
        stage = next((c for c in df.columns if "research_stage" in c.lower()), None)
        print(f"  id column   : {pid}")
        print(f"  stage column: {stage}")
        if pid:
            print(f"  n unique ids: {df[pid].nunique():,}")
        show = [c for c in df.columns if not c.endswith("_parquet")][:18]
        print(f"  columns[:18]: {show}")
        print("  dtypes of key cols:")
        for c in [x for x in (pid, stage, "collection_date", "collection_timestamp") if x and x in df.columns]:
            print(f"      {c:<26} {str(df[c].dtype)}")
    except Exception as exc:
        flag(f"{name}: FAILED TO LOAD -- {type(exc).__name__}: {exc}")


# =========================================================================== #
section("2. VISIT STRUCTURE AND ORDERING")
# =========================================================================== #
gut = loaded.get("gut_microbiome")
if gut is None:
    print("  gut_microbiome unavailable; skipping section")
else:
    gpid = next(c for c in gut.columns if c in ("participant_id", "study_id"))
    gstage = next(c for c in gut.columns if "research_stage" in c.lower())
    gdate = next((c for c in gut.columns
                  if c in ("collection_date", "collection_timestamp")), None)
    print(f"  id={gpid}  stage={gstage}  date={gdate}")

    g = gut.rename(columns={gpid: "study_id", gstage: "visit"}).copy()
    print(f"\n  distinct research_stage values: {sorted(g['visit'].dropna().unique().tolist())[:10]}")

    nvis = g.groupby("study_id")["visit"].nunique()
    repeat_ids = set(nvis[nvis >= 2].index)
    print(f"  participants total          : {g['study_id'].nunique():,}")
    print(f"  participants with >=2 visits: {len(repeat_ids):,}")

    gr = g[g["study_id"].isin(repeat_ids)].copy()

    # --- visit_n by DATE (what the pipeline should use) ---
    if gdate:
        gr[gdate] = pd.to_datetime(gr[gdate], errors="coerce")
        n_bad_date = gr[gdate].isna().sum()
        if n_bad_date:
            flag(f"{n_bad_date} gut rows have unparseable {gdate}")
        gr = gr.sort_values(["study_id", gdate])
        gr["visit_n_date"] = gr.groupby("study_id").cumcount() + 1
    # --- visit_n by STAGE RANK (the fallback / older method) ---
    gr["visit_n_stage"] = gr.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    if gdate:
        mism = (gr["visit_n_date"] != gr["visit_n_stage"]).sum()
        pct = 100 * mism / len(gr)
        print(f"\n  visit_n date-order vs stage-rank disagreement: {mism:,} rows ({pct:.2f}%)")
        if pct > 1:
            flag(f"visit_n ordering methods disagree on {pct:.1f}% of rows -- "
                 f"t1/t2 may be swapped for some participants")
        else:
            print("  -> orderings agree; stage labels sort chronologically here")

    dupes = gr.groupby(["study_id", "visit_n_date" if gdate else "visit_n_stage"]).size()
    n_dupe = (dupes > 1).sum()
    print(f"  (study_id, visit_n) pairs with >1 row: {n_dupe:,}")
    if n_dupe:
        print(f"      -> pipeline averages these; max rows in one cell = {dupes.max()}")

    # --- visit interval ---
    if gdate:
        vn = "visit_n_date"
        t1 = gr[gr[vn] == 1].groupby("study_id")[gdate].min()
        t2 = gr[gr[vn] == 2].groupby("study_id")[gdate].min()
        iv = ((t2 - t1).dt.days / 365.25).dropna()
        print(f"\n  INTERVAL  n={len(iv):,}  median={iv.median():.4f} yr  "
              f"IQR=[{iv.quantile(.25):.4f}, {iv.quantile(.75):.4f}]  "
              f"min={iv.min():.3f}  max={iv.max():.3f}")
        print(f"    paper claims: median 2.11, IQR [2.01, 2.27]")
        if abs(iv.median() - 2.11) > 0.02:
            flag(f"interval median {iv.median():.3f} != paper's 2.11")


# =========================================================================== #
section("3. INDEPENDENT RECOMPUTATION")
# =========================================================================== #

def icc31(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(3,1) two-way mixed, consistency. Written fresh, not copied."""
    x = np.column_stack([a, b]).astype(float)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ss_t = ((x - gm) ** 2).sum()
    ms_e = (ss_t - (n - 1) * ms_r - (k - 1) * ms_c) / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


sub("3a. Species table and diversity")
sp_df = None
if gut is not None:
    ptr = next((c for c in gut.columns
                if "species" in c.lower() and "parquet" in c.lower()), None)
    print(f"  species pointer column: {ptr}")
    if ptr:
        paths = gut[ptr].dropna().unique()
        print(f"  distinct pointer values: {len(paths)}")
        cand = DATA_ROOT / "gut_microbiome" / str(paths[0]).lstrip("./")
        print(f"  resolved path: {cand}")
        print(f"  exists: {cand.exists()}")
        if cand.exists():
            sp_df = flat(cand)
            print(f"  species table shape: {sp_df.shape}")
            spid = next((c for c in sp_df.columns
                         if c in ("participant_id", "study_id")), None)
            sstage = next((c for c in sp_df.columns
                           if "research_stage" in c.lower()), None)
            print(f"  species id={spid} stage={sstage}")
            meta = {"participant_id", "study_id", "cohort", "research_stage",
                    "array_index", "sample_name", "collection_timestamp",
                    "collection_date", "timezone"}
            acols = [c for c in sp_df.columns
                     if c not in meta and str(sp_df[c].dtype).startswith("float")]
            print(f"  float abundance columns: {len(acols)}")
            A = sp_df[acols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
            rs = A.sum(axis=1)
            print(f"  row sums: min={rs.min():.4f} median={rs.median():.4f} max={rs.max():.4f}")
            if not (0.9 < rs.median() < 1.1) and not (95 < rs.median() < 105):
                flag(f"abundance row sums median={rs.median():.3f} -- neither ~1 "
                     f"(proportions) nor ~100 (percent); check normalisation")
            prev = (A > 0).mean(axis=0)
            print(f"  species with prevalence >5%: {(prev > 0.05).sum()}  "
                  f"(paper claims 511)")
            if (prev > 0.05).sum() != 511:
                flag(f"prevalence>5% count = {(prev>0.05).sum()}, paper says 511")

            # Shannon + richness, computed fresh
            P = A.div(rs.replace(0, np.nan), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.where(P > 0, np.log(P), 0.0)
            sh = -(P.values * lg).sum(axis=1)
            ri = (A > 0).sum(axis=1).astype(float)
            div = pd.DataFrame({"study_id": sp_df[spid], "visit": sp_df[sstage],
                                "shannon": sh, "richness": ri})
            print(f"  Shannon  mean={np.nanmean(sh):.4f} sd={np.nanstd(sh):.4f}")
            print(f"  Richness mean={np.nanmean(ri):.4f} sd={np.nanstd(ri):.4f}")
            print(f"    paper Table 1: Shannon v1 3.570+/-0.586, richness v1 179.0+/-42.6")

sub("3b. Diversity ICC (independent)")
if sp_df is not None and gut is not None and gdate:
    vis = gr[["study_id", "visit", "visit_n_date"]].drop_duplicates()
    d2 = div.merge(vis, on=["study_id", "visit"], how="inner")
    d2 = d2.groupby(["study_id", "visit_n_date"], as_index=False)[["shannon", "richness"]].mean()
    for metric, claimed in (("shannon", 0.626), ("richness", 0.770)):
        w1 = d2[d2["visit_n_date"] == 1].set_index("study_id")[metric]
        w2 = d2[d2["visit_n_date"] == 2].set_index("study_id")[metric]
        common = w1.index.intersection(w2.index)
        v = icc31(w1.loc[common].values, w2.loc[common].values)
        r = float(np.corrcoef(w1.loc[common], w2.loc[common])[0, 1])
        print(f"  {metric:<9} ICC(3,1)={v:.4f}  Pearson r={r:.4f}  n={len(common):,}"
              f"   [paper: {claimed}]")
        if abs(v - claimed) > 0.02:
            flag(f"{metric} ICC {v:.4f} differs from paper's {claimed}")

sub("3c. Age derivation -- fixed 2020 vs actual visit date")
socio = [v for k, v in loaded.items() if k.startswith("sociodem/") and "DOB" in v.columns]
if socio:
    sd = pd.concat(socio, ignore_index=True)
    spid2 = next((c for c in sd.columns if c in ("participant_id", "study_id")), None)
    sd = sd[[spid2, "DOB"]].rename(columns={spid2: "study_id"}).drop_duplicates("study_id")
    sd["DOB"] = pd.to_numeric(sd["DOB"], errors="coerce")
    sd["age_2020"] = 2020 - sd["DOB"]
    print(f"  DOB non-null: {sd['DOB'].notna().sum():,}")
    print(f"  DOB range   : {sd['DOB'].min():.0f} .. {sd['DOB'].max():.0f}")
    print(f"  age (2020 - DOB): mean={sd['age_2020'].mean():.2f} "
          f"sd={sd['age_2020'].std():.2f}   [paper: 50.4 +/- 7.5]")
    if gut is not None and gdate:
        t1d = gr[gr["visit_n_date"] == 1].groupby("study_id")[gdate].min()
        chk = sd.set_index("study_id").join(t1d.rename("d1"), how="inner")
        chk["age_actual"] = chk["d1"].dt.year - chk["DOB"]
        diff = (chk["age_actual"] - chk["age_2020"]).dropna()
        print(f"\n  age using ACTUAL visit-1 year: mean={chk['age_actual'].mean():.2f}")
        print(f"  difference (actual - fixed2020): mean={diff.mean():+.2f} yr  "
              f"sd={diff.std():.2f}  range=[{diff.min():+.0f}, {diff.max():+.0f}]")
        if abs(diff.mean()) > 0.5:
            flag(f"age is systematically off by {diff.mean():+.2f} yr because the "
                 f"pipeline uses a FIXED year 2020 while visit-1 dates span "
                 f"{t1d.dt.year.min()}-{t1d.dt.year.max()}")
else:
    flag("no sociodemographics file with a DOB column found")

sub("3d. Outcome column selection -- what is actually being modelled")
for nm in ("hand_grip", "blood_pressure", "anthropometrics"):
    df = loaded.get(nm)
    if df is None:
        continue
    num = [c for c in df.columns
           if str(df[c].dtype).startswith(("float", "int"))
           and c not in ("array_index",)]
    print(f"\n  {nm}: numeric columns and non-null counts")
    for c in num[:14]:
        print(f"      {c:<44} {df[c].notna().sum():>7,}")
    if nm == "blood_pressure":
        sys_cols = [c for c in df.columns if "systolic" in c.lower()]
        dia_cols = [c for c in df.columns if "diastolic" in c.lower()]
        print(f"    systolic columns present : {sys_cols}")
        print(f"    diastolic columns present: {dia_cols}")
        if not sys_cols and dia_cols:
            flag("blood_pressure has NO systolic column -- pipeline falls back to "
                 "DIASTOLIC, which is non-monotonic with age and a weaker aging "
                 "marker. The paper reports 'lying_blood_pressure_diastolic'; "
                 "confirm this is intended.")

sub("3e. Spearman-Brown arithmetic")
for name, rho in (("AgeAccel", 0.483), ("Shannon", 0.626), ("Richness", 0.770)):
    r2 = 2 * rho / (1 + rho)
    ratio = np.sqrt(r2 / rho)
    print(f"  {name:<9} ICC1={rho:.3f}  ICC2={r2:.4f}  amplification={ratio:.4f}")
print(f"    paper claims for AgeAccel: ICC2=0.651, predicted ratio 1.161, observed 1.167")
r2_chk = 2 * 0.483 / 1.483
print(f"    recomputed: ICC2={r2_chk:.4f}, ratio={np.sqrt(r2_chk/0.483):.4f}")
print("\n  samples needed for target reliability, k = R(1-rho) / (rho(1-R)):")
for name, rho in (("Richness", 0.770), ("Shannon", 0.626), ("AgeAccel", 0.483)):
    ks = {R: R * (1 - rho) / (rho * (1 - R)) for R in (0.7, 0.8, 0.9)}
    print(f"    {name:<9} " + "  ".join(
        f"R={R}: k={v:.2f} (use {int(np.ceil(v))})" for R, v in ks.items()))


# =========================================================================== #
section("4. CROSS-CHECK AGAINST SAVED RESULT CSVs")
# =========================================================================== #
CHECKS = [
    (RESULTS / "cross_lag_all_paths.csv", "cross-lag main"),
    (RESULTS / "table2_crosslag.csv", "cross-lag publication table"),
    (RESULTS / "diversity_icc.csv", "diversity ICC"),
    (RESULTS / "visit_intervals.csv", "visit intervals"),
    (RESULTS / "incident_disease.csv", "incident disease"),
    (RESULTS / "path_b_asymmetry.csv", "path B asymmetry"),
    (RESULTS_LONG / "stability_summary.csv", "stability summary"),
    (RESULTS_LONG / "species_stability.csv", "species stability atlas"),
    (RESULTS_LONG / "trait_state_disease.csv", "trait/state disease"),
]
for path, label in CHECKS:
    if not path.exists():
        print(f"  MISSING: {label:<30} {path}")
        continue
    try:
        df = pd.read_csv(path)
        print(f"  OK     : {label:<30} shape={str(df.shape):<12} {path}")
        if "cross_lag_all_paths" in path.name:
            print(f"             rows={len(df)} (expect 18), "
                  f"min p={df['p'].min():.4f}, "
                  f"n significant q<0.05={int((df['q_fdr']<0.05).sum())}")
            if len(df) != 18:
                flag(f"cross_lag_all_paths has {len(df)} rows, expected 18")
        if "species_stability" in path.name:
            sc = next((c for c in df.columns if "stab" in c.lower()), None)
            if sc:
                print(f"             {sc}: n={df[sc].notna().sum()}, "
                      f"mean={df[sc].mean():.4f}, "
                      f"range=[{df[sc].min():.3f}, {df[sc].max():.3f}]")
        if "stability_summary" in path.name:
            for c in ("icc_age_accel", "icc_shannon", "icc_richness",
                      "median_interval_yr", "n_pairs"):
                if c in df.columns:
                    print(f"             {c:<22} = {df[c].iloc[0]}")
    except Exception as exc:
        flag(f"{label}: unreadable -- {exc}")


# =========================================================================== #
section("5. RED FLAGS")
# =========================================================================== #
if not FLAGS:
    print("\n  None. Every structural and numerical check passed.\n")
else:
    print(f"\n  {len(FLAGS)} item(s) need attention:\n")
    for i, f_ in enumerate(FLAGS, 1):
        print(f"   {i:>2}. {f_}")
print()
