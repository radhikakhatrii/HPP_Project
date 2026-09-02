#!/usr/bin/env python3
"""
audit_blood_meds.py
===============================================================================
Targeted follow-up to audit_data_inventory.py. Two open questions from that
scan:

  1. blood_tests.parquet (69 cols) -- does it carry kidney function markers
     (creatinine, eGFR) or inflammation markers (CRP, IL-6)? The keyword scan
     found platelets/lymphocytes but nothing renal or inflammatory yet.

  2. medications.parquet -- is `medication` a coded/categorical field we can
     filter for antibiotics, metformin, PPIs, or is it free text that would
     need a separate mapping step first?

Full column list plus a value sample for each. Small, fast, no schema-only
shortcuts needed -- both tables are well under a megabyte.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python audit_blood_meds.py 2>&1 | tee audit_blood_meds.log
===============================================================================
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))

RENAL_KW = ["creatinine", "egfr", "gfr", "urea", "bun", "cystatin", "urine",
            "albumin_urine", "acr", "uacr"]
INFLAM_KW = ["crp", "c_reactive", "il6", "il_6", "il-6", "esr", "sed_rate",
            "ferritin", "fibrinogen", "wbc", "white_blood", "neutrophil"]
MED_CLASS_KW = ["antibiotic", "metformin", "ppi", "proton_pump", "omeprazole",
               "statin", "aspirin", "nsaid", "antidepressant", "insulin"]


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main():
    section("1. blood_tests.parquet")
    path = DATA_ROOT / "blood_tests" / "blood_tests.parquet"
    if not path.exists():
        log(f"  NOT FOUND at {path}")
    else:
        df = pd.read_parquet(path)
        if df.index.names != [None]:
            df = df.reset_index()
        log(f"  shape: {df.shape}")
        log(f"\n  ALL {len(df.columns)} columns:")
        for c in sorted(df.columns):
            log(f"    {c}")

        log("\n  --- renal marker search ---")
        renal_hits = [c for c in df.columns
                     if any(k in c.lower() for k in RENAL_KW)]
        if renal_hits:
            for c in renal_hits:
                s = df[c]
                log(f"    FOUND: {c}  (dtype={s.dtype}, non-null={s.notna().sum():,})")
                if pd.api.types.is_numeric_dtype(s):
                    log(f"      range: [{s.min():.3g}, {s.max():.3g}]  "
                        f"mean={s.mean():.3g}")
        else:
            log("    NONE FOUND. No creatinine/eGFR-type column in this table.")

        log("\n  --- inflammation marker search ---")
        inflam_hits = [c for c in df.columns
                      if any(k in c.lower() for k in INFLAM_KW)]
        if inflam_hits:
            for c in inflam_hits:
                s = df[c]
                log(f"    FOUND: {c}  (dtype={s.dtype}, non-null={s.notna().sum():,})")
                if pd.api.types.is_numeric_dtype(s):
                    log(f"      range: [{s.min():.3g}, {s.max():.3g}]  "
                        f"mean={s.mean():.3g}")
        else:
            log("    NONE FOUND. No CRP/IL-6-type column in this table.")

    section("2. medications.parquet")
    path = DATA_ROOT / "medications" / "medications.parquet"
    if not path.exists():
        log(f"  NOT FOUND at {path}")
    else:
        df = pd.read_parquet(path)
        if df.index.names != [None]:
            df = df.reset_index()
        log(f"  shape: {df.shape}")
        log(f"  columns: {list(df.columns)}")

        log("\n  --- ATC classification codes, checked first (this is the answer to")
        log("  --- 'can we classify by drug class': printed before anything else can crash ---")
        for c in ("atc3", "atc4", "atc5", "api"):
            if c not in df.columns:
                log(f"    '{c}': NOT IN COLUMNS")
                continue
            s = df[c]
            log(f"    '{c}': dtype={s.dtype}, non-null={s.notna().sum():,}")
            try:
                nun = s.nunique()
                log(f"      {nun:,} unique values")
                if nun <= 80:
                    for v in sorted(s.dropna().unique().astype(str))[:80]:
                        log(f"        {v}")
                else:
                    for v in s.dropna().unique()[:15]:
                        log(f"        {v}")
            except TypeError:
                log(f"      unhashable/array-valued, sample: {s.dropna().head(5).tolist()}")

        # A column can be dtype=object because it holds LISTS/ARRAYS per cell
        # (e.g. one participant on several supplements stored as one array),
        # not because it holds plain strings. nunique() calls hash() on every
        # value and crashes on an unhashable numpy.ndarray. We detect that
        # case from a sample value first and handle it separately instead of
        # letting one such column kill the whole scan.
        array_valued_cols = []
        scalar_text_cols = []
        for c in df.columns:
            if str(df[c].dtype) not in ("object", "category"):
                continue
            sample = df[c].dropna()
            is_array_valued = False
            if len(sample):
                first = sample.iloc[0]
                is_array_valued = isinstance(first, (list, tuple)) or \
                    (hasattr(first, "__len__") and not isinstance(first, str))
            if is_array_valued:
                array_valued_cols.append(c)
            else:
                scalar_text_cols.append(c)

        for c in array_valued_cols:
            s = df[c].dropna()
            lens = s.apply(lambda x: len(x) if hasattr(x, "__len__") else 1)
            log(f"\n  '{c}': dtype={df[c].dtype}, ARRAY-VALUED per cell "
                f"(not a simple category), non-null={s.notna().sum():,}")
            log(f"    entries per row: min={lens.min()}, max={lens.max()}, "
                f"mean={lens.mean():.1f}")
            log(f"    sample raw values:")
            for v in s.head(5):
                log(f"       {v}")

        for c in scalar_text_cols:
            try:
                nun = df[c].nunique()
            except TypeError as exc:
                log(f"\n  '{c}': dtype={df[c].dtype} -- SKIPPED, still unhashable "
                    f"despite scalar-looking sample ({exc})")
                continue
            log(f"\n  '{c}': dtype={df[c].dtype}, {nun:,} unique values, "
                f"non-null={df[c].notna().sum():,}")
            if nun <= 60:
                log(f"    -> looks CODED/categorical. All values:")
                for v in sorted(df[c].dropna().unique().astype(str))[:60]:
                    log(f"       {v}")
            else:
                log(f"    -> too many unique values to be a small code set; "
                    f"likely free text or drug-level granularity. Sample:")
                for v in df[c].dropna().unique()[:20]:
                    log(f"       {v}")

        log("\n  --- searching all SCALAR text columns for named drug classes ---")
        text_cols = scalar_text_cols
        for kw in MED_CLASS_KW:
            hit_cols = []
            for c in text_cols:
                vals = df[c].dropna().astype(str).str.lower()
                if vals.str.contains(kw, na=False).any():
                    n = int(vals.str.contains(kw, na=False).sum())
                    hit_cols.append(f"{c} ({n} rows)")
            if hit_cols:
                log(f"  '{kw}': found in {hit_cols}")
            else:
                log(f"  '{kw}': not found in any text column")

    section("DONE")
    log("  If renal/inflammation markers came back NONE FOUND here, check")
    log("  whether a separate 'labs' or 'clinical_chemistry' table exists")
    log("  elsewhere in the tree before concluding they're unavailable.")
    log("  If medication class keywords found few hits, the field is likely")
    log("  coded (e.g. ATC codes or drug IDs) rather than free text, and will")
    log("  need a lookup/mapping table to classify by class.")


if __name__ == "__main__":
    main()
