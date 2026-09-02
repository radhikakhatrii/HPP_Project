#!/usr/bin/env python3
"""
audit_pathway_data.py
===============================================================================
Locates the HUMAnN pathway-abundance table before we build anything on top of
it. The earlier full inventory scan found columns named humann_pathway_
abundance_pathway_level_parquet etc., but gut_microbiome.parquet's confirmed
49-column schema does NOT include them -- so they live in a different table.
This script finds exactly which one, resolves the linked-parquet pointer
(same mechanism already used for metaphlan species), and reports the
resolved table's shape and a handful of column names, WITHOUT loading the
full pathway matrix (it can be very wide).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python audit_pathway_data.py 2>&1 | tee audit_pathway_data.log
===============================================================================
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def flat(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.names != [None]:
        df = df.reset_index()
    return df


def main():
    section("1. Every gut_microbiome and oral_microbiome parquet file, by name")
    for sub in ("gut_microbiome", "oral_microbiome"):
        d = DATA_ROOT / sub
        if not d.exists():
            log(f"  {sub}/ does not exist")
            continue
        files = sorted(d.rglob("*.parquet"))
        log(f"\n  {sub}/: {len(files)} parquet file(s)")
        for f in files:
            log(f"    {f.relative_to(DATA_ROOT)}  ({f.stat().st_size/1e6:.1f} MB)")

    section("2. Searching for a humann pointer column in each top-level table")
    candidates = []
    for sub in ("gut_microbiome", "oral_microbiome"):
        d = DATA_ROOT / sub
        if not d.exists():
            continue
        # top-level files only (not the aggregated/ subfolder, which holds the
        # RESOLVED tables, not pointer tables)
        for f in sorted(d.glob("*.parquet")):
            try:
                import pyarrow.parquet as pq
                cols = list(pq.read_schema(f).names)
            except Exception as exc:
                log(f"  {f.name}: schema read failed ({exc})")
                continue
            humann_cols = [c for c in cols if "humann" in c.lower()]
            log(f"\n  {f.relative_to(DATA_ROOT)}: {len(cols)} columns, "
                f"{len(humann_cols)} humann-related")
            for c in humann_cols:
                log(f"    {c}")
                candidates.append((f, c))

    section("3. Resolving the best pathway-ABUNDANCE candidate (not coverage, not genes)")
    # Prefer: pathway (not gene_families), abundance (not coverage),
    # pathway_level (not microbe_level -- that one is per-microbe-per-pathway,
    # much larger and not what we want for a first pass).
    def score(col: str) -> int:
        c = col.lower()
        s = 0
        if "pathway" in c: s += 4
        if "abundance" in c: s += 4
        if "coverage" in c: s -= 10
        if "gene" in c: s -= 10
        if "pathway_level" in c: s += 3
        if "microbe_level" in c or "microbiome_level" in c: s -= 2
        if c.endswith("_parquet"): s += 1
        return s

    if not candidates:
        log("  NO humann columns found in any top-level gut_microbiome/oral_microbiome "
            "table. The pathway data may live under a different directory entirely "
            "(check the full data_inventory.txt output again for the exact parent path).")
        return

    candidates.sort(key=lambda fc: score(fc[1]), reverse=True)
    log("  candidates ranked (best first):")
    for f, c in candidates[:10]:
        log(f"    score={score(c):>3}  {f.name} :: {c}")

    best_file, best_col = candidates[0]
    log(f"\n  using: {best_file.relative_to(DATA_ROOT)} :: {best_col}")

    df = flat(best_file)
    if best_col not in df.columns:
        log(f"  FATAL: '{best_col}' not found after loading {best_file.name}")
        return
    ptr_vals = df[best_col].dropna().unique()
    log(f"  distinct pointer values: {len(ptr_vals)}")
    for v in ptr_vals[:5]:
        log(f"    {v}")

    if len(ptr_vals) == 0:
        log("  Pointer column is entirely null. Cannot resolve.")
        return

    section("4. Resolving the pointer and inspecting the pathway table")
    rel = str(ptr_vals[0]).lstrip("./")
    resolved = (best_file.parent / rel)
    if not resolved.exists():
        resolved = DATA_ROOT / rel
    log(f"  resolved path: {resolved}")
    log(f"  exists: {resolved.exists()}")
    if not resolved.exists():
        log("  Trying a directory search for the filename anywhere under DATA_ROOT...")
        matches = list(DATA_ROOT.rglob(Path(rel).name))
        for m in matches[:5]:
            log(f"    found: {m.relative_to(DATA_ROOT)}")
        if matches:
            resolved = matches[0]

    if resolved.exists():
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(resolved)
            log(f"  resolved table: {len(schema.names)} columns")
            log(f"  first 15 column names:")
            for c in schema.names[:15]:
                log(f"    {c}")
            log(f"  last 10 column names:")
            for c in schema.names[-10:]:
                log(f"    {c}")
            log(f"  file size: {resolved.stat().st_size/1e6:.1f} MB")
        except Exception as exc:
            log(f"  schema read failed: {exc}")

    section("DONE")
    log("  If a pathway table resolved successfully above, the next script will")
    log("  build a pathway-level microbiome clock the same way the species clock")
    log("  was built: leak-free training on single-visit participants, AgeAccel")
    log("  as the residual, tested in the same cross-lagged framework.")


if __name__ == "__main__":
    main()
