#!/usr/bin/env python3
"""
audit_data_inventory.py
===============================================================================
One-shot inventory of everything available in DATA_ROOT, so we know which of
Adinath's suggested analyses (multidomain aging score, pathway-level microbiome
data, medication covariates, frailty/hospitalization endpoints, metabolomics,
immune profiling) are actually possible with existing HPP data versus needing
data we don't have.

Reads SCHEMA ONLY where possible (pyarrow, no data loaded) so this is fast and
safe even on very large tables. Falls back to pandas for anything pyarrow
can't open directly.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python audit_data_inventory.py > data_inventory.txt 2>&1
    less data_inventory.txt

Writes nothing except stdout.
===============================================================================
"""
from __future__ import annotations

import itertools
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))

# Keywords we're hunting for, grouped by what they'd unlock.
KEYWORD_GROUPS = {
    "MEDICATIONS (point 3)": [
        "medication", "drug", "antibiotic", "metformin", "ppi", "proton_pump",
        "statin", "prescri",
    ],
    "MULTIDOMAIN AGING SCORE (point 5)": [
        "cognit", "frailty", "frail", "multimorbid", "cardiovascular", "cardiac",
        "liver", "kidney", "renal", "hepatic", "body_composition", "dexa", "dxa",
        "inflamm", "crp", "il6", "il-6",
    ],
    "PATHWAY-LEVEL MICROBIOME (point 6)": [
        "humann", "pathway", "pathabundance", "genefamil", "ko_", "kegg",
        "metacyc", "ec_",
    ],
    "PROSPECTIVE ENDPOINTS (point 8)": [
        "hospital", "admission", "frailty", "functional_decline", "mortality",
        "death",
    ],
    "METABOLOMICS / IMMUNE (deep biology)": [
        "metabolom", "metabolite", "scfa", "short_chain", "bile_acid",
        "indole", "tmao", "cytokine", "immune", "lymphocyte", "flow_cytom",
    ],
    "TECHNICAL / BATCH (point 3)": [
        "batch", "sequenc", "read_count", "read_depth", "plate", "run_name",
        "instrument", "extraction",
    ],
}


def log(m=""):
    print(m, flush=True)


def get_columns(path: Path) -> list[str] | None:
    """
    Schema-only, always. NEVER falls back to a full pd.read_parquet() --
    that fallback is what made the first version of this script hang: HPP's
    species-abundance tables are hundreds of MB, and loading one fully just
    to read its column names can take minutes per file. If pyarrow can't
    read the schema, we report that and move on rather than pay that cost.
    """
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        return list(schema.names)
    except Exception as exc:
        log(f"    SKIPPED (schema unreadable, not falling back to full load): {exc}")
        return None


def main():
    log("=" * 78)
    log(f"  DATA INVENTORY: {DATA_ROOT}")
    log("=" * 78)

    if not DATA_ROOT.exists():
        log(f"FATAL: {DATA_ROOT} does not exist.")
        return

    top_dirs = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    log(f"\ntop-level modalities found: {len(top_dirs)}\n")
    for d in top_dirs:
        log(f"  {d.name}")

    log("\n" + "=" * 78)
    log("  PER-TABLE SCHEMA")
    log("=" * 78)

    # Directory name fragments that signal a huge per-participant shard tree
    # (CGM, actigraphy, heart rate, sleep, raw sequencing, ...). These are
    # skipped by NAME before we ever list their contents -- os.walk and even
    # os.scandir must enumerate every entry in a directory to know what's in
    # it, so on a folder with tens of thousands of files, the listing itself
    # is the slow part, independent of how many of those files we then open.
    # We take exactly ONE sample file via scandir with an early break, which
    # only pays for the first few directory entries, never the rest.
    SKIP_BY_NAME = ("fastq", "bam", "urs_mapped", "genotek", "single_bacteria",
                    "cgm", "actigraph", "heart_rate", "heartrate", "sleep",
                    "accelerom", "ecg", "ppg", "wearable", "timeseries",
                    "time_series", "continuous_glucose")
    SAMPLE_ONLY = 3   # files to grab from a flagged directory before stopping

    def sample_parquet(dirpath: Path, limit: int) -> list[Path]:
        """Lazily grab up to `limit` .parquet files from dirpath without
        listing the rest of a potentially huge directory."""
        try:
            with os.scandir(dirpath) as it:
                matches = (Path(e.path) for e in it
                          if e.is_file() and e.name.endswith(".parquet"))
                return list(itertools.islice(matches, limit))
        except Exception as exc:
            log(f"    could not scan {dirpath}: {exc}")
            return []

    all_columns: dict[str, list[str]] = {}
    for d in top_dirs:
        log(f"\n--- scanning {d.name}/ ---")
        # Breadth-first walk with a manual stack, so we can inspect and
        # prune a directory by NAME before descending, and so any directory
        # matching SKIP_BY_NAME gets sampled instead of fully listed.
        stack = [d]
        n_dirs_visited = 0
        while stack:
            cur = stack.pop()
            n_dirs_visited += 1
            if n_dirs_visited % 20 == 0:
                log(f"    ...{n_dirs_visited} subdirectories visited under {d.name}/")

            if any(h in cur.name.lower() for h in SKIP_BY_NAME):
                sample = sample_parquet(cur, SAMPLE_ONLY)
                if sample:
                    rel = sample[0].relative_to(DATA_ROOT)
                    log(f"  {cur.relative_to(DATA_ROOT)}/  -- name matches a known "
                        f"large-shard pattern. Sampling {len(sample)} file(s) "
                        f"WITHOUT listing the full directory.")
                    cols = get_columns(sample[0])
                    if cols is not None:
                        all_columns[f"{cur.relative_to(DATA_ROOT)}/* (sharded, "
                                   f"sampled)"] = cols
                        log(f"    {rel.name}  ({len(cols)} cols): {cols}")
                continue  # do NOT descend further into a flagged directory

            try:
                with os.scandir(cur) as it:
                    entries = list(it)
            except Exception as exc:
                log(f"    could not list {cur}: {exc}")
                continue

            pq_here = [Path(e.path) for e in entries
                      if e.is_file() and e.name.endswith(".parquet")]
            subdirs = [Path(e.path) for e in entries if e.is_dir()]

            if len(pq_here) > 20:
                sample = pq_here[:SAMPLE_ONLY]
                rel = sample[0].relative_to(DATA_ROOT)
                log(f"  {cur.relative_to(DATA_ROOT)}/  -- {len(pq_here)} parquet "
                    f"files found, sharded. Reading schema of one representative "
                    f"file instead of all {len(pq_here)}.")
                cols = get_columns(sample[0])
                if cols is not None:
                    all_columns[f"{cur.relative_to(DATA_ROOT)}/* "
                               f"({len(pq_here)} sharded files)"] = cols
                    log(f"    {rel.name}  ({len(cols)} cols): {cols}")
            else:
                for p in sorted(pq_here):
                    rel = p.relative_to(DATA_ROOT)
                    size_mb = p.stat().st_size / 1e6
                    cols = get_columns(p)
                    if cols is None:
                        continue
                    all_columns[str(rel)] = cols
                    log(f"  {rel}  ({size_mb:.1f} MB, {len(cols)} cols)")

            stack.extend(subdirs)

    log("\n" + "=" * 78)
    log("  KEYWORD MATCHES  --  what unlocks which analysis")
    log("=" * 78)

    for group, keywords in KEYWORD_GROUPS.items():
        log(f"\n### {group} ###")
        found_any = False
        for table, cols in all_columns.items():
            hits = [c for c in cols if any(kw in c.lower() for kw in keywords)]
            if hits:
                found_any = True
                log(f"  {table}")
                for h in hits:
                    log(f"      {h}")
        if not found_any:
            log("  NOTHING FOUND. This analysis is not possible with current data.")

    log("\n" + "=" * 78)
    log("  SUMMARY")
    log("=" * 78)
    log(f"  Total tables scanned: {len(all_columns)}")
    log(f"  Total distinct columns: {len(set(c for cs in all_columns.values() for c in cs)):,}")
    log("\n  Read the KEYWORD MATCHES section above: any group with real hits is")
    log("  worth pursuing; any group showing 'NOTHING FOUND' should be dropped")
    log("  from the plan or explicitly flagged as needing data we don't have.")


if __name__ == "__main__":
    main()
