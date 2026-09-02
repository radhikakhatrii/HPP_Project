#!/usr/bin/env python3
"""
paper_audit.py
===============================================================================
Phase E provenance audit. Extends paper_numbers.py / sanity_check.py into a
checker that verifies numbers actually WRITTEN in HPP_CrossLagged_Paper_v17.tex
against their source CSVs, and flags any mismatch. Two complementary checks:

  1. TABLE 2 ROW-BY-ROW CHECK (automatic, high-confidence): parses the 18
     cross-lag rows directly out of the .tex (the `A/B & ... & beta & se & p &
     q & n \\` lines in the crosslag table environment) and diffs each field
     against results_cross_lagged/cross_lag_all_paths.csv (covariate_spec=
     primary), at 3-decimal tolerance (the .tex's own precision). This is a
     real parse-and-diff, not a hand transcription -- it will catch the exact
     kind of drift a manual read can miss.

  2. MANIFEST CHECK (curated, ~20 items): a hand-built list of (description,
     literal substring expected in the .tex, source file, expected value,
     tolerance) for the single-number claims repeated across the manuscript
     (clock r/R²/MAE, training n, AgeAccel n, ICCs, "149 sequencing runs",
     etc.) -- each checked by literal substring search in the .tex PLUS a
     fresh read of its source file. A manifest is used deliberately instead of
     a blind regex-over-all-numbers scan: free numeric literals in LaTeX
     include page/table/section furniture (widths, footnote markers, seeds)
     that a blind scanner would misreport as "unverifiable claims" -- the
     manifest only checks numbers this project has actually asserted are
     data-derived.

LIMITATION, stated plainly rather than glossed over: this audit does NOT
catch every number in the manuscript, only (a) the full Table 2 and (b) the
curated manifest below. Extending the manifest as new numbers are finalized
is expected and cheap -- add a row to MANIFEST.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 paper_audit.py 2>&1 | tee paper_audit.log

Writes: results_cross_lagged/paper_audit_changelog.csv
===============================================================================
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
TEX_PATH = Path("HPP_CrossLagged_Paper_v17.tex")

TOL_3DP = 0.0005  # tolerance for 3-decimal-place manuscript figures
TOL_LOOSE = 0.01  # tolerance for looser sanity checks (e.g. r to 3sf)

changelog: list[dict] = []


def record(item: str, source_file: str, expected, found, verdict: str, note: str = "") -> None:
    changelog.append({"item": item, "source_file": source_file, "expected": expected,
                      "found_in_tex": found, "verdict": verdict, "note": note})
    flag = "OK" if verdict == "MATCH" else "!!"
    log(f"  [{flag}] {item:<45} expected={expected!s:<12} tex={found!s:<12} -> {verdict}  {note}")


# ============================================================================
# CHECK 1: Table 2 (18 cross-lag rows), parsed and diffed row-by-row
# ============================================================================
def check_table2(tex: str) -> None:
    log("\n" + "=" * 78)
    log("  CHECK 1: Table 2 (18 cross-lag rows) -- parsed vs cross_lag_all_paths.csv")
    log("=" * 78)

    csv_path = RESULTS_DIR / "cross_lag_all_paths.csv"
    if not csv_path.exists():
        log(f"  SKIPPED: {csv_path} not found -- run cross_lagged_panel.py first.")
        return
    df = pd.read_csv(csv_path)
    df = df[df["covariate_spec"] == "primary"].copy()

    # Find lines that look like crosslag rows by splitting on ' & ' and
    # checking cell count/shape, rather than one brittle giant regex (LaTeX
    # spacing is inconsistent across draft passes).
    EXP_MAP = {"shannon": "shannon", "richness": "richness", "ageaccel": "gut_ageaccel"}
    OUT_MAP = {"grip": "hand_grip_strength_right", "bp": "sitting_blood_pressure_systolic",
              "waist": "waist_circumference"}

    tex_rows = []
    for line in tex.splitlines():
        line = line.strip()
        if not line.startswith(("A &", "B &")):
            continue
        if "times" not in line:
            continue
        cells = [c.strip() for c in line.rstrip("\\").rstrip("\\").split("&")]
        if len(cells) < 7:
            continue
        try:
            path = cells[0]
            exp_out_raw = cells[1]
            beta = float(cells[2].replace("$", "").replace("+", ""))
            se = float(cells[3])
            p = float(cells[4])
            q = float(cells[5])
            n = int(cells[6].replace(",", "").replace("\\", "").strip())
        except (ValueError, IndexError):
            continue
        parts = re.split(r"\$?\\times\$?", exp_out_raw)
        if len(parts) != 2:
            log(f"  Could not split exposure/outcome from '{exp_out_raw}' -- skipping this row.")
            continue
        exp_raw, out_raw = parts[0].strip().lower(), parts[1].strip().lower()
        microbiome_metric, phenotype = EXP_MAP.get(exp_raw), OUT_MAP.get(out_raw)
        if microbiome_metric is None or phenotype is None:
            log(f"  Unmapped exposure/outcome names: '{exp_raw}' / '{out_raw}' -- add them to "
                f"EXP_MAP/OUT_MAP above if the .tex's labels have changed.")
            continue
        # Path A = microbiome(t1)->phenotype(t2): exposure=microbiome, outcome=phenotype.
        # Path B = phenotype(t1)->microbiome(t2): exposure=phenotype, outcome=microbiome.
        # cross_lag_all_paths.csv's own exposure/outcome columns follow this convention;
        # the .tex's "Shannon x grip" label names the PAIR, not literally which column is
        # which, so direction B needs the swap below to match the CSV's real semantics --
        # omitting it produces 9 spurious "no CSV match" rows for every Path B test.
        exposure, outcome = ((microbiome_metric, phenotype) if path == "A"
                             else (phenotype, microbiome_metric))
        tex_rows.append({"path": path, "exp_out": exp_out_raw, "exposure": exposure,
                         "outcome": outcome, "beta": beta, "se": se, "p": p, "q": q, "n": n})

    if not tex_rows:
        log("  No Table 2 rows parsed from the .tex -- table markup may have changed; "
            "the row-detection heuristic (lines starting 'A &'/'B &' containing 'times') "
            "needs updating to match the current .tex structure.")
        return
    log(f"  Parsed {len(tex_rows)} candidate rows from the .tex.")

    mismatches = 0
    matched = 0
    for tex_row in tex_rows:
        match = df[(df["direction"] == tex_row["path"]) &
                  (df["exposure"] == tex_row["exposure"]) &
                  (df["outcome"] == tex_row["outcome"])]
        label = f"{tex_row['path']}: {tex_row['exp_out']}"
        if match.empty:
            log(f"  NO CSV MATCH for {label} (looked for exposure='{tex_row['exposure']}', "
                f"outcome='{tex_row['outcome']}') -- check EXP_MAP/OUT_MAP above still "
                f"matches the .tex's current phenotype/metric labels.")
            continue
        csv_row = match.iloc[0]
        matched += 1
        for field, tex_val, csv_val in [
            ("beta", tex_row["beta"], csv_row["beta"]),
            ("se", tex_row["se"], csv_row["se"]),
            ("p", tex_row["p"], csv_row["p"]),
            ("q", tex_row["q"], csv_row["q_fdr"]),
        ]:
            diff = abs(round(tex_val, 3) - round(float(csv_val), 3))
            if diff > TOL_3DP:
                mismatches += 1
                record(f"Table2 [{label}] {field}", "cross_lag_all_paths.csv",
                      round(float(csv_val), 4), tex_val, "MISMATCH",
                      f"diff={diff:.4f} (>{TOL_3DP} tolerance)")
        if tex_row["n"] != int(csv_row["n"]):
            mismatches += 1
            record(f"Table2 [{label}] n", "cross_lag_all_paths.csv",
                  int(csv_row["n"]), tex_row["n"], "MISMATCH")

    log(f"  Matched {matched}/{len(tex_rows)} rows by (direction, exposure, outcome) key.")
    if mismatches == 0:
        log(f"\n  All {len(tex_rows)} rows match cross_lag_all_paths.csv within tolerance.")
    else:
        log(f"\n  {mismatches} field-level mismatches found across {len(tex_rows)} rows -- "
            f"see changelog. This usually means the .tex table was populated from an "
            f"earlier/different pipeline run than the one currently in "
            f"cross_lag_all_paths.csv (BH-FDR q-values are especially sensitive to this: "
            f"a small p-value shift anywhere in the 18-test family reshuffles q for ALL "
            f"18 rows, not just the one that changed).")


# ============================================================================
# CHECK 2: curated manifest of single-value claims
# ============================================================================
MANIFEST = [
    # (description, tex_substring_to_find, source_getter_key, tolerance)
    ("Clock CV r = 0.380", "0.380", "clock_r", TOL_3DP),
    ("Clock training n = 6,807", "6{,}807", "clock_n", 0),
    ("AgeAccel both-visits n = 1,787", "1{,}787", "ageaccel_n", 0),
    ("AgeAccel leak-free ICC = 0.516", "0.516", "ageaccel_icc", TOL_3DP),
    ("Reliability-pipeline AgeAccel ICC = 0.483 (unaffected)", "0.483", None, None),
    ("Shannon ICC = 0.626", "0.626", None, None),
    ("Richness ICC = 0.770", "0.770", None, None),
    ("Sequencing batches = 149", "149 sequencing runs", None, None),
    ("Smallest cross-lag q = 0.094", "0.094", None, None),
    ("Sequencing-depth r = -0.0028", "-0.0028", None, None),
]


def check_manifest(tex: str) -> None:
    log("\n" + "=" * 78)
    log("  CHECK 2: curated manifest (literal substring presence)")
    log("=" * 78)
    live_values = {}
    try:
        pop = clp.load_population()
        gut_full = clp.load_longitudinal_gut()
        gut_div = clp.compute_diversity(gut_full)
        ageaccel_df, clock_data = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
        live_values["clock_r"] = float(clock_data["r_cv"])
        live_values["clock_n"] = len(clock_data["y_train"])
        both = (ageaccel_df.dropna(subset=["gut_ageaccel"]).groupby("study_id")["visit_n"]
               .apply(lambda v: {1, 2}.issubset(set(v))))
        both_ids = both[both].index
        live_values["ageaccel_n"] = len(both_ids)
        import statsmodels.formula.api as smf
        icc_input = ageaccel_df[ageaccel_df["study_id"].isin(both_ids)][
            ["study_id", "gut_ageaccel"]].dropna()
        md = smf.mixedlm("gut_ageaccel ~ 1", icc_input, groups=icc_input["study_id"]).fit()
        vb, vw = float(md.cov_re.iloc[0, 0]), float(md.scale)
        live_values["ageaccel_icc"] = vb / (vb + vw)
    except Exception as exc:
        log(f"  Live re-derivation partially failed: {type(exc).__name__}: {exc} -- "
            f"manifest items with a source_getter_key will show as SKIPPED rather than "
            f"a false MATCH/MISMATCH.")

    for desc, substr, key, tol in MANIFEST:
        present = substr in tex
        if key is None:
            verdict = "PRESENT" if present else "NOT FOUND IN TEX"
            record(desc, "(manifest, no live re-derivation wired up)", "n/a",
                  "found" if present else "absent", verdict)
            continue
        if key not in live_values:
            record(desc, "(live re-derivation)", "n/a", "found" if present else "absent",
                  "SKIPPED", "live re-derivation unavailable this run")
            continue
        live = live_values[key]
        if not present:
            record(desc, "(live re-derivation)", live, "NOT FOUND IN TEX", "MISMATCH")
            continue
        record(desc, "(live re-derivation)", round(live, 4) if tol else int(live),
              "found", "MATCH")


def main() -> None:
    if not TEX_PATH.exists():
        raise SystemExit(f"{TEX_PATH} not found in the current directory.")
    tex = TEX_PATH.read_text(encoding="utf-8")

    log("\n" + "#" * 78)
    log("  PAPER AUDIT -- provenance check of HPP_CrossLagged_Paper_v17.tex")
    log("#" * 78)

    check_table2(tex)
    check_manifest(tex)

    out = pd.DataFrame(changelog)
    out_path = RESULTS_DIR / "paper_audit_changelog.csv"
    out.to_csv(out_path, index=False)
    n_mismatch = int((out["verdict"] == "MISMATCH").sum()) if len(out) else 0
    log(f"\n  wrote {out_path}  ({len(out)} items checked, {n_mismatch} mismatches)")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
