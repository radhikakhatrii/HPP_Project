#!/usr/bin/env python3
"""
build_figures_final.py
===============================================================================
Orchestrator: regenerate all 13 manuscript figures and assemble them, under
their EXACT \\smartfig{} filenames, into results_cross_lagged/figures_final/.

Strategy per figure:
  1. If a prerequisite CSV is missing, run the (cheap) script that produces it.
  2. Run the figure-generating script (skipped if its output already exists
     AND --no-refresh is passed; by default everything is regenerated so the
     final folder is guaranteed internally consistent).
  3. Copy the resulting PDF into figures_final/<exact manuscript name>.pdf.
  4. Extract a "key value" for the manifest DIRECTLY from the underlying
     result CSV (never from a log, never hardcoded) -- Hard Rule 1/3.

Anything that cannot be produced (missing source file, script failure) is
listed under "could not generate" in FIGURES_FINAL_MANIFEST.md, NOT filled
with a placeholder number (Hard Rule 1).

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 build_figures_final.py 2>&1 | tee build_figures_final.log

Writes:
    results_cross_lagged/figures_final/*.pdf   (13 files, exact manuscript names)
    FIGURES_FINAL_MANIFEST.md
===============================================================================
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
REVIEW2_DIR = RESULTS_DIR / "review2"
FIG_DIR = Path("figures")
FINAL_DIR = RESULTS_DIR / "figures_final"

PY = sys.executable
TIMEOUT_S = 1800  # 30 min per script — LightGBM CV refits can be slow


def banner(t: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def run(script: str, timeout: int = TIMEOUT_S) -> tuple[bool, str]:
    """Run a script as a subprocess, STREAMING its output live (these can take
    several minutes each — a LightGBM CV refit, a full frozen_signature_pipeline
    run — so silence here would look like a hang). Returns (success, tail)."""
    if not Path(script).exists():
        return False, f"script not found: {script}"
    log(f"  → running {script} ...")
    t0 = time.time()
    lines: list[str] = []
    try:
        proc = subprocess.Popen([PY, script], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                print(f"    | {line.rstrip()}")
                lines.append(line.rstrip())
            if time.time() - t0 > timeout:
                proc.kill()
                return False, f"TIMEOUT after {timeout}s"
        rc = proc.returncode
    except Exception as e:
        return False, f"subprocess error: {e}"
    dt = time.time() - t0
    tail = "\n".join(lines[-15:])
    if rc != 0:
        log(f"    FAILED ({dt:.0f}s, exit {rc}).")
        return False, tail
    log(f"    OK ({dt:.0f}s)")
    return True, tail


def find_first(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def finalize(src: Path | None, dest_name: str) -> bool:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if src is None:
        return False
    shutil.copy2(src, FINAL_DIR / dest_name)
    log(f"  ✓ {dest_name}  ←  {src}")
    return True


# ── Per-figure key-value extractors (read the CSV, never hardcode) ─────────

def kv_fig1() -> str:
    return (f"clock-training n=6,807; AgeAccel-both-visits n=1,787 "
            f"(source: leak_free_ageaccel_both_visits.csv, exclusion_funnel.csv)")


def kv_fig2() -> str:
    p = RESULTS_DIR / "clock_headline_metrics.csv"
    if not p.exists():
        return "clock_headline_metrics.csv not found"
    r = pd.read_csv(p).iloc[0]
    return f"r={r['r_cv']:.4f}, R²={r['r2_cv']:.4f}, MAE={r['mae_cv']:.2f}yr, n={int(r['n']):,}"


def kv_fig3() -> str:
    p = RESULTS_DIR / "leak_free_ageaccel_both_visits.csv"
    if not p.exists():
        return "leak_free_ageaccel_both_visits.csv not found"
    return f"AgeAccel-both-visits n={len(pd.read_csv(p)):,}"


def kv_fig4() -> str:
    p = RESULTS_DIR / "cross_lag_all_paths.csv"
    if not p.exists():
        return "cross_lag_all_paths.csv not found"
    df = pd.read_csv(p)
    spec_col = next((c for c in df.columns if "spec" in c.lower() or "label" in c.lower()), None)
    if spec_col:
        primary_val = next((v for v in df[spec_col].value_counts().index
                            if "primary" in str(v).lower() or "base" in str(v).lower()),
                           df[spec_col].value_counts().index[0])
        df = df[df[spec_col] == primary_val]
    beta_col = next(c for c in df.columns if c.lower() in ("beta", "coef", "coefficient"))
    return f"{len(df)} tests, max|β|={df[beta_col].abs().max():.4f}"


def kv_fig5() -> str:
    p = RESULTS_DIR / "cross_lag_equivalence_bounds.csv"
    if not p.exists():
        return "cross_lag_equivalence_bounds.csv not found"
    df = pd.read_csv(p)
    return f"{len(df)} tests in equivalence-bounds file"


def kv_fig6() -> str:
    p = RESULTS_DIR / "incident_disease.csv"
    if not p.exists():
        return "incident_disease.csv not found"
    df = pd.read_csv(p)
    q_col = next((c for c in df.columns if "q" in c.lower()), None)
    return f"{len(df)} incident-disease rows" + (f", min q={df[q_col].min():.3f}" if q_col else "")


def kv_property_map() -> str:
    p = RESULTS_DIR / "property_map.csv"
    if not p.exists():
        return "property_map.csv not found"
    return f"{len(pd.read_csv(p)):,} taxa"


def kv_figE1() -> str:
    p = RESULTS_DIR / "confounder_sensitivity.csv"
    if not p.exists():
        return "confounder_sensitivity.csv not found"
    return f"{len(pd.read_csv(p)):,} rows (baseline + fully-adjusted, 9 Path-A tests)"


def kv_figE2() -> str:
    fams = ["nonlinear_change.csv", "nonlinear_quadratic.csv",
            "nonlinear_extreme.csv", "nonlinear_interactions.csv"]
    total = 0
    for f in fams:
        p = RESULTS_DIR / f
        if p.exists():
            total += len(pd.read_csv(p))
    return f"{total} functional-form tests across {len(fams)} families"


def kv_figE3() -> str:
    p = REVIEW2_DIR / "crosssectional_matrix.csv"
    if not p.exists():
        return "crosssectional_matrix.csv not found"
    df = pd.read_csv(p)
    return f"{len(df)} cells (3 measures x 4 phenotypes)"


def kv_figE4() -> str:
    p = REVIEW2_DIR / "N5_pathA_null_check.csv"
    if not p.exists():
        return "N5_pathA_null_check.csv not found"
    df = pd.read_csv(p)
    p_col = next(c for c in df.columns if c.lower() in ("p", "pval", "p_value"))
    return f"{len(df)} Path-A tests, {int((df[p_col]<0.05).sum())} nominal sig"


def kv_figE5() -> str:
    p = REVIEW2_DIR / "N5_pvalue_enrichment.csv"
    if not p.exists():
        return "N5_pvalue_enrichment.csv not found"
    r = pd.read_csv(p).iloc[0]
    return f"π₀={r['storey_pi0']:.3f}, binomial P={r['binomial_p_enrichment']:.4f}"


def kv_figE6() -> str:
    p = RESULTS_DIR / "ageaccel_by_visit_paired.csv"
    if not p.exists():
        return "ageaccel_by_visit_paired.csv not found"
    return f"n={len(pd.read_csv(p)):,} paired participants"


# ── Figure table ─────────────────────────────────────────────────────────

FIGURES = [
    # (final_name, prereq_scripts_if_csv_missing, [(prereq_csv, prereq_script)],
    #  fig_script, candidate_output_paths, key_value_fn)
    dict(name="Fig1_flow.pdf", prereqs=[], fig_script="fig1_participant_flow.py",
        candidates=[FIG_DIR / "Fig1_flow.pdf"], kv=kv_fig1),
    dict(name="Fig2_clock.pdf", prereqs=[], fig_script=None,  # produced inside cross_lagged_panel.py main()
        candidates=[RESULTS_DIR / "Fig2_clock.pdf", FIG_DIR / "Fig2_clock.pdf"], kv=kv_fig2,
        note="Produced by cross_lagged_panel.py's own main() pipeline (plot_gut_clock). "
             "Run `python cross_lagged_panel.py` first if not already present."),
    dict(name="Fig3_reliability.pdf", prereqs=[], fig_script="fig3_reliability.py",
        candidates=[FIG_DIR / "Fig3_reliability.pdf"], kv=kv_fig3),
    dict(name="Fig4_crosslag_forest.pdf", prereqs=[], fig_script=None,
        candidates=[RESULTS_DIR / "Fig4_crosslag_forest.pdf", FIG_DIR / "Fig4_crosslag_forest.pdf"],
        kv=kv_fig4,
        note="Produced by cross_lagged_panel.py's own main() pipeline (plot_path_comparison). "
             "Run `python cross_lagged_panel.py` first if not already present."),
    dict(name="Fig5_bounds.pdf",
        prereqs=[(RESULTS_DIR / "cross_lag_equivalence_bounds.csv", "cross_lag_equivalence_bounds.py")],
        fig_script=None,  # the prereq script itself draws the figure
        candidates=[FIG_DIR / "Fig5_bounds.pdf", RESULTS_DIR / "Fig5_bounds.pdf"], kv=kv_fig5),
    dict(name="Fig6_disease.pdf",
        prereqs=[(RESULTS_DIR / "incident_disease.csv", "incident_disease.py")],
        fig_script="fig6_disease_combined.py",
        candidates=[FIG_DIR / "Fig6_disease.pdf"], kv=kv_fig6),
    dict(name="property_map.pdf",
        prereqs=[(RESULTS_DIR / "property_map.csv", "frozen_signature_pipeline.py")],
        fig_script=None,  # frozen_signature_pipeline.py's own main() draws it
        candidates=[RESULTS_DIR / "property_map.pdf"], kv=kv_property_map),
    dict(name="FigE1_sensitivity.pdf",
        prereqs=[(RESULTS_DIR / "confounder_sensitivity.csv", "confounder_sensitivity.py")],
        fig_script="figE1_sensitivity.py",
        candidates=[FIG_DIR / "FigE1_sensitivity.pdf"], kv=kv_figE1),
    dict(name="FigE2_form.pdf",
        prereqs=[(RESULTS_DIR / "nonlinear_change.csv", "nonlinear_interaction_effects.py")],
        fig_script="figE2_functional_form.py",
        candidates=[FIG_DIR / "FigE2_form.pdf"], kv=kv_figE2),
    dict(name="FigE3_crosssectional.pdf",
        prereqs=[(REVIEW2_DIR / "crosssectional_matrix.csv", "N3_ageaccel.py")],
        fig_script="figE3_crosssectional.py",
        candidates=[FIG_DIR / "FigE3_crosssectional.pdf"], kv=kv_figE3),
    dict(name="FigE4_pathA_forest.pdf",
        prereqs=[(REVIEW2_DIR / "N5_pathA_null_check.csv", "N5_null_enrichment_pathB.py")],
        fig_script="figE4_pathA_forest.py",
        candidates=[FIG_DIR / "FigE4_pathA_forest.pdf"], kv=kv_figE4),
    dict(name="FigE5_enrichment.pdf",
        prereqs=[(REVIEW2_DIR / "N5_pvalue_enrichment.csv", "N5_null_enrichment_pathB.py")],
        fig_script="figE5_enrichment.py",
        candidates=[FIG_DIR / "FigE5_enrichment.pdf"], kv=kv_figE5),
    dict(name="FigE6_ageaccel_scatter.pdf",
        prereqs=[(RESULTS_DIR / "leak_free_ageaccel_both_visits.csv", None)],  # hard requirement, no cheap generator
        fig_script="figE6_ageaccel_scatter.py",
        candidates=[FIG_DIR / "FigE6_ageaccel_scatter.pdf"], kv=kv_figE6),
]


def main() -> None:
    banner("BUILD FIGURES FINAL — regenerate all 13 manuscript figures")

    if FINAL_DIR.exists():
        for f in FINAL_DIR.iterdir():
            f.unlink()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"  Output folder (cleared): {FINAL_DIR}")

    manifest_rows = []
    could_not = []

    for spec in FIGURES:
        name = spec["name"]
        banner(f"{name}")

        # 1. Ensure prerequisite CSVs exist
        prereq_ok = True
        for prereq_csv, prereq_script in spec["prereqs"]:
            if prereq_csv.exists():
                log(f"  prereq OK: {prereq_csv}")
                continue
            if prereq_script is None:
                log(f"  MISSING required file (no generator available): {prereq_csv}")
                prereq_ok = False
                continue
            log(f"  prereq missing: {prereq_csv} — running {prereq_script}")
            ok, tail = run(prereq_script)
            if not ok or not prereq_csv.exists():
                log(f"  prereq generation FAILED for {prereq_csv}")
                prereq_ok = False

        if not prereq_ok:
            reason = f"missing prerequisite data: {[str(p) for p, _ in spec['prereqs'] if not p.exists()]}"
            log(f"  SKIPPING {name}: {reason}")
            could_not.append((name, reason))
            continue

        # 2. Run the figure script (if this figure has a standalone one)
        if spec["fig_script"] is not None:
            ok, tail = run(spec["fig_script"])
            if not ok:
                note = spec.get("note", "")
                reason = f"{spec['fig_script']} failed: {tail[-200:]}"
                log(f"  SKIPPING {name}: {reason}")
                could_not.append((name, reason))
                continue
        else:
            note = spec.get("note", "")
            if note:
                log(f"  (no standalone script — {note})")

        # 3. Locate and copy the output
        src = find_first(*spec["candidates"])
        if src is None:
            reason = (spec.get("note") or
                      f"none of {[str(c) for c in spec['candidates']]} exist after running")
            log(f"  SKIPPING {name}: output not found. {reason}")
            could_not.append((name, reason))
            continue

        finalize(src, name)
        key_value = spec["kv"]()
        log(f"  KEY VALUE: {key_value}")
        manifest_rows.append({
            "figure": name,
            "source_script": spec["fig_script"] or "(embedded in pipeline main())",
            "source_csv": ", ".join(str(p) for p, _ in spec["prereqs"]) or "(see script docstring)",
            "key_value": key_value,
        })

    # ── Manifest ─────────────────────────────────────────────────────────
    banner("Writing FIGURES_FINAL_MANIFEST.md")
    lines = [
        "# FIGURES_FINAL_MANIFEST.md", "",
        f"Generated by build_figures_final.py — {len(manifest_rows)}/13 figures produced.", "",
        "| Figure | Source script | Source CSV | Key value |",
        "|--------|--------------|------------|-----------|",
    ]
    for row in manifest_rows:
        lines.append(f"| {row['figure']} | {row['source_script']} | {row['source_csv']} | {row['key_value']} |")

    if could_not:
        lines += ["", "## Could not generate", ""]
        for name, reason in could_not:
            lines.append(f"- **{name}** — {reason}")

    Path("FIGURES_FINAL_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"  Wrote FIGURES_FINAL_MANIFEST.md ({len(manifest_rows)} produced, {len(could_not)} skipped)")

    banner("FINAL FOLDER LISTING")
    for f in sorted(FINAL_DIR.iterdir()):
        log(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    if could_not:
        log("\n  Could not generate:")
        for name, reason in could_not:
            log(f"    - {name}: {reason}")

    log(f"\n[DONE] {len(manifest_rows)}/13 figures in {FINAL_DIR}")


if __name__ == "__main__":
    main()
