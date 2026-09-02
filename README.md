# HPP Gut-Microbiome / Biological-Age Pipeline — full code

Everything needed to **re-run** the HPP cross-sectional + cross-lagged gut-microbiome-ageing
analysis end-to-end, including the Nastya-review-round extensions (N1–N5) and every
manuscript figure. No data files are included — see **Data (not included)** below.

## ⚠️ Data is intentionally NOT in this folder

HPP participant data must never leave MBZUAI. This folder contains **code only**. Before
pushing this folder to GitHub, double-check no `.parquet`/`.csv`/`.xpt` file or anything
under a data directory got swept in — if you ever see one, delete it before committing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # or: conda create -n pheno python=3.11
pip install -r requirements.txt
```

Set the two environment variables the pipeline reads (both have sane defaults if unset —
see `cross_lagged_panel.py` lines ~144–146):

```bash
export HPP_DATA_ROOT=~/studies/hpp_datasets      # root of the raw HPP parquet files
export HPP_RESULTS_DIR=results_cross_lagged      # where every script writes its output
```

`HPP_DATA_ROOT` must contain the per-domain parquet folders the scripts read from —
`gut_microbiome/`, `hand_grip/`, `blood_pressure/`, `anthropometrics/`, plus the
population/sociodemographic files (`initial_medical.parquet`, `ukbb.parquet`) and the
metaphlan4 species-abundance parquet
(`gut_microbiome/abundance/metaphlan4/aggregated/metaphlan_abundance_species.parquet`).

`RANDOM_SEED = 42` is hardcoded in `cross_lagged_panel.py` and reused by every script that
imports it — this is what makes every CV split, clock fit, and bootstrap reproducible.

## Execution order

Everything below imports `cross_lagged_panel` as `clp` — a single shared library, not just
the main script — so it must be importable (same folder / on `PYTHONPATH`) for anything
else to run.

**1. Core pipeline (run first — everything downstream reads its output CSVs):**
```
python cross_lagged_panel.py
```
Trains the leak-free gut-clock, computes AgeAccel, runs the 18 primary cross-lagged tests
(both covariate specs), the exclusion funnel, incident disease, and writes
`Fig2_clock.pdf`, `Fig4_crosslag_forest.pdf`, `clock_headline_metrics.csv`,
`cross_lag_all_paths.csv`, `leak_free_ageaccel_both_visits.csv`, `exclusion_funnel.csv`
and more into `$HPP_RESULTS_DIR/`.

**2. Signature / property-map pipeline (independent of step 1's outputs):**
```
python frozen_signature_pipeline.py
```
Writes `property_map.csv` / `property_map.pdf` and the E1–E3 discovery/stability/
prospective signature results.

**3. Nastya-review-round analyses (N1–N5), each reads step 1's CSVs:**
```
python N1_outcome_reliability.py
python N2_change_table.py
python build_single_visit_ageaccel.py    # or: python N3_ageaccel.py (does both A+B+C)
python N3_crosssectional.py
python N4_elasticnet_clock.py
python N5_null_enrichment_pathB.py
```
`N3_ageaccel.py` is the fixed/complete version — it applies the frozen clock to the
single-visit stratum, residualises on chronological age, and runs the incremental-R² model
comparison in one script (superseding the older `build_single_visit_ageaccel.py` +
`N3_crosssectional.py` two-step path, which is kept only because some earlier-generated
CSVs still reference it).

**4. Supporting analyses each figure needs (skip any whose figure you don't need):**
```
python confounder_sensitivity.py         # -> FigE1
python nonlinear_interaction_effects.py  # -> FigE2
python incident_disease.py               # -> Fig6
python cross_lag_equivalence_bounds.py   # -> Fig5 (writes the figure itself too)
python prospective_outcomes.py           # Table E7/E8 numbers
python longitudinal_microbiome_age.py    # trait_state_disease.csv, Fig6 panel (a)
```

**5. All 13 manuscript figures, assembled automatically:**
```
python build_figures_final.py
```
This is the orchestrator — it runs everything in step 4 itself if the prerequisite CSV is
missing, then runs every `fig*.py` / `figE*.py` script, and copies all 13 PDFs into
`$HPP_RESULTS_DIR/figures_final/` under the exact filenames the manuscript's `\smartfig{}`
calls expect. Also writes `FIGURES_FINAL_MANIFEST.md`. Prefer running this over the
individual figure scripts by hand.

**6. Tables / workbook (optional, run after step 1):**
```
python build_table1_cohort.py
python build_workbook.py
python build_workbook_review.py
```

## One-off / diagnostic scripts (not part of the reproducible chain above)

These were written during development to audit specific bugs or sanity-check specific
numbers. They are **not required** to reproduce the manuscript's figures or tables, but are
included per "leave nothing out" — several were how specific bugs described in
`MANUSCRIPT_UPDATE_LEDGER.md` were originally found, and re-running one may be the fastest
way to re-diagnose a similar issue on new data:

`audit_blood_meds.py`, `audit_data_and_numbers.py`, `audit_data_inventory.py`,
`audit_pathway_data.py`, `bmi_signflip_diagnostic.py`, `check_pa_fix.py`,
`check_s3_access.py`, `clock_validation.py`, `equivalence_tests.py`,
`heldout_clock_validation.py`, `make_review_figures.py`, `multidomain_aging.py`,
`N_review_figures.py`, `paper_analysis.py`, `paper_audit.py`, `paper_numbers.py`,
`participant_flow.py`, `path_b_asymmetry.py`, `review_followup.py`, `sanity_check.py`.

## Unrelated project bundled in the same repo

`prototype.py` and `hpp_prototype.py` are an **earlier, separate** proof-of-concept
(NHANES-based PhenoAge + oral-microbiome causal inference — see the top-level `README.md`).
They use a completely different dataset (public NHANES, not HPP) and are unrelated to the
cross-lagged pipeline above. Included for completeness only.

## ⚠️ If you re-run this on a longer / bigger dataset (e.g. a future 6-year HPP follow-up)

The pipeline is **not hardcoded to the current n's** — every count (single-visit stratum,
repeat-visit stratum, analytic n per outcome, etc.) is computed live from whatever data is
under `$HPP_DATA_ROOT`. Two things to know before trusting a re-run on materially different
data, though:

1. **Ledger cross-checks will fire loud "MISMATCH" warnings — by design, not a bug.**
   Several scripts (`fig1_participant_flow.py::verify_counts()`, `fig3_reliability.py`,
   `clock_validation.py`) hardcode the numbers currently reported in the manuscript
   (e.g. `CLOCK_TRAINING_N = 6807`, `AGEACCEL_BOTH_N = 1787`, ICC fallbacks 0.626/0.770,
   `PUBLISHED_R`) purely so a silent drift gets caught. On a new dataset these will not
   match and the scripts will print a `*** MISMATCH ***` line and keep going with the
   **live**, correctly-recomputed value — read the mismatch, don't suppress it, and update
   the hardcoded constant once the new number is confirmed correct.

2. **Only the first two visits per participant are used, everywhere.** `visit_n` is derived
   via `rank(method="dense")` on each participant's collection dates, and the repeat-visit /
   cross-lagged / ICC / Spearman-Brown / FigE6 logic throughout is written for exactly two
   time points (`visit_n == 1` vs `visit_n == 2`). A 6-year dataset with 3+ visits per
   participant will **not crash** — it will silently use only each participant's first two
   dated stool samples and ignore any later visits. Extending this to a genuine 3+ wave
   panel (e.g. a proper multi-wave cross-lagged model, or an N-visit ICC) is a real design
   change to `compute_diversity()`, `build_gut_clock_ageaccel()`, and
   `run_cross_lagged_panel()` in `cross_lagged_panel.py` — not something that happens
   automatically by pointing `HPP_DATA_ROOT` at more data.

## Provenance / numbers documentation (not code, kept for context)

Not required to run anything, but explains what every number in the manuscript traces back
to: `MANUSCRIPT_UPDATE_LEDGER.md`, `REVIEW2_NUMBERS.md`, `N3_AGEACCEL_NUMBERS.md`,
`V19_NUMBERS_CHANGELOG.md`, `CLAUDE_CODE_BRIEF*.md` (in the parent folder — not copied here
since they're documentation, not code the pipeline imports or runs).
