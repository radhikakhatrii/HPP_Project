# HPP Pipeline — 2-Year (as run), and How to Extend to 4-Year / 6-Year Follow-Up

This documents (1) the exact step-by-step pipeline used for the current manuscript, which
was run on **2 visits ≈ 2.1 years apart**, and (2) what changes — and what doesn't — if HPP
collects a 3rd visit (~4-year follow-up) or 4th visit (~6-year follow-up), with a concrete,
stepwise pipeline for each.

Every script named below lives in the project root (and is copied into
[`to go on github/`](to%20go%20on%20github/)). All of them `import cross_lagged_panel as clp`
— it is the shared library everything else depends on, not just a script in its own right.

---

## Part 1 — The 2-year pipeline, as actually run

Two dated stool metagenomes per repeat-visit participant (`visit_n == 1`, `visit_n == 2`),
~2.1 years apart. This is the study the manuscript reports.

```
STEP 1   python cross_lagged_panel.py
         → trains the leak-free gut clock (single-visit stratum, n=6,807),
           computes AgeAccel, runs the 18 primary cross-lagged tests (t1→t2, both
           directions, both covariate specs), writes cross_lag_all_paths.csv,
           leak_free_ageaccel_both_visits.csv, exclusion_funnel.csv,
           clock_headline_metrics.csv, Fig2_clock.pdf, Fig4_crosslag_forest.pdf.

STEP 2   python frozen_signature_pipeline.py
         → E1 (discovery) / E2 (stability) / E3 (prospective) signature analysis,
           property_map.csv/.pdf. Independent of Step 1's output.

STEP 3   python N1_outcome_reliability.py
         python N2_change_table.py
         python N3_ageaccel.py
         python N4_elasticnet_clock.py
         python N5_null_enrichment_pathB.py
         → the Nastya-review-round analyses (outcome ICC, grip-decline-vs-noise,
           cross-sectional matrix + AgeAccel incremental value, elastic-net
           baseline clock, null re-check + enrichment + Path-B). Each reads
           Step 1's CSVs.

STEP 4   python confounder_sensitivity.py
         python nonlinear_interaction_effects.py
         python incident_disease.py
         python cross_lag_equivalence_bounds.py
         python prospective_outcomes.py
         python longitudinal_microbiome_age.py
         → one supporting CSV per downstream figure/table.

STEP 5   python build_figures_final.py
         → runs Step 4 itself for anything still missing, then every fig*.py /
           figE*.py script, and assembles all 13 manuscript PDFs into
           results_cross_lagged/figures_final/ under their exact \smartfig{} names.

STEP 6   python build_table1_cohort.py
         python build_workbook.py
         python build_workbook_review.py
         → Table 1 and the full numbers workbook.
```

`RANDOM_SEED = 42` throughout (`cross_lagged_panel.py`), so every step above is
deterministic and reproducible on the same data.

### What "2 visits" is baked into

Everything above works because the study genuinely only has 2 stool-collection timepoints.
Several places encode that assumption **explicitly**, not just incidentally:

| Component | 2-visit assumption |
|---|---|
| `build_gut_clock_ageaccel()` [CHECK-2a] (`cross_lagged_panel.py`) | "both visits" = `1 in v and 2 in v` — hardcoded to visit numbers 1 and 2 |
| `run_cross_lagged_panel()` (`cross_lagged_panel.py`) | fits `Y_t2 ~ Y_t1 + X_t1 + covariates` — a single-wave-pair (2-timepoint) cross-lagged model; there is no t3/t4 term |
| Spearman-Brown amplification (`fig3_reliability.py::spearman_brown()`) | `2*ICC1/(1+ICC1)` — the **2-visit-mean** reliability formula specifically |
| `N2_change_table.py` | one Δ = visit2 − visit1 per person; no trajectory |
| `FigE6_ageaccel_scatter.pdf` (`figE6_ageaccel_scatter.py`) | a single visit-1-vs-visit-2 scatter |
| ICC(3,1) computation (`mixedlm` variance-ratio, used in `fig3_reliability.py`, `N1_outcome_reliability.py`, `figE6_ageaccel_scatter.py`) | works in **long format** (`melt` over whatever visit columns exist) — this one is *already* visit-count-agnostic, see Part 2 |

---

## Part 2 — Extending to 4-year (3 visits) and 6-year (4 visits) follow-up

Assume HPP keeps collecting a stool sample roughly every 2 years, so a 4-year mark gives a
**3rd visit** and a 6-year mark gives a **4th visit** for participants who stay enrolled
that long. `visit_n` is already derived generically
(`groupby("study_id")["visit"].rank(method="dense")`), so visit_n = 3, 4, … appear
automatically once the raw parquet has more rows — nothing needs to change to make new
visits *visible* to the code. The question is what to do with them.

### 2a. Runs unchanged, automatically stronger

Re-point `HPP_DATA_ROOT` at the updated data and re-run **Steps 1–2 as-is**:

- **The clock itself** (`build_gut_clock_ageaccel`) trains on the *single-visit* stratum —
  unaffected by how many follow-up visits repeat participants have. If total enrollment also
  grew, the clock trains on more people and should get *more* accurate (higher CV r²), not
  less. No code change needed to benefit from this.
- **ICC / reliability** (`fig3_reliability.py`'s live AgeAccel ICC, `N1_outcome_reliability.py`)
  already `melt()`s into long format and fits `mixedlm("~1", groups=study_id)` — this is a
  correct ICC estimator for *any* number of repeated measurements per person, not just 2.
  Re-running these as-is on 3- or 4-visit data gives a genuinely better-powered ICC estimate
  automatically (more repeats per person → tighter within-person variance estimate).
- **Incident disease / prospective outcomes** (`incident_disease.py`,
  `prospective_outcomes.py`) — a longer follow-up window means more accrued events, which
  can only help power. No code change required; some phenotypes currently excluded for low
  EPV (e.g. hypertension, EPV=9.0 < 10 at 2 years) may newly clear the threshold at 4–6
  years — re-check the EPV filter output rather than assuming the same phenotypes drop out.

### 2b. Needs a code change before it will use the new visits correctly

These will **run without crashing** on 3–4 visit data (visit_n just goes up to 3 or 4), but
will silently only use each participant's first two visits unless changed — meaning a
naive re-run at "6 years" would still only be reporting the same 2-year cross-lag result.

**STEP A — decide the cross-lag design** (a methods decision, not just code):
Two defensible options for `run_cross_lagged_panel()` in `cross_lagged_panel.py`:
  - *(i) Pairwise replication* — keep the exact same 2-wave model, but run it once per
    consecutive wave-pair (t1→t2, t2→t3, t3→t4) as independent replications. Cheapest
    change: parameterise `make_wide()`'s `visit_n` filter (currently hardcoded to 1/2) to
    accept an arbitrary `(t_early, t_late)` pair, and loop over
    `[(1,2), (2,3), (3,4)]` once the data has that many waves. Answers "does the same
    microbiome→phenotype pattern replicate as follow-up lengthens?"
  - *(ii) True multi-wave CLPM* — fit all waves jointly (a proper cross-lagged panel model
    with stability paths across all timepoints, e.g. via `semopy`/`lavaan`-style SEM, not
    `statsmodels.formula.api.ols`). Statistically stronger (uses all waves' information
    at once) but a real implementation project, not a same-day patch.
  For a 4-year (3-visit) re-run, option (i) is the pragmatic path; only move to (ii) once
  4+ waves exist and the extra complexity is worth it.

**STEP B — generalize Spearman-Brown to k visits.** `spearman_brown()` in
`fig3_reliability.py` implements the 2-visit-mean formula only. For k visits, replace it
with the general form:
```python
def spearman_brown_k(icc1: float, k: int) -> float:
    return k * icc1 / (1 + (k - 1) * icc1)
```
and pass `k=3` (4-year) or `k=4` (6-year) instead of the hardcoded 2-visit call.

**STEP C — generalize the "both visits present" check.** `[CHECK-2a]`'s
`1 in v and 2 in v` in `build_gut_clock_ageaccel()` should become "has ≥2 of the visits
you care about" (e.g. `{1,2}.issubset(v)` for the 2-year comparison specifically, or
`len(v) >= k` for "any k visits") depending on which comparison you're running.

**STEP D — extend the change table beyond a single Δ.** `N2_change_table.py` currently
computes one visit2−visit1 delta. At 3–4 visits this should become a per-person trajectory
(e.g. linear slope across visits, or Δ for each consecutive pair) — a genuine rewrite of
that script's core loop, not a parameter tweak.

**STEP E — FigE6 becomes a scatter-matrix.** `figE6_ageaccel_scatter.py`'s single v1-vs-v2
panel generalizes to an all-pairs scatter matrix (v1×v2, v1×v3, v2×v3, …) once 3+ visits
exist — straightforward to extend (loop the existing panel body over
`itertools.combinations(visit_ns, 2)`), not a design decision like Step A.

### 2c. Stepwise pipeline — 4-year follow-up (3 visits)

```
STEP 0   Confirm the new data actually has a 3rd dated stool sample per repeat
         participant: check visit_n now reaches 3 for a meaningful subset (not just 1–2).

STEP 1   Apply code changes B, C, E above (Step A only if you want the joint t1→t3 /
         t2→t3 cross-lag, not just the replication check).

STEP 2   python cross_lagged_panel.py
         → re-run as-is first. Confirm [CHECK-1] and the ledger cross-checks in
           fig1_participant_flow.py::verify_counts() — expect *** MISMATCH ***
           lines (the hardcoded 6,807/1,787/etc. are 2-year numbers); read the
           live values they print rather than trusting the hardcoded constant.

STEP 3   Re-run the pairwise cross-lag for the new wave-pairs (t1→t2 as a
         sanity-check replication of the manuscript, t2→t3 as the new 4-year test)
         using the Step-A parameterised make_wide()/run_cross_lagged_panel().

STEP 4   python N1_outcome_reliability.py  (now benefits from 3 repeats/person)
         python N5_null_enrichment_pathB.py  (re-run against the new
             cross_lag_all_paths.csv — now has both wave-pairs' tests in it)

STEP 5   python build_figures_final.py
         → regenerates all 13 figures against the new data. FigE6 will need its
           code change (2b Step E) applied first or it will only show v1-vs-v2.

STEP 6   Update REVIEW2_NUMBERS.md / MANUSCRIPT_UPDATE_LEDGER.md-style provenance
         notes with the new n's and effect sizes — don't silently overwrite the
         2-year numbers, add a new dated section (see how REVIEW2_NUMBERS.md itself
         is structured: value → source CSV → column, one section per analysis).
```

### 2d. Stepwise pipeline — 6-year follow-up (4 visits)

Same as 4-year, with two differences:

```
STEP 1   Same code changes as 4-year (B, C, E), plus extend N2_change_table.py's
         trajectory logic (2b Step D) — with 4 timepoints a single pairwise Δ is
         no longer an adequate summary of "change."

STEP 3   Now 3 wave-pairs exist: t1→t2 (replicates the manuscript), t2→t3
         (4-year test), t3→t4 (6-year test). Decide here whether to also attempt
         the joint multi-wave CLPM (2b Step A option ii) — with 4 waves this is
         the point where the joint model starts being clearly worth the extra
         implementation effort over 3 separate pairwise replications.

STEP 4   Re-run N1 with k=4 in spearman_brown_k(); the ICC itself needs no change
         (mixedlm already generalizes), only the amplification-prediction panel.
```

---

## Summary: what never needs to change

`build_gut_clock_ageaccel()`'s clock **training** step, `compute_diversity()`,
`_compute_visit_age()`, the ICC(3,1) `mixedlm` estimator, and every audit/table/workbook
script that just reads a CSV and formats it — all already visit-count-agnostic. The actual
work in going from 2→4→6 years is entirely in the **cross-lag model design** (Part 2b Step
A) and the handful of scripts that hardcode "exactly 2 visits" (Steps B–E). Everything else
in [`to go on github/`](to%20go%20on%20github/README.md) runs as documented in Part 1.
