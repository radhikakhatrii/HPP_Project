#!/usr/bin/env python3
"""
build_workbook_review.py
===============================================================================
Sheets added to HPP_Analysis_Statistics.xlsx in response to review (Adinath's
points 2, 3, 4, 5, 7 and 8), plus a figure/table index reflecting the numbers
the manuscript actually prints.

Imported by build_workbook.py -- run THAT, not this:

    python build_workbook.py

PROVENANCE. Every number here was transcribed from verified terminal output of
a named script, recorded in the "Source" column of each sheet. Nothing is
recomputed here, so this file cannot silently disagree with the pipeline; it
can only disagree with the manuscript, and the figure index exists to make
that comparison mechanical.

FIGURE NUMBERING. sn-jnl numbers main-text floats 1..7 and appendix floats
A1..A3. The captions do not carry their own numbers (an earlier draft had
"Extended Data Fig. E1" hardcoded in the caption, which printed alongside the
class's own "Fig. A1" as a doubled label). The Index sheet therefore records
the printed number, the internal \\label, and the PNG filename, because these
three no longer coincide -- FIG09 prints as Fig. 3, FIG11 as Fig. 7.
===============================================================================
"""

# --------------------------------------------------------------------- data

FIGURES = [
    # printed, label, file, section, note
    ("Fig. 1", "figflow", "(TikZ, drawn in LaTeX - no image file)",
     "2.2 Participant flow", "CONSORT-style derivation of the analytic samples"),
    ("Fig. 2", "fig1", "FIG02_gut_clock_predicted_vs_chronological_age.png",
     "2.3 Clock", "Predicted vs chronological age, training stratum"),
    ("Fig. 3", "figcalib", "FIG09_clock_calibration.png",
     "2.4 Clock validation", "NEW - calibration, decile residuals, stratified MAE"),
    ("Fig. 4", "fig2", "FIG03_longitudinal_stability_ICC_3panel.png",
     "2.5 Stability", "AgeAccel retest, aging rate distribution"),
    ("Fig. 5", "fig3", "FIG05_forest_plot_pathA_vs_pathB.png",
     "2.6 Cross-lagged", "Path A vs Path B coefficients with CIs"),
    ("Fig. 6", "fig4", "FIG06_direction_verdicts_neglog10p.png",
     "2.6 Cross-lagged", "Directional verdicts, -log10 p"),
    ("Fig. 7", "figprosp", "FIG11_robustness_forest.png",
     "2.11 Prospective outcomes", "NEW - forest of the 12 prospective tests"),
    ("Fig. A1", "figE1", "FIG07_effect_wheel_magnitudes.png",
     "Extended Data", "Radial comparison of effect magnitudes"),
    ("Fig. A2", "figE2", "FIG08_richness_visit1_vs_visit2_3D.png",
     "Extended Data", "Richness visit 1 vs visit 2, 3D density"),
    ("Fig. A3", "figE4", "FIG10_interaction_surface_3D.png",
     "Extended Data", "NEW - interaction response surface, fitted vs null"),
    ("(not printed)", "figE3", "FIG04_crosslag_heatmap_pathA_pathB.png",
     "removed from appendix",
     "COMMENTED OUT - PNG predates the Path-B pivot fix and under-reports "
     "6 of 9 Path-B cells while its caption claims it matches Table 3. "
     "Do not re-enable without regenerating."),
]

TABLES = [
    ("Table 1", "tab1", "Characteristics of the analytic samples", "2.1 Cohort", ""),
    ("Table 2", "tabclockval", "Clock performance stratified by sex and age tertile",
     "2.4 Clock validation", "NEW"),
    ("Table 3", "tab2", "All 18 cross-lagged coefficients", "2.6 Cross-lagged", ""),
    ("Table 4", "tab3", "Power / equivalence bounds for the nine Path A tests",
     "2.7 Power", ""),
    ("Table 5", "tabconf", "Confounder and technical sensitivity, nine Path A tests",
     "2.8 Confounders", "NEW"),
    ("Table 6", "tabnonlin", "Nonlinearity, extremes, change coupling, effect modification",
     "2.9 Nonlinearity", "NEW"),
    ("Table 7", "tabmulti", "Baseline microbiome vs change in the multidomain composite",
     "2.10 Multidomain", "NEW"),
    ("Table 8", "tab4", "AgeAccel vs 12 disease phenotypes", "2.11 Disease", ""),
    ("Table 9", "tabprosp", "Baseline microbiome vs subsequent clinical events",
     "2.12 Prospective", "NEW"),
]

CONFOUNDER = [
    ("Shannon", "Grip strength", 0.580, 0.472, "stable null"),
    ("Shannon", "Systolic BP", 0.448, 0.338, "stable null"),
    ("Shannon", "Waist circumference", 0.646, 0.694, "stable null"),
    ("Richness", "Grip strength", 0.226, 0.100, "null, moved closer"),
    ("Richness", "Systolic BP", 0.124, 0.081, "null, largest shift"),
    ("Richness", "Waist circumference", 0.492, 0.702, "stable null"),
    ("AgeAccel", "Grip strength", 0.843, 0.808, "stable null"),
    ("AgeAccel", "Systolic BP", 0.829, 0.615, "stable null"),
    ("AgeAccel", "Waist circumference", 0.691, 0.685, "stable null"),
]

CLOCK_STRATA = [
    ("All participants", "38-70", 5188, 0.375, 5.61),
    ("Male", "-", 2507, 0.347, 5.69),
    ("Female", "-", 2681, 0.393, 5.53),
    ("Youngest age tertile", "38-46", 1801, 0.126, 6.52),
    ("Middle age tertile", "47-53", 1696, 0.136, 2.59),
    ("Oldest age tertile", "54-70", 1691, 0.158, 7.67),
]

NONLINEAR = [
    ("Change-to-change", "delta exposure -> delta outcome", 9, 0, "0.199"),
    ("Nonlinearity", "quadratic exposure term", 9, 0, "0.199"),
    ("Extreme exposure", "top + bottom decile vs middle 80%", 9, 0, "all p > 0.06"),
    ("Effect modification", "exposure x modifier interaction", 45, 0, "0.246"),
]

MULTIDOMAIN = [
    ("Forward", "AgeAccel (t1) -> delta composite", -0.0003, 0.994, 354),
    ("Forward", "Shannon (t1) -> delta composite", -0.0131, 0.779, 354),
    ("Forward", "Richness (t1) -> delta composite", -0.0499, 0.275, 354),
    ("Reverse", "Composite (t1) -> AgeAccel (t2)", -0.0760, 0.137, 354),
]

PROSPECTIVE = [
    ("Multimorbidity accrual (count)", "AgeAccel", "beta", 0.030, 0.497, 512, 204),
    ("Multimorbidity accrual (count)", "Shannon", "beta", -0.030, 0.504, 512, 204),
    ("Multimorbidity accrual (count)", "Richness", "beta", -0.065, 0.143, 512, 204),
    ("Any new diagnosis (binary)", "AgeAccel", "OR", 1.040, 0.661, 512, 204),
    ("Any new diagnosis (binary)", "Shannon", "OR", 0.906, 0.275, 512, 204),
    ("Any new diagnosis (binary)", "Richness", "OR", 0.874, 0.137, 512, 204),
    ("Incident hospitalization", "AgeAccel", "OR", 1.098, 0.598, 508, 34),
    ("Incident hospitalization", "Shannon", "OR", 0.927, 0.659, 508, 34),
    ("Incident hospitalization", "Richness", "OR", 1.027, 0.880, 508, 34),
    ("Functional decline (grip -10%)", "AgeAccel", "OR", 0.970, 0.734, 509, 254),
    ("Functional decline (grip -10%)", "Shannon", "OR", 0.993, 0.939, 509, 254),
    ("Functional decline (grip -10%)", "Richness", "OR", 1.038, 0.679, 509, 254),
]


# Point-by-point response. "Addressed" means an analysis was run and is
# reported; "Not done" means exactly that, with the reason stated rather than
# softened. Nothing here is marked addressed on the strength of an argument
# alone.
RESPONSE = [
    ("1", "Replace ICC-division MDE with equivalence testing or a "
          "measurement-error model",
     "Addressed",
     "TOST equivalence bounds are now primary; the ICC-division bound is "
     "retained only as a labelled sensitivity check. All nine Path A tests "
     "establish equivalence at +/-0.10.",
     "Manuscript 2.7 + Table 4; sheet 05_Power_MDE"),

    ("2", "Participant-flow diagram; compare included vs excluded; consider "
          "IPW or multiple imputation",
     "Addressed",
     "Flow diagram added. 0 of 11 comparisons significant, 0 with |d| >= 0.20, "
     "largest 0.059. IPW/MI deliberately NOT applied: weighting on covariates "
     "showing no imbalance adds variance without removing bias.",
     "Manuscript 2.2 + Fig. 1; sheet 13_Selection_Bias"),

    ("3", "Sensitivity models for confounders and technical factors",
     "Addressed (partial)",
     "Diet, smoking, alcohol, activity, sequencing depth and medication use "
     "added; 0 of 9 cross into significance. LIMIT: antibiotics, PPIs and "
     "metformin are not separately coded in the source medication table and "
     "are absorbed into a general medication indicator.",
     "Manuscript 2.8 + Table 5; sheet 14_Confounder_Sensitivity"),

    ("4", "Validate the clock fully - R2, MAE, calibration, stratified "
          "performance, batch, leakage",
     "Addressed",
     "Calibration slope 0.151; sex- and tertile-stratified performance; no "
     "depth artefact (r = -0.006, p = 0.689); 139 batches within 0.58 residual "
     "SD; leakage re-confirmed. Gated on a reproduction check that matched the "
     "published r to 0.0013.",
     "Manuscript 2.4 + Table 2 + Fig. 3; sheet 15_Clock_Validation"),

    ("5", "Build a multidomain biological-aging score and test whether "
          "AgeAccel predicts change in it",
     "Addressed",
     "Seven domains from 17 measures, validated against chronological age "
     "(r = 0.308) before use. AgeAccel predicts change in it at beta = -0.0003, "
     "p = 0.994.",
     "Manuscript 2.10 + Table 7; sheet 16_Multidomain_Composite"),

    ("6", "Go beyond diversity and AgeAccel - species, pathways, functional "
          "modules, metabolite potential",
     "NOT DONE - blocked",
     "The cohort's HUMAnN pathway tables exist only for the ORAL microbiome, "
     "not the gut, and are stored as per-sample objects the analysis "
     "environment cannot reach (valid credentials, but no network route - "
     "ConnectTimeoutError). Two separate obstacles: wrong tissue, and "
     "unreachable storage. Named in the manuscript as the single most "
     "valuable extension of this work.",
     "Manuscript Discussion/Limitations, first item"),

    ("7", "Within-person change-to-change; nonlinear effects; extreme "
          "AgeAccel; interactions by subgroup",
     "Addressed",
     "72 tests across four families; 0 survive FDR within family. LIMIT: "
     "frailty could not be used as a modifier - no validated frailty index "
     "exists in this cohort. Recorded as unexamined, not excluded.",
     "Manuscript 2.9 + Table 6 + Fig. A3; sheet 17_Nonlinear_Interactions"),

    ("8", "Redesign the prospective analysis so only baseline microbiome "
          "predicts genuinely subsequent outcomes",
     "Addressed (partial)",
     "12 tests over multimorbidity accrual, incident hospitalization and "
     "functional decline; 0 significant. NOT DELIVERED: medication initiation "
     "(only 43 participants both medication-naive at t1 and with baseline "
     "microbiome) and frailty progression (not measurable in this cohort).",
     "Manuscript 2.12 + Table 9 + Fig. 7; sheet 18_Prospective_Outcomes"),

    ("9", "Replicate the main findings in an independent external cohort",
     "NOT DONE",
     "No second cohort with two-wave metagenomics and a comparable phenotype "
     "battery was available. Stated plainly in Limitations as the strongest "
     "remaining check on a null of this kind.",
     "Manuscript Discussion/Limitations"),

    ("Extra", "Metabolomics and immune/inflammatory profiling (SCFAs, bile "
              "acids, TMAO, CRP, IL-6)",
     "Not attempted",
     "CRP and IL-6 are absent from blood_tests and from every other table "
     "searched; only ferritin, WBC and neutrophils are available as coarse "
     "inflammation proxies. Nightingale metabolomics and serum lipidomics "
     "tables do exist in the cohort and have not been used - a genuine "
     "opportunity, but a new workstream rather than a revision.",
     "Not in manuscript; raised here for discussion"),
]


def add_review_sheets(wb, sheet, title, header, put, note,
                      FONT, DARK, HDRF, SUBF, ACCF, NULF, BORD):
    """Append the review-response sheets, then reorder the workbook."""

    P3 = '0.000'
    P4 = '0.0000'

    # ------------------------------------------------ 00 RESPONSE TO REVIEW
    ws = sheet("00_Response_To_Review")
    r = title(ws, 1, "Point-by-point response to review")
    r = note(ws, r, "Seven of the nine points are addressed with new analysis. "
                    "Two are not done, and are marked as such with the reason "
                    "rather than an argument for why they were unnecessary. "
                    "Every 'Addressed' row points to both the manuscript "
                    "location and the sheet carrying the underlying numbers.")
    r += 1
    r = header(ws, r, ["#", "Point raised", "Status", "What was done / why not",
                       "Where to find it"], [6, 44, 20, 76, 46])
    for num, point, status, detail, where in RESPONSE:
        if status.startswith("NOT"):
            f = {3: ACCF}
        elif "partial" in status or status == "Not attempted":
            f = {3: SUBF}
        else:
            f = None
        r = put(ws, r, [num, point, status, detail, where], fills=f)
    r += 1
    r = note(ws, r, "Headline: the null survives every specification tested. "
                    "99 further tests were run across confounder adjustment, "
                    "within-person change, functional form, extreme exposure, "
                    "effect modification, a multidomain outcome and prospective "
                    "clinical events. None survived correction.")

    # ------------------------------------------------ 00 FIGURE/TABLE INDEX
    ws = sheet("00_Figure_Table_Index")
    r = title(ws, 1, "Figure and table index - numbers as printed by sn-jnl")
    r = note(ws, r, "The printed number, the internal LaTeX label and the PNG "
                    "filename are three different things and no longer line up: "
                    "FIG09 prints as Fig. 3, FIG11 as Fig. 7, because the new "
                    "figures were inserted mid-document. Cite the printed number "
                    "in correspondence; use the label in the .tex.")
    r += 1
    r = header(ws, r, ["Printed as", "LaTeX label", "Image file", "Section",
                       "Contents / status"], [13, 13, 52, 26, 62])
    for printed, lab, f, sec, nt in FIGURES:
        fills = {1: ACCF} if "NEW" in nt else ({1: NULF} if "not printed" in printed else None)
        r = put(ws, r, [printed, lab, f, sec, nt], fills=fills)
    r += 1

    r = title(ws, r, "Tables")
    r = header(ws, r, ["Printed as", "LaTeX label", "Caption", "Section", "Status"],
               [13, 13, 62, 26, 12])
    for printed, lab, cap, sec, st in TABLES:
        r = put(ws, r, [printed, lab, cap, sec, st],
                fills={1: ACCF} if st == "NEW" else None)
    r += 1
    r = note(ws, r, "Highlighted rows were added in response to review. Nine "
                    "figures print (7 main + 3 appendix, one of which is drawn "
                    "in LaTeX rather than supplied as a PNG) and nine tables.")

    # ------------------------------------------------ 13 SELECTION BIAS
    ws = sheet("13_Selection_Bias")
    r = title(ws, 1, "Point 2 - participant flow and included-vs-excluded comparison")
    r = note(ws, r, "Source: participant_flow.py. The comparison contrasts the "
                    "participants entering each cross-lagged test against the "
                    "repeat-visit participants excluded from that same test.")
    r += 1
    r = header(ws, r, ["Stage", "n", "Note"], [46, 12, 74])
    flow = [
        ("Full HPP population", 12014, "Reference cohort"),
        ("Single-visit stratum (clock training)", 5188,
         "Held out deliberately - no participant here contributes to a longitudinal test"),
        ("Repeat-visit stratum", 1835, "Two dated stool metagenomes"),
        ("Diversity metrics at both visits", 1823, "12 lost to uncomputable diversity"),
        ("AgeAccel at both visits", 1817, "6 further lost to missing age at collection"),
        ("Analytic - grip strength", 509, "Requires the phenotype at both visits"),
        ("Analytic - systolic blood pressure", 521, ""),
        ("Analytic - waist circumference", 522, ""),
    ]
    for a, b, c in flow:
        r = put(ws, r, [a, b, c], fmts={2: '#,##0'})
    r += 1
    r = header(ws, r, ["Result", "Value"], [46, 74])
    for a, b in [
        ("Comparisons run", "11 (age, sex, BMI, baseline Shannon, baseline richness)"),
        ("Significant at p < 0.05", "0 of 11"),
        ("With |Cohen's d| >= 0.20", "0 of 11"),
        ("Largest standardised difference", "0.059 (sex, hand-grip sample)"),
        ("All other comparisons", "below 0.035"),
        ("Conclusion", "No detectable selection on measured characteristics. "
                       "IPW and multiple imputation were therefore not applied - "
                       "weighting on covariates that show no imbalance adds "
                       "variance without removing bias."),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})
    r += 1
    r = note(ws, r, "Limit of this result: it addresses selection on MEASURED "
                    "characteristics only, and cannot exclude selection on an "
                    "unmeasured determinant of both attendance and aging.")

    # ------------------------------------------------ 14 CONFOUNDERS
    ws = sheet("14_Confounder_Sensitivity")
    r = title(ws, 1, "Point 3 - sensitivity to lifestyle and technical confounders")
    r = note(ws, r, "Source: confounder_sensitivity.py. Adds diet quality, "
                    "smoking, alcohol, physical activity, sequencing depth and a "
                    "medication-use indicator to each Path A model.")
    r += 1
    r = header(ws, r, ["Exposure", "Outcome", "p (age + sex)",
                       "p (fully adjusted)", "Verdict"], [16, 24, 16, 18, 26])
    for e, o, p1, p2, v in CONFOUNDER:
        r = put(ws, r, [e, o, p1, p2, v],
                fmts={3: P3, 4: P3},
                fills={4: ACCF} if "largest" in v else None)
    r += 1
    for a, b in [
        ("Crossing into significance", "0 of 9"),
        ("Largest coefficient movement",
         "Richness -> systolic BP, beta -0.0493 to -0.0578 (p 0.124 to 0.081)"),
        ("Sample retained", "486-498 adjusted, against 509-522 minimally adjusted (4-5% loss)"),
        ("Antibiotics / PPI / metformin",
         "Not separately coded in the source medication table; absorbed into the "
         "general medication indicator. A data limitation, not a modelling choice."),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})
    r += 1
    r = note(ws, r, "OPEN ITEM: the minimally adjusted column is re-estimated "
                    "inside confounder_sensitivity.py and differs in the third "
                    "decimal from Table 3 for two rows (richness->grip 0.226 vs "
                    "0.201; richness->BP 0.124 vs 0.146). Confirm both pipelines "
                    "build the analytic frame identically before submission.")

    # ------------------------------------------------ 15 CLOCK VALIDATION
    ws = sheet("15_Clock_Validation")
    r = title(ws, 1, "Point 4 - clock calibration, stratified performance, artefacts")
    r = note(ws, r, "Source: clock_validation.py, an independent reimplementation "
                    "of the training-frame construction. It was gated on a "
                    "reproduction check and its stratified results were treated as "
                    "uninterpretable until that check passed.")
    r += 1
    r = header(ws, r, ["Reproduction check", "Reproduced", "Published", "Difference"],
               [40, 16, 16, 16])
    r = put(ws, r, ["Training frame size", 5188, 5188, 0], fmts={2: '#,##0', 3: '#,##0'})
    r = put(ws, r, ["Cross-validated r", 0.3747, 0.3760, 0.0013],
            fmts={2: P4, 3: P4, 4: P4}, fills={4: ACCF})
    r = note(ws, r, "Tolerance was 0.02; the check passed at 0.0013.")
    r += 1
    r = header(ws, r, ["Calibration", "Value", "Interpretation"], [30, 16, 62])
    for a, b, c in [
        ("Slope (predicted on chronological)", 0.1509,
         "Predictions span roughly a seventh of the range of true age"),
        ("Intercept (years)", 42.8536, ""),
        ("Mean residual, youngest decile", 8.92, "Ages 38.0-41.0; over-predicts the young"),
        ("Mean residual, oldest decile", -12.62, "Ages 61.0-70.0; under-predicts the old"),
    ]:
        r = put(ws, r, [a, b, c], fmts={2: P4})
    r = note(ws, r, "Compression is expected for a predictor at r = 0.376 and does "
                    "NOT propagate downstream: AgeAccel is the residual of predicted "
                    "on chronological age, orthogonal to age by construction.")
    r += 1
    r = header(ws, r, ["Stratum", "Range", "n", "r", "MAE (years)"],
               [26, 12, 12, 12, 14])
    for s, rng, n, rr, mae in CLOCK_STRATA:
        r = put(ws, r, [s, rng, n, rr, mae], fmts={3: '#,##0', 4: P3, 5: '0.00'},
                bold=(s == "All participants"))
    r = note(ws, r, "Within-tertile r is attenuated by range restriction and is not "
                    "comparable to the full-range value; MAE is the interpretable "
                    "column within strata. The middle tertile's low MAE (2.59) is "
                    "compression, not better prediction.")
    r += 1
    r = header(ws, r, ["Technical artefact", "Result"], [30, 78])
    for a, b in [
        ("Sequencing depth", "residual vs raw read count r = -0.0056, p = 0.689, n = 5,188"),
        ("Batch", "139 runs with >=10 samples; batch mean residuals -4.00 to +3.28 yr "
                  "against residual SD 6.93 yr (most extreme run = 0.58 SD)"),
        ("Leakage", "One sample per training participant, zero overlap with the "
                    "longitudinal stratum; r changes by 0.009 under participant-grouped CV"),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})

    # ------------------------------------------------ 16 MULTIDOMAIN
    ws = sheet("16_Multidomain_Composite")
    r = title(ws, 1, "Point 5 - multidomain composite of biological aging")
    r = note(ws, r, "Source: multidomain_aging.py. Seven domains (metabolic, renal, "
                    "hepatic, inflammatory, cardiovascular, body composition, "
                    "physical function) from 17 constituent measures.")
    r += 1
    r = header(ws, r, ["Construction", "Value"], [40, 74])
    for a, b in [
        ("Domains required per participant", ">= 4 of 7"),
        ("Person-visits retained", "3,058"),
        ("Standardisation", "Both visits against VISIT 1 parameters, so real change "
                            "survives; per-visit z-scoring would force mean change to "
                            "zero by construction and test an artefact"),
        ("Validation gate", "r = 0.308 with chronological age, p = 6.28e-12, n = 477 "
                            "(pre-specified threshold r >= 0.15)"),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})
    r += 1
    r = header(ws, r, ["Direction", "Model", "beta (std)", "p", "n"],
               [12, 42, 14, 12, 10])
    for d, m, b, p, n in MULTIDOMAIN:
        r = put(ws, r, [d, m, b, p, n], fmts={3: P4, 4: P3})
    r += 1
    for a, b in [
        ("FDR", "All six models (three forward, three reverse) null; every q > 0.59"),
        ("Equivalence bounds", "0.075 to 0.160 standardised units"),
        ("Sample constraint", "n = 354, below the main models' 509-522, because it "
                              "requires BOTH composite availability and leak-free "
                              "AgeAccel (523-participant ceiling)"),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})

    # ------------------------------------------------ 17 NONLINEAR
    ws = sheet("17_Nonlinear_Interactions")
    r = title(ws, 1, "Point 7 - nonlinearity, extremes, change coupling, effect modification")
    r = note(ws, r, "Source: nonlinear_interaction_effects.py. Four families across "
                    "the nine microbiome-phenotype pairings. BH correction applied "
                    "WITHIN each family.")
    r += 1
    r = header(ws, r, ["Family", "Specification", "Tests", "FDR significant",
                       "Smallest q"], [24, 40, 10, 16, 18])
    tot = 0
    for f, s, n, sig, q in NONLINEAR:
        tot += n
        r = put(ws, r, [f, s, n, sig, q])
    r = put(ws, r, ["TOTAL", "", tot, 0, ""], bold=True, fills={1: SUBF, 3: SUBF, 4: SUBF})
    r += 1
    for a, b in [
        ("Modifiers tested", "Chronological age, sex, metabolic health (glucose + "
                             "HbA1c composite), medication use, baseline disease burden"),
        ("Strongest nominal signal", "Richness x age -> waist circumference, p = 0.011, "
                                     "q = 0.246 across the 45-test family"),
        ("Frailty", "NOT TESTABLE - the cohort contains no validated frailty index. "
                    "Recorded as unexamined rather than excluded."),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})
    r += 1
    r = note(ws, r, "Fig. A3 renders the strongest interaction as a fitted response "
                    "surface beside the same model with the interaction zeroed. The "
                    "two differ by at most 0.25 SD of the outcome, reached only at "
                    "the joint extremes of both axes. An independent refit for that "
                    "figure, with a smaller covariate set, reproduced p = 0.011.")

    # ------------------------------------------------ 18 PROSPECTIVE
    ws = sheet("18_Prospective_Outcomes")
    r = title(ws, 1, "Point 8 - baseline microbiome vs genuinely subsequent events")
    r = note(ws, r, "Source: prospective_outcomes.py. Exposure measured at visit 1 "
                    "only; participants already meeting each outcome definition at "
                    "visit 1 are excluded, so only new events count.")
    r += 1
    r = header(ws, r, ["Outcome", "Exposure", "Effect", "Estimate", "p", "n", "Events"],
               [32, 12, 10, 12, 12, 10, 10])
    for o, e, t, est, p, n, ev in PROSPECTIVE:
        r = put(ws, r, [o, e, t, est, p, n, ev], fmts={4: P3, 5: P3})
    r += 1
    for a, b in [
        ("Tests / FDR significant", "12 / 0 (smallest q = 0.860)"),
        ("Multimorbidity definition", "New diagnoses across 15 curated conditions, each "
                                      "contributing only participants free of it at visit 1; "
                                      "participants at risk for < 5 of 15 dropped, leaving "
                                      "7,444 with adequate coverage (11.1% acquired >= 1)"),
        ("Hospitalization", "Explicit hospitalization-since-recruitment field; 1,699 at "
                            "risk, 4.1% hospitalised subsequently"),
        ("Functional decline", "Grip strength fall >= 10% between visits; 42.0% of 2,509"),
        ("Medication initiation", "NOT RUN - only 43 participants were both medication-naive "
                                  "at visit 1 and had baseline microbiome, below the "
                                  "threshold of 150. Reported rather than omitted."),
        ("Frailty progression", "NOT MEASURABLE - no validated frailty index in this cohort"),
    ]:
        r = put(ws, r, [a, b], fills={1: SUBF})
    r += 1
    r = note(ws, r, "Why curated phenotypes rather than raw diagnosis counts: "
                    "medical_conditions.parquet mixes a comprehensive baseline intake "
                    "(mean 4.06 diagnosis rows per participant) with lighter follow-up "
                    "phone contacts (mean 1.07-1.46), and visit_n is a per-participant "
                    "relative rank, so raw counts are not comparable across visits. "
                    "Counting them directly produced a spurious mean DECREASE of 2.02 "
                    "conditions. That approach was discarded.")

    # Reorder: README, then the index, then the analysis sheets in number
    # order, with the regeneration script last. move_sheet takes an offset
    # relative to the sheet's CURRENT position, so it has to be computed from
    # wb.sheetnames.index() at the moment of the call -- an offset derived
    # from len(sheetnames) overshoots and openpyxl silently leaves the sheet
    # where it was, which is how the index ended up buried mid-workbook.
    def move_to(name, target):
        if name not in wb.sheetnames:
            return
        cur = wb.sheetnames.index(name)
        if cur != target:
            wb.move_sheet(name, offset=target - cur)

    move_to("00_Response_To_Review", 1)          # what a reviewer opens first
    move_to("00_Figure_Table_Index", 2)
    move_to("Regeneration script", len(wb.sheetnames) - 1)  # last

    assert wb.sheetnames[:3] == ["README", "00_Response_To_Review",
                                 "00_Figure_Table_Index"], wb.sheetnames[:4]
    assert wb.sheetnames[-1] == "Regeneration script", wb.sheetnames[-3:]
    return wb
