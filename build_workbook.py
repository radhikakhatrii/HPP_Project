#!/usr/bin/env python3
"""
build_workbook.py
===============================================================================
Builds the consolidated statistics workbook for the HPP gut-microbiome / aging
project.

PROVENANCE: values here were transcribed from verified pipeline output (the
final systolic-corrected run). Derived quantities (MDE, disattenuated bounds,
Spearman-Brown factors, samples-needed) are written as EXCEL FORMULAS so they
recalculate from their inputs rather than being frozen.

To regenerate authoritatively from the CSVs on the analysis machine instead of
from these transcriptions, see the README sheet.
===============================================================================
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = r"C:\Users\radhi\OneDrive\Desktop\Radhika\MBZUAI\HPP_Analysis_Statistics.xlsx"

FONT = "Arial"
DARK = "0D2B3E"
HDRF = PatternFill("solid", fgColor=DARK)
SUBF = PatternFill("solid", fgColor="DCE6EC")
ACCF = PatternFill("solid", fgColor="FDEBD8")
NULF = PatternFill("solid", fgColor="EEF3F6")
THIN = Side(style="thin", color="B7C6D0")
BORD = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

wb = Workbook()
wb.remove(wb.active)


def sheet(name):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    return ws


def title(ws, row, text, span=8):
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(name=FONT, size=13, bold=True, color=DARK)
    return row + 2


def header(ws, row, cols, widths=None):
    for i, h in enumerate(cols, start=1):
        c = ws.cell(row=row, column=i, value=h)
        c.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        c.fill = HDRF
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORD
    ws.row_dimensions[row].height = 30
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
    return row + 1


def put(ws, row, values, fills=None, bold=False, fmts=None):
    for i, v in enumerate(values, start=1):
        c = ws.cell(row=row, column=i, value=v)
        c.font = Font(name=FONT, size=10, bold=bold)
        c.border = BORD
        c.alignment = Alignment(vertical="center",
                                horizontal="left" if isinstance(v, str) else "right")
        if fills and fills.get(i):
            c.fill = fills[i]
        if fmts and fmts.get(i):
            c.number_format = fmts[i]
    return row + 1


def note(ws, row, text, col=1):
    c = ws.cell(row=row, column=col, value=text)
    c.font = Font(name=FONT, size=9, italic=True, color="5A7184")
    c.alignment = Alignment(wrap_text=True, vertical="top")
    return row + 1


# =========================================================== README
ws = sheet("README")
ws.column_dimensions["A"].width = 30
ws.column_dimensions["B"].width = 105
r = title(ws, 1, "HPP Gut Microbiome and Functional Aging — Consolidated Statistics")
rows = [
    ("Project", "Cross-lagged panel analysis testing whether the gut microbiome precedes functional aging, extending the HPP archetype analysis (Li et al., arXiv:2510.12384)."),
    ("Cohort", "Human Phenotype Project. 1,823 participants with two dated stool metagenomes, median 2.11 years apart."),
    ("Headline result", "0 of 18 cross-lagged coefficients significant. 0 of 18 incident-disease tests significant after correction across the full family. Bounded null."),
    ("Headline positive", "Two-year test-retest reliability of microbiome-derived aging measures, plus an a priori Spearman-Brown prediction (1.161) confirmed at 1.167."),
    ("", ""),
    ("DATA VINTAGE", "All values reflect the FINAL systolic-corrected run. An earlier version of the pipeline silently used diastolic blood pressure; every blood-pressure number here supersedes that."),
    ("TRANSCRIPTION", "Values were transcribed from verified pipeline output. Derived columns (MDE, disattenuated bounds, Spearman-Brown, samples-needed) are live Excel formulas, not frozen numbers."),
    ("FORMULA NOTE", "Formula cells will show as blank until opened in Excel, which recalculates on load. This build environment had no spreadsheet engine available to pre-cache them."),
    ("", ""),
    ("TO REGENERATE FROM SOURCE", "On the analysis machine, run the script printed on the 'Regeneration script' sheet. It rebuilds every table directly from the result CSVs, removing any transcription risk."),
    ("", ""),
    ("SOURCE FILES", "results_cross_lagged/: cross_lag_all_paths.csv, table2_crosslag.csv, incident_disease.csv, path_b_asymmetry.csv, diversity_icc.csv, visit_intervals.csv"),
    ("", "results_longitudinal/: stability_summary.csv, species_stability.csv, trait_state_disease.csv, stability_weighted_clock.csv"),
]
for k, v in rows:
    c1 = ws.cell(row=r, column=1, value=k)
    c1.font = Font(name=FONT, size=10, bold=True, color=DARK)
    c1.alignment = Alignment(vertical="top")
    c2 = ws.cell(row=r, column=2, value=v)
    c2.font = Font(name=FONT, size=10)
    c2.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[r].height = 30 if v else 8
    r += 1

r += 1
r = title(ws, r, "Sheet index")
idx = [
    ("00_Response_To_Review", "Point-by-point response to the 9 review points - START HERE"),
    ("00_Figure_Table_Index", "Printed figure/table numbers, LaTeX labels, image filenames"),
    ("01_Cohort_Design", "Sample sizes, intervals, cohort characteristics"),
    ("02_Clock_Performance", "Microbiome age clock accuracy and the feature-selection experiment"),
    ("03_Reliability_ICC", "Test-retest ICC for all microbiome measures"),
    ("04_CrossLag_Full", "All 18 cross-lagged coefficients with CIs, R-squared, FDR"),
    ("05_Power_MDE", "Minimum detectable effects, observed and disattenuated"),
    ("06_Incident_Disease", "Prospective incident-disease logistic models, 18 tests"),
    ("07_TraitState_Disease", "AgeAccel vs 12 phenotypes plus comorbidity count"),
    ("08_PathB_Asymmetry", "Host aging to microbiome-shift models"),
    ("09_Reliability_Theory", "Spearman-Brown, attenuation, samples needed"),
    ("10_Species_Stability", "Per-species within-person stability summary"),
    ("11_Audit_Verification", "Independent re-derivation checks"),
    ("12_Bugs_Fixed", "Defects found and corrected during verification"),
    ("13_Selection_Bias", "Point 2 - participant flow, included vs excluded"),
    ("14_Confounder_Sensitivity", "Point 3 - diet, smoking, alcohol, activity, depth, medication"),
    ("15_Clock_Validation", "Point 4 - calibration, stratified performance, batch/depth"),
    ("16_Multidomain_Composite", "Point 5 - seven-domain aging score"),
    ("17_Nonlinear_Interactions", "Point 7 - 72 tests of form, extremes, modification"),
    ("18_Prospective_Outcomes", "Point 8 - 12 tests, baseline microbiome to later events"),
    ("Regeneration script", "Python to rebuild this workbook from source CSVs"),
]
r = header(ws, r, ["Sheet", "Contents"], [30, 105])
for a, b in idx:
    r = put(ws, r, [a, b])

# =========================================================== 01 COHORT
ws = sheet("01_Cohort_Design")
r = title(ws, 1, "Cohort and study design")
r = header(ws, r, ["Quantity", "Value", "Unit", "Source / note"], [42, 16, 14, 62])
coh = [
    ("CLOCK-TRAINING STRATUM", None, None, None),
    ("Participants (single visit only)", 5188, "n", "Disjoint from longitudinal stratum; ensures leak-free clock"),
    ("Species columns in source parquet", 2088, "n", "metaphlan_abundance_species.parquet"),
    ("Species retained (prevalence > 5%)", 511, "n", "Feature set for the clock"),
    ("LONGITUDINAL STRATUM", None, None, None),
    ("Repeat-visit participants", 1835, "n", "stability_summary.csv"),
    ("Pairs with diversity at both visits", 1823, "n", "diversity_icc.csv"),
    ("Pairs with AgeAccel, reliability pipeline", 1817, "n", "stability_summary.csv n_pairs"),
    ("Pairs with AgeAccel, cross-lag pipeline", 523, "n", "Leak-free clock; OPEN ITEM — see note below"),
    ("Median visit interval", 2.1136208, "years", "visit_intervals.csv, n=1,834"),
    ("Interval IQR lower", 2.01, "years", ""),
    ("Interval IQR upper", 2.27, "years", ""),
    ("CROSS-LAG TEST SAMPLES", None, None, None),
    ("Hand grip strength", 509, "pairs", "Requires phenotype at both visits"),
    ("Systolic blood pressure", 521, "pairs", ""),
    ("Waist circumference", 522, "pairs", ""),
    ("PHENOTYPE DISTRIBUTIONS", None, None, None),
    ("Shannon diversity, visit 1 (mean)", 3.570, "", "table1_characteristics.csv"),
    ("Shannon diversity, visit 1 (SD)", 0.586, "", ""),
    ("Shannon diversity, visit 2 (mean)", 3.591, "", ""),
    ("Shannon diversity, visit 2 (SD)", 0.594, "", ""),
    ("Species richness, visit 1 (mean)", 179.0, "", ""),
    ("Species richness, visit 1 (SD)", 42.6, "", ""),
    ("Age at visit 1, analytic sample (mean)", 52.47, "years", "Audit: (visit-1 year) minus DOB"),
    ("FULL HPP POPULATION (reference)", None, None, None),
    ("Participants", 12014, "n", "population.parquet"),
    ("Age (mean)", 50.4, "years", "2020 minus DOB, n=7,383"),
    ("Age (SD)", 7.5, "years", ""),
    ("Sex, female", 52.5, "%", ""),
]
for a, b, c, d in coh:
    if b is None:
        r = put(ws, r, [a, "", "", ""], fills={1: SUBF, 2: SUBF, 3: SUBF, 4: SUBF}, bold=True)
    else:
        r = put(ws, r, [a, b, c, d], fmts={2: "0.####"})
r += 1
r = note(ws, r, "OPEN ITEM: AgeAccel is available for 1,817 pairs in the reliability pipeline but only 523 in the cross-lag pipeline. 523 x 2 = 1,046, close to the 1,050 non-null rows reported, which suggests AgeAccel is computed after a merge that has already dropped other participants. If recoverable, the AgeAccel cross-lag tests would gain roughly 3.5x sample.")

# =========================================================== 02 CLOCK
ws = sheet("02_Clock_Performance")
r = title(ws, 1, "Microbiome age clock")
r = header(ws, r, ["Metric", "Value", "Note"], [40, 16, 78])
clock = [
    ("Cross-validated Pearson r", 0.376, "Five-fold CV, out-of-fold predictions"),
    ("R-squared (squared Pearson r)", 0.142, "Value annotated on the published figure"),
    ("R-squared (1 - SSres/SStot)", 0.141, "Lower because predictions are compressed toward the cohort mean"),
    ("Mean absolute error", 5.61, "years"),
    ("Training n", 5188, "Single-visit participants only"),
    ("Features", 511, "CLR-transformed species, prevalence > 5%"),
    ("Model", "LightGBM", "Gradient-boosted regression"),
]
for a, b, c in clock:
    r = put(ws, r, [a, b, c], fmts={2: "0.###"})

r += 2
r = title(ws, r, "Feature-selection experiment (separate pipeline, longitudinal script)")
r = header(ws, r, ["Method", "Features", "Clock r", "MAE (yr)", "ICC", "Retest r", "SD between", "SD within"],
           [38, 11, 11, 11, 11, 11, 12, 12])
r = put(ws, r, ["A: age-correlation only", 278, 0.3754237, 5.8444669, 0.4893649, 0.4805199, 2.1627873, 2.2092900],
        fmts={3: "0.0000", 4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0000"})
r = put(ws, r, ["B: age-correlation + stability", 139, 0.3204284, 5.9988104, 0.4817661, 0.4736887, 1.9723248, 2.0456120],
        fmts={3: "0.0000", 4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0000"})
r += 1
r = note(ws, r, "Informative negative: filtering to temporally stable species DEGRADED prediction (r 0.375 to 0.320) without improving reliability (ICC 0.489 to 0.482). Temporal stability is not a useful feature-selection criterion here.")

# =========================================================== 03 ICC
ws = sheet("03_Reliability_ICC")
r = title(ws, 1, "Test-retest reliability over the two-year interval")
r = header(ws, r, ["Measure", "ICC(3,1)", "Pearson retest r", "n pairs", "Source"], [30, 14, 18, 12, 46])
icc = [
    ("Microbiome AgeAccel", 0.4825361, 0.4789516, 1817, "stability_summary.csv"),
    ("Shannon diversity", 0.6372045, None, 1817, "stability_summary.csv"),
    ("Species richness", 0.7758597, None, 1817, "stability_summary.csv"),
    ("Shannon diversity", 0.6259689, 0.6260206, 1823, "diversity_icc.csv (independent recompute)"),
    ("Species richness", 0.7698515, 0.7698868, 1823, "diversity_icc.csv (independent recompute)"),
    ("Microbiome predicted age", None, 0.5426139, 1817, "stability_summary.csv"),
]
for a, b, c, d, e in icc:
    r = put(ws, r, [a, b, c, d, e], fmts={2: "0.0000", 3: "0.0000"})

r += 2
r = title(ws, r, "Variance decomposition, AgeAccel")
r = header(ws, r, ["Component", "Value", "Unit"], [30, 14, 14])
r = put(ws, r, ["Between-person SD", 1.9913405, "years"], fmts={2: "0.0000"})
r = put(ws, r, ["Within-person SD", 2.0621518, "years"], fmts={2: "0.0000"})
r += 1
r = note(ws, r, "Roughly half the variance in AgeAccel is between persons and persistent; half is within-person fluctuation and measurement error. Values reported for AgeAccel in the manuscript use the stability_summary figure (0.483). The diversity_icc recompute uses a slightly different pair set (1,823 vs 1,817), which accounts for the small difference in the Shannon and richness rows.")

# =========================================================== 04 CROSSLAG
ws = sheet("04_CrossLag_Full")
r = title(ws, 1, "All 18 cross-lagged coefficients (systolic-corrected run)")
r = header(ws, r, ["Path", "Exposure (t1)", "Outcome (t2)", "Dataset", "Beta (std)", "SE",
                   "p", "q (FDR)", "n", "Model R2", "95% CI low", "95% CI high", "MDE 80% power"],
           [7, 30, 30, 17, 11, 10, 10, 10, 8, 10, 12, 12, 14])
CL = [
    ("A", "shannon", "hand_grip_strength_right", "hand_grip", 0.0088, 0.0162, 0.5872, 0.8481, 509, 0.8716, -0.0230, 0.0406),
    ("B", "hand_grip_strength_right", "shannon", "hand_grip", -0.0125, 0.0594, 0.8328, 0.8481, 509, 0.3941, -0.1292, 0.1041),
    ("A", "shannon", "sitting_blood_pressure_systolic", "blood_pressure", -0.0243, 0.0322, 0.4504, 0.8481, 521, 0.4809, -0.0876, 0.0390),
    ("B", "sitting_blood_pressure_systolic", "shannon", "blood_pressure", 0.0075, 0.0380, 0.8426, 0.8481, 521, 0.3911, -0.0671, 0.0822),
    ("A", "shannon", "waist_circumference", "anthropometrics", 0.0102, 0.0223, 0.6482, 0.8481, 522, 0.7538, -0.0336, 0.0539),
    ("B", "waist_circumference", "shannon", "anthropometrics", -0.0293, 0.0376, 0.4367, 0.8481, 522, 0.3945, -0.1032, 0.0446),
    ("A", "richness", "hand_grip_strength_right", "hand_grip", 0.0206, 0.0161, 0.2009, 0.8481, 509, 0.8720, -0.0110, 0.0522),
    ("B", "hand_grip_strength_right", "richness", "hand_grip", -0.0095, 0.0498, 0.8481, 0.8481, 509, 0.5778, -0.1074, 0.0883),
    ("A", "richness", "sitting_blood_pressure_systolic", "blood_pressure", -0.0466, 0.0320, 0.1458, 0.8481, 521, 0.4824, -0.1095, 0.0163),
    ("B", "sitting_blood_pressure_systolic", "richness", "blood_pressure", 0.0375, 0.0319, 0.2400, 0.8481, 521, 0.5725, -0.0251, 0.1002),
    ("A", "richness", "waist_circumference", "anthropometrics", -0.0146, 0.0220, 0.5073, 0.8481, 522, 0.7539, -0.0578, 0.0286),
    ("B", "waist_circumference", "richness", "anthropometrics", 0.0080, 0.0314, 0.7980, 0.8481, 522, 0.5754, -0.0536, 0.0696),
    ("A", "gut_ageaccel", "hand_grip_strength_right", "hand_grip", 0.0035, 0.0160, 0.8257, 0.8481, 509, 0.8716, -0.0279, 0.0350),
    ("B", "hand_grip_strength_right", "gut_ageaccel", "hand_grip", 0.1143, 0.0658, 0.0832, 0.8481, 509, 0.2545, -0.0151, 0.2436),
    ("A", "gut_ageaccel", "sitting_blood_pressure_systolic", "blood_pressure", 0.0075, 0.0319, 0.8154, 0.8481, 521, 0.4803, -0.0553, 0.0702),
    ("B", "sitting_blood_pressure_systolic", "gut_ageaccel", "blood_pressure", -0.0376, 0.0420, 0.3712, 0.8481, 521, 0.2554, -0.1202, 0.0449),
    ("A", "gut_ageaccel", "waist_circumference", "anthropometrics", -0.0085, 0.0221, 0.6989, 0.8481, 522, 0.7538, -0.0519, 0.0348),
    ("B", "waist_circumference", "gut_ageaccel", "anthropometrics", -0.0116, 0.0417, 0.7810, 0.8481, 522, 0.2547, -0.0935, 0.0703),
]
first = r
for row in CL:
    f = {1: NULF} if row[0] == "A" else None
    r = put(ws, r, list(row) + [f"=F{r}*2.8"],
            fills=f, fmts={5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0000",
                           10: "0.0000", 11: "0.0000", 12: "0.0000", 13: "0.0000"})
ws.freeze_panes = f"A{first}"
r += 1
r = header(ws, r, ["Summary", "Value", "Note"], None)
summ = [
    ("Tests run", 18, "3 microbiome metrics x 3 phenotypes x 2 directions"),
    ("Significant at nominal alpha = 0.05", 0, "Smallest p = 0.0832"),
    ("Significant after BH-FDR", 0, "All 18 q-values identical at 0.8481"),
    ("Expected false positives at alpha 0.05", 0.9, "18 x 0.05"),
    ("Smallest p-value", 0.0832, "Path B: grip strength predicting AgeAccel"),
    ("Largest p-value", 0.8481, "Path B: grip strength predicting richness"),
    ("Beta range, minimum", -0.0466, "Path A: richness predicting systolic BP"),
    ("Beta range, maximum", 0.1143, "Path B: grip strength predicting AgeAccel"),
    ("Autoregressive beta, minimum", 0.4915, "Positive control"),
    ("Autoregressive beta, maximum", 0.8137, "Positive control"),
    ("Autoregressive largest p", 1.841672e-32, "All autoregressive paths overwhelmingly significant"),
]
for a, b, c in summ:
    r = put(ws, r, [a, b, c], fmts={2: "0.000000"})
r += 1
r = note(ws, r, "All 18 beta values were independently verified against the regenerated cross-lag heatmap figure, an independent rendering of the same source data. BH monotonicity assigns every test the same q-value because the largest raw p (0.8481) is itself the rank-18 value.")

# =========================================================== 05 POWER
ws = sheet("05_Power_MDE")
r = title(ws, 1, "Minimum detectable effects — Path A (microbiome to aging)")
r = note(ws, r, "MDE = SE x 2.80 (two-sided alpha 0.05, 80% power). Disattenuated = MDE / ICC, correcting for exposure measurement error.")
r += 1
r = header(ws, r, ["Exposure", "Outcome", "Exposure ICC", "Observed |beta|", "SE", "MDE observed", "MDE disattenuated"],
           [22, 30, 14, 15, 11, 15, 18])
PA = [
    ("Shannon", "Hand grip strength", 0.626, 0.0088, 0.0162),
    ("Shannon", "Systolic blood pressure", 0.626, 0.0243, 0.0322),
    ("Shannon", "Waist circumference", 0.626, 0.0102, 0.0223),
    ("Richness", "Hand grip strength", 0.770, 0.0206, 0.0161),
    ("Richness", "Systolic blood pressure", 0.770, 0.0466, 0.0320),
    ("Richness", "Waist circumference", 0.770, 0.0146, 0.0220),
    ("AgeAccel", "Hand grip strength", 0.483, 0.0035, 0.0160),
    ("AgeAccel", "Systolic blood pressure", 0.483, 0.0075, 0.0319),
    ("AgeAccel", "Waist circumference", 0.483, 0.0085, 0.0221),
]
for a, b, c, d, e in PA:
    r = put(ws, r, [a, b, c, d, e, f"=E{r}*2.8", f"=F{r}/C{r}"],
            fmts={3: "0.000", 4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0000"})
r += 1
r = note(ws, r, "Conclusion: true standardised microbiome-to-aging effects above roughly 0.06 to 0.18 are inconsistent with these data at 80% power. Smaller effects are not excluded. Bounds are mildly conservative because only the exposure is disattenuated; the outcomes also carry measurement error.")

r += 2
r = title(ws, r, "Path B standard errors (less well powered)")
r = header(ws, r, ["Exposure", "Outcome", "SE", "MDE observed"], [30, 22, 11, 15])
PB = [
    ("Hand grip strength", "Shannon", 0.0594), ("Systolic BP", "Shannon", 0.0380),
    ("Waist circumference", "Shannon", 0.0376), ("Hand grip strength", "Richness", 0.0498),
    ("Systolic BP", "Richness", 0.0319), ("Waist circumference", "Richness", 0.0314),
    ("Hand grip strength", "AgeAccel", 0.0658), ("Systolic BP", "AgeAccel", 0.0420),
    ("Waist circumference", "AgeAccel", 0.0417),
]
for a, b, c in PB:
    r = put(ws, r, [a, b, c, f"=C{r}*2.8"], fmts={3: "0.0000", 4: "0.0000"})
r += 1
r = note(ws, r, "Path B MDE range 0.088 to 0.184 on the observed scale. The Path B null is correspondingly weaker evidence and the manuscript does not rely on it.")

# =========================================================== 06 INCIDENT
ws = sheet("06_Incident_Disease")
r = title(ws, 1, "Prospective incident-disease models (prevalent cases excluded at baseline)")
r = header(ws, r, ["Phenotype", "Predictor", "n", "Incident cases", "OR", "CI low", "CI high",
                   "p", "q (all tests)", "Covariates", "Full adj.", "EPV"],
           [26, 11, 8, 14, 10, 10, 10, 11, 13, 18, 10, 8])
INC = [
    ("hypertension", "richness", 669, 36, 0.6170, 0.4386, 0.8680, 0.005547, 0.099846, "sex+bmi_t1", "No", 9.0),
    ("Abdominal_Adiposity", "richness", 578, 217, 0.8195, 0.6909, 0.9720, 0.022236, 0.101560, "sex", "No", 72.3),
    ("abdominal_adiposity", "richness", 578, 217, 0.8195, 0.6909, 0.9720, 0.022236, 0.101560, "sex", "No", 72.3),
    ("hypertension", "shannon", 669, 36, 0.7230, 0.5462, 0.9570, 0.023368, 0.101560, "sex+bmi_t1", "No", 9.0),
    ("osteoporosis", "richness", 432, 25, 0.6811, 0.4573, 1.0145, 0.058842, 0.211831, "sex+age+bmi_t1", "Yes", 5.0),
    ("osa", "shannon", 1281, 215, 1.1083, 0.9505, 1.2922, 0.189682, 0.569046, "sex+age+bmi_t1", "Yes", 53.8),
    ("osa", "richness", 1281, 215, 1.0945, 0.9423, 1.2712, 0.236808, 0.593748, "sex+age+bmi_t1", "Yes", 53.8),
    ("depression", "shannon", 503, 2, 7.3925, 0.1728, 316.1909, 0.296521, 0.593748, "sex+age+bmi_t1", "Yes", 0.4),
    ("diabetes__before_filter", "richness", 428, 30, 1.2140, 0.8433, 1.7478, 0.296874, 0.593748, "sex+age+bmi_t1", "Yes", 6.0),
    ("diabetes", "richness", 438, 31, 1.1830, 0.8299, 1.6864, 0.352818, 0.616798, "sex+age+bmi_t1", "Yes", 6.2),
    ("osteoporosis", "shannon", 432, 25, 0.8387, 0.5671, 1.2403, 0.378242, 0.616798, "sex+age+bmi_t1", "Yes", 5.0),
    ("bmi", "shannon", 220, 35, 0.8688, 0.6121, 1.2333, 0.431423, 0.616798, "sex+age", "Yes", 8.8),
    ("depression", "richness", 503, 2, 1.7709, 0.4080, 7.6868, 0.445465, 0.616798, "sex+age+bmi_t1", "Yes", 0.4),
    ("diabetes__before_filter", "shannon", 428, 30, 1.1037, 0.7451, 1.6348, 0.622627, 0.696278, "sex+age+bmi_t1", "Yes", 6.0),
    ("Abdominal_Adiposity", "shannon", 578, 217, 0.9624, 0.8124, 1.1401, 0.657596, 0.696278, "sex", "No", 72.3),
    ("abdominal_adiposity", "shannon", 578, 217, 0.9624, 0.8124, 1.1401, 0.657596, 0.696278, "sex", "No", 72.3),
    ("bmi", "richness", 220, 35, 0.9840, 0.8072, 1.1995, 0.873170, 0.897217, "sex+age", "Yes", 8.8),
    ("diabetes", "shannon", 438, 31, 1.0141, 0.7012, 1.4665, 0.940861, 0.940861, "sex+age+bmi_t1", "Yes", 6.2),
]
first = r
for row in INC:
    f = {1: ACCF, 8: ACCF, 9: ACCF} if row[7] < 0.05 else None
    r = put(ws, r, list(row), fills=f,
            fmts={5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.000000", 9: "0.000000", 12: "0.0"})
ws.freeze_panes = f"A{first}"
r += 1
notes_inc = [
    "0 of 18 tests significant after FDR correction across the full family (q column above).",
    "Highlighted rows reach nominal p < 0.05 but none survives correction.",
    "EPV = events per model variable. Below 10 the estimate is fragile regardless of p-value; depression rows at EPV 0.4 are uninterpretable and should not be read.",
    "'Full adj.' = No means the full sex+age+BMI set could not be fitted at that sample size and a reduced covariate set was used. Age has only ~61% coverage in this cohort.",
    "Abdominal_Adiposity and abdominal_adiposity are two source files for the same phenotype shipped under different naming conventions; they are not independent tests.",
    "10 of 19 curated conditions never reached a model: fewer than 30 incident cases, or no identifiable disease-free reference category.",
]
for n_ in notes_inc:
    r = note(ws, r, n_)

# =========================================================== 07 TRAIT/STATE
ws = sheet("07_TraitState_Disease")
r = title(ws, 1, "AgeAccel vs disease phenotypes (partial correlations, adjusted for age, sex, BMI)")
r = header(ws, r, ["Outcome", "n", "r visit 1", "p visit 1", "r visit 2", "p visit 2",
                   "r 2-visit mean", "p mean", "q (FDR) mean", "r instability", "p instability", "q (FDR) instab."],
           [24, 8, 11, 11, 11, 11, 13, 11, 13, 13, 13, 14])
TS = [
    ("n_conditions", 1802, 0.048334, 0.040214, 0.042849, 0.068983, 0.053216, 0.023881, None, 0.050292, 0.032800, None),
    ("hypertension", 1463, -0.018551, 0.478305, -0.091727, 0.000443, -0.064890, 0.013048, 0.1566, -0.006104, 0.815558, 0.8897),
    ("abdominal_adiposity", 1802, -0.031471, 0.181762, -0.030698, 0.192741, -0.036276, 0.123721, 0.2474, -0.008293, 0.847000, 0.8472),
    ("diabetes", 1802, 0.043492, 0.064917, 0.024141, 0.305735, 0.039506, 0.093637, 0.2474, -0.018091, 0.442795, 0.8472),
    ("mafld", 1655, -0.047568, 0.053015, -0.020987, 0.393530, -0.040106, 0.102887, 0.2474, -0.016816, 0.494197, 0.8472),
    ("osteoporosis", 1801, 0.027085, 0.250616, 0.043696, 0.063744, 0.041258, 0.080039, 0.2474, -0.009074, 0.808008, 0.8897),
    ("migraine", 1800, 0.037844, 0.108485, 0.026261, 0.265461, 0.037431, 0.112393, 0.2474, -0.027746, 0.239300, 0.8472),
    ("anxiety", 1802, 0.024312, 0.302311, 0.027411, 0.244831, 0.030172, 0.200471, 0.3437, 0.032876, 0.163021, 0.8472),
    ("ckd", 138, 0.058877, 0.492742, 0.063440, 0.459775, 0.070303, 0.412575, 0.6116, -0.074122, 0.387586, 0.8472),
    ("depression", 1802, -0.012721, 0.589423, -0.017232, 0.464746, -0.017467, 0.458682, 0.6116, -0.025002, 0.220000, 0.8472),
    ("ischemic_heart_disease", 1800, 0.020923, 0.374978, 0.007890, 0.973314, 0.012710, 0.589954, 0.6881, 0.000840, 0.971591, 0.9716),
    ("osa", 1799, 0.003190, 0.892441, 0.014762, 0.531491, 0.010456, 0.657629, 0.6881, -0.018880, 0.423535, 0.8472),
    ("sleep_quality", 1762, -0.010634, 0.655553, -0.005742, 0.809673, -0.009570, 0.688107, 0.6881, 0.007740, 0.777395, 0.8897),
]
first = r
fm = {i: "0.000000" for i in range(3, 13)}
for row in TS:
    f = {1: SUBF} if row[0] == "n_conditions" else None
    r = put(ws, r, list(row), fills=f, fmts=fm)
ws.freeze_panes = f"A{first}"
r += 1
r = note(ws, r, "0 of 12 curated phenotypes significant after BH-FDR for either the 2-visit mean or the instability exposure. Minimum q = 0.157 (hypertension, and in the counter-intuitive negative direction).")
r = note(ws, r, "n_conditions (comorbidity count) is the primary outcome and is excluded from the 12-phenotype FDR family. Its nominal association (partial r = +0.053, p = 0.024) is NOT supported by a negative binomial model: beta = +0.0137, p = 0.194.")

# =========================================================== 08 PATH B
ws = sheet("08_PathB_Asymmetry")
r = title(ws, 1, "Path B asymmetry test — does host aging precede microbiome change?")
r = header(ws, r, ["Model", "n", "Beta (std)", "SE", "p", "q (FDR)", "MDE 80%", "Significant"],
           [42, 8, 12, 11, 11, 11, 12, 12])
PBA = [
    ("aging_t1 -> Aitchison shift", 523, 0.04216, 0.04684, 0.368411, 0.491214, 0.13114, "No"),
    ("aging_t1 -> Bray-Curtis shift", 523, 0.09885, 0.04745, 0.037720, 0.104794, 0.13287, "No"),
    ("shannon_t1 -> aging_t2 (adj. aging_t1)", 523, -0.01806, 0.02718, 0.506619, 0.506619, 0.07611, "No"),
    ("richness_t1 -> aging_t2 (adj. aging_t1)", 523, -0.07501, 0.03858, 0.052397, 0.104794, 0.10802, "No"),
]
for row in PBA:
    r = put(ws, r, list(row), fmts={3: "0.00000", 4: "0.00000", 5: "0.000000", 6: "0.000000", 7: "0.00000"})

r += 2
r = title(ws, r, "Composite aging-score coverage (why power did not increase)")
r = header(ws, r, ["Component / group", "n", "Sign", "Note"], [30, 10, 8, 62])
COV = [
    ("hand_grip", 2510, -1, "Negated: grip declines with age"),
    ("blood_pressure", 2542, 1, "Systolic; rises with age"),
    ("anthropometrics", 2533, 1, "Waist circumference"),
    ("Participants with >= 1 component", 2542, None, ""),
    ("Exactly 1 component", 1, None, ""),
    ("Exactly 2 components", 39, None, ""),
    ("Exactly 3 components", 2502, None, "98.4% have all three"),
]
for a, b, c, d in COV:
    r = put(ws, r, [a, b, c, d])
r += 1
r = note(ws, r, "The composite was built to recover participants that single-phenotype tests discard. It did not: missingness across the three phenotypes is correlated, not complementary, so the analysis frame stayed at n=523, statistically identical to the original Path B tests (n=509-522).")

r += 1
r = title(ws, r, "Microbiome compositional shift, visit 1 to visit 2")
r = header(ws, r, ["Distance metric", "n", "Mean", "SD"], [22, 10, 12, 12])
r = put(ws, r, ["Aitchison", 1823, 61.771, 11.434], fmts={3: "0.000", 4: "0.000"})
r = put(ws, r, ["Bray-Curtis", 1823, 0.451, 0.135], fmts={3: "0.000", 4: "0.000"})
r += 1
r = note(ws, r, "The two distance metrics disagree (Aitchison p = 0.368, Bray-Curtis p = 0.038) and neither survives correction. Disagreement between two measures of the same construct is the behaviour of noise, not of a real effect.")
r = note(ws, r, "Held-out replication could not be run: the discovery set (participants with a complete phenotype at both visits) covered 2,541 of 2,542, leaving 0 held-out participants.")

# =========================================================== 09 RELIABILITY THEORY
ws = sheet("09_Reliability_Theory")
r = title(ws, 1, "Spearman-Brown: the a priori prediction")
r = header(ws, r, ["Measure", "ICC, 1 sample", "ICC, 2-visit mean", "Predicted amplification"],
           [24, 16, 20, 22])
SB = [("AgeAccel", 0.483), ("Shannon diversity", 0.626), ("Species richness", 0.770)]
sb_first = r
for a, b in SB:
    r = put(ws, r, [a, b, f"=2*B{r}/(1+B{r})", f"=SQRT(C{r}/B{r})"],
            fmts={2: "0.000", 3: "0.0000", 4: "0.0000"})
r += 1
r = header(ws, r, ["Check", "Value", "Note"], [30, 14, 70])
r = put(ws, r, ["Predicted amplification (AgeAccel)", 1.1613, "sqrt(0.6514 / 0.483)"], fmts={2: "0.0000"})
r = put(ws, r, ["OBSERVED median amplification", 1.167, "Median across disease outcomes"],
        fills={1: ACCF, 2: ACCF, 3: ACCF}, bold=True, fmts={2: "0.000"})
r = put(ws, r, ["Attenuation: 1 visit vs 2-visit mean", 0.1389, "1 - 1/1.1613"], fmts={2: "0.0000"})
r = put(ws, r, ["Attenuation: 1 visit vs error-free", 0.3050, "1 - sqrt(0.483)"], fmts={2: "0.0000"})
r += 1
r = note(ws, r, "The predicted factor was fixed by the ICC before any association was estimated, and could have failed. Caveat: the attenuation formula describes shrinkage of a genuine effect, and most individual disease associations here are null, so the ratio is a median over small and individually uncertain quantities.")

r += 2
r = title(ws, r, "Samples needed for a target reliability")
r = note(ws, r, "k = R(1 - rho) / (rho(1 - R)), where rho is the single-sample ICC and R the target.")
r += 1
r = header(ws, r, ["Measure", "ICC (rho)", "k for R=0.7", "k for R=0.8", "k for R=0.9",
                   "Use (0.7)", "Use (0.8)", "Use (0.9)"],
           [22, 12, 13, 13, 13, 11, 11, 11])
for a, b in [("Species richness", 0.770), ("Shannon diversity", 0.626), ("AgeAccel", 0.483)]:
    r = put(ws, r, [a, b,
                    f"=0.7*(1-B{r})/(B{r}*0.3)", f"=0.8*(1-B{r})/(B{r}*0.2)", f"=0.9*(1-B{r})/(B{r}*0.1)",
                    f"=ROUNDUP(C{r},0)", f"=ROUNDUP(D{r},0)", f"=ROUNDUP(E{r},0)"],
            fmts={2: "0.000", 3: "0.00", 4: "0.00", 5: "0.00"})
r += 1
r = note(ws, r, "Practical reading: richness needs 2 samples for ICC 0.8, Shannon 3, AgeAccel 5. To reach ICC 0.9, AgeAccel needs 10.")

# =========================================================== 10 SPECIES
ws = sheet("10_Species_Stability")
r = title(ws, 1, "Per-species within-person stability across two years")
r = header(ws, r, ["Statistic", "Value"], [34, 16])
for a, b in [("Species with a stability estimate", 509), ("Mean stability", 0.3678),
             ("Minimum", -0.011), ("Maximum", 0.830)]:
    r = put(ws, r, [a, b], fmts={2: "0.0000"})
r += 2
r = title(ws, r, "Selected species (illustrative extract)")
r = header(ws, r, ["Species", "Within-person stability"], [58, 22])
SP = [
    ("Archaea (unclassified) SGB1024", 0.7061512),
    ("Eggerthellaceae bacterium", 0.5417003),
    ("Bifidobacterium bifidum", 0.5351984),
    ("Collinsella bouchesdurhonensis", 0.4287734),
    ("Olsenella sp GAM18", 0.4707484),
    ("Bifidobacterium pseudocatenulatum", 0.4028988),
    ("Bifidobacterium longum", 0.3895093),
    ("Parolsenella catena", 0.3720945),
    ("Collinsella aerofaciens", 0.3410291),
    ("Slackia isoflavoniconvertens", 0.3071140),
    ("Bifidobacterium adolescentis", 0.2943097),
    ("Eggerthella lenta", 0.2760860),
    ("Gordonibacter pamelaeae", 0.2678830),
    ("Methanosphaera stadtmanae", 0.2545224),
    ("Senegalimassilia anaerobia", 0.2506042),
    ("Methanobrevibacter smithii", 0.2488757),
    ("Adlercreutzia equolifaciens", 0.2460793),
    ("Senegalimassilia faecalis", 0.2377661),
    ("Bifidobacterium catenulatum", 0.2184467),
    ("Bifidobacterium angulatum", 0.1857919),
    ("Bifidobacterium animalis", 0.0807708),
]
for a, b in SP:
    r = put(ws, r, [a, b], fmts={2: "0.0000"})
r += 1
r = note(ws, r, "Extract only. The full 509-species table is in results_longitudinal/species_stability.csv. Stability varies more than 70-fold across taxa, so the number of samples needed to study a given species depends heavily on which species it is.")

# =========================================================== 10b REVIEW FOLLOW-UP
ws = sheet("10b_Review_Followup")
r = title(ws, 1, "Follow-up analyses requested at HPP review")
r = note(ws, r, "Three points were raised: whether the age predictor was trained on the same individuals it later scores; whether participant-grouped cross-validation would enlarge the usable sample; and whether an objectively measured continuous endpoint would be better powered than partly self-reported disease. All three were tested. Source: results_cross_lagged/followup_*.csv")
r += 2

r = title(ws, r, "A. Leakage check - is the training set person-disjoint?", span=7)
r = header(ws, r, ["Property", "Value", "Interpretation"], [40, 16, 60])
for a, b, c in [
    ("Training rows", "5,177", "One sample per participant"),
    ("Distinct training participants", "5,177", "Ratio exactly 1.000"),
    ("Overlap with repeat-visit set", "0", "No scored individual is in training"),
]:
    r = put(ws, r, [a, b, c])
r += 1
r = header(ws, r, ["Cross-validation scheme", "r", "R-squared", "MAE (yr)"], [40, 12, 14, 12])
r = put(ws, r, ["Ordinary five-fold", 0.3800, 0.1439, 5.58], fmts={2: "0.0000", 3: "0.0000", 4: "0.00"})
r = put(ws, r, ["Grouped by participant", 0.3706, 0.1365, 5.61], fmts={2: "0.0000", 3: "0.0000", 4: "0.00"})
r = put(ws, r, ["Difference in r", 0.0094, None, None], fmts={2: "0.0000"}, bold=True)
r += 1
r = note(ws, r, "A 0.009 difference confirms no within-person leakage, because there was never more than one sample per training participant to leak. The concern was correctly raised but does not apply to this design.")
r += 2

r = title(ws, r, "B. Person-grouped clock trained across the full population", span=7)
r = header(ws, r, ["Quantity", "Grouped design", "Published disjoint design"], [40, 20, 26])
for a, b, c, fmt in [
    ("Training samples", 6585, 5177, "#,##0"),
    ("Distinct participants", 6058, 5177, "#,##0"),
    ("Clock r", 0.3691, 0.3760, "0.0000"),
    ("Clock MAE (years)", 5.679, 5.61, "0.000"),
    ("Repeat participants with AgeAccel at both visits", 523, 523, "#,##0"),
    ("AgeAccel retest r", 0.4874, 0.4790, "0.0000"),
    ("AgeAccel ICC(3,1)", 0.4874, 0.4825, "0.0000"),
]:
    r = put(ws, r, [a, b, c], fmts={2: fmt, 3: fmt})
r += 1
r = note(ws, r, "The alternative design recovers the identical 523 participants, so the sample ceiling is a genuine data constraint (age at collection is missing for part of the cohort) rather than an artefact of the training split. The two ICC estimates agree to two decimal places, confirming that the reliability figure used to disattenuate the power bounds transfers to the cross-lagged AgeAccel.")
r += 2

r = title(ws, r, "C. Change in BMI - objectively measured continuous endpoint", span=7)
r = note(ws, r, "n = 521 with BMI and AgeAccel at both visits. Mean BMI change +0.183 kg/m2 (SD 1.553, range -9.4 to +11.4). Adjusted for age, sex, baseline BMI and visit interval.")
r += 1
r = header(ws, r, ["Model", "beta (std)", "SE", "p", "q (FDR)", "MDE 80%", "Significant"],
           [46, 12, 10, 12, 12, 12, 12])
DBMI = [
    ("AgeAccel(t1) -> change in BMI", -0.0375, 0.0422, 0.3739, 0.5153, 0.1180, "No"),
    ("Change in AgeAccel <-> change in BMI", 0.0336, 0.0418, 0.4227, 0.5153, 0.1172, "No"),
    ("Change in BMI -> AgeAccel(t2), adj. AgeAccel(t1)", 0.0250, 0.0384, 0.5153, 0.5153, 0.1074, "No"),
    ("AgeAccel(t1) <-> BMI(t1), cross-sectional", -0.1383, 0.0426, 0.0013, 0.0051, 0.1194, "Yes"),
]
for a, b, c, d, e, f, g in DBMI:
    r = put(ws, r, [a, b, c, d, e, f, g],
            fmts={2: "0.0000", 3: "0.0000", 4: "0.0000", 5: "0.0000", 6: "0.0000"})
r += 1
r = note(ws, r, "All three temporal models are null and sit well inside their minimum detectable effects. The single significant association is cross-sectional: higher microbiome age acceleration accompanies lower BMI at the same visit, consistent with the established inverse relationship between adiposity and microbial diversity. An objectively measured endpoint therefore reproduces the temporal null while recovering an expected same-visit association.")

# =========================================================== 11 AUDIT
ws = sheet("11_Audit_Verification")
r = title(ws, 1, "Independent verification")
r = note(ws, r, "An audit script re-derived every headline number directly from the raw parquet files, deliberately importing no pipeline code, so a pipeline bug could not reproduce itself and hide.")
r += 1
r = header(ws, r, ["Check", "Pipeline value", "Independent value", "Verdict"], [46, 18, 20, 16])
AUD = [
    ("Species at prevalence > 5%", "511", "511", "Match"),
    ("Median visit interval (years)", "2.1136", "2.1136", "Match"),
    ("Shannon ICC(3,1)", "0.6260", "0.6260", "Match"),
    ("Richness ICC(3,1)", "0.7699", "0.7699", "Match"),
    ("Shannon mean", "3.570", "3.573", "Match (rounding)"),
    ("Richness mean", "179.0", "178.8", "Match (rounding)"),
    ("Spearman-Brown ICC2 (AgeAccel)", "0.651", "0.6514", "Match"),
    ("Spearman-Brown amplification", "1.161", "1.1613", "Match"),
    ("Cross-lag rows", "18", "18", "Match"),
    ("visit_n by date vs research-stage rank", "-", "agree", "No t1/t2 swap"),
    ("Research-stage alignment", "-", "99.8%", "Pass"),
    ("Age: fixed-2020 vs actual visit year", "-", "+0.06 yr (SD 0.23)", "Approximation valid"),
]
for a, b, c, d in AUD:
    r = put(ws, r, [a, b, c, d])
r += 1
r = note(ws, r, "All structural and numerical checks passed. Separately, all 18 Table 2 beta values were verified against the regenerated heatmap figure, an independent rendering of the same data.")

# =========================================================== 12 BUGS
ws = sheet("12_Bugs_Fixed")
r = title(ws, 1, "Defects found and corrected during verification")
r = header(ws, r, ["Defect", "Impact", "Resolution"], [38, 52, 46])
BUGS = [
    ("Blood pressure used diastolic, not systolic",
     "A word-order mismatch in a column-name list meant no systolic candidate ever matched, so the code fell through to diastolic. Diastolic peaks near the sixth decade then declines, making it a poor aging marker. Affected 6 of 18 cross-lag tests.",
     "Corrected column names; full pipeline re-run. Null held; the one near-threshold signal disappeared."),
    ("Cross-lag heatmap hid 6 of 9 Path B cells",
     "The Path B panel pivoted on the wrong column, collapsing to its diagonal and averaging over microbiome metrics.",
     "Pivot corrected; both panels now render all nine coefficients."),
    ("Missing values counted as disease-positive",
     "In the incident-disease script, ~isin() returns True for NaN, so every missing phenotype value was labelled diseased, inflating prevalence and incidence.",
     "Explicit NaN handling before labelling."),
    ("Duplicate study_id column crash",
     "population.parquet carries both participant_id and its own study_id. Renaming without checking produced two columns of the same name, crashing every phenotype that reached the demographic merge.",
     "Deduplication helper applied at every rename point."),
    ("Age silently absent from all models",
     "population.parquet has only month/year of birth, which the age matcher never recognised, so age entered no model in any run despite appearing in covariate lists.",
     "Age now sourced from sociodemographics DOB, matching the main pipeline (mean 50.4)."),
    ("Visit ordering by research-stage string",
     "Dense-ranking a stage label sorts alphabetically, not chronologically; where labels disagree this silently swaps t1 and t2 and turns incident cases into remissions.",
     "Ordering by collection date; audit confirms both methods now agree."),
    ("Full covariate adjustment discarded tests",
     "Requiring sex+age+BMI dropped hypertension from n=670 to n=160, below the sample floor, so it was silently skipped and produced no result at all.",
     "Fallback ladder: fullest covariate set that clears n>=200, with the set used recorded per test."),
    ("Pipeline runtime",
     "Step 12 fits one causal forest per intervention-phenotype pair (90 forests), each cross-fitting two single-threaded nuisance models.",
     "Histogram-based threaded learners plus configurable skip; 10-30x faster."),
]
for a, b, c in BUGS:
    r = put(ws, r, [a, b, c])
    for col in (1, 2, 3):
        ws.cell(row=r - 1, column=col).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[r - 1].height = 58

# =========================================================== REGEN SCRIPT
ws = sheet("Regeneration script")
ws.column_dimensions["A"].width = 120
r = title(ws, 1, "Rebuild this workbook directly from the source CSVs")
r = note(ws, r, "Run on the analysis machine. This removes any transcription risk by reading the result CSVs directly.")
r += 1
SCRIPT = '''cd ~/studies/HPP_Prototype && python3 - <<'EOF'
import pandas as pd
from pathlib import Path

CL = Path("results_cross_lagged")
LG = Path("results_longitudinal")

sources = {
    "01_CrossLag_Full":      CL / "cross_lag_all_paths.csv",
    "02_CrossLag_Table2":    CL / "table2_crosslag.csv",
    "03_Incident_Disease":   CL / "incident_disease.csv",
    "04_PathB_Asymmetry":    CL / "path_b_asymmetry.csv",
    "05_PathB_Coverage":     CL / "path_b_composite_coverage.csv",
    "06_Diversity_ICC":      CL / "diversity_icc.csv",
    "07_Visit_Intervals":    CL / "visit_intervals.csv",
    "08_MDE_By_Test":        CL / "mde_by_test.csv",
    "09_Direction_Summary":  CL / "causal_direction_summary.csv",
    "10_Table1_Characteristics": CL / "table1_characteristics.csv",
    "11_Stability_Summary":  LG / "stability_summary.csv",
    "12_Species_Stability":  LG / "species_stability.csv",
    "13_TraitState_Disease": LG / "trait_state_disease.csv",
    "14_Weighted_Clock":     LG / "stability_weighted_clock.csv",
}

out = "HPP_Analysis_Statistics_FROM_SOURCE.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    for sheet, path in sources.items():
        if path.exists():
            df = pd.read_csv(path)
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
            print(f"  {sheet:<28} {df.shape}")
        else:
            print(f"  MISSING: {path}")
print("wrote", out)
EOF'''
for line in SCRIPT.split("\n"):
    c = ws.cell(row=r, column=1, value=line)
    c.font = Font(name="Courier New", size=9)
    r += 1

# ===========================================================================
# REVIEW-RESPONSE ADDITIONS (Adinath points 2, 3, 4, 5, 7, 8)
#
# Everything below post-dates the original workbook. Values are transcribed
# from verified pipeline output exactly as the sheets above were; the
# provenance column on each sheet names the script that produced it, so any
# number can be traced back to a specific run.
#
# The figure and table index is built from the manuscript as it now stands.
# sn-jnl numbers appendix floats A1, A2, A3 -- the captions no longer carry
# their own numbers, so these are the numbers that actually print.
# ===========================================================================
from build_workbook_review import add_review_sheets  # noqa: E402

add_review_sheets(wb, sheet, title, header, put, note,
                  FONT, DARK, HDRF, SUBF, ACCF, NULF, BORD)

wb.save(OUT)
print("WROTE", OUT)
print("sheets:", len(wb.sheetnames))
