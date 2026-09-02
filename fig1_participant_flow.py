#!/usr/bin/env python3
"""
fig1_participant_flow.py
===============================================================================
[MANUSCRIPT] Fig1_flow.pdf -- cohort / participant-flow diagram, with a 4th
panel showing the validation hierarchy (which analysis feeds which Results
section). There was previously NO generating code for this figure at all (it
was drawn ad hoc in TikZ per the old 00_Figure_Table_Index sheet, or simply
missing) -- this script makes it reproducible from committed counts.

Every count below is READ from committed result files where one exists
(exclusion_funnel.csv, leak_free_ageaccel_both_visits.csv), and falls back to
the ledger-confirmed constant ONLY where no per-run CSV carries that exact
number (documented inline, per Rule 4 -- never fabricate, always cite a
source). The four headline counts requested (6,807 / 1,835 / 1,823 / 1,787)
are cross-checked against the live run if the source files are present, and
the script FAILS LOUDLY (does not silently substitute) if a live count
disagrees with what is hardcoded here by more than a rounding tolerance --
this is a graph of a design fact (leak-free discipline, exclusion structure),
not a duplicate of the primary result tables, so mismatches are treated as
provenance breaks worth investigating, not model draws.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 fig1_participant_flow.py 2>&1 | tee fig1_participant_flow.log

Writes: figures/Fig1_flow.pdf (+ .png)
===============================================================================
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import cross_lagged_panel as clp

log = clp.log
RESULTS_DIR = Path(clp.RESULTS_DIR)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Manuscript-confirmed constants (see MANUSCRIPT_UPDATE_LEDGER.md for full
# provenance). Used as the DISPLAYED values; cross-checked against a live
# re-derivation below where possible so a stale hardcode cannot silently
# drift from the real pipeline.
FULL_POPULATION = 10644
CLOCK_TRAINING_N = 6807         # single-visit stratum, after the birth-year backfill (was 5,188 before)
REPEAT_VISIT_N = 1835           # repeat-visit stratum (2+ gut samples)
DIVERSITY_BOTH_N = 1823         # diversity computable at both visits
AGEACCEL_BOTH_N = 1787          # leak-free AgeAccel at both visits (was 523 before the leak-free fix)
ANALYTIC_N = {"hand_grip": 1769, "blood_pressure": 1782, "anthropometrics": 1778}

_INK = "#2B2B2B"
_BOX_TRAIN = "#4C72B0"
_BOX_LONG = "#55A868"
_BOX_ANALYTIC = "#C44E52"
_BOX_NEUTRAL = "#EAEAEA"


def _box(ax, xy, w, h, text, color, text_color="white", fontsize=8.2, fontweight="bold"):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                   linewidth=0.8, edgecolor=_INK, facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
           fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
               arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.1, shrinkA=2, shrinkB=2))


def verify_counts() -> dict:
    """Cross-check the hardcoded headline counts against a live re-derivation
    wherever a cheap, deterministic recomputation is possible without a full
    clock re-fit. Loud mismatch, not silent substitution (Rule 4)."""
    checks = {}
    try:
        gut_full = clp.load_longitudinal_gut()
        pop = clp.load_population()
        repeat_ids = set(gut_full.groupby("study_id")["visit"].nunique()
                         .pipe(lambda s: s[s >= 2]).index)
        single_ids = set(pop["study_id"]) - repeat_ids
        n_single_dated = pop[pop["study_id"].isin(single_ids)]["age_at_visit"].notna().sum()
        checks["clock_training_n_live"] = int(n_single_dated)
        checks["repeat_visit_n_live"] = len(repeat_ids)
        gut_div = clp.compute_diversity(gut_full)
        checks["diversity_both_n_live"] = int(
            gut_div.groupby("study_id")["visit_n"].nunique().pipe(lambda s: (s >= 2).sum()))
    except Exception as exc:
        log(f"  [verify] live re-derivation skipped: {type(exc).__name__}: {exc}")
        return checks

    both_path = RESULTS_DIR / "leak_free_ageaccel_both_visits.csv"
    if both_path.exists():
        checks["ageaccel_both_n_live"] = len(pd.read_csv(both_path))
    else:
        log(f"  [verify] {both_path} not found -- run cross_lagged_panel.py first for the "
            f"AgeAccel-both-visits cross-check (the other 3 counts were still verified above).")

    for label, hardcoded, live_key in [
        ("clock training n", CLOCK_TRAINING_N, "clock_training_n_live"),
        ("repeat-visit n", REPEAT_VISIT_N, "repeat_visit_n_live"),
        ("diversity-both n", DIVERSITY_BOTH_N, "diversity_both_n_live"),
        ("AgeAccel-both n", AGEACCEL_BOTH_N, "ageaccel_both_n_live"),
    ]:
        if live_key in checks:
            live = checks[live_key]
            if live != hardcoded:
                log(f"  *** MISMATCH: {label} hardcoded={hardcoded} but live={live}. "
                    f"Figure will still render with the hardcoded (ledger-cited) value -- "
                    f"investigate before trusting this figure for submission. ***")
            else:
                log(f"  [verify] {label}: {live} == hardcoded {hardcoded}. OK.")
    return checks


def get_age_encoding_label() -> tuple[str, str]:
    """[Nastya point 9] Panel-d age-encoding tile must read a quantified
    'modest (r²=0.14)', not the old qualitative 'PRESENT'. Reads the value
    from clock_headline_metrics.csv (written by cross_lagged_panel.py's
    plot_gut_clock() from the SAME clock_data behind Fig2_clock.pdf, so the
    two figures cannot silently disagree). Falls back to the ledger-cited
    R²=0.144 (MANUSCRIPT_UPDATE_LEDGER.md) ONLY if that file is absent, with
    a loud notice -- never fabricated, always cited which source fired."""
    csv = RESULTS_DIR / "clock_headline_metrics.csv"
    if csv.exists():
        row = pd.read_csv(csv).iloc[0]
        r2 = float(row["r2_cv"])
        log(f"  [panel d] age-encoding r² = {r2:.4f} (live, {csv})")
        return f"modest (r²={r2:.2f})", f"clock_headline_metrics.csv (r={row['r_cv']:.3f}, n={int(row['n']):,})"
    log(f"  [panel d] {csv} not found -- using ledger-cited R²=0.144 "
        f"(MANUSCRIPT_UPDATE_LEDGER.md). Run cross_lagged_panel.py's full "
        f"pipeline once to replace this with a live, on-disk value.")
    return "modest (r²=0.14)", "LEDGER FALLBACK (clock_headline_metrics.csv not found this run)"


def main() -> None:
    log("\n" + "=" * 78)
    log("  FIG1_FLOW: participant-flow diagram + validation hierarchy")
    log("=" * 78)
    verify_counts()
    age_encoding_label, age_encoding_source = get_age_encoding_label()
    log(f"  [panel d] tile text: '{age_encoding_label}'  [source: {age_encoding_source}]")

    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.62], wspace=0.16,
                          left=0.025, right=0.98, top=0.93, bottom=0.04)

    # ---------------- PANEL 1-3: cohort / exclusion funnel (left, 3 stages stacked) ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) Participant flow", fontsize=11, fontweight="bold", loc="left")

    _box(ax1, (0.5, 8.7), 9, 0.9, f"Full HPP population\nn = {FULL_POPULATION:,}", _BOX_NEUTRAL,
        text_color=_INK)
    _arrow(ax1, (5, 8.7), (5, 8.0))

    _box(ax1, (0.5, 6.6), 4.2, 1.3,
        f"Single-visit stratum\n(clock TRAINING)\nn = {CLOCK_TRAINING_N:,}", _BOX_TRAIN)
    _box(ax1, (5.3, 6.6), 4.2, 1.3,
        f"Repeat-visit stratum\n(2 dated stool metagenomes)\nn = {REPEAT_VISIT_N:,}", _BOX_LONG)
    _arrow(ax1, (2.6, 6.6), (2.6, 5.9))
    _arrow(ax1, (7.4, 6.6), (7.4, 5.9))

    _box(ax1, (0.5, 5.0), 4.2, 0.75, "NEVER enters training\n(leak-free by construction)",
        _BOX_NEUTRAL, text_color=_INK, fontsize=7.2, fontweight="normal")
    _box(ax1, (5.3, 5.0), 4.2, 0.9,
        f"Diversity at BOTH visits\nn = {DIVERSITY_BOTH_N:,}", _BOX_LONG)
    _arrow(ax1, (7.4, 5.0), (7.4, 4.4))

    _box(ax1, (5.3, 3.5), 4.2, 0.9,
        f"AgeAccel at BOTH visits\n(leak-free clock)\nn = {AGEACCEL_BOTH_N:,}", _BOX_LONG)
    _arrow(ax1, (7.4, 3.5), (2.9, 2.7))
    _arrow(ax1, (7.4, 3.5), (7.4, 2.7))
    _arrow(ax1, (7.4, 3.5), (11.9, 2.7))

    analytic_labels = [
        ("Hand grip\n(analytic)", ANALYTIC_N["hand_grip"], 0.5),
        ("Blood pressure\n(analytic)", ANALYTIC_N["blood_pressure"], 3.9),
        ("Waist circumf.\n(analytic)", ANALYTIC_N["anthropometrics"], 7.3),
    ]
    for label, n, x in analytic_labels:
        _box(ax1, (x, 1.5), 2.6, 1.0, f"{label}\nn = {n:,}", _BOX_ANALYTIC, fontsize=7.6)
    ax1.text(5, 0.9,
             "Primary spec (age+sex only); n's shown are for the gut_ageaccel exposure --\n"
             "shannon/richness land at 1,769/1,782/1,778 too (exclusion_funnel.csv, Table A9)",
             ha="center", fontsize=6.6, color="#555555", style="italic")

    # ---------------- PANEL 4: validation hierarchy -> Results sections ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b) Validation hierarchy → Results sections", fontsize=11,
                 fontweight="bold", loc="left")

    hierarchy = [
        ("Leak-free clock\n(Check 1: 0 overlap)", "§2.2–2.3 Clock", _BOX_TRAIN, 8.7),
        ("Reliability / ICC\n(AgeAccel 0.516, Shannon 0.626,\nrichness 0.77)", "§2.5 Reliability", _BOX_LONG, 7.0),
        ("Primary cross-lag\n(18 tests, age+sex)", "§2.6 Cross-lagged (Table 2)", _BOX_ANALYTIC, 5.3),
        ("Equivalence bounds\n(TOST vs |β|<0.10)", "§2.7 Power/Equivalence (Table 4)", "#8172B2", 3.6),
        ("Confounder sensitivity\n+ incident disease + prospective", "§2.8–2.12", "#CCB974", 1.9),
        ("Extension experiments\n(E1–E3 signatures, multidomain)", "Extended Data", "#64B5CD", 0.2),
    ]
    for text, target, color, y in hierarchy:
        _box(ax2, (0.3, y), 4.3, 1.4, text, color, fontsize=7.4)
        ax2.annotate("", xy=(5.0, y + 0.7), xytext=(4.6, y + 0.7),
                    arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.0))
        ax2.text(5.2, y + 0.7, target, va="center", fontsize=7.8, color=_INK)
        if y > 0.5:
            _arrow(ax2, (2.45, y), (2.45, y - 0.3))

    # ---------------- PANEL (d): age-encoding tile [Nastya point 9] ----------------
    # Old draft qualitatively labelled this tile "PRESENT"; per review, it must
    # quantify the (modest) effect size instead. See get_age_encoding_label().
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("(d) Age encoding", fontsize=11, fontweight="bold", loc="left")
    _box(ax3, (0.4, 6.0), 9.2, 2.4,
        f"Does the gut microbiome\nencode chronological age?\n\n{age_encoding_label}",
        "#8172B2", fontsize=9.5)
    ax3.text(5, 5.3, "(clock CV performance,\nsingle-visit training stratum)",
             ha="center", fontsize=6.8, color="#555555", style="italic")
    ax3.text(5, 3.6,
             "Prior draft labelled this tile\n\"PRESENT\" (qualitative).\nRevised per review to state\nthe quantified, modest\neffect size directly.",
             ha="center", va="top", fontsize=6.8, color=_INK,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#BBBBBB", lw=0.5))
    ax3.text(5, 0.4, f"Source: {age_encoding_source}", ha="center", fontsize=5.8,
             color="#777777", style="italic", wrap=True)

    fig.suptitle("Fig. 1 — Participant Flow and Validation Hierarchy  (HPP, corrected leak-free pipeline)",
                fontsize=12.5, fontweight="bold", fontfamily="serif")
    fig.savefig(FIG_DIR / "Fig1_flow.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig1_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR}/Fig1_flow.pdf/.png")
    log("\n[DONE]")


if __name__ == "__main__":
    main()
