#!/usr/bin/env python3
"""
paper_analysis.py

Nature-journal-quality figures and comprehensive statistics for:
  "Temporal directionality between gut microbiome diversity and functional
   aging outcomes: a cross-lagged panel analysis of the Human Phenotype Project"

Inputs:
  $HPP_DATA_ROOT/        — raw HPP parquets (for descriptive stats & diversity)
  $HPP_RESULTS_DIR/cross_lag_all_paths.csv
  $HPP_RESULTS_DIR/causal_direction_summary.csv

Outputs:
  paper_stats.json            — every inline number for manuscript text
  table1_characteristics.csv  — participant descriptive statistics
  table2_crosslag.csv         — full cross-lag results (publication-ready)
  fig1_study_design.pdf/png   — cohort flow + CLPM schematic
  fig2_microbiome.pdf/png     — Shannon characterisation + stability
  fig3_crosslag.pdf/png       — main result: forest plot Path A vs B
  fig4_comparison.pdf/png     — Path A vs B p-value scatter + verdict
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path(os.environ.get("HPP_DATA_ROOT",  Path.home() / "studies" / "hpp_datasets"))
RESULTS_DIR = Path(os.environ.get("HPP_RESULTS_DIR", Path.home() / "studies" / "HPP_Prototype" / "results_cross_lagged"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_FDR = 0.05

# ── Nature journal style ───────────────────────────────────────────────────────
NATURE = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size":          7,
    "axes.labelsize":     8,
    "axes.titlesize":     8.5,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.7,
    "pdf.fonttype":       42,   # embed TrueType in PDF
    "ps.fonttype":        42,
    "savefig.dpi":        300,
    "figure.dpi":         150,
    "axes.grid":          False,
}
plt.rcParams.update(NATURE)

# ── Colour palette (Nature / NEJM inspired) ────────────────────────────────────
C = {
    "blue":       "#2166AC",
    "red":        "#D6604D",
    "light_blue": "#92C5DE",
    "light_red":  "#F4A582",
    "gray":       "#878787",
    "dark":       "#1A1A1A",
    "green":      "#4DAC26",
    "purple":     "#762A83",
    "orange":     "#E08533",
    "bg":         "#F7F7F7",
}

# Figure widths in inches (Nature single-col = 3.5 in, double-col = 7.2 in)
W1 = 3.5
W2 = 7.2

def log(m): print(m, flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        p = RESULTS_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
    log(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


def panel_label(ax, letter, x=-0.12, y=1.05, size=10):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight="bold", va="top", ha="left")


def despine(ax, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)


def parse_ci95(val):
    """Parse ci95 from CSV string '(lo, hi)' → (lo, hi)."""
    try:
        t = ast.literal_eval(str(val))
        return float(t[0]), float(t[1])
    except Exception:
        return np.nan, np.nan


# ── Step 0: Load cross-lag results ────────────────────────────────────────────

def load_results():
    paths_csv = RESULTS_DIR / "cross_lag_all_paths.csv"
    dir_csv   = RESULTS_DIR / "causal_direction_summary.csv"

    if not paths_csv.exists():
        log("cross_lag_all_paths.csv not found — running cross_lagged_panel.py now...")
        import sys, importlib
        # Locate cross_lagged_panel.py relative to this file
        clp_path = Path(__file__).parent / "cross_lagged_panel.py"
        if not clp_path.exists():
            raise FileNotFoundError(
                f"cross_lagged_panel.py not found at {clp_path}. "
                "Place it alongside paper_analysis.py."
            )
        spec = importlib.util.spec_from_file_location("cross_lagged_panel", clp_path)
        clp  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clp)
        # Force cross_lagged_panel to write into OUR results directory
        clp.RESULTS_DIR = RESULTS_DIR
        clp.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        clp.main()
        if not paths_csv.exists():
            raise RuntimeError(
                "cross_lagged_panel.py ran but did not produce cross_lag_all_paths.csv. "
                "Check its output above for errors."
            )

    paths = pd.read_csv(paths_csv)
    # Parse ci95 tuple stored as string
    paths[["ci_lo","ci_hi"]] = paths["ci95"].apply(
        lambda v: pd.Series(parse_ci95(v)))
    direction = pd.read_csv(dir_csv) if dir_csv.exists() else pd.DataFrame()
    log(f"Loaded {len(paths)} cross-lag path rows, {len(direction)} direction rows")
    return paths, direction


# ── Step 1: Load raw diversity for descriptive stats ──────────────────────────

def load_raw_diversity():
    """
    Load Shannon diversity per participant per visit from species parquet.
    Returns DataFrame with columns: study_id, visit_n, shannon, richness
    """
    log("Loading raw diversity from species parquet...")
    gut = pd.read_parquet(DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut.index.names != [None]: gut = gut.reset_index()
    pid   = next(c for c in gut.columns if c in ("participant_id","study_id"))
    stage = next(c for c in gut.columns if "research_stage" in c.lower())
    gut   = gut.rename(columns={pid: "study_id", stage: "visit"})
    counts = gut.groupby("study_id")["visit"].nunique()
    repeat_ids = counts[counts >= 2].index
    gut = gut[gut["study_id"].isin(repeat_ids)].copy()
    gut["visit_n"] = gut.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    # Load species parquet
    ptr_col = next((c for c in gut.columns if "species" in c.lower() and "parquet" in c.lower()), None)
    sp_path_val = gut[ptr_col].dropna().iloc[0]
    sp_path = DATA_ROOT / "gut_microbiome" / str(sp_path_val).lstrip("./")
    sp = pd.read_parquet(sp_path)
    if sp.index.names != [None]: sp = sp.reset_index()

    sp_meta = {"participant_id","study_id","cohort","research_stage",
               "array_index","sample_name","collection_timestamp","collection_date","timezone"}
    abund_cols = [c for c in sp.columns if c not in sp_meta and
                  str(sp[c].dtype).startswith("float")]
    abund = sp[abund_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs()
    row_sums = abund.sum(axis=1)
    sp = sp.copy()
    sp["richness"] = (abund > 0).sum(axis=1).astype(float)
    row_sums_safe = row_sums.replace(0, np.nan)
    p = abund.div(row_sums_safe, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    sp["shannon"] = -(p.values * lp).sum(axis=1)
    sp.loc[row_sums == 0, "shannon"] = np.nan

    sp_pid   = next(c for c in ("participant_id","study_id") if c in sp.columns)
    sp_stage = next(c for c in sp.columns if "research_stage" in c.lower())
    sp_small = sp[[sp_pid, sp_stage, "shannon", "richness"]].rename(
        columns={sp_pid: "study_id", sp_stage: "visit"})
    merged = gut[["study_id","visit","visit_n"]].merge(sp_small, on=["study_id","visit"], how="left")
    div = (merged.groupby(["study_id","visit_n"], as_index=False)
                 .agg({"shannon": "mean", "richness": "mean"}))
    log(f"  Diversity: {len(div):,} rows, {div['study_id'].nunique():,} participants")
    return div


def load_sex_age():
    pop = pd.read_parquet(DATA_ROOT / "population" / "population.parquet")
    if pop.index.names != [None]: pop = pop.reset_index()
    if pop.columns.duplicated().any():
        seen = {}
        new = []
        for c in pop.columns:
            if c in seen:
                seen[c] += 1; new.append(f"__dup_{c}_{seen[c]}")
            else:
                seen[c] = 0; new.append(c)
        pop.columns = new
        pop = pop[[c for c in pop.columns if not c.startswith("__dup_")]]
    pid = next((c for c in pop.columns if c in ("participant_id","study_id")), None)
    sex_col = next((c for c in pop.columns if "sex" in c.lower()), None)
    dob_col = next((c for c in pop.columns if any(k in c.lower() for k in ("birth","dob","born"))), None)
    if pid:
        pop = pop.rename(columns={pid: "study_id"})
    demo = pop[["study_id"]].copy()
    if sex_col:
        demo["sex"] = pd.to_numeric(pop[sex_col], errors="coerce")
    if dob_col:
        dob = pd.to_datetime(pop[dob_col], errors="coerce")
        demo["age"] = (pd.Timestamp("2020-01-01") - dob).dt.days / 365.25
    return demo.dropna(subset=["study_id"]).drop_duplicates("study_id")


# ── Step 2: Compute and save comprehensive stats ───────────────────────────────

def compute_paper_stats(paths: pd.DataFrame, direction: pd.DataFrame,
                         div: pd.DataFrame, demo: pd.DataFrame) -> dict:
    log("Computing paper statistics...")
    s = {}

    # Cohort
    n_repeat = div["study_id"].nunique()
    s["n_repeat_gut"] = int(n_repeat)

    div1 = div[div["visit_n"] == 1]
    div2 = div[div["visit_n"] == 2]
    s["n_shannon_t1"] = int(div1["shannon"].notna().sum())
    s["n_shannon_t2"] = int(div2["shannon"].notna().sum())

    # Shannon stats
    sh1 = div1["shannon"].dropna()
    sh2 = div2["shannon"].dropna()
    s["shannon_t1_mean"] = round(float(sh1.mean()), 3)
    s["shannon_t1_sd"]   = round(float(sh1.std()), 3)
    s["shannon_t1_med"]  = round(float(sh1.median()), 3)
    s["shannon_t2_mean"] = round(float(sh2.mean()), 3)
    s["shannon_t2_sd"]   = round(float(sh2.std()), 3)

    # Shannon temporal stability (ICC proxy = r between t1 and t2)
    paired = div1.merge(div2, on="study_id", suffixes=("_t1","_t2"))
    paired = paired.dropna(subset=["shannon_t1","shannon_t2"])
    r_sh, p_sh = pearsonr(paired["shannon_t1"], paired["shannon_t2"])
    s["shannon_pearson_r"]   = round(float(r_sh), 3)
    s["shannon_pearson_p"]   = round(float(p_sh), 6)
    s["n_paired_shannon"]    = int(len(paired))

    # Richness
    r1 = div1["richness"].dropna()
    s["richness_t1_mean"] = round(float(r1.mean()), 1)
    s["richness_t1_sd"]   = round(float(r1.std()), 1)

    # Demographics
    if "sex" in demo.columns:
        sex_vals = demo["sex"].dropna()
        s["n_sex_available"]   = int(len(sex_vals))
        s["pct_female"]        = round(float((sex_vals == 0).mean() * 100), 1)
        s["pct_male"]          = round(float((sex_vals == 1).mean() * 100), 1)
    if "age" in demo.columns:
        age_vals = demo["age"].dropna()
        s["age_mean"] = round(float(age_vals.mean()), 1)
        s["age_sd"]   = round(float(age_vals.std()), 1)
        s["age_min"]  = round(float(age_vals.min()), 1)
        s["age_max"]  = round(float(age_vals.max()), 1)

    # Cross-lag results
    valid = paths.dropna(subset=["p","beta"])
    s["n_cross_lag_tests"]    = int(len(valid))
    s["n_fdr_significant"]    = int(valid["significant"].sum()) if "significant" in valid.columns else 0
    s["n_path_A"]             = int((valid["direction"] == "A").sum())
    s["n_path_B"]             = int((valid["direction"] == "B").sum())

    # Median n across pairs
    s["n_median_per_pair"]    = int(valid["n"].median()) if "n" in valid.columns else 0
    s["n_min_per_pair"]       = int(valid["n"].min()) if "n" in valid.columns else 0
    s["n_max_per_pair"]       = int(valid["n"].max()) if "n" in valid.columns else 0

    # Most nominal finding
    nominal = valid[valid["p"] < 0.1].sort_values("p")
    if not nominal.empty:
        top = nominal.iloc[0]
        s["nominal_top_path"]  = str(top.get("path",""))
        s["nominal_top_p"]     = round(float(top["p"]), 4)
        s["nominal_top_beta"]  = round(float(top["beta"]), 4)
        s["nominal_top_q"]     = round(float(top.get("q_fdr", np.nan)), 4)
        s["nominal_top_n"]     = int(top["n"])
        s["nominal_top_dir"]   = str(top.get("direction",""))

    # Direction verdicts
    if not direction.empty and "verdict" in direction.columns:
        vc = direction["verdict"].value_counts().to_dict()
        s["direction_verdicts"] = {k: int(v) for k, v in vc.items()}
        s["n_pairs_total"]      = int(len(direction))

    out_path = RESULTS_DIR / "paper_stats.json"
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)
    log(f"  Saved: paper_stats.json")

    # Table 1: participant characteristics
    tbl1 = pd.DataFrame([
        {"Characteristic": "Gut microbiome repeat-visit participants",
         "Value": f"{s.get('n_repeat_gut','')}", "Unit": "n"},
        {"Characteristic": "Shannon diversity (visit 1)",
         "Value": f"{s.get('shannon_t1_mean','')} ± {s.get('shannon_t1_sd','')}",
         "Unit": "mean ± SD"},
        {"Characteristic": "Shannon diversity (visit 2)",
         "Value": f"{s.get('shannon_t2_mean','')} ± {s.get('shannon_t2_sd','')}",
         "Unit": "mean ± SD"},
        {"Characteristic": "Shannon temporal stability (Pearson r)",
         "Value": f"{s.get('shannon_pearson_r','')}",
         "Unit": f"n={s.get('n_paired_shannon','')}"},
        {"Characteristic": "Species richness (visit 1)",
         "Value": f"{s.get('richness_t1_mean','')} ± {s.get('richness_t1_sd','')}",
         "Unit": "mean ± SD"},
        {"Characteristic": "Sex (% female)",
         "Value": f"{s.get('pct_female','')}%",
         "Unit": f"n={s.get('n_sex_available','')}"},
        {"Characteristic": "Age (years)",
         "Value": f"{s.get('age_mean','')} ± {s.get('age_sd','')}",
         "Unit": "mean ± SD"},
    ])
    tbl1.to_csv(RESULTS_DIR / "table1_characteristics.csv", index=False)

    # Table 2: cross-lag results
    tbl2 = valid[["direction","exposure","outcome","dataset","beta","se","p","q_fdr",
                   "n","r2","ci_lo","ci_hi"]].copy()
    tbl2 = tbl2.rename(columns={
        "direction": "Path", "exposure": "Exposure",
        "outcome": "Outcome", "dataset": "Dataset",
        "beta": "β (std)", "se": "SE", "p": "p-value",
        "q_fdr": "q (FDR)", "n": "n", "r2": "R²",
        "ci_lo": "95% CI lower", "ci_hi": "95% CI upper",
    })
    for col in ["β (std)","SE","p-value","q (FDR)","R²","95% CI lower","95% CI upper"]:
        if col in tbl2.columns:
            tbl2[col] = tbl2[col].round(4)
    tbl2.to_csv(RESULTS_DIR / "table2_crosslag.csv", index=False)
    log(f"  Saved: table1_characteristics.csv, table2_crosslag.csv")
    return s


# ── Figure 1: Study Design ─────────────────────────────────────────────────────

def fig1_study_design(stats: dict):
    log("Plotting Figure 1: Study design...")
    fig = plt.figure(figsize=(W2, 5.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12, width_ratios=[1, 1.2])
    ax_flow  = fig.add_subplot(gs[0])
    ax_model = fig.add_subplot(gs[1])

    # ── Panel A: CONSORT-style flow diagram ───────────────────────────────────
    ax_flow.set_xlim(0, 1); ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    ax_flow.set_title("a", loc="left", fontweight="bold", fontsize=10, pad=4)

    BOX_W, BOX_H = 0.72, 0.09
    CX = 0.50
    BLUE  = "#2166AC"; GRAY = "#AAAAAA"; RED = "#D6604D"

    def flow_box(ax, x, y, text, color=BLUE, alpha=0.12, textcol="black",
                 bold=False):
        bx = x - BOX_W / 2
        rect = FancyBboxPatch((bx, y), BOX_W, BOX_H,
                              boxstyle="round,pad=0.012",
                              linewidth=0.8, edgecolor=color,
                              facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x, y + BOX_H / 2, text, ha="center", va="center",
                fontsize=7, color=textcol,
                fontweight="bold" if bold else "normal",
                multialignment="center")

    def flow_arrow(ax, x, y_top, y_bot, color="black"):
        ax.annotate("", xy=(x, y_bot + BOX_H + 0.005),
                    xytext=(x, y_top - 0.005),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.8, mutation_scale=8))

    n_rep = stats.get("n_repeat_gut", "")
    n_sh  = stats.get("n_shannon_t1", "")
    n_med = stats.get("n_median_per_pair", "")

    boxes = [
        (0.85, "Human Phenotype Project (HPP)\ncohort", BLUE, True),
        (0.70, f"Participants with gut microbiome\nprofile at ≥2 visits  (n={n_rep})", BLUE, False),
        (0.55, f"Shannon diversity computed\nfrom MetaPhlAn species profiles  (n={n_sh})", BLUE, False),
        (0.40, "Linked with repeated-measure\nfunctional outcomes (grip, BP, BMI)", BLUE, False),
        (0.25, f"Cross-lagged panel models\n(median n per pair={n_med})", RED, True),
    ]
    for yi, txt, col, bold in boxes:
        flow_box(ax_flow, CX, yi, txt, color=col, bold=bold,
                 alpha=0.15 if col == RED else 0.10)

    for i in range(len(boxes) - 1):
        y_top = boxes[i][0]
        y_bot = boxes[i+1][0]
        flow_arrow(ax_flow, CX, y_top, y_bot)

    ax_flow.text(0.50, 0.10,
        "Exposures: Shannon diversity, species richness\n"
        "Outcomes: grip strength, blood pressure, BMI",
        ha="center", va="center", fontsize=6.5, color="#555555",
        style="italic")

    # ── Panel B: CLPM schematic ───────────────────────────────────────────────
    ax_model.set_xlim(0, 1); ax_model.set_ylim(0, 1)
    ax_model.axis("off")
    ax_model.set_title("b", loc="left", fontweight="bold", fontsize=10, pad=4)

    def node(ax, x, y, label, sublabel="", color=BLUE):
        circ = plt.Circle((x, y), 0.12, color=color, alpha=0.15,
                          transform=ax.transData, zorder=2)
        ax.add_patch(circ)
        circ2 = plt.Circle((x, y), 0.12, color=color, fill=False,
                           linewidth=1.0, transform=ax.transData, zorder=3)
        ax.add_patch(circ2)
        ax.text(x, y + 0.02, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color, zorder=4)
        if sublabel:
            ax.text(x, y - 0.05, sublabel, ha="center", va="center",
                    fontsize=6.5, color="#555555", zorder=4)

    def arrow(ax, x0, y0, x1, y1, label="", color="black", lw=1.2,
              lstyle="-", rad=0.0, fontsize=7):
        ax.annotate("",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=lw, mutation_scale=8,
                            connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my + 0.04, label, ha="center", fontsize=fontsize,
                    color=color, fontweight="bold")

    # Node positions (x, y)
    pos = {
        "M1": (0.22, 0.68), "M2": (0.78, 0.68),
        "O1": (0.22, 0.32), "O2": (0.78, 0.32),
    }
    node(ax_model, *pos["M1"], "M(t₁)", "Microbiome\ndiversity", C["blue"])
    node(ax_model, *pos["M2"], "M(t₂)", "Microbiome\ndiversity", C["blue"])
    node(ax_model, *pos["O1"], "Y(t₁)", "Functional\noutcome", C["red"])
    node(ax_model, *pos["O2"], "Y(t₂)", "Functional\noutcome", C["red"])

    # Auto-regressive (stability) arrows
    arrow(ax_model, pos["M1"][0]+0.12, pos["M1"][1],
          pos["M2"][0]-0.12, pos["M2"][1], color=C["gray"], lw=0.9)
    arrow(ax_model, pos["O1"][0]+0.12, pos["O1"][1],
          pos["O2"][0]-0.12, pos["O2"][1], color=C["gray"], lw=0.9)

    # Path A: M(t1) → Y(t2)
    arrow(ax_model, pos["M1"][0]+0.08, pos["M1"][1]-0.06,
          pos["O2"][0]-0.08, pos["O2"][1]+0.06,
          label="Path A", color=C["blue"], rad=0.1, lw=1.5)

    # Path B: Y(t1) → M(t2)
    arrow(ax_model, pos["O1"][0]+0.08, pos["O1"][1]+0.06,
          pos["M2"][0]-0.08, pos["M2"][1]-0.06,
          label="Path B", color=C["red"], rad=-0.1, lw=1.5)

    # Time axis
    ax_model.annotate("", xy=(0.92, 0.10), xytext=(0.08, 0.10),
                       arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0))
    ax_model.text(0.22, 0.07, "Visit 1 (t₁)", ha="center", fontsize=7, color="#333333")
    ax_model.text(0.78, 0.07, "Visit 2 (t₂)", ha="center", fontsize=7, color="#333333")
    ax_model.text(0.50, 0.025, "Time", ha="center", fontsize=7.5,
                  color="#333333", fontstyle="italic")

    # Legend
    lA = mlines.Line2D([], [], color=C["blue"], lw=1.5,
                        label="Path A: Microbiome(t₁)→Outcome(t₂)")
    lB = mlines.Line2D([], [], color=C["red"],  lw=1.5,
                        label="Path B: Outcome(t₁)→Microbiome(t₂)")
    lG = mlines.Line2D([], [], color=C["gray"], lw=0.9,
                        label="Auto-regression (stability)")
    ax_model.legend(handles=[lA, lB, lG], loc="upper right",
                    fontsize=6, frameon=True, framealpha=0.9,
                    edgecolor="#CCCCCC", bbox_to_anchor=(1.0, 0.99))

    fig.suptitle(
        "Cross-Lagged Panel Analysis of the Human Phenotype Project",
        fontsize=9, fontweight="bold", y=1.01,
    )
    save_fig(fig, "fig1_study_design")


# ── Figure 2: Microbiome Characterisation ─────────────────────────────────────

def fig2_microbiome(div: pd.DataFrame, stats: dict):
    log("Plotting Figure 2: Microbiome characterisation...")
    div1 = div[div["visit_n"] == 1].dropna(subset=["shannon"])
    div2 = div[div["visit_n"] == 2].dropna(subset=["shannon"])
    paired = div1.merge(div2, on="study_id", suffixes=("_v1","_v2"))
    paired = paired.dropna(subset=["shannon_v1","shannon_v2"])

    fig = plt.figure(figsize=(W2, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: Shannon distribution at both visits ──────────────────────────
    bins = np.linspace(div["shannon"].min() - 0.1, div["shannon"].max() + 0.1, 35)
    ax1.hist(div1["shannon"], bins=bins, color=C["blue"], alpha=0.65,
             label=f"Visit 1 (n={len(div1):,})", density=True, linewidth=0)
    ax1.hist(div2["shannon"], bins=bins, color=C["red"],  alpha=0.55,
             label=f"Visit 2 (n={len(div2):,})", density=True, linewidth=0)

    # KDE overlay
    from scipy.stats import gaussian_kde
    for data, col in [(div1["shannon"], C["blue"]), (div2["shannon"], C["red"])]:
        kde = gaussian_kde(data, bw_method=0.3)
        xs  = np.linspace(bins[0], bins[-1], 200)
        ax1.plot(xs, kde(xs), color=col, lw=1.2)

    mu1, mu2 = div1["shannon"].mean(), div2["shannon"].mean()
    ax1.axvline(mu1, color=C["blue"], lw=0.9, ls="--", alpha=0.8)
    ax1.axvline(mu2, color=C["red"],  lw=0.9, ls="--", alpha=0.8)

    ax1.set_xlabel("Shannon diversity index")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, fontsize=6.5)
    t_stat, t_p = stats_test = stats.get("shannon_t1_mean",""), ""
    # t-test between visits
    t_res = stats_module_ttest(div1["shannon"].values, div2["shannon"].values)
    ax1.text(0.97, 0.97,
             f"Visit 1: {mu1:.2f}±{div1['shannon'].std():.2f}\n"
             f"Visit 2: {mu2:.2f}±{div2['shannon'].std():.2f}\n"
             f"p={t_res.pvalue:.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.8))
    panel_label(ax1, "a")
    despine(ax1)

    # ── Panel B: Temporal stability scatter ────────────────────────────────────
    # Plot a random subset for visibility
    rng = np.random.default_rng(42)
    idx = rng.choice(len(paired), size=min(500, len(paired)), replace=False)
    sub = paired.iloc[idx]

    ax2.scatter(sub["shannon_v1"], sub["shannon_v2"],
                s=4, alpha=0.4, color=C["blue"], linewidths=0)
    # Regression line
    r, p_r = pearsonr(paired["shannon_v1"], paired["shannon_v2"])
    xs = np.linspace(paired["shannon_v1"].min(), paired["shannon_v1"].max(), 100)
    m, b = np.polyfit(paired["shannon_v1"], paired["shannon_v2"], 1)
    ax2.plot(xs, m * xs + b, color=C["dark"], lw=1.2)
    # Diagonal (perfect stability)
    lim = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
           max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lim, lim, color=C["gray"], lw=0.7, ls="--", alpha=0.7)

    ax2.text(0.05, 0.95,
             f"r = {r:.3f}\np = {p_r:.2e}\nn = {len(paired):,}",
             transform=ax2.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.9))
    ax2.set_xlabel("Shannon diversity — Visit 1")
    ax2.set_ylabel("Shannon diversity — Visit 2")
    panel_label(ax2, "b")
    despine(ax2)

    # ── Panel C: Species richness distribution ─────────────────────────────────
    rich1 = div[div["visit_n"] == 1]["richness"].dropna()
    rich2 = div[div["visit_n"] == 2]["richness"].dropna()
    data   = [rich1.values, rich2.values]
    labels = ["Visit 1", "Visit 2"]
    parts  = ax3.violinplot(data, positions=[1, 2],
                             showmedians=True, showextrema=False)
    for i, (pc, col) in enumerate(zip(parts["bodies"], [C["blue"], C["red"]])):
        pc.set_facecolor(col); pc.set_alpha(0.5); pc.set_edgecolor(col)
    parts["cmedians"].set_color([C["blue"], C["red"]])
    parts["cmedians"].set_linewidth(1.5)

    for i, (d, pos) in enumerate(zip(data, [1, 2])):
        q1, q3 = np.percentile(d, 25), np.percentile(d, 75)
        ax3.vlines(pos, q1, q3, color=[C["blue"],C["red"]][i], lw=2.5, alpha=0.7)

    ax3.set_xticks([1, 2]); ax3.set_xticklabels(labels)
    ax3.set_xlabel("Visit")
    ax3.set_ylabel("Species richness (# species detected)")
    r_rich1 = rich1.mean(); r_rich2 = rich2.mean()
    ax3.text(0.97, 0.97,
             f"V1: {r_rich1:.0f}±{rich1.std():.0f}\n"
             f"V2: {r_rich2:.0f}±{rich2.std():.0f}",
             transform=ax3.transAxes, ha="right", va="top", fontsize=6.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.8))
    panel_label(ax3, "c")
    despine(ax3)

    fig.suptitle("Gut Microbiome Diversity Characterisation — HPP Repeat Visitors",
                 fontsize=9, fontweight="bold", y=1.02)
    save_fig(fig, "fig2_microbiome")


def stats_module_ttest(a, b):
    return stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)])


# ── Figure 3: Main Result — Forest Plot ────────────────────────────────────────

OUTCOME_LABELS = {
    "grip_strength":                    "Grip Strength",
    "grip_composite":                   "Grip Strength",
    "hand_grip_strength":               "Grip Strength",
    "finger_grip_strength_left":        "Grip Strength",
    "systolic":                         "Systolic BP",
    "systolic_bp":                      "Systolic BP",
    "lying_systolic_blood_pressure":    "Systolic BP",
    "lying_blood_pressure_diastolic":   "Diastolic BP",
    "diastolic_bp":                     "Diastolic BP",
    "bmi":                              "BMI",
    "body_mass_index":                  "BMI",
    "waist_circumference":              "Waist Circumference",
    "outcome":                          "Functional Outcome",
}

EXPOSURE_LABELS = {
    "shannon":      "Shannon Diversity",
    "richness":     "Species Richness",
    "gut_ageaccel": "Gut Age Acceleration",
}


def clean_label(s, mapping):
    return mapping.get(str(s), str(s).replace("_", " ").title())


def fig3_forest(paths: pd.DataFrame, stats: dict):
    log("Plotting Figure 3: Forest plot (main result)...")
    valid = paths.dropna(subset=["beta","ci_lo","ci_hi","p"])

    # Separate Path A and B
    pathA = valid[valid["direction"] == "A"].copy()
    pathB = valid[valid["direction"] == "B"].copy()

    # Build row labels
    pathA["row_label"] = (pathA["exposure"].map(EXPOSURE_LABELS).fillna(pathA["exposure"])
                          + "  →  " +
                          pathA["outcome"].map(OUTCOME_LABELS).fillna(pathA["outcome"]))
    pathB["row_label"] = (pathB["exposure"].map(OUTCOME_LABELS).fillna(pathB["exposure"])
                          + "  →  " +
                          pathB["outcome"].map(EXPOSURE_LABELS).fillna(pathB["outcome"]))

    pathA = pathA.sort_values(["exposure","dataset"])
    pathB = pathB.sort_values(["exposure","dataset"])

    n_rows = max(len(pathA), len(pathB))
    fig = plt.figure(figsize=(W2, max(5.5, n_rows * 0.55 + 1.5)))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06)
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])

    def draw_forest(ax, df, color, title, mirror=False):
        df = df.reset_index(drop=True)
        n  = len(df)
        ys = np.arange(n)

        for i, row in df.iterrows():
            sig   = bool(row.get("significant", False))
            alpha = 1.0 if sig else 0.65
            marker= "D" if sig else "o"
            msize = 40  if sig else 28
            # CI line
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                    color=color, lw=1.2, alpha=alpha, solid_capstyle="round")
            # Point estimate
            ax.scatter([row["beta"]], [i], color=color, s=msize,
                       zorder=5, marker=marker, alpha=alpha, linewidths=0)
            # p annotation
            p_str = f"p={row['p']:.3f}" if row['p'] >= 0.001 else "p<0.001"
            n_str = f"n={int(row['n']):,}"
            xmax  = ax.get_xlim()[1] if not mirror else ax.get_xlim()[0]
            ax.text(1.01 if not mirror else -0.01, i,
                    f"{p_str}  {n_str}",
                    transform=ax.get_yaxis_transform(),
                    va="center", fontsize=5.5,
                    ha="left" if not mirror else "right",
                    color="#444444")

        ax.axvline(0, color="#333333", lw=0.8, ls="--", alpha=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(df["row_label"], fontsize=6.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlabel("Standardised β (95% CI)", fontsize=7.5)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        despine(ax)
        # Zero label
        ax.text(0, -0.85, "0", ha="center", fontsize=6, color="#333333",
                transform=ax.get_xaxis_transform())

    draw_forest(ax_A, pathA, C["blue"],
                "Path A: Microbiome(t₁) → Outcome(t₂)\n[Does microbiome drive aging?]")
    draw_forest(ax_B, pathB, C["red"],
                "Path B: Outcome(t₁) → Microbiome(t₂)\n[Does aging reshape microbiome?]",
                mirror=False)

    # Shared subtitle
    n_sig  = stats.get("n_fdr_significant", 0)
    n_tot  = stats.get("n_cross_lag_tests", 0)
    n_med  = stats.get("n_median_per_pair", "")
    fig.suptitle(
        f"Cross-Lagged Panel Results: Gut Microbiome ↔ Functional Aging\n"
        f"FDR-significant: {n_sig}/{n_tot} tests  |  Median n per pair: {n_med}  "
        f"|  ◆ FDR q<0.05  ● p-nominal",
        fontsize=8, fontweight="bold", y=1.02,
    )
    # Add FDR legend
    dot_sig  = mlines.Line2D([], [], color=C["blue"], marker="D", markersize=5,
                              linestyle="None", label="FDR q<0.05")
    dot_nom  = mlines.Line2D([], [], color=C["blue"], marker="o", markersize=4,
                              linestyle="None", label="p-nominal (not FDR sig.)",
                              alpha=0.65)
    ax_A.legend(handles=[dot_sig, dot_nom], fontsize=6, frameon=True,
                framealpha=0.9, edgecolor="#CCCCCC", loc="lower right")

    panel_label(ax_A, "a", x=-0.02)
    panel_label(ax_B, "b", x=-0.02)
    save_fig(fig, "fig3_crosslag")


# ── Figure 4: Path A vs B Comparison ──────────────────────────────────────────

def fig4_comparison(paths: pd.DataFrame, direction: pd.DataFrame, stats: dict):
    log("Plotting Figure 4: Path A vs B comparison...")

    fig = plt.figure(figsize=(W2, 4.0))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40, width_ratios=[1.3, 1, 0.9])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: p-value scatter (Path A p vs Path B p) ──────────────────────
    if not direction.empty and "p_A" in direction.columns:
        dir2 = direction.dropna(subset=["p_A","p_B"])
        neg_log_A = -np.log10(dir2["p_A"].clip(1e-10))
        neg_log_B = -np.log10(dir2["p_B"].clip(1e-10))

        for i, row in dir2.iterrows():
            label = (f"{clean_label(row.get('exposure',''), EXPOSURE_LABELS)}"
                     f"\n× {clean_label(row.get('outcome',''), OUTCOME_LABELS)}")
            nA = -np.log10(max(row["p_A"], 1e-10))
            nB = -np.log10(max(row["p_B"], 1e-10))
            ax1.scatter([nA], [nB], s=28, color=C["gray"], zorder=3, linewidths=0)

        # Diagonal (equal evidence for A and B)
        lim = max(max(neg_log_A.max(), neg_log_B.max()) + 0.3, 2.5)
        ax1.plot([0, lim], [0, lim], color=C["gray"], lw=0.8, ls="--", alpha=0.6)

        sig_thresh = -np.log10(0.05)
        ax1.axhline(sig_thresh, color=C["red"],  lw=0.7, ls=":", alpha=0.6,
                    label=f"p=0.05 threshold")
        ax1.axvline(sig_thresh, color=C["blue"], lw=0.7, ls=":", alpha=0.6)
        ax1.set_xlabel("−log₁₀(p) Path A\n[Microbiome → Aging]", fontsize=7.5)
        ax1.set_ylabel("−log₁₀(p) Path B\n[Aging → Microbiome]", fontsize=7.5)
        ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)

        # Shade regions
        ax1.fill_betweenx([0, lim], sig_thresh, lim,
                          color=C["blue"], alpha=0.04)
        ax1.fill_between([0, lim], sig_thresh, lim,
                         color=C["red"],  alpha=0.04)
        ax1.text(lim * 0.97, lim * 0.03,
                 "Aging →\nMicrobiome", ha="right", va="bottom",
                 fontsize=6.5, color=C["red"], fontstyle="italic")
        ax1.text(lim * 0.03, lim * 0.97,
                 "Microbiome →\nAging", ha="left", va="top",
                 fontsize=6.5, color=C["blue"], fontstyle="italic")
    despine(ax1)
    panel_label(ax1, "a")

    # ── Panel B: β_A vs β_B scatter ──────────────────────────────────────────
    if not direction.empty and "beta_A" in direction.columns:
        dir2 = direction.dropna(subset=["beta_A","beta_B"])
        ax2.scatter(dir2["beta_A"], dir2["beta_B"],
                    s=32, color=C["gray"], alpha=0.8, linewidths=0, zorder=3)
        ax2.axhline(0, color="#333333", lw=0.7, ls="--", alpha=0.6)
        ax2.axvline(0, color="#333333", lw=0.7, ls="--", alpha=0.6)
        ax2.set_xlabel("β (Path A)\n[Microbiome → Aging]", fontsize=7.5)
        ax2.set_ylabel("β (Path B)\n[Aging → Microbiome]", fontsize=7.5)
    despine(ax2)
    panel_label(ax2, "b")

    # ── Panel C: Verdict bar chart ────────────────────────────────────────────
    if not direction.empty and "verdict" in direction.columns:
        vc = direction["verdict"].value_counts()
        color_map = {
            "MICROBIOME → AGING":    C["blue"],
            "AGING → MICROBIOME":    C["red"],
            "BIDIRECTIONAL":         C["purple"],
            "no significant effect": C["gray"],
        }
        bar_colors = [color_map.get(v, C["gray"]) for v in vc.index]
        bars = ax3.barh(range(len(vc)), vc.values, color=bar_colors,
                        height=0.55, linewidth=0)
        ax3.set_yticks(range(len(vc)))
        ax3.set_yticklabels(
            [v.replace(" → ", "\n→ ").replace(" effect","") for v in vc.index],
            fontsize=6.5)
        ax3.set_xlabel("Number of pairs (FDR q<0.05)", fontsize=7)
        for bar, val in zip(bars, vc.values):
            ax3.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=7, color="#333333")
        ax3.set_xlim(0, max(vc.values) + 1.5)
    despine(ax3)
    panel_label(ax3, "c")

    nominal_path = stats.get("nominal_top_path","")
    nominal_p    = stats.get("nominal_top_p", "")
    nominal_dir  = stats.get("nominal_top_dir","")
    fig.suptitle(
        f"Causal Direction Analysis — No FDR-Significant Effects Detected\n"
        f"Most nominal: '{nominal_path}' (Path {nominal_dir}, p={nominal_p})",
        fontsize=8, fontweight="bold", y=1.03,
    )
    save_fig(fig, "fig4_comparison")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("PAPER ANALYSIS: Nature-quality figures + comprehensive stats")
    log(f"Results dir: {RESULTS_DIR}")
    log("=" * 65)

    # Load cross-lag results
    paths, direction = load_results()

    # Load raw diversity for descriptive stats
    div  = load_raw_diversity()
    demo = load_sex_age()

    # Compute & save all statistics
    stats = compute_paper_stats(paths, direction, div, demo)

    log(f"\nKey statistics:")
    log(f"  n (repeat-visit gut): {stats.get('n_repeat_gut')}")
    log(f"  Shannon t1: {stats.get('shannon_t1_mean')} ± {stats.get('shannon_t1_sd')}")
    log(f"  Shannon stability r: {stats.get('shannon_pearson_r')}")
    log(f"  Cross-lag tests: {stats.get('n_cross_lag_tests')}")
    log(f"  FDR-significant: {stats.get('n_fdr_significant')}")
    log(f"  Most nominal: {stats.get('nominal_top_path')} p={stats.get('nominal_top_p')}")

    # Generate figures
    log("\nGenerating figures...")
    fig1_study_design(stats)
    fig2_microbiome(div, stats)
    fig3_forest(paths, stats)
    fig4_comparison(paths, direction, stats)

    log("\n" + "=" * 65)
    log("DONE — all outputs saved to:")
    log(f"  {RESULTS_DIR}/")
    log("  paper_stats.json, table1_characteristics.csv, table2_crosslag.csv")
    log("  fig1_study_design, fig2_microbiome, fig3_crosslag, fig4_comparison")
    log("  (each as .pdf + .png at 300 dpi)")
    log("=" * 65)


if __name__ == "__main__":
    main()
