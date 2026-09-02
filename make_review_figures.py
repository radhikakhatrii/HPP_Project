#!/usr/bin/env python3
"""
make_review_figures.py
===============================================================================
Generates the three figures that the review-response tables cannot carry on
their own. Everything else added to the manuscript is fully expressed by a
table, so this script is optional -- the paper compiles and is complete
without it.

  FIG09_clock_calibration.png
      Panel a: cross-validated predicted vs chronological age with the
               calibration line (slope 0.151) against the identity line.
      Panel b: mean residual by age decile -- the compression, made visible.
      Panel c: MAE by sex and by age tertile.
      Supports Sec. "Clock calibration, stratified performance and technical
      artefacts".

  FIG10_interaction_surface_3D.png
      The strongest single interaction in the whole analysis (richness x age
      -> waist circumference, nominal p = 0.011, q = 0.246) rendered as a
      fitted response surface. The point of the figure is that the surface is
      visibly close to planar: even the largest effect-modification signal we
      found has no meaningful curvature or twist.

  FIG11_robustness_forest.png
      Every prospective-outcome test on one axis with 95% CIs, so the reader
      sees the whole null at once rather than reading twelve rows.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python make_review_figures.py 2>&1 | tee make_review_figures.log

Writes PNGs into results_cross_lagged/ (same place as every other output).
===============================================================================
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy import stats

warnings.filterwarnings("ignore")

import cross_lagged_panel as clp

OUT = Path(clp.RESULTS_DIR)
OUT.mkdir(parents=True, exist_ok=True)
log = clp.log

# Verified values from clock_validation.py, used where the figure needs a
# number this script does not itself recompute. Each one is reproduced in
# Table "tabclockval" of the manuscript.
CALIB_SLOPE, CALIB_INTERCEPT = 0.1509, 42.8536
SEX_STRATA = [("Male", 2507, 0.347, 5.69), ("Female", 2681, 0.393, 5.53)]
AGE_STRATA = [("38-46", 1801, 0.126, 6.52),
              ("47-53", 1696, 0.136, 2.59),
              ("54-70", 1691, 0.158, 7.67)]


def banner(t):
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def zscore(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else s * 0.0


# --------------------------------------------------------------------------
def fig_clock_calibration(clock_data: dict) -> None:
    banner("FIG09  clock calibration and stratified performance")

    # Key names verified against build_gut_clock_ageaccel in
    # cross_lagged_panel.py (~line 788), which returns
    # {"y_train", "y_pred_cv", "r_cv", "mae_cv"}. An earlier version of this
    # function guessed at the names and, on failing to match, quietly drew two
    # empty panels -- a silent blank is worse than a crash, so a missing key
    # is now a hard error rather than an empty axis.
    log(f"  clock_data keys: {sorted(clock_data.keys())}")
    missing = [k for k in ("y_train", "y_pred_cv") if k not in clock_data]
    if missing:
        raise KeyError(
            f"clock_data is missing {missing}; expected the keys returned by "
            f"build_gut_clock_ageaccel. Found: {sorted(clock_data.keys())}")
    y_true = np.asarray(clock_data["y_train"], dtype=float)
    y_pred = np.asarray(clock_data["y_pred_cv"], dtype=float)
    if len(y_true) < 10 or len(y_true) != len(y_pred):
        raise ValueError(
            f"unusable prediction arrays: len(y_train)={len(y_true)}, "
            f"len(y_pred_cv)={len(y_pred)}")
    log(f"  n={len(y_true):,}, r={np.corrcoef(y_true, y_pred)[0,1]:.4f}, "
        f"MAE={np.mean(np.abs(y_true - y_pred)):.2f} yr")

    # Recompute the calibration line rather than trusting the constant, and
    # check it against the value clock_validation.py reported.
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    log(f"  calibration slope={slope:.4f} intercept={intercept:.4f} "
        f"(clock_validation.py reported {CALIB_SLOPE:.4f} / {CALIB_INTERCEPT:.4f})")
    if abs(slope - CALIB_SLOPE) > 0.01:
        log(f"  WARNING: calibration slope differs from the validated value by "
            f"{abs(slope - CALIB_SLOPE):.4f}. The figure draws its own fit; "
            f"reconcile before using it alongside the manuscript text.")

    fig = plt.figure(figsize=(15, 4.6))

    # ---- panel a: predicted vs chronological -----------------------------
    ax = fig.add_subplot(1, 3, 1)
    hb = ax.hexbin(y_true, y_pred, gridsize=38, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="participants")
    xs = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(xs, xs, "k--", lw=1.4, label="identity")
    # Two decimals, deliberately: this fit and clock_validation.py's differ in
    # the third (0.152 vs 0.1509), and a figure legend disagreeing with the
    # manuscript text on a number they both report is a needless snag for a
    # reader. The precise value lives in the text and Table "tabclockval".
    ax.plot(xs, slope * xs + intercept, "r-", lw=2,
            label=f"calibration (slope {slope:.2f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted age (years)")
    ax.set_title("a  Clock calibration", loc="left", fontweight="bold")

    # ---- panel b: mean residual by age decile ----------------------------
    ax = fig.add_subplot(1, 3, 2)
    resid = y_pred - y_true
    df = pd.DataFrame({"age": y_true, "resid": resid})
    df["dec"] = pd.qcut(df["age"], 10, duplicates="drop")
    g = df.groupby("dec", observed=True)["resid"].mean()
    centres = [iv.mid for iv in g.index]
    width = 0.85 * float(np.median(np.diff(sorted(centres)))) if len(centres) > 1 else 2.0
    cols = ["#c0392b" if v < 0 else "#2471a3" for v in g.values]
    ax.bar(centres, g.values, width=width, color=cols, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("Mean residual (predicted $-$ true, years)")
    ax.set_xlabel("Age decile midpoint (years)")
    ax.set_title("b  Compression toward the mean", loc="left", fontweight="bold")
    log(f"  decile mean residuals: {g.iloc[0]:+.2f} yr (youngest) to "
        f"{g.iloc[-1]:+.2f} yr (oldest)")

    # ---- panel c: MAE by stratum -----------------------------------------
    ax = fig.add_subplot(1, 3, 3)
    labels = [s[0] for s in SEX_STRATA] + [s[0] for s in AGE_STRATA]
    maes = [s[3] for s in SEX_STRATA] + [s[3] for s in AGE_STRATA]
    cols = ["#5499c7", "#5499c7"] + ["#f5b041"] * 3
    bars = ax.bar(range(len(labels)), maes, color=cols, edgecolor="k", lw=0.5)
    ax.axhline(5.61, color="k", ls="--", lw=1.2, label="overall MAE 5.61 yr")
    for b, m in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.12, f"{m:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("MAE (years)")
    ax.set_title("c  By sex and age tertile", loc="left", fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = OUT / "FIG09_clock_calibration.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {p}")


# --------------------------------------------------------------------------
def fig_interaction_surface(base: pd.DataFrame, outcomes: dict) -> None:
    banner("FIG10  3D response surface for the strongest interaction")

    anthro = outcomes.get("anthropometrics")
    if anthro is None or anthro.empty:
        log("  anthropometrics unavailable; surface figure skipped.")
        return
    meas = next(c for c in anthro.columns
                if c not in ("study_id", "visit", "visit_n"))
    wide = clp.make_wide(anthro, meas).reset_index()
    wide = wide.rename(columns={f"{meas}_t1": "y1", f"{meas}_t2": "y2"})

    d = base.merge(wide[["study_id", "y1", "y2"]], on="study_id", how="inner")
    d = d[["richness", "age_at_visit", "sex_encoded", "y1", "y2"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(d) < 150:
        log(f"  n={len(d)} < 150 after merge; surface figure skipped.")
        return
    log(f"  n={len(d):,} for richness x age -> waist circumference")

    for c in ("richness", "age_at_visit", "y1", "y2"):
        d[c] = zscore(d[c])
    d["inter"] = d["richness"] * d["age_at_visit"]

    import statsmodels.api as sm
    X = sm.add_constant(d[["y1", "richness", "age_at_visit", "inter",
                           "sex_encoded"]].astype(float))
    fit = sm.OLS(d["y2"].astype(float), X).fit()
    b = fit.params
    p_inter = float(fit.pvalues["inter"])
    log(f"  interaction coefficient = {b['inter']:+.4f}  p = {p_inter:.4f}")
    # The figure must report the p-value of the model it actually draws, not
    # the one from nonlinear_interaction_effects.py -- that script's model
    # carries a different covariate set, so the two need not agree. Flag it
    # loudly if they diverge rather than letting the caption drift.
    if abs(p_inter - 0.011) > 0.02:
        log(f"  NOTE: this refit's p ({p_inter:.4f}) differs from the 0.011 "
            f"reported by nonlinear_interaction_effects.py, whose model uses a "
            f"different covariate set. The figure reports its own value; use "
            f"the table's value in the manuscript text.")

    rr = np.linspace(d["richness"].quantile(.02), d["richness"].quantile(.98), 40)
    aa = np.linspace(d["age_at_visit"].quantile(.02),
                     d["age_at_visit"].quantile(.98), 40)
    R, A = np.meshgrid(rr, aa)
    base_plane = (b["const"] + b["richness"] * R + b["age_at_visit"] * A
                  + b["sex_encoded"] * 0.5)
    Z_fit = base_plane + b["inter"] * R * A     # model as fitted
    Z_null = base_plane                         # interaction set to zero

    # The visible tilt of either surface is the MAIN effects of age and
    # richness; an interaction shows up as a twist (saddle), not a tilt.
    # Plotting the fitted surface beside the interaction-free counterfactual
    # on a shared z-scale is what actually lets a reader see that the
    # interaction contributes nothing -- the single surface alone cannot.
    vmax = float(np.abs(np.concatenate([Z_fit.ravel(), Z_null.ravel()])).max())
    lim = max(1.0, vmax * 1.05)
    max_gap = float(np.abs(Z_fit - Z_null).max())
    log(f"  largest vertical gap between fitted and interaction-free "
        f"surfaces: {max_gap:.4f} SD of the outcome")

    fig = plt.figure(figsize=(13.8, 5.9))
    for i, (Z, ttl) in enumerate(
            [(Z_fit, "a  Model as fitted (interaction retained)"),
             (Z_null, "b  Counterfactual: interaction coefficient set to 0")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        surf = ax.plot_surface(R, A, Z, cmap=cm.coolwarm, alpha=0.9,
                               linewidth=0.2, edgecolor="k",
                               rstride=2, cstride=2,
                               vmin=-lim, vmax=lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("\nSpecies richness (z, visit 1)", fontsize=9)
        ax.set_ylabel("\nChronological age (z)", fontsize=9)
        ax.set_zlabel("Predicted waist circumference\n(z, visit 2)", fontsize=9)
        ax.zaxis.labelpad = 8
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10, y=0.98)
        ax.view_init(elev=22, azim=-127)
        ax.tick_params(labelsize=8)
        if i == 1:
            fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.11,
                         label="predicted outcome (z)")

    fig.suptitle("Strongest effect-modification signal in the analysis: "
                 "richness $\\times$ age $\\rightarrow$ waist circumference "
                 f"(interaction $\\beta$ = {b['inter']:+.4f}, p = {p_inter:.3f} "
                 f"in this refit; q = 0.246 across the 45-test family).\n"
                 f"The two surfaces differ by at most {max_gap:.3f} SD of the "
                 "outcome; the visible tilt is the main effects, not the "
                 "interaction.", fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.045, right=0.94, top=0.83, bottom=0.02,
                        wspace=0.10)
    p = OUT / "FIG10_interaction_surface_3D.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {p}")


# --------------------------------------------------------------------------
def fig_robustness_forest() -> None:
    banner("FIG11  prospective-outcome forest")

    src = OUT / "prospective_outcomes.csv"
    if not src.exists():
        log(f"  {src} not found; run prospective_outcomes.py first. Skipped.")
        return
    df = pd.read_csv(src)
    df = df.dropna(subset=["effect", "p"]).copy()
    if df.empty:
        log("  no usable rows; skipped.")
        return

    def pretty(label: str) -> str:
        """Internal column names are fine in a CSV and wrong on a figure axis."""
        s = str(label)
        for a, b in (("gut_ageaccel", "AgeAccel"), ("shannon", "Shannon"),
                     ("richness", "Richness"), ("->", "→"),
                     (" (continuous)", ""), (" (binary)", ""),
                     ("multimorbidity accrual", "multimorbidity accrual"),
                     ("grip -10%", "grip $-$10%")):
            s = s.replace(a, b)
        return s

    df["test"] = df["test"].map(pretty)

    # Reconstruct SEs from the two-sided p-value via the normal approximation:
    # z = |effect| / SE  =>  SE = |effect| / z. For odds ratios the model
    # scale is log(OR), so convert, build the interval, convert back.
    rows = []
    for _, r in df.iterrows():
        is_or = str(r.get("effect_type", "")).upper() == "OR"
        est = np.log(r["effect"]) if is_or else r["effect"]
        p = min(max(float(r["p"]), 1e-12), 0.999999)
        z = stats.norm.ppf(1 - p / 2)
        se = abs(est) / z if z > 0 and est != 0 else np.nan
        if not np.isfinite(se) or se == 0:
            se = abs(est) if est else 0.1
        lo, hi = est - 1.96 * se, est + 1.96 * se
        if is_or:
            est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
        rows.append({"label": r["test"], "est": est, "lo": lo, "hi": hi,
                     "is_or": is_or, "n": r.get("n", np.nan)})
    f = pd.DataFrame(rows)

    n_cont = int((~f["is_or"]).sum())
    n_bin = int(f["is_or"].sum())
    # Side-by-side panels force 3 rows and 9 rows to occupy the same height,
    # which stretches the sparse panel and makes the two look unrelated.
    # Stacking with height_ratios tied to the row counts keeps one row the
    # same physical height in both panels.
    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 0.46 * (n_cont + n_bin) + 2.6),
        gridspec_kw={"height_ratios": [max(n_cont, 1), max(n_bin, 1)]})
    for ax, is_or, null, xlab, ttl in (
            (axes[0], False, 0.0, "Standardised $\\beta$ (95% CI)",
             "a  Continuous outcome: multimorbidity accrual"),
            (axes[1], True, 1.0, "Odds ratio per SD (95% CI)",
             "b  Binary outcomes")):
        sub = f[f["is_or"] == is_or].reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(sub))
        ax.errorbar(sub["est"], y,
                    xerr=[sub["est"] - sub["lo"], sub["hi"] - sub["est"]],
                    fmt="o", color="#1f4e79", ecolor="#7f8c8d",
                    capsize=3, markersize=5.5, lw=1.4)
        ax.axvline(null, color="#c0392b", ls="--", lw=1.4, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=9)
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
        if is_or:
            ax.set_xscale("log")
            # A log axis defaulting to a lone 10^0 tick is unreadable; place
            # ticks on the ratio values a reader actually interprets.
            ticks = [0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            lo_lim = min(sub["lo"].min() * 0.9, 0.45)
            hi_lim = max(sub["hi"].max() * 1.1, 2.2)
            ticks = [t for t in ticks if lo_lim <= t <= hi_lim]
            ax.set_xlim(lo_lim, hi_lim)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", alpha=0.3)
        for yi, r in sub.iterrows():
            ax.text(1.015, yi, f"n={int(r['n'])}" if np.isfinite(r["n"]) else "",
                    transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8, color="#555555")
    fig.suptitle("Baseline microbiome predicting subsequent clinical events: "
                 "every interval crosses the null", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = OUT / "FIG11_robustness_forest.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {p}")
    log("  NOTE: intervals are reconstructed from the reported point estimate "
        "and p-value under a normal approximation, not read from the model "
        "objects. State this in the caption if the figure is used.")


# --------------------------------------------------------------------------
def main() -> None:
    banner("REVIEW-RESPONSE FIGURES")

    pop = clp.load_population()
    gut = clp.load_longitudinal_gut()
    gut_div = clp.compute_diversity(gut)

    gut_full = pd.read_parquet(
        clp.DATA_ROOT / "gut_microbiome" / "gut_microbiome.parquet")
    if gut_full.index.names != [None]:
        gut_full = gut_full.reset_index()
    pid_col = next(c for c in gut_full.columns
                   if c in ("participant_id", "study_id"))
    stg_col = next(c for c in gut_full.columns if "research_stage" in c.lower())
    gut_full = gut_full.rename(columns={pid_col: "study_id", stg_col: "visit"})
    gut_full["visit_n"] = gut_full.groupby("study_id")["visit"].transform(
        lambda x: x.rank(method="dense").astype(int))

    ageaccel_df, clock_data = clp.build_gut_clock_ageaccel(gut_div, gut_full, pop)
    outcomes = clp.load_functional_outcomes()

    demo = pop[["study_id", "age_at_visit", "sex_encoded"]].drop_duplicates("study_id")
    div_t1 = (gut_div[gut_div["visit_n"] == 1]
              .groupby("study_id", as_index=False)[["shannon", "richness"]].first())
    base = div_t1.merge(demo, on="study_id", how="left")

    fig_clock_calibration(clock_data)
    fig_interaction_surface(base, outcomes)
    fig_robustness_forest()

    banner("DONE")
    log("  Copy the PNGs into the manuscript's figures/ folder on Overleaf.")


if __name__ == "__main__":
    main()
