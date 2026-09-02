#!/usr/bin/env python3
"""
Fast, standalone check of the physical-activity coding fix in
hpp_prototype._build_pa(). Does NOT run confounder_sensitivity.py's full
pipeline (which trains the leak-free clock via LightGBM CV and is slow) --
loads only the "lifestyle" table and calls _build_pa() directly. _build_pa()
already prints the mean/SD/median/max/hard-ceiling diagnostic internally, so
no extra code is needed here beyond calling it.

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python3 check_pa_fix.py
"""
import hpp_prototype as hproto

lif = hproto._load("lifestyle")
cols = ["study_id"] + [c for c in
        ("activity_vigorous_days_weekly", "activity_vigorous_minutes_daily",
         "activity_moderate_days_weekly", "activity_moderate_minutes_daily",
         "activity_walking_10min_days_weekly", "activity_walking_minutes_daily")
        if c in lif.columns]
print(f"lifestyle table: {len(lif):,} rows, activity columns found: {len(cols)-1}/6")
lif_small = lif[cols].drop_duplicates("study_id")
pa = hproto._build_pa(lif_small)
print("\nDONE -- if the line above shows mean/SD/median/max, the fix is live.")
