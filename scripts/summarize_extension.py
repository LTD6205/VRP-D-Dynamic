#!/usr/bin/env python3
"""Summarize the extension experiments (drone model, hybrid trigger, dynamic degree, plan gap).

    python3 scripts/summarize_extension.py            # prints markdown tables used in README §4.5-4.8

Inputs (results/): ext_drone_main.csv, ext_drone_solomon.csv, ext_dod_main.csv,
ext_dod_solomon.csv, ext_pomdp_gap.csv. Missing files are skipped.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
KEY = ["instance", "scenario", "seed", "drone_model", "surge_scale"]


def load(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def rel_vs(df, base, other, col):
    """Paired relative change (%) of strategy `other` vs `base` on column col."""
    a = df[df.strategy == base].set_index(KEY)[col]
    b = df[df.strategy == other].set_index(KEY)[col]
    j = a.index.intersection(b.index)
    return 100 * (b[j] - a[j]) / a[j].abs().clip(lower=1e-9)


def pval(x):
    x = x[np.abs(x) > 1e-12]
    return wilcoxon(x).pvalue if len(x) >= 6 else float("nan")


def drone_tables(df, label):
    print(f"\n#### {label}: drone model x strategy ({df.instance.nunique()} instances, "
          f"{len(df[df.strategy == 'static'])} runs per strategy)\n")
    print("| drone model | static Z1 | mean dwell (h) | drone trips | max battery | static unmet runs | gap0 Z1 (static) |"
          " ΔZ1 restart | ΔZ1 warm | ΔZ1 hybrid | Δf1 warm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for dm in ["centroid", "centroid+setup", "route+setup"]:
        d = df[df.drone_model == dm]
        if d.empty:
            continue
        st = d[d.strategy == "static"]
        print(f"| {dm} | {st.f3.mean():.2f} | {st.mean_wait.mean():.3f} | {st.total_drone_trips.mean():.1f} | "
              f"{d.max_battery_use.max():.3f} | {(st.unmet_ratio > 1e-9).sum()}/{len(st)} | "
              f"{100 * st.gap0_f3.mean():+.1f}% | "
              f"{rel_vs(d, 'static', 'restart', 'f3').mean():+.2f}% | {rel_vs(d, 'static', 'warm', 'f3').mean():+.2f}% | "
              f"{rel_vs(d, 'static', 'hybrid', 'f3').mean():+.2f}% | {rel_vs(d, 'static', 'warm', 'f1').mean():+.2f}% |")
    viol = int(df.battery_violations.sum())
    unmet_re = int((df[df.strategy != "static"].unmet_ratio > 1e-9).sum())
    print(f"\nbattery violations: {viol}; re-planning runs with unmet demand: {unmet_re}")
    # centroid -> route: how much the drone model itself changes the realized objective (static plan)
    c = df[(df.strategy == "warm") & (df.drone_model == "centroid")].set_index(["instance", "scenario", "seed"])
    r = df[(df.strategy == "warm") & (df.drone_model == "route+setup")].set_index(["instance", "scenario", "seed"])
    cs = df[(df.strategy == "warm") & (df.drone_model == "centroid+setup")].set_index(["instance", "scenario", "seed"])
    j = c.index.intersection(r.index)
    if len(j):
        print(f"warm, realized Z1: centroid+setup vs centroid {100 * (cs.f3[j] / c.f3[j] - 1).mean():+.1f}%, "
              f"route+setup vs centroid {100 * (r.f3[j] / c.f3[j] - 1).mean():+.1f}%; "
              f"f1 route+setup vs centroid {100 * (r.f1[j] / c.f1[j] - 1).mean():+.1f}%")


def hybrid_table(frames):
    print("\n#### Hybrid trigger vs. warm (paired, same runs)\n")
    print("| set | drone model | tol | epochs repaired | ΔZ1 vs warm | worse than warm | p | CPU vs warm |")
    print("|---|---|---|---|---|---|---|---|")
    for label, df in frames:
        for dm in ["centroid", "centroid+setup", "route+setup"]:
            d = df[df.drone_model == dm]
            if d.empty:
                continue
            w = d[d.strategy == "warm"].set_index(KEY)
            for hs, tol in (("hybrid", "2%"), ("hybrid@0.05", "5%")):
                h = d[d.strategy == hs].set_index(KEY)
                j = w.index.intersection(h.index)
                if not len(j):
                    continue
                dz = 100 * (h.f3[j] - w.f3[j]) / w.f3[j]
                rep = h.n_repaired[j].sum() / (2 * len(j))
                cpu = h.replan_cpu_s[j].sum() / max(w.replan_cpu_s[j].sum(), 1e-9)
                print(f"| {label} | {dm} | {tol} | {100 * rep:.0f}% | {dz.mean():+.2f}% | {100 * (dz > 1e-9).mean():.0f}% | "
                      f"{pval(dz):.2g} | {100 * cpu:.0f}% |")


def dod_table(frames):
    print("\n#### Dynamic degree (demand dynamics only, no road damage; route+setup drones)\n")
    print("| set | surge scale | realized DoD | static unmet (mean) | static runs with unmet | ΔZ1 warm vs static |"
          " Δf1 warm vs static | hybrid epochs repaired | ΔZ1 hybrid vs warm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, df in frames:
        for s in sorted(df.surge_scale.unique()):
            d = df[df.surge_scale == s]
            st = d[d.strategy == "static"]
            h = d[d.strategy == "hybrid"]
            print(f"| {label} | {s:g} | {100 * d[d.strategy == 'warm'].dynamic_degree.mean():.1f}% | "
                  f"{100 * st.unmet_ratio.mean():.2f}% | {(st.unmet_ratio > 1e-9).sum()}/{len(st)} | "
                  f"{rel_vs(d, 'static', 'warm', 'f3').mean():+.2f}% | {rel_vs(d, 'static', 'warm', 'f1').mean():+.2f}% | "
                  f"{100 * h.n_repaired.sum() / (2 * max(len(h), 1)):.0f}% | {rel_vs(d, 'warm', 'hybrid', 'f3').mean():+.2f}% |")


def gap_table(df):
    print("\n#### Plan-vs-actual gap per planner (warm, 30% damage, reveal 0.6, accuracy 0.8, demand noise 15%)\n")
    print("| planner | gap0 Z1 | gap_last Z1 | abs gap_last Z1 | gap_last f1 | ΔZ1 vs oracle |")
    print("|---|---|---|---|---|---|")
    o = df[df.planner == "oracle"].set_index(["instance", "scenario", "seed"])
    for pl in ["oracle", "naive", "belief", "belief_cc"]:
        d = df[df.planner == pl].set_index(["instance", "scenario", "seed"])
        j = d.index.intersection(o.index)
        print(f"| {pl} | {100 * d.gap0_f3.mean():+.1f}% | {100 * d.gap_last_f3.mean():+.2f}% | "
              f"{100 * d.gap_last_f3.abs().mean():.2f}% | {100 * d.gap_last_f1.mean():+.2f}% | "
              f"{100 * (d.f3[j] / o.f3[j] - 1).mean():+.2f}% |")


def main():
    pd.set_option("display.width", 200)
    dm, ds = load("ext_drone_main.csv"), load("ext_drone_solomon.csv")
    if dm is not None:
        drone_tables(dm, "Main set")
    if ds is not None:
        drone_tables(ds, "Solomon-based set")
    frames = [(lbl, x) for lbl, x in (("main", dm), ("Solomon", ds)) if x is not None]
    if frames:
        hybrid_table(frames)
    om, os_ = load("ext_dod_main.csv"), load("ext_dod_solomon.csv")
    f2 = [(lbl, x) for lbl, x in (("main (n<=15)", om), ("Solomon", os_)) if x is not None]
    if f2:
        dod_table(f2)
    g = load("ext_pomdp_gap.csv")
    if g is not None:
        gap_table(g)


if __name__ == "__main__":
    sys.exit(main())
