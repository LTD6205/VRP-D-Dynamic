#!/usr/bin/env python3
"""Summarize results/dynamic_*.csv produced by run_dynamic_experiments.py.

    python3 scripts/summarize_dynamic.py results/dynamic_so_z1.csv [results/dynamic_mo.csv]

Prints, per scale group and scenario, the mean relative change of each realized
objective vs. the static (never re-plan) baseline, and -- when paired HV columns are
present -- warm-vs-cold hypervolume on identical residual problems."""
import sys

import pandas as pd


def scale(n):
    return "small (n<=9)" if n <= 9 else ("medium (n 11-13)" if n <= 13 else "large (n 20-50)")


def summarize(path):
    df = pd.read_csv(path)
    df["scale"] = df["n"].apply(scale)
    key = ["instance", "scenario", "seed"]
    base = df[df.strategy == "static"].set_index(key)
    out = []
    for strat in ("restart", "warm"):
        d = df[df.strategy == strat].set_index(key)
        for m in ("f1", "f2", "f3"):
            d[f"d_{m}%"] = 100 * (d[m] - base[m]) / base[m]
        d["d_unmet_pp"] = 100 * (d["unmet_ratio"] - base["unmet_ratio"])
        d["strategy"] = strat
        out.append(d.reset_index())
    rel = pd.concat(out)
    mode = df["mode"].iloc[0]
    print(f"\n=== {path}  (mode={mode}; negative = better than static) ===")
    tab = rel.groupby(["scale", "scenario", "strategy"])[["d_f1%", "d_f2%", "d_f3%", "d_unmet_pp"]].mean().round(2)
    print(tab.to_string())
    print("\nOverall:")
    print(rel.groupby("strategy")[["d_f1%", "d_f2%", "d_f3%", "d_unmet_pp"]].mean().round(2).to_string())
    win = rel.groupby("strategy").apply(lambda g: (g["d_f3%"] < -1e-6).mean() * 100).round(1)
    print("\n% runs with realized Z1 strictly better than static:", win.to_dict())

    w = df[df.strategy == "warm"]
    if w["hv_warm_e1"].notna().any():
        rows = []
        for e in (1, 2):
            x = w[[f"hv_warm_e{e}", f"hv_cold_e{e}", "scale"]].dropna()
            x = x.assign(diff=x[f"hv_warm_e{e}"] - x[f"hv_cold_e{e}"])
            for sc, g in x.groupby("scale"):
                rows.append(dict(epoch=e, scale=sc, runs=len(g), hv_warm=g[f"hv_warm_e{e}"].mean(),
                                 hv_cold=g[f"hv_cold_e{e}"].mean(),
                                 warm_better=(g["diff"] > 1e-9).mean() * 100,
                                 tie=(g["diff"].abs() <= 1e-9).mean() * 100))
        print("\nPaired HV on identical residual problems (normalized, ref 1.1^3):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    return rel


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
