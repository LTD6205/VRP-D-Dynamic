#!/usr/bin/env python3
"""
Computes the fleet-performance metrics of paper Section 6 (Z1, C_ret, C_sat,
C_del, Z2_*, Z3, Z3_del, Tbar_del, m_min/max, util_mean/max) for every
solution in a directory, one CSV row per instance.

Usage:
    python3 scripts/compute_fleet_metrics.py --instances-dir instances/main \\
        --solutions-dir solutions/grasp_ils_pr_main --output results/fleet_metrics.csv
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import parse_instance, cluster_victims, compute_distances  # noqa: E402
from fleet_metrics import METRIC_COLS, TOL, analyze_solution, check_solution, compute_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-dir", required=True)
    ap.add_argument("--solutions-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows, problems = [], []
    for sol_path in sorted(glob.glob(os.path.join(args.solutions_dir, "*.json"))):
        solution = json.load(open(sol_path))
        name = solution["instance"]
        inst = parse_instance(os.path.join(args.instances_dir, name + ".txt"))
        clustered = cluster_victims(inst, random_state=0)
        dist = compute_distances(inst, clustered)
        routes = {k: v for k, v in solution["routes"].items()}
        drone_route = {s: [int(s), int(c)] for s, c in solution["assignment"].items()}
        truck_route = {k: [0] + v + [inst.n + 1] for k, v in routes.items()}
        sigma, sats, trucks = analyze_solution(inst, clustered, dist, truck_route, drone_route)
        D = float(np.asarray(inst.demand_m).sum())

        issues = check_solution(inst, D, sigma, sats, trucks)
        met = compute_metrics(inst, name, sigma, sats, trucks, D)
        if abs(met["Z1"] - solution["objective"]) > TOL * max(1.0, abs(solution["objective"])):
            issues.append(f"Z1 recomputed {met['Z1']!r} != solution objective {solution['objective']!r}")
        if issues:
            problems.append((name, issues))
        rows.append(met)

    df = pd.DataFrame(rows, columns=METRIC_COLS).sort_values("instance")
    df.to_csv(args.output, index=False, float_format="%.10g")
    print(f"Wrote {args.output} ({len(df)} rows)")
    if problems:
        print("\nCONSISTENCY ISSUES:")
        for name, issues in problems:
            for issue in issues:
                print(f"  {name}: {issue}")
        raise SystemExit(1)
    print("All consistency checks passed.")


if __name__ == "__main__":
    main()
