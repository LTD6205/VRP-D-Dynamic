#!/usr/bin/env python3
"""
Independent verification of a heuristic solution with Gurobi (paper Section 5.2).

Builds the MILP of src/milp.py, FIXES x, y, z (satellite->centroid assignment)
and delta to the solution's routes and assignment, and solves for the
remaining continuous variables (arrival/waiting times a, w). The solution is
verified if the fixed-variable model is feasible (status OPTIMAL) and the
resulting MILP objective equals the solution's own claimed objective within a
relative tolerance.

A solution file is JSON with:
  {"instance": "M-09-09-60-1", "objective": 17.27...,
   "routes": {"1": [8, 5, 2], "2": [3, 6], ...},        # per truck, satellites in visiting order (no depot)
   "assignment": {"8": 3, "5": 5, "2": 4, ...}}          # satellite -> centroid

Usage:
    # one solution
    python3 scripts/verify_solution.py --instance instances/main/M-09-09-60-1.txt \\
        --solution solutions/M-09-09-60-1.json

    # every solution in a directory (one <name>.json per <name>.txt)
    python3 scripts/verify_solution.py --instances-dir instances/main \\
        --solutions-dir solutions/ils --output results/verification_check.csv
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import milp  # noqa: E402

TOL = 1e-6


def verify_one(instance_path: str, solution: dict) -> dict:
    name = solution["instance"]
    rec = dict(instance=name, heuristic_objective=solution["objective"],
               milp_objective_fixed=None, status="", relative_difference=None, ok=False)
    inst = milp.parse_instance(instance_path)
    clustered = milp.cluster_victims(inst, random_state=0)
    dist = milp.compute_distances(inst, clustered)
    ctx = milp.build_model(inst, clustered, dist, mip_gap=0.0, time_limit=600, seed=0, threads=1, output_flag=0)
    try:
        routes = [list(map(int, v)) for _, v in sorted(solution["routes"].items(), key=lambda kv: int(kv[0]))]
        sigma = {int(s): int(c) for s, c in solution["assignment"].items()}
        if len(routes) != len(ctx.K):
            rec["status"] = f"NUM_ROUTES({len(routes)})_NE_K({len(ctx.K)})"
            return rec

        arcs, visited = {}, {}
        for k, route in enumerate(routes):
            path = [ctx.v0] + route + [ctx.vend]
            for a, b in zip(path[:-1], path[1:]):
                arcs[(a, b, k)] = 1
            for s in route:
                visited[(s, k)] = 1

        for (i, j, k), var in ctx.x.items():
            var.LB = var.UB = arcs.get((i, j, k), 0)
        for (i, k), var in ctx.y.items():
            var.LB = var.UB = visited.get((i, k), 0)
        for (i, j), var in ctx.z.items():
            var.LB = var.UB = 1 if sigma.get(i) == j else 0
        for (i, j, k), var in ctx.delta.items():
            var.LB = var.UB = 1 if (visited.get((i, k), 0) and sigma.get(i) == j) else 0
        ctx.mdl.update()
        ctx.mdl.optimize()

        rec["status"] = milp.STATUS_MAP.get(ctx.mdl.Status, f"STATUS_{ctx.mdl.Status}")
        if ctx.mdl.SolCount > 0:
            rec["milp_objective_fixed"] = ctx.mdl.ObjVal
            rec["relative_difference"] = (ctx.mdl.ObjVal - solution["objective"]) / solution["objective"]
            rec["ok"] = rec["status"] == "OPTIMAL" and abs(rec["relative_difference"]) <= TOL
    finally:
        ctx.mdl.dispose()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", help="Path to one instance .txt file.")
    ap.add_argument("--solution", help="Path to that instance's solution .json file.")
    ap.add_argument("--instances-dir", help="Batch mode: verify every solution against its instance.")
    ap.add_argument("--solutions-dir", help="Batch mode: directory of <name>.json solution files.")
    ap.add_argument("--output", default=None, help="Batch mode: write results to this CSV.")
    args = ap.parse_args()

    if args.instance and args.solution:
        solution = json.load(open(args.solution))
        rec = verify_one(args.instance, solution)
        print(rec)
        sys.exit(0 if rec["ok"] else 1)

    if args.instances_dir and args.solutions_dir:
        import pandas as pd
        rows = []
        for sol_path in sorted(glob.glob(os.path.join(args.solutions_dir, "*.json"))):
            solution = json.load(open(sol_path))
            instance_path = os.path.join(args.instances_dir, solution["instance"] + ".txt")
            rec = verify_one(instance_path, solution)
            rows.append(rec)
            print(f"{rec['instance']}: status={rec['status']} ok={rec['ok']}")
        df = pd.DataFrame(rows)
        print(f"\nVerified OK: {int(df.ok.sum())}/{len(df)}")
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote {args.output}")
        sys.exit(0 if df.ok.all() else 1)

    ap.error("give either --instance/--solution or --instances-dir/--solutions-dir")


if __name__ == "__main__":
    main()
