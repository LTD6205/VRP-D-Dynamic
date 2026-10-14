#!/usr/bin/env python3
"""
Run one of the three heuristics on a single instance and print/save its result.

Usage:
    python3 scripts/run_heuristic.py --method grasp-ils-pr \\
        --instance instances/main/M-09-09-60-1.txt --time 30 --seed 0

    # save the best solution as JSON (for scripts/verify_solution.py or
    # scripts/compute_fleet_metrics.py)
    python3 scripts/run_heuristic.py --method sa --instance instances/main/M-09-09-60-1.txt \\
        --time 30 --seed 0 --output solutions/M-09-09-60-1.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
from common import parse_instance, cluster_victims, compute_distances  # noqa: E402

METHODS = ("grasp-ils-pr", "grasp-evpr", "sa")


def build_solver(method, inst, clustered, dist, time_limit, use_evpr):
    if method == "grasp-ils-pr":
        from grasp_ils_pr import GraspILSConfig, GraspILSSolver
        cfg = GraspILSConfig(time_limit=time_limit, use_evpr=use_evpr)
        return GraspILSSolver(inst, clustered, dist, cfg)
    if method == "grasp-evpr":
        from grasp_evpr import GraspEvPRConfig, GraspEvPRSolver
        cfg = GraspEvPRConfig(time_limit=time_limit, use_evpr=use_evpr)
        return GraspEvPRSolver(inst, clustered, dist, cfg)
    from sa import SAConfig, SimulatedAnnealingSolver
    cfg = SAConfig(time_limit=time_limit)
    return SimulatedAnnealingSolver(inst, clustered, dist, cfg)


def to_solution_dict(inst_name, result):
    """Converts a solver's native RunResult (truck_route: truck -> full path
    including the depot at both ends; drone_route: satellite -> [satellite,
    centroid]) into the routes/assignment format used by
    scripts/verify_solution.py and scripts/compute_fleet_metrics.py."""
    routes = {str(k): [int(x) for x in v[1:-1]] for k, v in result.truck_route.items()}
    assignment = {str(int(v[0])): int(v[1]) for v in result.drone_route.values()}
    return dict(instance=inst_name, objective=result.objective, routes=routes, assignment=assignment)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--instance", required=True, help="Path to the instance .txt file.")
    ap.add_argument("--time", type=float, default=30.0, help="Time budget t_max in seconds.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-evpr", action="store_true", help="[grasp-ils-pr] enable EvPR (off by default).")
    ap.add_argument("--no-evpr", action="store_true", help="[grasp-evpr] disable EvPR (on by default).")
    ap.add_argument("--output", default=None, help="If set, write the best solution as JSON here.")
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    clustered = cluster_victims(inst, random_state=0)
    dist = compute_distances(inst, clustered)

    use_evpr = args.use_evpr if args.method == "grasp-ils-pr" else not args.no_evpr
    np.random.seed(args.seed)
    solver = build_solver(args.method, inst, clustered, dist, args.time, use_evpr)
    result = solver.solve()

    if result.objective is None:
        print(f"{inst.name}: FAILED (no feasible solution found)")
        raise SystemExit(1)

    print(f"{inst.name}: objective={result.objective:.6f} iterations={result.iterations} "
          f"runtime_s={result.runtime_s:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(to_solution_dict(inst.name, result), f, indent=1)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
