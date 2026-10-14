#!/usr/bin/env python3
"""
Batch-runs one heuristic over every instance in a folder, `--reps` independent
replications each, and writes one CSV row per (instance, replication) --
the same format as the CSVs in results/.

Seeding (must match to reproduce the paper's results exactly): instances are
sorted alphabetically; instance i (0-based) gets base_seed = (i+1) * reps;
replication r of that instance uses seed = base_seed + r. KMeans clustering
is frozen (see instances/README.md), not reseeded per replication.

Usage:
    python3 scripts/run_experiments.py --method grasp-ils-pr \\
        --folder instances/main --pattern "M-*.txt" --reps 10 --time 30 \\
        --workers 14 --output results/GRASP_ILS_PR_main.csv \\
        --solutions-dir solutions/grasp_ils_pr_main
"""
import argparse
import glob
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
from common import parse_instance, cluster_victims, compute_distances  # noqa: E402
from run_heuristic import build_solver, to_solution_dict  # noqa: E402

RUN_COLS = ["instance", "rep", "seed", "objective", "feasible", "objective_check", "iterations", "runtime_s"]


def instance_names(folder, pattern):
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(folder, pattern)))


def independent_check(inst, clustered, dist, routes, assignment, objective):
    """Recomputes feasibility and the objective from scratch, independently
    of the solver's own bookkeeping."""
    n, end = inst.n, inst.n + 1
    notes = []
    sats = [s for r in routes.values() for s in r]
    if sorted(sats) != list(range(1, n + 1)):
        notes.append("a satellite is not visited exactly once")
    sigma = {int(s): int(c) for s, c in assignment.items()}
    if sorted(sigma) != list(range(1, n + 1)) or sorted(sigma.values()) != list(range(1, n + 1)):
        notes.append("satellite->centroid assignment is not a bijection")
    obj_ok = False
    if not notes:
        dem = clustered.demand_n
        for r in routes.values():
            if sum(dem[sigma[i] - 1] for i in r) > inst.q + 1e-9:
                notes.append("truck capacity violated"); break
        if any(dem[c - 1] > inst.no_u * inst.p + 1e-9 for c in sigma.values()):
            notes.append("drone capacity violated")
        total, path = 0.0, {}
        for k, r in routes.items():
            full = [0] + r + [end]
            t = 0.0
            for a, b in zip(full[:-1], full[1:]):
                t += dist.time_n_0[a, b]
                total += t
                if b != end:
                    t += 2.0 * dist.time_n_m[b - 1, sigma[b] - 1]
        obj_ok = abs(total - objective) <= 1e-9 * max(1.0, abs(objective))
    return (not notes), obj_ok


def run_one(method, folder, name, rep, time_limit, use_evpr, base_seed):
    inst = parse_instance(os.path.join(folder, name + ".txt"))
    clustered = cluster_victims(inst, random_state=0)
    dist = compute_distances(inst, clustered)
    seed = base_seed + rep
    np.random.seed(seed)
    solver = build_solver(method, inst, clustered, dist, time_limit, use_evpr)
    result = solver.solve()

    row = dict(instance=name, rep=rep, seed=seed, objective=result.objective, feasible=False,
               objective_check=False, iterations=result.iterations, runtime_s=result.runtime_s)
    solution = None
    if result.objective is not None:
        solution = to_solution_dict(name, result)
        row["feasible"], row["objective_check"] = independent_check(
            inst, clustered, dist, solution["routes"], solution["assignment"], result.objective)
    return row, solution


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("grasp-ils-pr", "grasp-evpr", "sa"), required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="M-*.txt")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--time", type=float, default=30.0, help="Time budget t_max in seconds.")
    ap.add_argument("--use-evpr", action="store_true", help="[grasp-ils-pr] enable EvPR (off by default).")
    ap.add_argument("--no-evpr", action="store_true", help="[grasp-evpr] disable EvPR (on by default).")
    ap.add_argument("--workers", type=int, default=14, help="Parallel worker processes (paper: 14, single-threaded each).")
    ap.add_argument("--output", required=True, help="Per-replication CSV output path.")
    ap.add_argument("--solutions-dir", default=None, help="If set, save each instance's best solution as JSON here.")
    args = ap.parse_args()

    use_evpr = args.use_evpr if args.method == "grasp-ils-pr" else not args.no_evpr
    names = instance_names(args.folder, args.pattern)
    print(f"{len(names)} instances x {args.reps} reps, method={args.method}, t_max={args.time}s, {args.workers} workers")

    tasks = [(n, r, (i + 1) * args.reps) for i, n in enumerate(names) for r in range(args.reps)]
    pending = {n: {} for n in names}
    all_rows = []
    if args.solutions_dir:
        os.makedirs(args.solutions_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.method, args.folder, n, r, args.time, use_evpr, base): (n, r)
                   for n, r, base in tasks}
        done = 0
        for fut in as_completed(futures):
            n, r = futures[fut]
            row, solution = fut.result()
            pending[n][r] = (row, solution)
            all_rows.append(row)
            done += 1
            if len(pending[n]) == args.reps:
                reps = [pending[n][k] for k in range(args.reps)]
                feasible = [(row["objective"], sol) for row, sol in reps if row["objective"] is not None and row["feasible"]]
                if feasible and args.solutions_dir:
                    best_obj, best_sol = min(feasible, key=lambda t: t[0])
                    with open(os.path.join(args.solutions_dir, n + ".json"), "w") as f:
                        json.dump(best_sol, f, indent=1)
                objs = [row["objective"] for row, _ in reps if row["objective"] is not None]
                print(f"[{done}/{len(tasks)}] {n}: min={min(objs) if objs else None} "
                      f"mean={statistics.fmean(objs) if objs else None} failed={args.reps - len(objs)}")

    import csv
    all_rows.sort(key=lambda r: (r["instance"], r["rep"]))
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Done. Wrote {args.output}" + (f" and {args.solutions_dir}/" if args.solutions_dir else ""))


if __name__ == "__main__":
    main()
