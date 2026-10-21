#!/usr/bin/env python3
"""
POMDP experiment: how should the planner use noisy, partial observations?

Road reports: a changed road is reported with probability --reveal (default 0.6) and the
report is correct with probability --accuracy (default 0.8). Surge demand is reported with
relative noise --demand-noise (default 0.15) until a truck reaches the cluster.
Every run uses the warm strategy with the same budget; only the planner's reading of the
observations changes:

  oracle     knows the true road states and demand (upper bound)
  naive      trusts every report, assumes unreported roads unchanged, uses reported demand
  belief     Bayesian belief (Markov prediction + Bayes update), expected travel times,
             capacity/battery checked at mean + 1 sd of the demand belief
  belief_cc  belief + chance constraint (proposal slide 10): roads with
             P(Open)+P(Degraded) < theta are planned as detours

    python3 scripts/run_pomdp_experiment.py --workers 2 --output results/dynamic_pomdp.csv
"""
import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

PLANNERS = ("oracle", "naive", "belief", "belief_cc")
FIELDS = ["instance", "n", "K", "scenario", "seed", "planner", "f1", "f2", "f3", "C_del", "unmet_ratio",
          "n_redeliveries", "max_battery_use", "replan_cpu_s", "gap0_f1", "gap0_f3", "gap_last_f1", "gap_last_f3"]


def find_instance(name):
    for sub in ("main", "large", "solomon"):
        p = os.path.join(ROOT, "instances", sub, f"{name}.txt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def job(a):
    name, scenario, seed, t, reveal, acc, dnoise, theta = a
    os.chdir(ROOT)
    from dynamic_runner import load_world, run_closed_loop
    from dynamic_solver import ReoptConfig
    w = load_world(find_instance(name), os.path.join("instances", "dynamic_events", f"{name}.events.json"), scenario)
    base = ReoptConfig(mode="so", weights=(0, 0, 1))
    rows = []
    for pl in PLANNERS:
        out = run_closed_loop(w, ["warm"], replace(base, time_limit=2 * t), replace(base, time_limit=t),
                              seed=seed, planner=pl, theta=theta, reveal_prob=reveal, obs_accuracy=acc,
                              demand_noise=dnoise)
        m = out["warm"]["metrics"]
        rows.append(dict(instance=name, n=w.inst.n, K=w.inst.no_k, scenario=scenario, seed=seed, planner=pl,
                         **{k: m[k] for k in ("f1", "f2", "f3", "C_del", "unmet_ratio", "n_redeliveries",
                                              "max_battery_use", "replan_cpu_s", "gap0_f1", "gap0_f3",
                                              "gap_last_f1", "gap_last_f3")}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="damage_30pct,damage_50pct")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reveal", type=float, default=0.6)
    ap.add_argument("--accuracy", type=float, default=0.8)
    ap.add_argument("--demand-noise", type=float, default=0.15)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--time-small", type=float, default=1.0)
    ap.add_argument("--time-medium", type=float, default=1.5)
    ap.add_argument("--time-large", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    names = sorted(os.path.basename(f).replace(".events.json", "")
                   for f in glob.glob("instances/dynamic_events/*.events.json"))
    jobs = []
    for name in names:
        n = int(name.split("-")[2])
        t = args.time_small if n <= 15 else (args.time_medium if n <= 20 else args.time_large)
        for sc in args.scenarios.split(","):
            for seed in map(int, args.seeds.split(",")):
                jobs.append((name, sc, seed, t, args.reveal, args.accuracy, args.demand_noise, args.theta))
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(job, j): j for j in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                j = futs[fu]
                try:
                    rows = fu.result()
                except Exception as e:
                    print(f"[{i}/{len(jobs)}] {j[:3]} FAILED {e!r}", flush=True)
                    continue
                for r in rows:
                    w.writerow(r)
                f.flush()
                print(f"[{i}/{len(jobs)}] {j[0]} {j[1]} seed={j[2]} Z1: " +
                      " ".join(f"{r['planner']}={r['f3']:.3f}" for r in rows), flush=True)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
