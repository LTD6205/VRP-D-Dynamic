#!/usr/bin/env python3
"""
Operator-selector experiment: uniform vs adaptive (ALNS roulette) vs trained DQN.

Paired design: during each closed-loop run (warm strategy), the residual problem of every
epoch (t = 0 and each event) is solved once per selector with the same seed, the same
starting solution and the same CPU budget. The best objective each selector reaches on that
identical problem is recorded, so the comparison isolates the operator-selection policy.

    python3 scripts/run_selector_experiment.py --budgets 0.25,1.0 --output results/dynamic_selectors.csv
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

SELECTORS = ["uniform", "adaptive", "dqn"]
FIELDS = ["instance", "n", "K", "scenario", "seed", "budget", "epoch", "selector", "violation", "objective",
          "iterations"]


def find_instance(name):
    for sub in ("main", "large"):
        p = os.path.join(ROOT, "instances", sub, f"{name}.txt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def job(a):
    name, scenario, seed, budget, model = a
    os.chdir(ROOT)
    from dynamic_runner import load_world, run_closed_loop
    from dynamic_solver import ReoptConfig
    w = load_world(find_instance(name), os.path.join("instances", "dynamic_events", f"{name}.events.json"), scenario)
    base = ReoptConfig(mode="so", weights=(0, 0, 1))
    out = run_closed_loop(w, ["warm"], replace(base, time_limit=budget), replace(base, time_limit=budget),
                          seed=seed, selector="adaptive", paired_selectors=SELECTORS, selector_model=model)
    rows = []
    for e in out["warm"]["epochs"]:
        for sname, (viol, obj, iters) in (e.get("paired") or {}).items():
            rows.append(dict(instance=name, n=w.inst.n, K=w.inst.no_k, scenario=scenario, seed=seed,
                             budget=budget, epoch=e["epoch"], selector=sname, violation=viol, objective=obj,
                             iterations=iters))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="0.25,1.0")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--model", default="models/dqn_selector.npz")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.chdir(ROOT)
    from dynamic_env import load_scenarios
    names = sorted(os.path.basename(f).replace(".events.json", "")
                   for f in glob.glob("instances/dynamic_events/*.events.json"))
    jobs = []
    for name in names:
        for sc in load_scenarios(os.path.join("instances", "dynamic_events", f"{name}.events.json")):
            for seed in map(int, args.seeds.split(",")):
                for b in map(float, args.budgets.split(",")):
                    jobs.append((name, sc, seed, b, args.model))
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
                    print(f"[{i}/{len(jobs)}] {j[:4]} FAILED {e!r}", flush=True)
                    continue
                for r in rows:
                    w.writerow(r)
                f.flush()
                print(f"[{i}/{len(jobs)}] {j[0]} {j[1]} seed={j[2]} budget={j[3]}", flush=True)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
