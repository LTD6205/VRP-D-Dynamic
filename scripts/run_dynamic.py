#!/usr/bin/env python3
"""
Run the rolling-horizon dynamic 2E-CTVRP on one instance + one scenario.

    python3 scripts/run_dynamic.py --instance instances/main/M-09-09-60-1.txt \\
        --scenario damage_30pct --strategies static,restart,warm --time 3 --seed 0 -v

Options:
    --events      events file (default instances/dynamic_events/<name>.events.json)
    --mode        mo (Pareto archive, default) | so (single objective, --weights)
    --weights     weights for so mode / commit weights for mo mode, e.g. 0,0,1 (= paper's Z1)
    --selector    adaptive (ALNS roulette, default) | uniform
    --planner     oracle | naive | belief | belief_cc (POMDP: how observations are used)
    --reveal / --accuracy / --demand-noise   observation model (POMDP)
    --selector    adaptive | uniform | dqn (trained DRL controller)
    --paired-hv   at every epoch also solve the warm strategy's residual problem cold
                  and report the hypervolume of both archives
    --output      write full JSON (metrics, per-epoch records, final plan)
    --drone-model centroid (paper: drones fly to the cluster centroid) | route (drones visit
                  every victim: multi-trip CVRP per cluster)
    --setup       drone launch,recovery,drop times in minutes, e.g. 1,2,0.1 (default 0,0,0)
    --surge-scale dynamic degree: every surge multiplier m becomes 1 + s (m - 1)
    --strategies  may include hybrid (repair first, search only if needed) and hybrid@0.05
    --no-road-events   ignore road damage (demand dynamics only)
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dynamic_runner import load_world, run_closed_loop  # noqa: E402
from dynamic_solver import ReoptConfig  # noqa: E402
from drone_service import SetupTimes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--scenario", default="damage_30pct")
    ap.add_argument("--strategies", default="static,restart,warm")
    ap.add_argument("--time", type=float, default=3.0, help="CPU seconds per re-planning epoch")
    ap.add_argument("--time0", type=float, default=None, help="CPU seconds for the t=0 plan (default 2x --time)")
    ap.add_argument("--mode", choices=("mo", "so"), default="mo")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--selector", choices=("adaptive", "uniform", "dqn"), default="adaptive")
    ap.add_argument("--selector-model", default="models/dqn_selector.npz", help="trained DQN weights")
    ap.add_argument("--planner", choices=("oracle", "naive", "belief", "belief_cc"), default="belief",
                    help="how the planner reads (partial/noisy) observations")
    ap.add_argument("--reveal", type=float, default=None, help="probability a changed road is reported")
    ap.add_argument("--accuracy", type=float, default=None, help="probability a road report is correct")
    ap.add_argument("--demand-noise", type=float, default=0.0, help="relative s.d. of reported surge demand")
    ap.add_argument("--theta", type=float, default=0.8, help="chance-constraint threshold (belief_cc)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paired-hv", action="store_true")
    ap.add_argument("--no-recourse", action="store_true",
                    help="disable re-delivery jobs (surges after delivery then stay unmet)")
    ap.add_argument("--drone-model", choices=("centroid", "route"), default="centroid")
    ap.add_argument("--setup", default="0,0,0", help="drone launch,recovery,drop minutes")
    ap.add_argument("--surge-scale", type=float, default=1.0)
    ap.add_argument("--hybrid-tol", type=float, default=0.02)
    ap.add_argument("--no-road-events", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    name = os.path.splitext(os.path.basename(args.instance))[0]
    ev_dir = "solomon_events" if os.path.basename(os.path.dirname(os.path.abspath(args.instance))) == "solomon" \
        else "dynamic_events"
    events = args.events or os.path.join("instances", ev_dir, f"{name}.events.json")
    world = load_world(args.instance, events, args.scenario)

    w = tuple(float(x) for x in args.weights.split(",")) if args.weights else None
    base = ReoptConfig(mode=args.mode)
    if w is not None:
        base = ReoptConfig(mode=args.mode, weights=w, commit_weights=w)
    from dataclasses import replace
    cfg = replace(base, time_limit=args.time)
    cfg0 = replace(base, time_limit=args.time0 if args.time0 is not None else 2 * args.time)

    out = run_closed_loop(world, args.strategies.split(","), cfg0, cfg, seed=args.seed,
                          selector=args.selector, paired_hv_check=args.paired_hv, verbose=args.verbose,
                          recourse=not args.no_recourse, planner=args.planner, theta=args.theta,
                          reveal_prob=args.reveal, obs_accuracy=args.accuracy,
                          demand_noise=args.demand_noise, selector_model=args.selector_model,
                          drone_model=args.drone_model,
                          setup=SetupTimes.from_minutes(*map(float, args.setup.split(","))),
                          surge_scale=args.surge_scale, hybrid_tol=args.hybrid_tol,
                          road_events=not args.no_road_events)
    print(f"\n{name} / {args.scenario}  (realized after executing the whole horizon)")
    print(f"{'strategy':10s} {'f1 prio-wtd delivery':>21s} {'f2 spread':>10s} {'f3 Z1':>9s} "
          f"{'C_del':>7s} {'unmet%':>7s} {'re-deliv':>8s} {'replan s':>9s} {'drone trips':>11s} {'repaired':>8s}")
    for s, r in out.items():
        m = r["metrics"]
        print(f"{s:10s} {m['f1']:21.4f} {m['f2']:10.4f} {m['f3']:9.4f} {m['C_del']:7.3f} "
              f"{100 * m['unmet_ratio']:7.2f} {m['n_redeliveries']:8d} {m['replan_cpu_s']:9.2f} "
              f"{m['total_drone_trips']:11d} {m['n_repaired']:8d}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(instance=name, scenario=args.scenario, seed=args.seed, results=out), f, indent=1,
                      default=float)
        print("Wrote", args.output)


if __name__ == "__main__":
    main()
