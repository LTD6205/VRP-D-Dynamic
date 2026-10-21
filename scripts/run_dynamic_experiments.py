#!/usr/bin/env python3
"""
Batch experiment for the dynamic 2E-CTVRP: every instance that has an
.events.json x every scenario x seeds x {static, restart, warm}.

    python3 scripts/run_dynamic_experiments.py --mode so --weights 0,0,1 \\
        --seeds 0,1,2 --workers 2 --output results/dynamic_so_z1.csv
    python3 scripts/run_dynamic_experiments.py --mode mo --paired-hv \\
        --seeds 0,1,2 --workers 2 --output results/dynamic_mo.csv

Per-epoch CPU budget is scaled by instance size (--time-small for n <= 15,
--time-medium for n <= 20, --time-large otherwise); the t=0 plan gets 2x.

Extension options (defaults reproduce the original experiments):
    --events-dir instances/solomon_events      the Solomon-derived set
    --strategies static,restart,warm,hybrid,hybrid@0.05
    --drone-models centroid,centroid+setup,route+setup
                   centroid = paper (drones fly to the cluster centroid), route = drones visit
                   every victim (multi-trip CVRP); "+setup" adds launch / recovery / drop times
                   (--setup minutes, default 1,2,0.1)
    --surge-scales 0.5,1,2,4                   dynamic-degree sweep (surge m -> 1 + s (m - 1))
    --no-road-events                           demand dynamics only
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

from dynamic_env import load_scenarios  # noqa: E402

FIELDS = ["instance", "n", "K", "scenario", "seed", "mode", "strategy", "f1", "f2", "f3", "C_del", "C_ret",
          "unmet_ratio", "n_redeliveries", "replan_cpu_s", "free_at_e1", "free_at_e2", "hv_warm_e1", "hv_cold_e1",
          "hv_warm_e2", "hv_cold_e2",
          # extension columns
          "drone_model", "surge_scale", "road_events", "dynamic_degree", "n_repaired", "mean_wait",
          "total_drone_trips", "max_battery_use", "battery_violations", "gap0_f1", "gap0_f3", "gap_last_f1",
          "gap_last_f3"]


def find_instance(name):
    for sub in ("main", "large", "solomon"):
        p = os.path.join(ROOT, "instances", sub, f"{name}.txt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def job(args):
    (name, scenario, seed, mode, weights, t, selector, paired, recourse, events_dir, strategies, dmodel,
     setup_min, surge_scale, road_events) = args
    os.chdir(ROOT)
    from dynamic_runner import load_world, run_closed_loop
    from dynamic_solver import ReoptConfig
    from drone_service import SetupTimes
    world = load_world(find_instance(name), os.path.join(events_dir, f"{name}.events.json"), scenario)
    base = ReoptConfig(mode=mode) if weights is None else ReoptConfig(mode=mode, weights=weights,
                                                                       commit_weights=weights)
    kind, _, extra = dmodel.partition("+")
    setup = SetupTimes.from_minutes(*setup_min) if extra == "setup" else None
    out = run_closed_loop(world, strategies, replace(base, time_limit=2 * t),
                          replace(base, time_limit=t), seed=seed, selector=selector,
                          paired_hv_check=paired, recourse=recourse, drone_model=kind, setup=setup,
                          surge_scale=surge_scale, road_events=road_events)
    rows = []
    warm_ep = {e["epoch"]: e for e in out.get("warm", {"epochs": []})["epochs"]}
    for strat, r in out.items():
        m = r["metrics"]
        ep = {e["epoch"]: e for e in r["epochs"]}
        rows.append(dict(
            instance=name, n=world.inst.n, K=world.inst.no_k, scenario=scenario, seed=seed, mode=mode,
            strategy=strat, f1=m["f1"], f2=m["f2"], f3=m["f3"], C_del=m["C_del"], C_ret=m["C_ret"],
            unmet_ratio=m["unmet_ratio"], n_redeliveries=m["n_redeliveries"], replan_cpu_s=m["replan_cpu_s"],
            free_at_e1=ep.get(1, {}).get("n_free"), free_at_e2=ep.get(2, {}).get("n_free"),
            hv_warm_e1=warm_ep.get(1, {}).get("hv_warm"), hv_cold_e1=warm_ep.get(1, {}).get("hv_cold"),
            hv_warm_e2=warm_ep.get(2, {}).get("hv_warm"), hv_cold_e2=warm_ep.get(2, {}).get("hv_cold"),
            drone_model=dmodel, surge_scale=surge_scale, road_events=int(road_events),
            **{k: m[k] for k in ("dynamic_degree", "n_repaired", "mean_wait", "total_drone_trips",
                                 "max_battery_use", "battery_violations", "gap0_f1", "gap0_f3",
                                 "gap_last_f1", "gap_last_f3")},
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("mo", "so"), default="mo")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scenarios", default=None, help="comma list; default = all in each events file")
    ap.add_argument("--instances", default=None, help="comma list of instance names; default = all events files")
    ap.add_argument("--time-small", type=float, default=1.5)
    ap.add_argument("--time-medium", type=float, default=3.0)
    ap.add_argument("--time-large", type=float, default=5.0)
    ap.add_argument("--selector", default="adaptive")
    ap.add_argument("--paired-hv", action="store_true")
    ap.add_argument("--no-recourse", action="store_true")
    ap.add_argument("--events-dir", default=os.path.join("instances", "dynamic_events"))
    ap.add_argument("--strategies", default="static,restart,warm")
    ap.add_argument("--drone-models", default="centroid")
    ap.add_argument("--setup", default="1,2,0.1", help="launch,recovery,drop minutes for '+setup' models")
    ap.add_argument("--surge-scales", default="1")
    ap.add_argument("--no-road-events", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.chdir(ROOT)
    weights = tuple(float(x) for x in args.weights.split(",")) if args.weights else None
    files = sorted(glob.glob(os.path.join(args.events_dir, "*.events.json")))
    setup_min = tuple(float(x) for x in args.setup.split(","))
    names = [os.path.basename(f).replace(".events.json", "") for f in files]
    if args.instances:
        names = [x for x in names if x in args.instances.split(",")]
    jobs = []
    for name in names:
        scs = load_scenarios(os.path.join(args.events_dir, f"{name}.events.json"))
        n = next(iter(scs.values())).num_satellites
        t = args.time_small if n <= 15 else (args.time_medium if n <= 20 else args.time_large)
        for sc in (args.scenarios.split(",") if args.scenarios else scs.keys()):
            for seed in map(int, args.seeds.split(",")):
                for dm in args.drone_models.split(","):
                    for ss in map(float, args.surge_scales.split(",")):
                        jobs.append((name, sc, seed, args.mode, weights, t, args.selector, args.paired_hv,
                                     not args.no_recourse, args.events_dir, args.strategies.split(","), dm,
                                     setup_min, ss, not args.no_road_events))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(job, j): j for j in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                j = futs[fu]
                try:
                    rows = fu.result()
                except Exception as e:  # keep the batch going
                    print(f"[{i}/{len(jobs)}] {j[0]} {j[1]} seed={j[2]} FAILED: {e!r}", flush=True)
                    continue
                for r in rows:
                    w.writerow(r)
                f.flush()
                st = {r["strategy"]: r["f3"] for r in rows}
                print(f"[{i}/{len(jobs)}] {j[0]} {j[1]} seed={j[2]} {j[11]} s={j[13]} Z1: " +
                      " ".join(f"{k}={v:.3f}" for k, v in st.items()), flush=True)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
