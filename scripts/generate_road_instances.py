#!/usr/bin/env python3
"""
Generate the road-network benchmark of the proposal (Section 7):
Solomon geometries x sizes x road intensities x instance seeds, each with S scenarios
(reveal times + full road-state history).

    python3 scripts/generate_road_instances.py                       # full set
    python3 scripts/generate_road_instances.py --geoms R --sizes 25 --scenarios 2

Writes
    instances/road/<name>.json                 static instance (graph, zones, Q, anchors)
    instances/road/<name>.scen<k>.json         scenario k (reveal times, road events)
    instances/road/summary.csv                 one row per (instance, scenario)
    instances/road/summary_by_type.csv         averages per geometry x size x intensity

Only four Solomon geometries are distinct (C1, C2, R1 = R2, RC1 = RC2), so the 56
files reduce to C1, C2, R, RC; sizes 25 and 50 are the first customers of each file.
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_instances import (Params, build_instance, sample_scenario, save_json,  # noqa: E402
                            summarise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geoms", default="C1,C2,R,RC")
    ap.add_argument("--sizes", default="25,50,100")
    ap.add_argument("--intensities", default="low,medium,high")
    ap.add_argument("--seeds", default="1,2", help="instance (graph + zone) seeds")
    ap.add_argument("--scenarios", type=int, default=10)
    ap.add_argument("--graph", default="delaunay", choices=["delaunay", "knn"])
    ap.add_argument("--zone-share", type=float, default=0.2)
    ap.add_argument("--permanent-share", type=float, default=0.2)
    ap.add_argument("--road-model", default="zone", choices=["zone", "independent"])
    ap.add_argument("--drones", type=int, default=2)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--utilisation", type=float, default=0.8)
    ap.add_argument("--reveal", default="poisson", choices=["poisson", "solomon"])
    ap.add_argument("--solomon-dir", default=str(ROOT / "instances/solomon/raw"))
    ap.add_argument("--out", default=str(ROOT / "instances/road"))
    a = ap.parse_args()

    out = Path(a.out)
    rows = []
    for g in a.geoms.split(","):
        for n in map(int, a.sizes.split(",")):
            for inten in a.intensities.split(","):
                for s in map(int, a.seeds.split(",")):
                    p = Params(geometry=g, n=n, graph=a.graph, zone_share=a.zone_share,
                               permanent_share=a.permanent_share, intensity=inten, road_model=a.road_model,
                               drones=a.drones, delta=a.delta, utilisation=a.utilisation,
                               reveal=a.reveal, seed=s)
                    inst = build_instance(p, Path(a.solomon_dir))
                    save_json(asdict(inst), out / f"{inst.name}.json")
                    for k in range(1, a.scenarios + 1):
                        scen = sample_scenario(inst, k)
                        save_json(scen, out / f"{inst.name}.scen{k}.json")
                        r = summarise(inst, scen)
                        r.update(geometry=g, size=n, intensity=inten, inst_seed=s)
                        rows.append(r)
                    print(f"{inst.name}: feasible={inst.feasible} tries={inst.tries} "
                          f"roads={len(inst.edges)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    cols = ["roads", "mean_degree", "zone_roads_pct", "victims_in_zone_pct",
            "permanent_drone_only", "drone_only_t0_pct", "drone_only_mean_pct",
            "drone_only_max_pct", "ever_drone_only_pct", "road_events", "zone_level_changes", "max_anchor_km",
            "mean_p_hat_min", "last_reveal_min"]
    agg = df.groupby(["geometry", "size", "intensity"], sort=False)[cols].mean().round(1)
    agg["feasible_pct"] = df.groupby(["geometry", "size", "intensity"], sort=False)["feasible"].mean() * 100
    agg.to_csv(out / "summary_by_type.csv")
    print(agg.to_string())


if __name__ == "__main__":
    main()
