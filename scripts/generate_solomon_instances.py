#!/usr/bin/env python3
"""
Build a second benchmark set for the 2-echelon truck-drone relief problem from the
Solomon (1987) VRPTW instances.

Mapping
  Solomon customer          -> victim (location x10 so 1 Solomon unit = 100 m, demand kept)
  Solomon depot             -> truck depot
  k-means groups of victims -> n relief clusters (frozen in <name>.clusters.json)
  satellites                -> one per cluster, placed 1.0-2.5 km from that cluster's centroid
                               in a random direction (inside the 10 km x 10 km area), so the
                               satellite <-> cluster matching sigma is a real decision
  Solomon time windows      -> DROPPED: inside one geographic cluster they spread over most
                               of the day (they were made for one vehicle visiting customers one
                               by one), so they cannot be met by a single truck stop
  Solomon service times     -> dropped (drone drop time is a model parameter instead)

Fleet: K = 2 trucks for n = 8 clusters, K = 3 for n = 12; truck capacity Q = ceil(1.25 D / K)
(room for surges); U = 3 drones per truck, payload P = 60 (2-4 victims per trip);
truck 40 km/h, drone 60 km/h -- same speeds as the main set.

Only four Solomon geometries are distinct (C1, C2, R1 = R2, RC1 = RC2: the series inside a
class differ only in their time windows), so the set is
    {C1, C2, R, RC} x n in {8, 12} x satellite-placement seed in {1, 2}  = 16 instances,
named S-<geom>-<n>-<seed>, written in the repo's instance format to instances/solomon/.
Then:   python3 scripts/generate_dynamic_dataset.py --instances solomon/S-C1-08-1,... \\
                --out instances/solomon_events
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

GEOMS = {"C1": "c101.txt", "C2": "c201.txt", "R": "r101.txt", "RC": "rc101.txt"}
SCALE = 10.0           # Solomon unit -> instance unit (instance unit = 10 m)


def read_solomon(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 7 and p[0].isdigit():
            rows.append([float(x) for x in p])
    a = np.array(rows)
    return a[0, 1:3], a[1:, 1:3], a[1:, 3].astype(int)


def build(geom: str, n: int, seed: int, raw_dir: Path, out_dir: Path):
    depot, xy, dem = read_solomon(raw_dir / GEOMS[geom])
    depot, xy = depot * SCALE, xy * SCALE
    km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(xy)
    label, centers = km.labels_, km.cluster_centers_
    rng = np.random.default_rng(1000 * seed + n)
    sats = []
    for c in range(n):
        for _ in range(100):
            r = rng.uniform(100, 250)                      # 1.0 - 2.5 km (instance units of 10 m)
            ang = rng.uniform(0, 2 * math.pi)
            p = centers[c] + r * np.array([math.cos(ang), math.sin(ang)])
            if (0 <= p).all() and (p <= 100 * SCALE).all():
                break
        sats.append(np.clip(p, 0, 100 * SCALE))
    sats = np.array(sats)
    K = 2 if n <= 8 else 3
    D = int(dem.sum())
    Q = int(math.ceil(1.25 * D / K))
    U, P, v_t, v_d = 3, 60, 40, 60
    name = f"S-{geom}-{n:02d}-{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.txt", "w") as f:
        f.write(f"{n} {len(xy)} {K} {U} {Q} {P} {v_t} {v_d}\n")
        f.write(f"{depot[0]:.2f} {depot[1]:.2f}\n")
        for p in sats:
            f.write(f"{p[0]:.2f} {p[1]:.2f}\n")
        for p, q in zip(xy, dem):
            f.write(f"{p[0]:.2f} {p[1]:.2f} {q}\n")
    demand_n = [float(dem[label == c].sum()) for c in range(n)]
    with open(out_dir / f"{name}.clusters.json", "w") as f:
        json.dump(dict(instance=name, source=f"Solomon {GEOMS[geom]}", centers=centers.tolist(),
                       label=label.tolist(), demand_n=demand_n), f)
    return name, dict(K=K, Q=Q, D=D, cluster_demand=(min(demand_n), max(demand_n)),
                      victims_per_cluster=(int(np.bincount(label).min()), int(np.bincount(label).max())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="instances/solomon/raw")
    ap.add_argument("--out", default="instances/solomon")
    ap.add_argument("--sizes", default="8,12")
    ap.add_argument("--seeds", default="1,2")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    names = []
    for geom in GEOMS:
        for n in map(int, args.sizes.split(",")):
            for seed in map(int, args.seeds.split(",")):
                name, info = build(geom, n, seed, Path(args.raw), Path(args.out))
                names.append(name)
                print(name, info)
    print("\nnext: python3 scripts/generate_dynamic_dataset.py --instances " +
          ",".join(f"solomon/{x}" for x in names) + " --out instances/solomon_events")


if __name__ == "__main__":
    main()
