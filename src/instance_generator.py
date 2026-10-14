"""
Parameterized 2E-CTVRP instance generator.

Generates instances at an arbitrary number of satellites n, following the
spatial model in the paper (Section 5.1): victims clustered around n_c
randomly-placed centers within the affected area, two groups of satellites
on arcs facing the depot, and demand/truck-count/drone-count drawn to match
the paper's target utilization ranges.

Two points where the generator's actual behavior differs from the paper's
prose, confirmed against the published instance files and preserved here
for consistency with them:

  1. Cluster-size selection uses `np.random.randint(0, len(cluster_radius)-1)`,
     which in NumPy's [low, high) convention can only draw index 0..3, never
     4 -- so the largest option (radius=150, 130 victims) is unreachable.
     Confirmed consistent with all 10 existing families' m/n_c ratios.
  2. Satellite group 2's radius is 1200, not the 1100 stated in the paper
     draft -- confirmed directly against M_05_03_60_1.txt's real satellite
     coordinates (measured distance from center = 1200.00).

Two deliberate departures from the original, informed by directly comparing
the published files:
  - Full seeding. The original only ever called `np.random.seed(3)`
    immediately before demand generation (a notebook cell run interactively,
    "# %%" markers and all), leaving spatial layout unseeded and the exact
    sequence of cell re-executions -- and hence the RNG state at each
    "generate this file" moment -- unrecoverable. This version seeds once
    per family so every new instance is fully, deterministically
    reproducible from (n_c, n1, n2, seed).
  - Drone count U is computed once per family and shared across both fleet
    variants, matching the paper's stated formula (U depends on total
    demand D and satellite count n only, not on truck capacity). In the
    original files this sometimes held (e.g. M_05_03: U=6 for both fleets)
    and sometimes didn't (M_10_8: U=5 vs U=4) -- an artifact of interactive
    re-running, not a rule to preserve. Truck count K is still drawn
    independently per fleet, since it genuinely depends on Q.

Usage:
    from instance_generator import generate_family
    generate_family(n_c=20, n=20, n1=10, n2=10, seed=42,
                     out_dir="instances/large", name_prefix="M-20-20")
    # writes 10 files: M-20-20-{40,50,60,70,80}-{1,2}.txt (+ .clusters.json
    # companions once frozen -- see scripts/freeze_clustering.py)

Entry point: scripts/generate_instances.py.
"""

import math
import os
import numpy as np

# ---- fixed parameters, matching the paper / original script exactly ----
REG_RADIUS = 850          # radius within which cluster centers are placed
AFF_CENTER_X = 1000
AFF_CENTER_Y = 2000
CLUSTER_RAD_SIZE = [110, 120, 130, 140, 150]
NO_VICTIMS_SIZE = [80, 90, 100, 120, 130]   # index 4 (150/130) is unreachable -- see module docstring
GROUP1_RADIUS = 1000
GROUP2_RADIUS = 1200      # NOT 1100 -- confirmed against real published files
DEPOT_OFFSET_MIN = 2000
DEPOT_OFFSET_MAX = 3000
DEPOT_Y_JITTER = 500
DRONE_CAP = 1000
V_TRUCK = 40
TRUCK_CAPS = [12500, 3500]     # fleet 1 (large), fleet 2 (small)
V_DRONE_VALUES = [40, 50, 60, 70, 80]
TRUCK_UTIL_RANGE = (0.6, 0.9)   # matches original truck_generate()
DRONE_UTIL_RANGE = (0.3, 0.5)   # matches original drone_generate()


def create_centroids(n_centroids, radius, x_center, y_center):
    theta = np.linspace(0, 2 * np.pi, n_centroids)
    r = np.random.rand(n_centroids)
    x = x_center + r * radius * np.cos(theta)
    y = y_center + r * radius * np.sin(theta)
    return x, y


def victims_generate(x_centroid, y_centroid, cluster_radius, no_victims):
    item = np.random.randint(0, len(cluster_radius) - 1)  # bug preserved: index 4 unreachable
    rand_cluster_radius = cluster_radius[item]
    rand_no_victims = no_victims[item]
    theta_cluster = np.linspace(0, 2 * np.pi, rand_no_victims)
    r = np.random.rand(rand_no_victims)
    x = x_centroid + r * rand_cluster_radius * np.cos(theta_cluster)
    y = y_centroid + r * rand_cluster_radius * np.sin(theta_cluster)
    return x, y


def satellites_generate(n_satellites, radius, x_center, y_center):
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, n_satellites)
    x = x_center + radius * np.cos(theta)
    y = y_center + radius * np.sin(theta)
    return x, y


def depot_generate(x_center, y_center, aff_radius):
    x_depot = float(np.random.uniform(x_center + aff_radius + DEPOT_OFFSET_MIN,
                                       x_center + aff_radius + DEPOT_OFFSET_MAX))
    y_depot = float(np.random.uniform(y_center - DEPOT_Y_JITTER, y_center + DEPOT_Y_JITTER))
    return x_depot, y_depot


def demand_generate(n_victims):
    return [int(np.random.randint(0, 25)) for _ in range(n_victims)]


def truck_generate(total_demand, truck_capacity):
    util = np.random.uniform(*TRUCK_UTIL_RANGE)
    total_capacity = total_demand / util
    return math.ceil(total_capacity / truck_capacity)


def drone_generate(total_demand, n_satellites, drone_capacity):
    avg_demand = total_demand / n_satellites
    util = np.random.uniform(*DRONE_UTIL_RANGE)
    total_drone_capacity = avg_demand / util
    return math.ceil(total_drone_capacity / drone_capacity)


def _write_file(out_path, n_satellites, m, no_trucks, no_drones, truck_cap, v_drone,
                 depot_x, depot_y, sat_x1, sat_y1, sat_x2, sat_y2, vic_x, vic_y, demand):
    with open(out_path, "w") as f:
        f.write(f"{n_satellites} {m} {no_trucks} {no_drones} "
                f"{truck_cap} {DRONE_CAP} {V_TRUCK} {v_drone}\n")
        f.write(f"{round(depot_x,2)} {round(depot_y,2)}\n")
        for x, y in zip(sat_x1, sat_y1):
            f.write(f"{round(x,2)} {round(y,2)}\n")
        for x, y in zip(sat_x2, sat_y2):
            f.write(f"{round(x,2)} {round(y,2)}\n")
        for x, y, d in zip(vic_x, vic_y, demand):
            f.write(f"{round(x,2)} {round(y,2)} {d}\n")


def generate_family(n_c, n, n1, n2, seed, out_dir, name_prefix,
                     truck_caps=TRUCK_CAPS, v_drone_values=V_DRONE_VALUES):
    """Generates a full family of instances (all fleet x speed combinations)
    sharing one spatial layout, demand, and drone count U -- matching the
    confirmed structure of the published benchmark (see module docstring).
    Returns a summary dict with the family's characteristics for Table-1-style
    reporting.

    Writes `len(truck_caps) * len(v_drone_values)` files, named
    `{name_prefix}-{v_d}-{f}.txt` with f=1..len(truck_caps) in the order
    given (so the default TRUCK_CAPS=[12500,3500] gives f=1 -> large trucks,
    f=2 -> small trucks, matching the paper's convention)."""
    assert n1 + n2 == n, f"n1+n2 ({n1+n2}) must equal n ({n})"
    os.makedirs(out_dir, exist_ok=True)

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        centr_x, centr_y = create_centroids(n_c, REG_RADIUS, AFF_CENTER_X, AFF_CENTER_Y)

        vic_x, vic_y = [], []
        for i in range(len(centr_x)):
            vx, vy = victims_generate(centr_x[i], centr_y[i], CLUSTER_RAD_SIZE, NO_VICTIMS_SIZE)
            vic_x.extend(vx); vic_y.extend(vy)
        m = len(vic_x)

        sat_x1, sat_y1 = satellites_generate(n1, GROUP1_RADIUS, AFF_CENTER_X, AFF_CENTER_Y)
        sat_x2, sat_y2 = satellites_generate(n2, GROUP2_RADIUS, AFF_CENTER_X, AFF_CENTER_Y)

        depot_x, depot_y = depot_generate(AFF_CENTER_X, AFF_CENTER_Y, REG_RADIUS)

        demand = demand_generate(m)
        total_demand = sum(demand)

        no_drones = drone_generate(total_demand, n, DRONE_CAP)  # shared across fleets

        per_fleet_k = []
        for f_idx, truck_cap in enumerate(truck_caps, start=1):
            no_trucks = truck_generate(total_demand, truck_cap)
            per_fleet_k.append(no_trucks)
            for v_drone in v_drone_values:
                out_path = os.path.join(out_dir, f"{name_prefix}-{v_drone}-{f_idx}.txt")
                _write_file(out_path, n, m, no_trucks, no_drones, truck_cap, v_drone,
                            depot_x, depot_y, sat_x1, sat_y1, sat_x2, sat_y2, vic_x, vic_y, demand)

        return dict(name_prefix=name_prefix, n_c=n_c, n=n, n1=n1, n2=n2, m=m,
                    total_demand=total_demand, no_drones=no_drones,
                    fleets=list(zip(truck_caps, per_fleet_k)))
    finally:
        np.random.set_state(rng_state)
