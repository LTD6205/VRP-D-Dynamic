"""
2E-CTVRP exact MILP solver (Gurobi).

  1. parse an instance file
  2. read the frozen per-satellite centroids (see instances/README.md) --
     independent of src/common.py by design, so this module also serves as
     the ground truth against which the heuristics' feasibility/objective
     code is checked (see scripts/verify_solution.py)
  3. compute first-echelon (truck, Manhattan) and second-echelon (drone,
     Euclidean) travel times
  4. build and solve the arc-based MILP with Gurobi
  5. optionally dump a per-instance route report
  6. report instance, incumbent, best bound, MIP gap and runtime for a
     folder of instances to a CSV file

Entry point: scripts/run_milp.py.
"""

from __future__ import annotations

import glob
import os
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

import gurobipy as gp
from gurobipy import GRB, quicksum


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Instance:
    name: str
    n: int              # number of satellites
    m: int               # number of victim nodes
    no_k: int            # truck fleet size
    no_u: int             # drone fleet size (per satellite)
    q: int                # truck capacity
    p: int                 # drone capacity
    v_t: float             # truck speed
    v_d: float              # drone speed
    x0: float
    y0: float
    x_n: np.ndarray         # satellite x coordinates
    y_n: np.ndarray          # satellite y coordinates
    x_m: np.ndarray           # victim x coordinates
    y_m: np.ndarray            # victim y coordinates
    demand_m: np.ndarray        # victim demand


@dataclass
class ClusteredData:
    centers: np.ndarray    # centroid coordinates, shape (n, 2)
    label: np.ndarray      # cluster label of each victim node
    demand_n: List[float]  # total demand assigned to each centroid


@dataclass
class DistanceData:
    data_v: np.ndarray     # coordinates of [depot, satellites..., depot copy]
    time_n_0: np.ndarray   # truck travel time matrix over data_v (hours)
    time_n_m: np.ndarray   # drone travel time matrix, satellites x centroids (hours)


@dataclass
class ModelContext:
    mdl: "gp.Model"
    V: List[int]
    V_s: List[int]
    M: List[int]
    K: List[int]
    v0: int
    vend: int
    a: Dict
    w: Dict
    x: Dict
    y: Dict
    z: Dict
    delta: Dict


# ============================================================
# 1. INSTANCE PARSING
# ============================================================

def parse_instance(filepath: str) -> Instance:
    """Parse a 2E-CTVRP instance file (see Readme.txt for the format)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    line_1 = list(map(int, lines[0].split()))
    n, m, no_k, no_u, q, p, v_t, v_d = line_1

    x0, y0 = map(float, lines[1].split())

    x_n = np.zeros(n)
    y_n = np.zeros(n)
    for i in range(2, n + 2):
        x_n[i - 2], y_n[i - 2] = map(float, lines[i].split()[:2])

    x_m = np.zeros(m)
    y_m = np.zeros(m)
    demand_m = np.zeros(m, dtype=int)
    for i in range(n + 2, n + 2 + m):
        parts = lines[i].split()
        x_m[i - n - 2] = float(parts[0])
        y_m[i - n - 2] = float(parts[1])
        demand_m[i - n - 2] = int(parts[2])

    name = os.path.splitext(os.path.basename(filepath))[0]

    return Instance(
        name=name, n=n, m=m, no_k=no_k, no_u=no_u, q=q, p=p,
        v_t=v_t, v_d=v_d, x0=x0, y0=y0,
        x_n=x_n, y_n=y_n, x_m=x_m, y_m=y_m, demand_m=demand_m,
    )


# ============================================================
# 2. CLUSTERING
# ============================================================


_FROZEN_CLUSTER_DIRS = ["instances/main", "instances/large"]


def _frozen_cluster_path(name):
    """Looked up relative to the current working directory (the repository
    root), not this source file, so scripts can run from anywhere the repo
    is checked out."""
    for d in _FROZEN_CLUSTER_DIRS:
        p = os.path.join(d, f"{name}.clusters.json")
        if os.path.exists(p):
            return p
    return None


def _load_frozen_cluster(inst):
    """Every instance ships with its clustering already computed (see
    instances/README.md); this reads it instead of re-running K-Means, which
    is not perfectly reproducible across machines/thread counts (float
    tie-breaks in the KMeans implementation can differ at the ~1e-13 level,
    enough to change which local optimum a heuristic search converges to on
    harder instances)."""
    p = _frozen_cluster_path(inst.name)
    if p is None:
        return None
    with open(p) as f:
        data = json.load(f)
    return ClusteredData(centers=np.array(data["centers"]), label=np.array(data["label"]),
                         demand_n=list(data["demand_n"]))


def cluster_victims(inst: Instance, random_state: int = 0) -> ClusteredData:
    """Cluster victim nodes into `n` centroids (one per satellite) with K-Means++."""
    frozen = _load_frozen_cluster(inst)
    if frozen is not None:
        return frozen
    data_m = np.column_stack([inst.x_m, inst.y_m])

    kmeans = KMeans(n_clusters=inst.n, init="k-means++", random_state=random_state, n_init="auto")
    label = kmeans.fit_predict(data_m)
    centers = kmeans.cluster_centers_

    demand_n = [0.0] * inst.n
    for c in range(inst.n):
        demand_n[c] = float(inst.demand_m[label == c].sum())

    return ClusteredData(centers=centers, label=label, demand_n=demand_n)


# ============================================================
# 3. DISTANCE / TIME MATRICES
# ============================================================

def _manhattan_matrix(data: np.ndarray) -> np.ndarray:
    diff = np.abs(data[:, None, :] - data[None, :, :])
    return diff.sum(axis=2)


def _euclidean_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    diff = data1[:, None, :] - data2[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def compute_distances(inst: Instance, clustered: ClusteredData) -> DistanceData:
    """Build the truck (1st echelon, Manhattan) and drone (2nd echelon, Euclidean)
    travel-time matrices. 1 space unit = 10 m."""

    # V = [depot, satellite_1, ..., satellite_n, depot copy]
    data_v = np.zeros((inst.n + 2, 2))
    data_v[0] = [inst.x0, inst.y0]
    data_v[-1] = [inst.x0, inst.y0]
    data_v[1:-1, 0] = inst.x_n
    data_v[1:-1, 1] = inst.y_n

    distance_n_0 = _manhattan_matrix(data_v)
    time_n_0 = (distance_n_0 * 10) / (1000 * inst.v_t)

    data_v_s = data_v[1:-1]
    distance_n_m = _euclidean_matrix(data_v_s, clustered.centers)
    time_n_m = (distance_n_m * 10) / (1000 * inst.v_d)

    return DistanceData(data_v=data_v, time_n_0=time_n_0, time_n_m=time_n_m)


# ============================================================
# 4. MODEL CONSTRUCTION
# ============================================================

def build_model(
    inst: Instance,
    clustered: ClusteredData,
    dist: DistanceData,
    mip_gap: float = 0.0,
    time_limit: float = 3600.0,
    seed: int = 0,
    threads: Optional[int] = None,
    output_flag: int = 1,
) -> ModelContext:
    """Build the 2E-CTVRP MILP (arc-based truck routing + satellite/drone
    assignment, big-M time propagation for subtour elimination)."""

    n, no_k = inst.n, inst.no_k
    demand_n = clustered.demand_n
    time_n_0 = dist.time_n_0
    time_n_m = dist.time_n_m

    mdl = gp.Model("2ECTVRP")
    mdl.Params.OutputFlag = output_flag

    # --- Sets ---
    V = list(range(n + 2))
    v0 = 0
    vend = n + 1
    V_s = list(range(1, n + 1))
    M = V_s.copy()
    K = list(range(no_k))
    U = inst.no_u

    # --- Capacities ---
    Q = inst.q     # capacity of each truck
    P = inst.p     # capacity of one drone

    # --- Valid first-echelon arcs: depot->sat, sat->sat, sat->end depot ---
    E1 = [(v0, j) for j in V_s]
    E1 += [(i, j) for i in V_s for j in V_s if i != j]
    E1 += [(i, vend) for i in V_s]

    pred = {j: [i for (i, jj) in E1 if jj == j] for j in V}
    succ = {i: [j for (ii, j) in E1 if ii == i] for i in V}

    # --- Safe time bounds / Big-M ---
    max_truck_time = max(time_n_0[i, j] for (i, j) in E1)
    max_drone_wait = max(2 * time_n_m[i - 1, j - 1] for i in V_s for j in M)

    # A route can contain at most n satellites, n+1 truck arcs, n waits.
    A_UB = (n + 1) * max_truck_time + n * max_drone_wait
    BIG_M = A_UB + max_drone_wait + max_truck_time

    # ============================================================
    # VARIABLES
    # ============================================================

    a = {
        (i, k): mdl.addVar(lb=0, ub=A_UB, vtype=GRB.CONTINUOUS, name=f"a_{i}_{k + 1}")
        for k in K for i in V
    }

    w = {
        i: mdl.addVar(lb=0, ub=max_drone_wait, vtype=GRB.CONTINUOUS, name=f"w_{i}")
        for i in V_s
    }

    x = {
        (i, j, k): mdl.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k + 1}")
        for k in K for i, j in E1
    }

    y = {
        (i, k): mdl.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k + 1}")
        for k in K for i in V_s
    }

    z = {
        (i, j): mdl.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
        for i in V_s for j in M
    }

    # delta[i,j,k] linearizes y[i,k] * z[i,j] for the truck-capacity constraint
    delta = {
        (i, j, k): mdl.addVar(vtype=GRB.BINARY, name=f"delta_{i}_{j}_{k + 1}")
        for k in K for i in V_s for j in M
    }

    mdl.update()

    # ============================================================
    # OBJECTIVE -- Eq. (1)
    # ============================================================
    mdl.setObjective(
        quicksum(a[i, k] for k in K for i in V_s + [vend]),
        GRB.MINIMIZE,
    )

    # ============================================================
    # DEPOT CONSTRAINTS -- Eqs. (2)-(3)
    # ============================================================
    mdl.addConstrs((quicksum(x[v0, j, k] for j in V_s) == 1 for k in K), name="truck_start")
    mdl.addConstrs((quicksum(x[i, vend, k] for i in V_s) == 1 for k in K), name="truck_end")

    # ============================================================
    # EACH SATELLITE IS VISITED EXACTLY ONCE -- Eq. (4)
    # ============================================================
    mdl.addConstrs(
        (quicksum(x[i, j, k] for k in K for j in succ[i]) == 1 for i in V_s),
        name="satellite_exactly_once",
    )

    # ============================================================
    # FLOW CONSERVATION -- Eq. (5)
    # ============================================================
    mdl.addConstrs(
        (
            quicksum(x[i, j, k] for i in pred[j]) == quicksum(x[j, h, k] for h in succ[j])
            for j in V_s for k in K
        ),
        name="flow_conservation",
    )

    # ============================================================
    # x-y LINKING -- Eq. (6)
    # ============================================================
    mdl.addConstrs(
        (quicksum(x[i, j, k] for i in pred[j]) == y[j, k] for j in V_s for k in K),
        name="xy_link",
    )

    # ============================================================
    # SATELLITE-CENTROID LINKING -- Eq. (7)
    # ============================================================
    mdl.addConstrs(
        (z[i, j] <= quicksum(y[i, k] for k in K) for i in V_s for j in M),
        name="satellite_assignment_link",
    )

    # ============================================================
    # EACH VISITED SATELLITE SERVES ONE CENTROID -- Eq. (8)
    # ============================================================
    mdl.addConstrs(
        (quicksum(z[i, j] for j in M) == quicksum(y[i, k] for k in K) for i in V_s),
        name="satellite_centroid_assignment",
    )

    # ============================================================
    # EACH CENTROID IS SERVED EXACTLY ONCE -- Eq. (9)
    # ============================================================
    mdl.addConstrs((quicksum(z[i, j] for i in V_s) == 1 for j in M), name="centroid_exactly_once")

    # ============================================================
    # LINEARIZATION FOR TRUCK CAPACITY -- Eq. (10a)-(10c)
    # ============================================================
    mdl.addConstrs((delta[i, j, k] <= y[i, k] for i in V_s for j in M for k in K), name="delta_y")
    mdl.addConstrs((delta[i, j, k] <= z[i, j] for i in V_s for j in M for k in K), name="delta_z")
    mdl.addConstrs(
        (delta[i, j, k] >= y[i, k] + z[i, j] - 1 for i in V_s for j in M for k in K),
        name="delta_yz",
    )

    # ============================================================
    # TRUCK CAPACITY -- Eq. (10d)
    # ============================================================
    mdl.addConstrs(
        (
            quicksum(demand_n[j - 1] * delta[i, j, k] for i in V_s for j in M) <= Q
            for k in K
        ),
        name="truck_capacity",
    )

    # ============================================================
    # DRONE CAPACITY -- Eq. (11)
    # ============================================================
    mdl.addConstrs(
        (quicksum(demand_n[j - 1] * z[i, j] for j in M) <= U * P for i in V_s),
        name="drone_capacity",
    )

    # ============================================================
    # TIME PROPAGATION / SUBTOUR ELIMINATION -- Eq. (12)
    # ============================================================
    for k in K:
        for i, j in E1:
            waiting_i = 0 if i == v0 else w[i]
            mdl.addConstr(
                a[j, k] >= a[i, k] + waiting_i + time_n_0[i, j] - BIG_M * (1 - x[i, j, k]),
                name=f"time_{i}_{j}_{k}",
            )

    # ============================================================
    # ARRIVAL TIME ACTIVATION -- Eq. (13)
    # ============================================================
    mdl.addConstrs((a[v0, k] == 0 for k in K), name="start_time")
    mdl.addConstrs(
        (a[i, k] <= A_UB * y[i, k] for i in V_s for k in K),
        name="arrival_activation",
    )

    # ============================================================
    # WAITING TIME -- Eq. (14)
    # ============================================================
    mdl.addConstrs(
        (w[i] >= 2 * time_n_m[i - 1, j - 1] * z[i, j] for i in V_s for j in M),
        name="waiting_time",
    )

    # ============================================================
    # GUROBI PARAMETERS
    # ============================================================
    mdl.Params.MIPGap = mip_gap
    mdl.Params.TimeLimit = time_limit
    mdl.Params.Seed = seed
    if threads is not None:
        mdl.Params.Threads = threads

    mdl.update()

    assert mdl.NumQConstrs == 0, "Model still contains quadratic constraints."

    return ModelContext(
        mdl=mdl, V=V, V_s=V_s, M=M, K=K, v0=v0, vend=vend,
        a=a, w=w, x=x, y=y, z=z, delta=delta,
    )


# ============================================================
# 5. SOLUTION EXTRACTION / REPORTING
# ============================================================

def create_truck_route(truck_arcs: List[List[List[int]]]) -> List[List[int]]:
    """Reconstruct an ordered node sequence for each truck from its selected arcs."""
    truck_route = [[] for _ in truck_arcs]
    for k, route in enumerate(truck_arcs):
        if not route:
            continue
        copy_route = [None] * len(route)
        copy_route[0] = route[0]
        for i in range(len(copy_route) - 1):
            for arc in route:
                if arc[0] == copy_route[i][1]:
                    copy_route[i + 1] = arc
                    break
        for arc in copy_route:
            if arc is None:
                continue
            if arc[0] not in truck_route[k]:
                truck_route[k].append(arc[0])
            if arc[1] not in truck_route[k]:
                truck_route[k].append(arc[1])
    return truck_route


def extract_solution(ctx: ModelContext) -> Optional[Dict]:
    """Extract truck/drone routes and response time from a solved model.
    Returns None if no incumbent solution is available."""
    if ctx.mdl.SolCount == 0:
        return None

    truck_arcs = [[] for _ in ctx.K]
    for k in ctx.K:
        for i in ctx.V:
            for j in ctx.V:
                if i != j and (i, j, k) in ctx.x and ctx.x[i, j, k].X >= 0.99:
                    truck_arcs[k].append([i, j])

    truck_route = create_truck_route(truck_arcs)

    drone_route = []
    for i in ctx.V:
        for j in ctx.M:
            if (i, j) in ctx.z and ctx.z[i, j].X >= 0.99:
                drone_route.append([i, j])

    comp_time = max(ctx.a[i, k].X for k in ctx.K for i in ctx.V)

    return {
        "truck_route": truck_route,
        "drone_route": drone_route,
        "response_time": comp_time,
    }


def save_solution_report(path: str, inst: Instance, ctx: ModelContext, solution: Dict) -> None:
    """Write a human-readable route report, mirroring the notebook's output file."""
    with open(path, "w") as rm:
        rm.write("Results:\n")
        rm.write(f"Objective function (h): z = {ctx.mdl.ObjVal:.3f}\n")
        rm.write(f"Response time: r = {solution['response_time']:.3f}\n")
        rm.write(f"Calculation time (s): t = {ctx.mdl.Runtime:.3f}\n")
        for k in ctx.K:
            route = solution["truck_route"][k]
            rm.write(f"truck {k + 1}: " + " -> ".join(map(str, route)) + "\n")
        rm.write("\n")
        for idx, (i, j) in enumerate(solution["drone_route"]):
            rm.write(f"drone route {idx + 1}: {i} -> {j}\n")


# ============================================================
# 6. PER-INSTANCE PIPELINE
# ============================================================

STATUS_MAP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INTERRUPTED: "INTERRUPTED",
}


def solve_instance(
    filepath: str,
    mip_gap: float = 0.0,
    time_limit: float = 3600.0,
    seed: int = 0,
    threads: Optional[int] = None,
    output_flag: int = 1,
    report_dir: Optional[str] = None,
) -> Dict:
    """Run the full pipeline for a single instance file and return a result row."""
    inst = parse_instance(filepath)
    clustered = cluster_victims(inst, random_state=seed)
    dist = compute_distances(inst, clustered)

    ctx = build_model(
        inst, clustered, dist,
        mip_gap=mip_gap, time_limit=time_limit, seed=seed,
        threads=threads, output_flag=output_flag,
    )

    try:
        ctx.mdl.optimize()

        status_name = STATUS_MAP.get(ctx.mdl.Status, f"STATUS_{ctx.mdl.Status}")
        has_incumbent = ctx.mdl.SolCount > 0

        result = {
            "instance": inst.name,
            "n_satellites": inst.n,
            "n_victims": inst.m,
            "n_trucks": inst.no_k,
            "n_drones": inst.no_u,
            "status": status_name,
            "incumbent": ctx.mdl.ObjVal if has_incumbent else None,
            "best_bound": ctx.mdl.ObjBound,
            "mip_gap_pct": 100 * ctx.mdl.MIPGap if has_incumbent else None,
            "runtime_s": ctx.mdl.Runtime,
            "nodes": ctx.mdl.NodeCount,
            "sol_count": ctx.mdl.SolCount,
        }

        if has_incumbent and report_dir is not None:
            os.makedirs(report_dir, exist_ok=True)
            solution = extract_solution(ctx)
            save_solution_report(
                os.path.join(report_dir, f"{inst.name}_result.txt"), inst, ctx, solution
            )

        return result
    finally:
        ctx.mdl.dispose()


# ============================================================
# 7. BATCH DRIVER
# ============================================================

def find_instance_files(folder: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return files


def write_results_csv(results: List[Dict], output_csv: str) -> None:
    import csv

    fieldnames = [
        "instance", "n_satellites", "n_victims", "n_trucks", "n_drones",
        "status", "incumbent", "best_bound", "mip_gap_pct", "runtime_s",
        "nodes", "sol_count", "error",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_batch(
    folder: str,
    pattern: str,
    output_csv: str,
    mip_gap: float = 0.0,
    time_limit: float = 3600.0,
    seed: int = 0,
    threads: Optional[int] = None,
    output_flag: int = 0,
    report_dir: Optional[str] = None,
) -> List[Dict]:
    files = find_instance_files(folder, pattern)
    if not files:
        print(f"No instance files matching '{pattern}' found in '{folder}'.")
        return []

    results: List[Dict] = []
    for idx, filepath in enumerate(files, start=1):
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"[{idx}/{len(files)}] Solving {name} ...")
        try:
            row = solve_instance(
                filepath,
                mip_gap=mip_gap, time_limit=time_limit, seed=seed,
                threads=threads, output_flag=output_flag, report_dir=report_dir,
            )
            print(
                f"    status={row['status']} incumbent={row['incumbent']} "
                f"best_bound={row['best_bound']} gap={row['mip_gap_pct']} "
                f"runtime={row['runtime_s']:.2f}s"
            )
        except Exception as exc:
            traceback.print_exc()
            row = {"instance": name, "status": "ERROR", "error": str(exc)}

        results.append(row)
        # Flush after every instance so a crash on a later instance
        # doesn't lose the results collected so far.
        write_results_csv(results, output_csv)

    print(f"\nResults written to {output_csv}")
    return results
