"""
Shared building blocks for the 2E-CTVRP heuristics (GRASP-EvPR, GRASP-ILS-PR and
Simulated Annealing): instance parsing, K-Means clustering (with frozen-cluster
support -- see `_load_frozen_cluster`), distance/time matrices, the randomized
construction heuristic, the objective function, the capacity-repair procedure,
and the neighborhood operators used for local search and perturbation.

All three heuristics share this module so they search over exactly the same
solution representation, objective, and feasibility rules -- the baseline
comparisons in the paper depend on that.

Instance files ship with clustering already computed (`<name>.clusters.json`
next to each `.txt`, see instances/README.md), so `cluster_victims` reads the
frozen result instead of re-running K-Means: K-Means cluster assignments are
not perfectly reproducible across machines/thread counts (float tie-breaks
can differ at the ~1e-13 level, enough to change which local optimum a search
converges to on harder instances), so re-clustering would not reproduce the
paper's results exactly.
"""

from __future__ import annotations

import copy
import csv
import glob
import itertools
import os
import json
import statistics
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


# ============================================================
# DATA STRUCTURES (instance parsing / clustering / distances)
# ============================================================

@dataclass
class Instance:
    name: str
    n: int              # number of satellites
    m: int               # number of victim nodes
    no_k: int             # truck fleet size
    no_u: int              # drone fleet size (per satellite)
    q: int                  # truck capacity
    p: int                   # drone capacity
    v_t: float                # truck speed
    v_d: float                 # drone speed
    x0: float
    y0: float
    x_n: np.ndarray
    y_n: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    demand_m: np.ndarray


@dataclass
class ClusteredData:
    centers: np.ndarray
    label: np.ndarray
    demand_n: List[float]


@dataclass
class DistanceData:
    data_v: np.ndarray
    distance_n_0: np.ndarray   # 1st-echelon Manhattan distance matrix
    time_n_0: np.ndarray       # 1st-echelon (truck) travel time matrix, hours
    time_n_m: np.ndarray       # 2nd-echelon (drone) travel time matrix, hours


def parse_instance(filepath: str) -> Instance:
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



_FROZEN_CLUSTER_DIRS = ["instances/main", "instances/large", "instances/solomon"]


def _frozen_cluster_path(name):
    """Frozen-cluster files are looked up relative to the current working
    directory (the repository root), not relative to this source file, so
    scripts can be run from anywhere the repo is checked out."""
    for d in _FROZEN_CLUSTER_DIRS:
        p = os.path.join(d, f"{name}.clusters.json")
        if os.path.exists(p):
            return p
    return None


def _load_frozen_cluster(inst):
    """If <name>.cluster.json exists next to the instance file, load it instead of
    re-running KMeans. Centroid/label assignment is not perfectly thread-invariant
    (float tie-breaks in the C KMeans implementation can differ by ~1e-13 between
    thread counts), so every script must read the SAME frozen clustering, computed
    once, rather than each re-clustering independently -- see freeze_clustering.py."""
    p = _frozen_cluster_path(inst.name)
    if p is None:
        return None
    with open(p) as f:
        data = json.load(f)
    return ClusteredData(centers=np.array(data["centers"]), label=np.array(data["label"]),
                         demand_n=list(data["demand_n"]))


def cluster_victims(inst: Instance, random_state: int = 0) -> ClusteredData:
    frozen = _load_frozen_cluster(inst)
    if frozen is not None:
        return frozen
    data_m = np.column_stack([inst.x_m, inst.y_m])

    kmeans = KMeans(n_clusters=inst.n, init="k-means++", n_init="auto", random_state=random_state)
    label = kmeans.fit_predict(data_m)
    centers = kmeans.cluster_centers_

    demand_n = [float(inst.demand_m[label == c].sum()) for c in range(inst.n)]

    return ClusteredData(centers=centers, label=label, demand_n=demand_n)


def _manhattan_matrix(data: np.ndarray) -> np.ndarray:
    diff = np.abs(data[:, None, :] - data[None, :, :])
    return diff.sum(axis=2)


def _euclidean_matrix(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    diff = data1[:, None, :] - data2[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def compute_distances(inst: Instance, clustered: ClusteredData) -> DistanceData:
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

    return DistanceData(data_v=data_v, distance_n_0=distance_n_0, time_n_0=time_n_0, time_n_m=time_n_m)


# ============================================================
# RESULT / SUMMARY
# ============================================================

@dataclass
class RunResult:
    instance: str
    objective: Optional[float]
    runtime_s: float
    iterations: int
    truck_route: Optional[Dict[int, List[int]]] = None
    drone_route: Optional[Dict[int, List[int]]] = None
    # Optional per-run iteration breakdown (e.g. GRASP's construct/repair/local
    # search/improve counters, see GraspEvPRSolver.solve). None for solvers
    # that don't report one (e.g. the SA baseline).
    diagnostics: Optional[Dict[str, int]] = None
    # CPU seconds (since the start of solve()) at which the returned best solution was found.
    time_to_best: Optional[float] = None


@dataclass
class ReplicationSummary:
    instance: str
    n_reps: int
    n_failed: int
    mean_objective: Optional[float]
    std_objective: Optional[float]
    min_objective: Optional[float]
    max_objective: Optional[float]
    mean_runtime_s: float
    runs: List[RunResult] = field(default_factory=list)


# ============================================================
# SHARED SOLUTION REPRESENTATION / OPERATORS
#
# This is the part every algorithm converted from the notebooks shares:
# construction heuristic, objective, capacity repair and the neighborhood
# moves used for local search. A concrete metaheuristic (GRASP-PR/EvPR,
# Simulated Annealing, ...) subclasses this and only adds its own solve()
# loop (plus, for GRASP, path relinking / EvPR).
# ============================================================

# Insertion costs that are mathematically 0 (the insertion point lies exactly
# on the existing path) rarely come out as exact 0.0 in floating point --
# `a + b - c` from independently-looked-up matrix entries typically leaves a
# tiny positive residual instead. Treating anything below this tolerance as
# "free" avoids 1/c blowing up (and, in the limit, factorial becoming inf and
# producing a NaN probability) for insertions that are effectively free.
_FREE_INSERTION_EPS = 1e-9


class ECTVRPOperators:
    """State + moves shared by every 2E-CTVRP metaheuristic in this project.

    All state that used to be module-level globals in the source notebooks
    (n, q, demand_n, time_n_0, time_n_m, V_s, K, ...) is stored as instance
    attributes here so that many replications / instances can be solved
    safely (no shared mutable global state between runs).
    """

    def __init__(self, inst: Instance, clustered: ClusteredData, dist: DistanceData):
        self.inst = inst

        self.n = inst.n
        self.no_k = inst.no_k
        self.no_u = inst.no_u
        self.q = inst.q
        self.p = inst.p

        self.v0 = 0
        self.vend = inst.n + 1
        self.V = list(range(inst.n + 2))
        self.V_s = list(range(1, inst.n + 1))
        self.M = self.V_s
        self.K = list(range(inst.no_k))

        self.demand_n = list(clustered.demand_n)
        self.distance_n_0 = dist.distance_n_0
        self.time_n_0 = dist.time_n_0
        self.time_n_m = dist.time_n_m

    # -------------------- construction heuristic --------------------

    def ranked_satellites(self) -> List[int]:
        dis_neighbor_depot = self.distance_n_0[self.v0]
        order = np.argsort(dis_neighbor_depot, kind="stable")
        return [int(idx) for idx in order if idx != self.v0 and idx != self.vend]

    def create_init_tours(self, k: int, seed_node: int) -> Dict[int, List[int]]:
        truck_route: Dict[int, List[int]] = {}
        assigned: List[int] = []
        for i in range(k):
            truck_route[i] = [self.v0]
            if i == 0:
                truck_route[i].append(seed_node)
                assigned.append(seed_node)
            else:
                assigned_node = truck_route[i - 1][-1]
                neighbor = self.time_n_0[assigned_node]
                unvisited = [j for j in self.V_s if j not in assigned]
                truck_node = max(unvisited, key=lambda j: neighbor[j])
                truck_route[i].append(truck_node)
                assigned.append(truck_node)
        for i in range(k):
            truck_route[i].append(self.vend)
        return truck_route

    def mul_truck_create_init_tours(self, rank_satellites: List[int]) -> Dict[int, List[int]]:
        remain_nodes = len(self.V_s) - self.no_k
        single_routes = max(0, self.no_k - remain_nodes)
        truck_route: Dict[int, List[int]] = {}
        for i in range(self.no_k):
            truck_route[i] = [self.v0]
            if i < single_routes:
                truck_route[i].append(rank_satellites[i])
            else:
                truck_route[i].append(rank_satellites[i + remain_nodes])
            truck_route[i].append(self.vend)
        return truck_route

    def cheapest_insertion(self, truck_route: Dict[int, List[int]], insert_node: int) -> Dict[int, List[int]]:
        insertion_cost = []
        insert_position = []
        for i in range(len(truck_route)):
            for j in range(len(truck_route[i]) - 1):
                node_before, node_after = truck_route[i][j], truck_route[i][j + 1]
                cost = (
                    self.time_n_0[node_before, insert_node]
                    + self.time_n_0[insert_node, node_after]
                    - self.time_n_0[node_before, node_after]
                )
                insert_position.append([i, j + 1])
                insertion_cost.append(cost)

        has_nonpositive = any(c <= _FREE_INSERTION_EPS for c in insertion_cost)
        count_0 = sum(1 for c in insertion_cost if c <= _FREE_INSERTION_EPS) if has_nonpositive else 0
        factorial = 0.0 if has_nonpositive else sum(1 / c for c in insertion_cost)

        prob_to_insert = []
        for c in insertion_cost:
            if has_nonpositive:
                probability = 1 / count_0 if c <= _FREE_INSERTION_EPS else 0
            else:
                probability = (1 / c) / factorial
            prob_to_insert.append(probability)

        insert_infor = int(np.random.choice(range(len(insert_position)), p=prob_to_insert))
        insert_tour, insert_index = insert_position[insert_infor]
        truck_route[insert_tour].insert(insert_index, insert_node)
        return truck_route

    def add_satellites(self, truck_route: Dict[int, List[int]]) -> Dict[int, List[int]]:
        served = set()
        for route in truck_route.values():
            served.update(route)
        unserved = [x for x in self.V_s if x not in served]
        for node in unserved:
            self.cheapest_insertion(truck_route, node)
        return truck_route

    def order_drone_route(self, truck_route: Dict[int, List[int]]) -> List[int]:
        max_length = max(len(route) for route in truck_route.values())
        order = []
        seen = set()
        for i in range(max_length):
            for route in truck_route.values():
                if len(route) > i and route[i] not in seen:
                    order.append(route[i])
                    seen.add(route[i])
        order.remove(self.v0)
        order.remove(self.vend)
        return order

    def generate_drone_route(self, satellites_order: List[int]) -> Dict[int, List[int]]:
        drone_route: Dict[int, List[int]] = {}
        drone_served = set()
        for i in satellites_order:
            drone_route[i] = [i]
            neighbor = list(self.time_n_m[i - 1])
            candidates = copy.deepcopy(neighbor)
            for j in drone_served:
                candidates.remove(neighbor[j])
            factorial = sum(1 / c for c in candidates)
            probs = [(1 / c) / factorial for c in candidates]
            chosen = float(np.random.choice(candidates, p=probs))
            destination = neighbor.index(chosen)
            drone_route[i].append(destination + 1)
            drone_served.add(destination)
        return drone_route

    def construct_initial_solution(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        rank_satellites = self.ranked_satellites()
        remain_nodes = len(self.V_s) - self.no_k
        single_routes = max(0, self.no_k - remain_nodes)
        if single_routes == 0:
            truck_route = self.create_init_tours(self.no_k, rank_satellites[-1])
        else:
            truck_route = self.mul_truck_create_init_tours(rank_satellites)
        truck_route = self.add_satellites(truck_route)
        satellites_order = self.order_drone_route(truck_route)
        drone_route = self.generate_drone_route(satellites_order)
        return truck_route, drone_route

    def construct_feasible_solution(self, max_attempts: int = 20):
        """Construct + repair until capacity-feasible, or give up after
        `max_attempts` and return the last (possibly still infeasible) try.
        GRASP gets this "for free" because it restarts from scratch every
        iteration anyway; algorithms that run a single trajectory (e.g. SA)
        need it once, up front, to start from a workable solution."""
        truck_route, drone_route = self.construct_initial_solution()
        for _ in range(max_attempts):
            # A drone-capacity violation can't be fixed by truck-route repair
            # (it depends only on which centroid a satellite serves) -- only a
            # fresh construction (a different satellite<->centroid pairing) can
            # resolve it, so skip straight to retrying.
            if self.drone_capacity_verify(drone_route):
                demand = self.connect_demand(drone_route)
                if not self.capacity_verify(truck_route, demand):
                    truck_route = self.repair(truck_route, drone_route)
                if self.capacity_verify(truck_route, demand):
                    return truck_route, drone_route
            truck_route, drone_route = self.construct_initial_solution()
        return truck_route, drone_route

    # -------------------- objective / feasibility --------------------

    def flight_time_cal(self, satellite: int, drone_route: Dict[int, List[int]]) -> float:
        if satellite == self.v0 or satellite == self.vend:
            return 0.0
        for v in drone_route.values():
            if satellite == v[0]:
                return 2 * self.time_n_m[satellite - 1, v[1] - 1]
        return 0.0

    def sum_arr_time(self, truck_route: Dict[int, List[int]], drone_route: Dict[int, List[int]]) -> float:
        total = 0.0
        for route in truck_route.values():
            arr_time = 0.0
            route_total = 0.0
            for j in range(len(route) - 1):
                origin, dest = route[j], route[j + 1]
                arr_time += self.time_n_0[origin, dest]
                route_total += arr_time
                arr_time += self.flight_time_cal(dest, drone_route)
            total += route_total
        return total

    def connect_demand(self, drone_route: Dict[int, List[int]]) -> List[float]:
        demand = [self.demand_n[drone_route[i][1] - 1] for i in range(1, len(drone_route) + 1)]
        return [0] + demand + [0]

    def capacity_verify(self, truck_route: Dict[int, List[int]], satellites_demand: List[float]) -> bool:
        for route in truck_route.values():
            if sum(satellites_demand[j] for j in route) > self.q:
                return False
        return True

    def drone_capacity_verify(self, drone_route: Dict[int, List[int]]) -> bool:
        """MILP Eq. 11: the demand a satellite hands off to its drones must not
        exceed U*P (no_u drones of capacity p each, operating in parallel)."""
        cap = self.no_u * self.p
        return all(self.demand_n[destination - 1] <= cap for _, destination in drone_route.values())

    def is_feasible(self, truck_route: Dict[int, List[int]], drone_route: Dict[int, List[int]]) -> bool:
        """Combined truck- and drone-capacity feasibility check.

        Which centroid a satellite serves (drone_route) determines how much
        demand that satellite's truck ends up carrying, so any move that
        changes drone_route (bias_swap_drone, path relinking) can silently
        break truck capacity even though it never touches truck_route -- use
        this whenever accepting a candidate that changed drone_route."""
        if not self.drone_capacity_verify(drone_route):
            return False
        return self.capacity_verify(truck_route, self.connect_demand(drone_route))

    # -------------------- repair (capacity restoration) --------------------

    def check_tour_cap(self, tour: List[int], drone_route: Dict[int, List[int]]) -> float:
        return sum(self.demand_n[drone_route[tour[i]][1] - 1] for i in range(1, len(tour) - 1))

    def total_demand(self, nodes: List[int], drone_route: Dict[int, List[int]]) -> float:
        return sum(self.demand_n[drone_route[node][1] - 1] for node in nodes)

    def invalid_node(self, truck_route: Dict[int, List[int]], drone_route: Dict[int, List[int]]):
        new_route = copy.deepcopy(truck_route)
        invalid_nodes = []
        for tour in new_route.values():
            tour_cap = self.check_tour_cap(tour, drone_route)
            while tour_cap > self.q:
                invalid_nodes.append(tour.pop(-2))
                tour_cap = self.check_tour_cap(tour, drone_route)
        return invalid_nodes, new_route

    def check_cap(self, truck_route: Dict[int, List[int]], drone_route: Dict[int, List[int]]) -> List[float]:
        caps = []
        for tour in truck_route.values():
            caps.append(sum(self.demand_n[drone_route[tour[j]][1] - 1] for j in range(1, len(tour) - 1)))
        return caps

    def cheapest_insertion_with_cap(self, truck_route, insert_node, drone_route):
        insertion_cost = []
        insert_position = []
        added_demand = self.demand_n[drone_route[insert_node][1] - 1]
        for i in range(len(truck_route)):
            tour_cap = self.check_tour_cap(truck_route[i], drone_route)
            for j in range(len(truck_route[i]) - 1):
                node_before, node_after = truck_route[i][j], truck_route[i][j + 1]
                if tour_cap + added_demand > self.q:
                    cost = np.inf
                else:
                    cost = (
                        self.time_n_0[node_before, insert_node]
                        + self.time_n_0[insert_node, node_after]
                        - self.time_n_0[node_before, node_after]
                    )
                insert_position.append([i, j + 1])
                insertion_cost.append(cost)

        has_nonpositive = any(c <= _FREE_INSERTION_EPS for c in insertion_cost)
        count_0 = sum(1 for c in insertion_cost if c <= _FREE_INSERTION_EPS) if has_nonpositive else 0
        factorial = 0.0
        if not has_nonpositive:
            factorial = sum(0 if c == np.inf else 1 / c for c in insertion_cost)

        prob_to_insert = []
        for c in insertion_cost:
            if has_nonpositive:
                probability = 1 / count_0 if c <= _FREE_INSERTION_EPS else 0
            elif c == np.inf:
                probability = 0
            else:
                probability = (1 / c) / factorial
            prob_to_insert.append(probability)

        if sum(prob_to_insert) == 1:
            insert_infor = int(np.random.choice(range(len(insert_position)), p=prob_to_insert))
            insert_tour, insert_index = insert_position[insert_infor]
            truck_route[insert_tour].insert(insert_index, insert_node)
            return True, truck_route
        return False, truck_route

    def reinsert(self, invalid_nodes: List[int], input_route, drone_route):
        remaining = []
        for node in invalid_nodes:
            ok, input_route = self.cheapest_insertion_with_cap(input_route, node, drone_route)
            if not ok:
                remaining.append(node)
        return input_route, remaining

    def find_combine(self, truck_route, drone_route) -> List[List[int]]:
        tour_capacity = self.check_cap(truck_route, drone_route)
        sample = []
        for i, j in itertools.combinations(range(len(tour_capacity)), 2):
            if tour_capacity[i] + tour_capacity[j] < self.q:
                sample.append([i, j])
        return sample

    def repair_truck_route(self, truck_route, drone_route):
        new_truck_route: Dict[int, List[int]] = {}
        # Indices sorted by descending demand, consumed by index (not value) so
        # that tied demands don't collapse onto the same centroid via .index().
        order = sorted(range(len(self.demand_n)), key=lambda i: self.demand_n[i], reverse=True)
        allocated_node: List[List[int]] = []
        while order:
            first_idx = order.pop(0)
            first_pack = self.demand_n[first_idx]
            allocated_node.append([first_idx + 1])
            accumulate_cap = first_pack
            # Guard against the notebook's original IndexError: order can
            # already be empty here (single remaining item just consumed above).
            while order and accumulate_cap + self.demand_n[order[-1]] <= self.q:
                last_idx = order.pop()
                accumulate_cap += self.demand_n[last_idx]
                allocated_node[-1].append(last_idx + 1)
        # Reject not just "needs more bins than trucks available" but also
        # "needs fewer" -- returning fewer groups than len(truck_route) would
        # silently leave some trucks unused (no arc to/from a satellite at
        # all), which the MILP's formulation forbids (every truck must
        # depart to some satellite). Confirmed this let GRASP report
        # objectives below a proven MILP optimum (M_10_07_40_2: GRASP found
        # 16.97 using only 4 of 5 trucks against a proven optimum of 17.20).
        if len(allocated_node) != len(truck_route):
            return truck_route
        for i, nodes in enumerate(allocated_node):
            new_truck_route[i] = [self.v0]
            for center in nodes:
                satellite = next(k for k, v in drone_route.items() if v[1] == center)
                new_truck_route[i].append(satellite)
            new_truck_route[i].append(self.vend)
        return new_truck_route

    def repair(self, truck_route, drone_route):
        invalid_nodes, new_route = self.invalid_node(truck_route, drone_route)
        new_truck_route, remaining_nodes = self.reinsert(invalid_nodes, new_route, drone_route)
        if not remaining_nodes:
            return new_truck_route

        # A single node whose own demand exceeds truck capacity can never be
        # placed anywhere (no repair move splits one satellite's demand across
        # trucks) -- bail out immediately instead of churning below.
        if any(self.demand_n[drone_route[node][1] - 1] > self.q for node in remaining_nodes):
            return self._repair_fallback(truck_route, drone_route)

        # The merge/re-split loop below has no convergence guarantee: on a
        # structurally infeasible instance it can cycle forever (merging two
        # combinable routes always frees up a near-empty "stub" route that is
        # itself combinable with something else, ad infinitum). Cap the
        # iterations so it degrades to the existing give-up fallback instead
        # of hanging.
        max_combine_iters = max(50, 5 * len(new_truck_route) ** 2)
        combine = self.find_combine(new_truck_route, drone_route)
        combine_iters = 0
        # Loop only while there's still something to repair. The original
        # notebook's "for node in remaining_nodes: ... remaining_nodes.remove(node)"
        # mutates the list it's iterating over, silently dropping every other
        # node -- remaining_nodes then never truly empties, so this loop kept
        # running (via find_combine still finding unrelated combinable pairs)
        # and could turn OTHER, already-valid trucks into empty [v0, vend]
        # routes (confirmed: ~39% of repaired solutions on a tight-capacity
        # instance ended up with an idle truck this way). Draining
        # remaining_nodes correctly in one shot, and stopping once it's
        # empty, fixes both the dropped-node bug and the idle-truck bug.
        while combine and remaining_nodes and combine_iters < max_combine_iters:
            combine_iters += 1
            random_select = 0 if len(combine) == 1 else int(np.random.randint(0, len(combine) - 1))
            first_tour_index, second_tour_index = combine[random_select]
            new_tour = (
                copy.deepcopy(new_truck_route[first_tour_index][:-1])
                + copy.deepcopy(new_truck_route[second_tour_index][1:])
            )
            new_truck_route[first_tour_index] = new_tour
            if self.total_demand(remaining_nodes, drone_route) <= self.q:
                new_truck_route[second_tour_index] = [self.v0] + list(remaining_nodes) + [self.vend]
                remaining_nodes.clear()
            else:
                select_node = remaining_nodes[0]
                new_truck_route[second_tour_index] = [self.v0, select_node, self.vend]
                remaining_nodes.remove(select_node)
            combine = self.find_combine(new_truck_route, drone_route)

        if not remaining_nodes:
            return new_truck_route

        return self._repair_fallback(truck_route, drone_route)

    def _repair_fallback(self, truck_route, drone_route):
        """Last-resort repair attempt; returns the original (still possibly
        infeasible) route if even this doesn't work. The caller is expected
        to re-run capacity_verify and treat a failure as "discard this
        construction," not to trust this always returns a feasible route."""
        new_truck_route = self.repair_truck_route(truck_route, drone_route)
        satellites_demand = self.connect_demand(drone_route)
        if self.capacity_verify(new_truck_route, satellites_demand):
            return new_truck_route
        return truck_route

    # -------------------- local search neighborhoods --------------------

    def sum_cap(self, tour: List[int], drone_route: Dict[int, List[int]]) -> float:
        """Total demand a truck picks up along `tour`. A satellite's demand
        is whatever centroid drone_route currently assigns it to serve --
        NOT demand_n[node-1] (that would silently assume satellite i always
        serves centroid i, which is only true by coincidence)."""
        return sum(
            self.demand_n[drone_route[node][1] - 1]
            for node in tour if node != self.v0 and node != self.vend
        )

    def relocate_truck_node(self, truck_route, drone_route=None):
        """N1 - relocate a satellite within a truck route.

        Best-improving full scan: relocating a node within its own tour
        never changes that tour's total demand, so no capacity check is
        needed here -- every candidate is feasible by construction."""
        base_route = truck_route
        best_route = copy.deepcopy(base_route)
        best_val = self.sum_arr_time(best_route, drone_route) if drone_route is not None else None
        for ridx, tour in base_route.items():
            for node_index in range(1, len(tour) - 1):
                node = tour[node_index]
                without_node = tour[:node_index] + tour[node_index + 1:]
                for insert_pos in range(1, len(without_node) - 1):
                    candidate_tour = without_node[:insert_pos] + [node] + without_node[insert_pos:]
                    if candidate_tour == tour or drone_route is None:
                        continue
                    candidate_route = dict(base_route)
                    candidate_route[ridx] = candidate_tour
                    val = self.sum_arr_time(candidate_route, drone_route)
                    if val < best_val:
                        best_val = val
                        best_route = candidate_route
        return best_route

    def select_prob(self, node_1: int, sample_node_2: List[int]) -> List[float]:
        factorial = sum(1 / self.time_n_0[node_1, s] for s in sample_node_2)
        return [(1 / self.time_n_0[node_1, s]) / factorial for s in sample_node_2]

    def bias_swap_satellites(self, truck_route, drone_route=None):
        """N2 - bias swap satellites (intra-route).

        Best-improving full scan: swapping two nodes within the same tour
        never changes that tour's total demand, so no capacity check is
        needed -- every candidate is feasible by construction."""
        base_route = truck_route
        best_route = copy.deepcopy(base_route)
        best_val = self.sum_arr_time(best_route, drone_route) if drone_route is not None else None
        for ridx, tour in base_route.items():
            interior_positions = list(range(1, len(tour) - 1))
            for i, j in itertools.combinations(interior_positions, 2):
                if drone_route is None:
                    continue
                candidate_tour = list(tour)
                candidate_tour[i], candidate_tour[j] = candidate_tour[j], candidate_tour[i]
                candidate_route = dict(base_route)
                candidate_route[ridx] = candidate_tour
                val = self.sum_arr_time(candidate_route, drone_route)
                if val < best_val:
                    best_val = val
                    best_route = candidate_route
        return best_route

    def two_opt_intra(self, truck_route, drone_route=None):
        """N3 - 2-opt neighborhood (intra-route).

        Best-improving full scan: reversing a segment within a tour never
        changes that tour's total demand, so no capacity check is needed --
        every candidate is feasible by construction."""
        base_route = truck_route
        best_route = copy.deepcopy(base_route)
        best_val = self.sum_arr_time(best_route, drone_route) if drone_route is not None else None
        for ridx, tour in base_route.items():
            n_tour = len(tour)
            if n_tour <= 4:
                continue
            for first_edge in range(1, n_tour - 2):
                for second_edge in range(first_edge + 2, n_tour):
                    if drone_route is None:
                        continue
                    candidate_tour = (
                        tour[:first_edge]
                        + tour[second_edge - 1:first_edge - 1:-1]
                        + tour[second_edge:]
                    )
                    candidate_route = dict(base_route)
                    candidate_route[ridx] = candidate_tour
                    val = self.sum_arr_time(candidate_route, drone_route)
                    if val < best_val:
                        best_val = val
                        best_route = candidate_route
        return best_route

    #: Cap on how many candidate moves the expensive inter-route operators
    #: (N4, N5, T6) evaluate per call. Their neighborhoods grow combinatorially
    #: with instance size, so -- unlike the cheap intra-route operators above,
    #: which are cheap enough to scan exhaustively -- these sample a bounded
    #: number of candidates and return the best of the batch.
    INTER_OP_SAMPLE_LIMIT = 15

    def inter_swap(self, truck_route, drone_route):
        """N4 - swap satellites (inter-route): best of a bounded random sample."""
        base_route = truck_route
        candidates = []
        sample_tour = list(range(len(base_route)))
        while sample_tour and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
            tour_1_index = int(np.random.choice(sample_tour))
            tour_1 = base_route[tour_1_index]
            sample_node_1 = [x for x in tour_1 if x != self.v0 and x != self.vend]
            while sample_node_1 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                node_1 = int(np.random.choice(sample_node_1))
                node_1_index = tour_1.index(node_1)
                sample_tour_2 = [t for t in sample_tour if t != tour_1_index]
                while sample_tour_2 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                    tour_2_index = int(np.random.choice(sample_tour_2))
                    tour_2 = base_route[tour_2_index]
                    sample_node_2 = [x for x in tour_2 if x != self.v0 and x != self.vend]
                    while sample_node_2 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                        node_2 = int(np.random.choice(sample_node_2))
                        node_2_index = tour_2.index(node_2)
                        demand_1 = self.demand_n[drone_route[node_1][1] - 1]
                        demand_2 = self.demand_n[drone_route[node_2][1] - 1]
                        if (
                            self.sum_cap(tour_1, drone_route) - demand_1 + demand_2 <= self.q
                            and self.sum_cap(tour_2, drone_route) - demand_2 + demand_1 <= self.q
                        ):
                            candidate_route = copy.deepcopy(base_route)
                            candidate_route[tour_1_index][node_1_index] = node_2
                            candidate_route[tour_2_index][node_2_index] = node_1
                            val = self.sum_arr_time(candidate_route, drone_route)
                            candidates.append((val, candidate_route))
                        sample_node_2.remove(node_2)
                    sample_tour_2.remove(tour_2_index)
                sample_node_1.remove(node_1)
            sample_tour.remove(tour_1_index)
        if not candidates:
            return copy.deepcopy(base_route)
        return min(candidates, key=lambda c: c[0])[1]

    def two_opt_inter(self, truck_route, drone_route):
        """N5 - 2-opt* neighborhood (inter-route): best of a bounded random sample.

        For each sampled reconnection, both the non-inverted and inverted
        variant are scored against the real `drone_route` and the cheaper of
        the two is kept as that sample's candidate."""
        base_route = truck_route
        candidates = []
        sample_tour_index = list(range(len(base_route)))
        while sample_tour_index and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
            tour_1_index = int(np.random.choice(sample_tour_index))
            tour_1 = base_route[tour_1_index]
            sample_tour_2 = [t for t in sample_tour_index if t != tour_1_index]
            while sample_tour_2 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                tour_2_index = int(np.random.choice(sample_tour_2))
                tour_2 = base_route[tour_2_index]
                edges_set_1 = list(range(len(tour_1)))[1:len(tour_1) - 1]
                while edges_set_1 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                    first_edge = int(np.random.choice(edges_set_1))
                    edges_set_2 = list(range(len(tour_2)))[2:len(tour_2) - 1]
                    while edges_set_2 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                        second_edge = int(np.random.choice(edges_set_2))
                        if (
                            self.sum_cap(tour_1[:first_edge] + tour_2[second_edge:], drone_route) <= self.q
                            and self.sum_cap(tour_2[:second_edge] + tour_1[first_edge:], drone_route) <= self.q
                            and self.sum_cap(tour_1[:first_edge] + list(reversed(tour_2[:second_edge])), drone_route) <= self.q
                            and self.sum_cap(list(reversed(tour_1[first_edge:])) + tour_2[second_edge:], drone_route) <= self.q
                        ):
                            mod_route_no_inv = copy.deepcopy(base_route)
                            mod_route_no_inv[tour_1_index] = tour_1[:first_edge] + tour_2[second_edge:]
                            mod_route_no_inv[tour_2_index] = tour_2[:second_edge] + tour_1[first_edge:]

                            mod_route_inv = copy.deepcopy(base_route)
                            new_route_1_inv = tour_1[:first_edge] + list(reversed(tour_2[:second_edge]))
                            new_route_2_inv = list(reversed(tour_1[first_edge:])) + tour_2[second_edge:]
                            new_route_1_inv[-1] = self.vend
                            new_route_2_inv[0] = self.v0
                            mod_route_inv[tour_1_index] = new_route_1_inv
                            mod_route_inv[tour_2_index] = new_route_2_inv

                            val_no_inv = self.sum_arr_time(mod_route_no_inv, drone_route)
                            val_inv = self.sum_arr_time(mod_route_inv, drone_route)
                            if val_no_inv < val_inv:
                                candidates.append((val_no_inv, mod_route_no_inv))
                            else:
                                candidates.append((val_inv, mod_route_inv))
                        edges_set_2.remove(second_edge)
                    edges_set_1.remove(first_edge)
                sample_tour_2.remove(tour_2_index)
            sample_tour_index.remove(tour_1_index)
        if not candidates:
            return copy.deepcopy(base_route)
        return min(candidates, key=lambda c: c[0])[1]

    def inter_relocate(self, truck_route, drone_route):
        """T6 - relocate a satellite from one truck route to another: best of
        a bounded random sample."""
        base_route = truck_route
        candidates = []
        sample_tour_index = list(range(len(base_route)))
        while sample_tour_index and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
            tour_1_index = int(np.random.choice(sample_tour_index))
            tour_1 = base_route[tour_1_index]
            sample_insert_node = [x for x in tour_1 if x != self.v0 and x != self.vend]
            while len(sample_insert_node) > 1 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                insert_node = int(np.random.choice(sample_insert_node))
                sample_tour_2 = [t for t in sample_tour_index if t != tour_1_index]
                while sample_tour_2 and len(candidates) < self.INTER_OP_SAMPLE_LIMIT:
                    tour_2_index = int(np.random.choice(sample_tour_2))
                    tour_2 = base_route[tour_2_index]
                    if self.sum_cap(tour_2, drone_route) + self.demand_n[drone_route[insert_node][1] - 1] <= self.q:
                        sample_insert_index = [i for i in range(len(tour_2)) if i != 0 and i != len(tour_2) - 1]
                        if sample_insert_index:
                            insert_index = int(np.random.choice(sample_insert_index))
                            candidate_route = copy.deepcopy(base_route)
                            candidate_route[tour_1_index].remove(insert_node)
                            candidate_route[tour_2_index].insert(insert_index, insert_node)
                            val = self.sum_arr_time(candidate_route, drone_route)
                            candidates.append((val, candidate_route))
                    sample_tour_2.remove(tour_2_index)
                sample_insert_node.remove(insert_node)
            sample_tour_index.remove(tour_1_index)
        if not candidates:
            return copy.deepcopy(base_route)
        return min(candidates, key=lambda c: c[0])[1]

    def select_prob_drone(self, origin: int, sample_destination: List[int]) -> List[float]:
        factorial = sum(1 / self.time_n_m[origin - 1, s - 1] for s in sample_destination)
        return [(1 / self.time_n_m[origin - 1, s - 1]) / factorial for s in sample_destination]

    def bias_swap_drone(self, drone_route):
        """D1 - bias swap drone routes."""
        route = copy.deepcopy(drone_route)
        sample_tour_index = list(route.keys())
        while sample_tour_index:
            tour_1_index = int(np.random.choice(sample_tour_index))
            tour_1 = route[tour_1_index]
            satellite_1 = tour_1[0]
            sample_satellites_2 = [v[0] for v in route.values()]
            sample_satellites_2.remove(satellite_1)
            if sample_satellites_2:
                probs = self.select_prob_drone(satellite_1, sample_satellites_2)
                satellite_2 = int(np.random.choice(sample_satellites_2, p=probs))
                tour_2 = route[satellite_2]
                tour_1[1], tour_2[1] = tour_2[1], tour_1[1]
                return route
            sample_tour_index.remove(tour_1_index)
        return route

    # -------------------- generic local search descent --------------------

    #: the 6 truck-route operators + the 1 drone operator, in the order the
    #: original GRASP-PR notebook applies them in its first-improvement pass.
    NEIGHBORHOOD_OPERATORS: Tuple[str, ...] = (
        "relocate_truck_node", "bias_swap_satellites", "two_opt_intra",
        "inter_swap", "two_opt_inter", "inter_relocate",
    )

    def local_search_descent(self, truck_route, drone_route, obj_val):
        """First-improvement descent: repeatedly sweep all neighborhoods,
        keeping any improving move, until a full sweep finds none."""
        improved = True
        while improved:
            improved = False
            for op_name in self.NEIGHBORHOOD_OPERATORS:
                candidate = getattr(self, op_name)(truck_route, drone_route)
                val = self.sum_arr_time(candidate, drone_route)
                if val < obj_val:
                    truck_route, obj_val, improved = candidate, val, True

            candidate_drone = self.bias_swap_drone(drone_route)
            val = self.sum_arr_time(truck_route, candidate_drone)
            # Unlike the truck-route operators above, bias_swap_drone can
            # change which centroid's demand a satellite's truck carries, so
            # an improving swap still needs a capacity check before it's
            # accepted (see is_feasible's docstring).
            if val < obj_val and self.is_feasible(truck_route, candidate_drone):
                drone_route, obj_val, improved = candidate_drone, val, True

        return truck_route, drone_route, obj_val


# ============================================================
# GENERIC REPLICATION / BATCH DRIVER
#
# Shared by every metaheuristic script: run N independent replications of a
# solver on one instance, write a per-instance CSV, and batch this over every
# instance file in a folder into one combined summary CSV.
# ============================================================

SolverFactory = Callable[[Instance, ClusteredData, DistanceData], "ECTVRPOperators"]


def run_replications(
    filepath: str,
    solver_factory: SolverFactory,
    n_reps: int = 10,
    seed: Optional[int] = 0,
    cluster_seed: int = 0,
    verbose: bool = False,
) -> ReplicationSummary:
    inst = parse_instance(filepath)
    clustered = cluster_victims(inst, random_state=cluster_seed)
    dist = compute_distances(inst, clustered)

    runs: List[RunResult] = []
    for rep in range(n_reps):
        if seed is not None:
            np.random.seed(seed + rep)
        solver = solver_factory(inst, clustered, dist)
        result = solver.solve()
        runs.append(result)
        if verbose:
            obj_str = f"{result.objective:.3f}" if result.objective is not None else "FAILED"
            print(f"    rep {rep + 1}/{n_reps}: objective={obj_str} runtime={result.runtime_s:.2f}s")
            if result.diagnostics:
                diag_str = " ".join(f"{k}={v}" for k, v in result.diagnostics.items())
                print(f"      diagnostics: {diag_str}")

    objectives = [r.objective for r in runs if r.objective is not None]
    n_failed = len(runs) - len(objectives)
    mean_runtime = statistics.fmean(r.runtime_s for r in runs)

    return ReplicationSummary(
        instance=inst.name,
        n_reps=n_reps,
        n_failed=n_failed,
        mean_objective=statistics.fmean(objectives) if objectives else None,
        std_objective=statistics.pstdev(objectives) if len(objectives) > 1 else (0.0 if objectives else None),
        min_objective=min(objectives) if objectives else None,
        max_objective=max(objectives) if objectives else None,
        mean_runtime_s=mean_runtime,
        runs=runs,
    )


def write_instance_csv(summary: ReplicationSummary, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{summary.instance}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "objective", "runtime_s"])
        for i, r in enumerate(summary.runs, start=1):
            writer.writerow([i, r.objective if r.objective is not None else "", f"{r.runtime_s:.6f}"])
        writer.writerow(["mean", summary.mean_objective if summary.mean_objective is not None else "", f"{summary.mean_runtime_s:.6f}"])
        writer.writerow(["std", summary.std_objective if summary.std_objective is not None else "", ""])
        writer.writerow(["min", summary.min_objective if summary.min_objective is not None else "", ""])
        writer.writerow(["max", summary.max_objective if summary.max_objective is not None else "", ""])
    return path


def write_summary_csv(
    summaries: List[ReplicationSummary],
    summary_csv: str,
    extra_columns: Optional[Dict[str, object]] = None,
) -> None:
    extra_columns = extra_columns or {}
    fieldnames = [
        "instance", "n_reps", "n_failed",
        "mean_objective", "std_objective", "min_objective", "max_objective",
        "mean_runtime_s",
    ] + list(extra_columns.keys())
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {
                "instance": s.instance,
                "n_reps": s.n_reps,
                "n_failed": s.n_failed,
                "mean_objective": s.mean_objective,
                "std_objective": s.std_objective,
                "min_objective": s.min_objective,
                "max_objective": s.max_objective,
                "mean_runtime_s": s.mean_runtime_s,
            }
            row.update(extra_columns)
            writer.writerow(row)


def find_instance_files(folder: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))


def run_batch(
    folder: str,
    pattern: str,
    solver_factory: SolverFactory,
    n_reps: int,
    output_dir: str,
    summary_csv: str,
    seed: Optional[int] = 0,
    cluster_seed: int = 0,
    verbose: bool = False,
    extra_summary_columns: Optional[Dict[str, object]] = None,
    progress_label: str = "",
) -> List[ReplicationSummary]:
    files = find_instance_files(folder, pattern)
    if not files:
        print(f"No instance files matching '{pattern}' found in '{folder}'.")
        return []

    summaries: List[ReplicationSummary] = []
    for idx, filepath in enumerate(files, start=1):
        name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"[{idx}/{len(files)}] {name}: running {n_reps} replications{progress_label} ...")
        try:
            instance_seed = None if seed is None else seed + idx * n_reps
            summary = run_replications(
                filepath, solver_factory, n_reps=n_reps, seed=instance_seed,
                cluster_seed=cluster_seed, verbose=verbose,
            )
            write_instance_csv(summary, output_dir)
            print(
                f"    mean={summary.mean_objective} std={summary.std_objective} "
                f"min={summary.min_objective} max={summary.max_objective} "
                f"mean_runtime={summary.mean_runtime_s:.2f}s failed={summary.n_failed}/{summary.n_reps}"
            )
        except Exception:
            traceback.print_exc()
            summary = ReplicationSummary(
                instance=name, n_reps=n_reps, n_failed=n_reps,
                mean_objective=None, std_objective=None, min_objective=None, max_objective=None,
                mean_runtime_s=0.0, runs=[],
            )

        summaries.append(summary)
        # Flush the combined summary after every instance so a crash later
        # in the batch doesn't lose the results collected so far.
        write_summary_csv(summaries, summary_csv, extra_summary_columns)

    print(f"\nPer-instance CSVs written to {output_dir}/")
    print(f"Summary CSV written to {summary_csv}")
    return summaries
