"""
GRASP-ILS-PR: the paper's proposed metaheuristic for the 2E-CTVRP.

Constructs one initial solution, then runs a single-trajectory iterated local
search (cheap random-move perturbation, accept-if-not-worse) with two pieces
of GRASP-PR machinery layered in periodically rather than on every iteration:
a full best-of-`k_candidates` local-search sweep every `local_search_freq`
iterations, and forward/backward path relinking against an elite pool every
`pr_freq` iterations. EvPR (periodic evolutionary path relinking within the
elite pool) is implemented and configurable (`use_evpr`) but disabled by
default: an ablation study found no benefit from it in this architecture at
any time budget tested. A full reconstruction runs after `restart_patience`
iterations without improvement.

See the paper (Section 4, Table 7) for the full description and the default
parameter values. Entry point: scripts/run_heuristic.py.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import ClusteredData, DistanceData, ECTVRPOperators, Instance, RunResult


def _fast_copy(d: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Much faster than copy.deepcopy for this codebase's dict-of-lists shape."""
    return {k: list(v) for k, v in d.items()}


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GraspILSConfig:
    time_limit: Optional[float] = 30.0   # t_max in the paper (Table 7): primary stop, matched against SA/MILP
    n_max: int = 100_000_000             # n_max in the paper (Table 7): safety cap on iterations (never binding at t_max=30s)
    k_candidates: int = 5                # best-of-K sampling within the periodic full local-search sweep
    local_search_freq: int = 30          # run a full local-search sweep every N iterations
    pr_freq: int = 10                    # run forward/backward path relinking every N iterations
    mu: float = 0.007                    # elite-pool acceptance threshold
    maxps: int = 20                      # elite-pool size
    evpr_freq: int = 200                 # iterations between EvPR calls
    use_evpr: bool = False               # off by default: an ablation study found no benefit at any budget tested; opt in with use_evpr=True
    restart_patience: int = 2000         # iterations without improvement before a full reconstruction


# ============================================================
# SOLVER
# ============================================================

class GraspILSSolver(ECTVRPOperators):
    def __init__(self, inst: Instance, clustered: ClusteredData, dist: DistanceData, config: GraspILSConfig):
        super().__init__(inst, clustered, dist)
        self.config = config

    # -------------------- speed fixes (same as v2) --------------------

    def flight_time_cal(self, satellite: int, drone_route: Dict[int, List[int]]) -> float:
        if satellite == self.v0 or satellite == self.vend:
            return 0.0
        entry = drone_route.get(satellite)
        if entry is None:
            return 0.0
        return 2 * self.time_n_m[satellite - 1, entry[1] - 1]

    def generate_drone_route(self, satellites_order: List[int]) -> Dict[int, List[int]]:
        drone_route: Dict[int, List[int]] = {}
        drone_served = set()
        for i in satellites_order:
            drone_route[i] = [i]
            neighbor = list(self.time_n_m[i - 1])
            candidates = list(neighbor)
            for j in drone_served:
                candidates.remove(neighbor[j])
            factorial = sum(1 / c for c in candidates)
            probs = [(1 / c) / factorial for c in candidates]
            chosen = float(np.random.choice(candidates, p=probs))
            destination = neighbor.index(chosen)
            drone_route[i].append(destination + 1)
            drone_served.add(destination)
        return drone_route

    # -------------------- neighborhoods (same as v2: fast copies) --------------------

    def relocate_truck_node(self, truck_route, drone_route=None):
        route = _fast_copy(truck_route)
        tour_indices = list(range(len(route)))
        while tour_indices:
            ridx = int(np.random.choice(tour_indices))
            tour = route[ridx]
            sample_node = list(tour)
            sample_node.remove(self.v0)
            sample_node.remove(tour[-1])
            while len(sample_node) > 2:
                node = int(np.random.choice(sample_node))
                insert_index_sample = list(range(len(tour)))
                node_index = tour.index(node)
                insert_index_sample.remove(node_index)
                insert_index_sample.remove(0)
                insert_index_sample.remove(len(tour) - 1)
                if insert_index_sample:
                    insert_index = int(np.random.choice(insert_index_sample))
                    tour.remove(node)
                    tour.insert(insert_index, node)
                    return route
                sample_node.remove(node_index)
            tour_indices.remove(ridx)
        return route

    def bias_swap_satellites(self, truck_route, drone_route=None):
        route = _fast_copy(truck_route)
        tour_indices = list(range(len(route)))
        while tour_indices:
            ridx = int(np.random.choice(tour_indices))
            tour = route[ridx]
            sample_node_1 = list(tour)
            sample_node_1.remove(self.v0)
            sample_node_1.remove(tour[-1])
            while len(sample_node_1) > 0:
                node_1 = int(np.random.choice(sample_node_1))
                node_1_index = tour.index(node_1)
                sample_node_2 = list(sample_node_1)
                sample_node_2.remove(node_1)
                if sample_node_2:
                    probs = self.select_prob(node_1, sample_node_2)
                    node_2 = int(np.random.choice(sample_node_2, p=probs))
                    node_2_index = tour.index(node_2)
                    tour[node_1_index] = node_2
                    tour[node_2_index] = node_1
                    return route
                sample_node_1.remove(node_1)
            tour_indices.remove(ridx)
        return route

    def two_opt_intra(self, truck_route, drone_route=None):
        route = _fast_copy(truck_route)
        tour_indices = list(range(len(route)))
        while tour_indices:
            tidx = int(np.random.choice(tour_indices))
            select_tour = route[tidx]
            sample_edge = list(range(len(select_tour)))
            while len(sample_edge) > 4:
                first_edge_set = list(sample_edge)[1:len(sample_edge) - 2]
                first_edge = int(np.random.choice(first_edge_set))
                second_edge_set = list(sample_edge)[first_edge + 2:]
                if second_edge_set:
                    second_edge = int(np.random.choice(second_edge_set))
                    new_tour = (
                        select_tour[:first_edge]
                        + select_tour[second_edge - 1:first_edge - 1:-1]
                        + select_tour[second_edge:]
                    )
                    route[tidx] = new_tour
                    return route
                sample_edge.remove(first_edge)
            tour_indices.remove(tidx)
        return route

    def inter_swap(self, truck_route, drone_route):
        route = _fast_copy(truck_route)
        sample_tour = list(range(len(route)))
        while sample_tour:
            tour_1_index = int(np.random.choice(sample_tour))
            tour_1 = route[tour_1_index]
            sample_node_1 = list(tour_1)
            sample_node_1.remove(self.v0)
            sample_node_1.remove(self.vend)
            while sample_node_1:
                node_1 = int(np.random.choice(sample_node_1))
                node_1_index = tour_1.index(node_1)
                sample_tour_2 = list(sample_tour)
                sample_tour_2.remove(tour_1_index)
                while sample_tour_2:
                    tour_2_index = int(np.random.choice(sample_tour_2))
                    tour_2 = route[tour_2_index]
                    sample_node_2 = list(tour_2)
                    sample_node_2.remove(self.v0)
                    sample_node_2.remove(self.vend)
                    while sample_node_2:
                        node_2 = int(np.random.choice(sample_node_2))
                        node_2_index = tour_2.index(node_2)
                        demand_1 = self.demand_n[drone_route[node_1][1] - 1]
                        demand_2 = self.demand_n[drone_route[node_2][1] - 1]
                        if (
                            self.sum_cap(tour_1, drone_route) - demand_1 + demand_2 <= self.q
                            and self.sum_cap(tour_2, drone_route) - demand_2 + demand_1 <= self.q
                        ):
                            tour_1[node_1_index] = node_2
                            tour_2[node_2_index] = node_1
                            return route
                        sample_node_2.remove(node_2)
                    sample_tour_2.remove(tour_2_index)
                sample_node_1.remove(node_1)
            sample_tour.remove(tour_1_index)
        return route

    def two_opt_inter(self, truck_route, drone_route):
        mod_route_no_inv = _fast_copy(truck_route)
        mod_route_inv = _fast_copy(truck_route)
        sample_tour_index = list(range(len(mod_route_no_inv)))
        while sample_tour_index:
            tour_1_index = int(np.random.choice(sample_tour_index))
            tour_1 = mod_route_no_inv[tour_1_index]
            sample_tour_2 = list(sample_tour_index)
            sample_tour_2.remove(tour_1_index)
            while sample_tour_2:
                tour_2_index = int(np.random.choice(sample_tour_2))
                tour_2 = mod_route_no_inv[tour_2_index]
                edges_set_1 = list(range(len(tour_1)))[1:len(tour_1) - 1]
                while edges_set_1:
                    first_edge = int(np.random.choice(edges_set_1))
                    edges_set_2 = list(range(len(tour_2)))[2:len(tour_2) - 1]
                    while edges_set_2:
                        second_edge = int(np.random.choice(edges_set_2))
                        if (
                            self.sum_cap(tour_1[:first_edge] + tour_2[second_edge:], drone_route) <= self.q
                            and self.sum_cap(tour_2[:second_edge] + tour_1[first_edge:], drone_route) <= self.q
                            and self.sum_cap(tour_1[:first_edge] + list(reversed(tour_2[:second_edge])), drone_route) <= self.q
                            and self.sum_cap(list(reversed(tour_1[first_edge:])) + tour_2[second_edge:], drone_route) <= self.q
                        ):
                            new_route_1_no_inv = tour_1[:first_edge] + tour_2[second_edge:]
                            new_route_2_no_inv = tour_2[:second_edge] + tour_1[first_edge:]
                            mod_route_no_inv[tour_1_index] = new_route_1_no_inv
                            mod_route_no_inv[tour_2_index] = new_route_2_no_inv
                            order_no_inv = self.order_drone_route(mod_route_no_inv)
                            drone_route_no_inv = self.generate_drone_route(order_no_inv)

                            new_route_1_inv = tour_1[:first_edge] + list(reversed(tour_2[:second_edge]))
                            new_route_2_inv = list(reversed(tour_1[first_edge:])) + tour_2[second_edge:]
                            mod_route_inv[tour_1_index] = new_route_1_inv
                            mod_route_inv[tour_1_index][-1] = self.vend
                            mod_route_inv[tour_2_index] = new_route_2_inv
                            mod_route_inv[tour_2_index][0] = self.v0
                            order_inv = self.order_drone_route(mod_route_inv)
                            drone_route_inv = self.generate_drone_route(order_inv)

                            delta_1 = self.sum_arr_time(mod_route_no_inv, drone_route_no_inv)
                            delta_2 = self.sum_arr_time(mod_route_inv, drone_route_inv)
                            return mod_route_no_inv if delta_1 > delta_2 else mod_route_inv
                        edges_set_2.remove(second_edge)
                    edges_set_1.remove(first_edge)
                sample_tour_2.remove(tour_2_index)
            sample_tour_index.remove(tour_1_index)
        return mod_route_no_inv

    def inter_relocate(self, truck_route, drone_route):
        route = _fast_copy(truck_route)
        sample_tour_index = list(range(len(route)))
        while sample_tour_index:
            tour_1_index = int(np.random.choice(sample_tour_index))
            tour_1 = route[tour_1_index]
            sample_insert_node = list(tour_1)
            sample_insert_node.remove(self.v0)
            sample_insert_node.remove(self.vend)
            while len(sample_insert_node) > 1:
                insert_node = int(np.random.choice(sample_insert_node))
                sample_tour_2 = list(sample_tour_index)
                sample_tour_2.remove(tour_1_index)
                while sample_tour_2:
                    tour_2_index = int(np.random.choice(sample_tour_2))
                    tour_2 = route[tour_2_index]
                    if self.sum_cap(tour_2, drone_route) + self.demand_n[drone_route[insert_node][1] - 1] <= self.q:
                        sample_insert_index = list(range(len(tour_2)))
                        sample_insert_index.remove(0)
                        sample_insert_index.remove(len(tour_2) - 1)
                        if sample_insert_index:
                            insert_index = int(np.random.choice(sample_insert_index))
                            tour_1.remove(insert_node)
                            tour_2.insert(insert_index, insert_node)
                            return route
                        sample_tour_2.remove(tour_2_index)
                    else:
                        sample_tour_2.remove(tour_2_index)
                sample_insert_node.remove(insert_node)
            sample_tour_index.remove(tour_1_index)
        return route

    def bias_swap_drone(self, drone_route):
        route = _fast_copy(drone_route)
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

    # -------------------- cheap single-move perturbation (like SA's) --------------------

    def _random_neighbor(self, truck_route, drone_route):
        """Uniformly pick one of the 6 truck-route neighborhoods or the
        drone-swap neighborhood and apply it once -- one cheap move, not a
        full sweep. Mirrors sa.py's _random_neighbor exactly."""
        n_ops = len(self.NEIGHBORHOOD_OPERATORS)
        move = int(np.random.randint(0, n_ops + 1))
        if move < n_ops:
            candidate_truck = getattr(self, self.NEIGHBORHOOD_OPERATORS[move])(truck_route, drone_route)
            return candidate_truck, drone_route
        candidate_drone = self.bias_swap_drone(drone_route)
        return truck_route, candidate_drone

    # -------------------- strengthened local search: best-of-K (same as v2) --------------------

    def local_search_descent(self, truck_route, drone_route, obj_val):
        k = max(1, self.config.k_candidates)
        improved = True
        while improved:
            improved = False
            for op_name in self.NEIGHBORHOOD_OPERATORS:
                op = getattr(self, op_name)
                best_candidate, best_val = None, obj_val
                for _ in range(k):
                    candidate = op(truck_route, drone_route)
                    val = self.sum_arr_time(candidate, drone_route)
                    if val < best_val:
                        best_candidate, best_val = candidate, val
                if best_candidate is not None:
                    truck_route, obj_val, improved = best_candidate, best_val, True

            best_drone, best_val = None, obj_val
            for _ in range(k):
                candidate_drone = self.bias_swap_drone(drone_route)
                val = self.sum_arr_time(truck_route, candidate_drone)
                if val < best_val and self.is_feasible(truck_route, candidate_drone):
                    best_drone, best_val = candidate_drone, val
            if best_drone is not None:
                drone_route, obj_val, improved = best_drone, best_val, True

        return truck_route, drone_route, obj_val

    # -------------------- path relinking / EvPR (same as v2/GRASP) --------------------

    def sym_diff_cal(self, initial_solution, final_solution) -> List[int]:
        return [i for i in self.V_s if initial_solution[i] != final_solution[i]]

    def path_relinking(self, truck_route, initial_drone_route, final_drone_route):
        initial_drone_route = {k: v for k, v in sorted(initial_drone_route.items(), key=lambda item: item[1])}
        final_drone_route = {k: v for k, v in sorted(final_drone_route.items(), key=lambda item: item[1])}
        current = _fast_copy(initial_drone_route)

        f_initial = self.sum_arr_time(truck_route, current)
        f_final = self.sum_arr_time(truck_route, final_drone_route)
        f_star = min(f_initial, f_final)
        # NOTE: must be an independent snapshot, not an alias -- `current` is
        # mutated in place for the rest of this function as the walk
        # proceeds, and `final_drone_route` is never copied by the caller
        # either. Aliasing here let x_star's *content* silently drift away
        # from the f_star value it's paired with by the time this returns.
        x_star = _fast_copy(current) if f_star == f_initial else _fast_copy(final_drone_route)

        sym_diff = self.sym_diff_cal(current, final_drone_route)
        while sym_diff:
            first_ex_index = sym_diff[0]
            target_dest = final_drone_route[first_ex_index][1]
            second_ex_index = first_ex_index
            for i in sym_diff:
                if current[i][1] == target_dest:
                    second_ex_index = i
            current[first_ex_index][1], current[second_ex_index][1] = (
                current[second_ex_index][1],
                current[first_ex_index][1],
            )
            sym_diff = self.sym_diff_cal(current, final_drone_route)
            current_sol = _fast_copy(current)
            f_current = self.sum_arr_time(truck_route, current_sol)
            if f_current < f_star:
                f_star = f_current
                x_star = current_sol
        return x_star, f_star

    def evpr(self, truck_route, P: List[Dict[int, List[int]]]) -> List[Dict[int, List[int]]]:
        for i in range(len(P)):
            for j in range(len(P)):
                if P[i] != P[j]:
                    new_elite, _ = self.path_relinking(truck_route, P[i], P[j])
                    replace_index = 0
                    min_diff = np.inf
                    for k in range(len(P)):
                        diff = self.sym_diff_cal(P[k], new_elite)
                        if len(diff) < min_diff:
                            min_diff = len(diff)
                            replace_index = k
                    P[replace_index] = new_elite
        return P

    def _elite_pool_update(self, elite_pool: List[Dict[int, List[int]]], drone_route, obj_val, best_obj) -> None:
        if obj_val < (1 + self.config.mu) * best_obj and drone_route not in elite_pool:
            if len(elite_pool) < self.config.maxps:
                elite_pool.append(drone_route)
            elif len(elite_pool) == self.config.maxps:
                del elite_pool[0]
                elite_pool.append(drone_route)

    # -------------------- main ILS + GRASP-PR/EvPR loop --------------------

    def solve(self) -> RunResult:
        cfg = self.config
        st = time.process_time()

        current_truck, current_drone = self.construct_feasible_solution()
        if not self.is_feasible(current_truck, current_drone):
            # ConstructFeasible failed after R_max attempts: report a failed run
            # instead of searching from (and possibly returning) an infeasible solution.
            return RunResult(
                instance=self.inst.name, objective=None,
                runtime_s=time.process_time() - st, iterations=0,
                truck_route=None, drone_route=None,
                diagnostics={"restarts": 0, "restarts_skipped": 0},
            )
        current_obj = self.sum_arr_time(current_truck, current_drone)
        pre_truck, pre_drone, pre_obj = current_truck, current_drone, current_obj
        current_truck, current_drone, current_obj = self.local_search_descent(current_truck, current_drone, current_obj)
        if not self.is_feasible(current_truck, current_drone):        # defensive: S* must be feasible
            current_truck, current_drone, current_obj = pre_truck, pre_drone, pre_obj

        best_truck, best_drone, best_obj = current_truck, current_drone, current_obj
        t_best = time.process_time() - st
        n_restarts = n_restarts_skipped = 0
        elite_pool: List[Dict[int, List[int]]] = [current_drone]

        iteration = 0
        since_improvement = 0
        it2evpr = cfg.evpr_freq

        while iteration < cfg.n_max:
            iteration += 1

            if cfg.time_limit is not None and (time.process_time() - st) > cfg.time_limit:
                iteration -= 1
                break

            # -- cheap perturbation (the SA-like part) --
            cand_truck, cand_drone = self._random_neighbor(current_truck, current_drone)
            cand_obj = self.sum_arr_time(cand_truck, cand_drone)
            drone_changed = cand_drone is not current_drone
            candidate_ok = self.is_feasible(cand_truck, cand_drone) if drone_changed else True
            if candidate_ok and cand_obj <= current_obj:
                current_truck, current_drone, current_obj = cand_truck, cand_drone, cand_obj

            # -- periodic full local-search intensification --
            if iteration % cfg.local_search_freq == 0:
                current_truck, current_drone, current_obj = self.local_search_descent(
                    current_truck, current_drone, current_obj
                )

            # -- periodic path relinking against the elite pool (GRASP-PR's edge) --
            if elite_pool and iteration % cfg.pr_freq == 0:
                target = elite_pool[int(np.random.randint(0, len(elite_pool)))]
                x_star_fwd, f_star_fwd = self.path_relinking(current_truck, current_drone, target)
                x_star_bwd, f_star_bwd = self.path_relinking(current_truck, target, current_drone)
                x_star, f_star = (x_star_fwd, f_star_fwd) if f_star_fwd < f_star_bwd else (x_star_bwd, f_star_bwd)
                if x_star not in elite_pool:
                    if len(elite_pool) < cfg.maxps:
                        elite_pool.append(x_star)
                    elif len(elite_pool) == cfg.maxps:
                        del elite_pool[0]
                        elite_pool.append(x_star)
                if f_star < current_obj and self.is_feasible(current_truck, x_star):
                    current_drone, current_obj = x_star, f_star

            self._elite_pool_update(elite_pool, current_drone, current_obj, best_obj)

            if it2evpr < 1:
                if cfg.use_evpr:
                    elite_pool = self.evpr(current_truck, elite_pool)
                it2evpr = cfg.evpr_freq + 1
            it2evpr -= 1

            # Defense in depth: unlike v1/v2 (which discard truck_route/drone_route
            # every iteration via a fresh construction), ILS carries current_obj
            # forward across iterations, so any bookkeeping drift would compound
            # over time instead of being discarded next iteration. Re-verify
            # against a fresh computation every iteration and self-correct --
            # cheap (O(n)) and this is also what actually gets recorded as best.
            temp_obj_val = self.sum_arr_time(current_truck, current_drone)
            current_obj = temp_obj_val
            if temp_obj_val < best_obj and self.is_feasible(current_truck, current_drone):
                best_obj = temp_obj_val
                best_truck, best_drone = current_truck, current_drone
                t_best = time.process_time() - st
                since_improvement = 0
            else:
                since_improvement += 1

            # -- diversification: full reconstruction only when genuinely stuck --
            if cfg.restart_patience and since_improvement >= cfg.restart_patience:
                new_truck, new_drone = self.construct_feasible_solution()
                if self.is_feasible(new_truck, new_drone):
                    new_obj = self.sum_arr_time(new_truck, new_drone)
                    new_truck, new_drone, new_obj = self.local_search_descent(new_truck, new_drone, new_obj)
                    current_truck, current_drone, current_obj = new_truck, new_drone, new_obj
                    n_restarts += 1
                else:
                    n_restarts_skipped += 1          # ConstructFeasible failed: keep the current solution
                since_improvement = 0

        runtime_s = time.process_time() - st

        return RunResult(
            instance=self.inst.name,
            objective=best_obj,
            runtime_s=runtime_s,
            iterations=iteration,
            truck_route=best_truck,
            drone_route=best_drone,
            diagnostics={"restarts": n_restarts, "restarts_skipped": n_restarts_skipped},
            time_to_best=t_best,
        )
