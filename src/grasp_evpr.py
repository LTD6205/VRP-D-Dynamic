"""
GRASP-PR(+EvPR): the predecessor baseline metaheuristic.

Every iteration does a full randomized greedy construction, a full
local-search descent over all seven neighborhoods, and forward/backward path
relinking against an elite pool (Section 4 of the paper). EvPR (periodic
path relinking between every pair of elite-pool solutions) can be switched
off with `use_evpr=False` to run a plain GRASP-PR ablation baseline.

Shares construction, repair, the objective and the neighborhood operators
with the other two heuristics via src/common.py, so all three search over
exactly the same solution space. Entry point: scripts/run_heuristic.py.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from common import ClusteredData, DistanceData, ECTVRPOperators, Instance, RunResult


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GraspEvPRConfig:
    n_max: int = 100_000_000   # safety cap on iterations (never binding once time_limit is set)
    mu: float = 0.008          # beta in the paper (Table 7): elite-pool acceptance threshold
    maxps: int = 20            # omega in the paper (Table 7): elite-pool (P) size
    evpr_freq: int = 100       # iterations between EvPR calls (paper: every 100 iterations)
    use_evpr: bool = True      # ablation switch: evolve the elite pool with EvPR
    time_limit: Optional[float] = None  # primary stop (CPU s); set via scripts/run_heuristic.py --time
    record_history: bool = False  # opt-in: log (iteration, cpu_time, best_so_far) on every improvement


# ============================================================
# GRASP-PR(+EvPR) SOLVER
# ============================================================

class GraspEvPRSolver(ECTVRPOperators):
    def __init__(self, inst: Instance, clustered: ClusteredData, dist: DistanceData, config: GraspEvPRConfig):
        super().__init__(inst, clustered, dist)
        self.config = config

    # -------------------- path relinking / EvPR --------------------

    def sym_diff_cal(self, initial_solution, final_solution) -> List[int]:
        return [i for i in self.V_s if initial_solution[i] != final_solution[i]]

    def path_relinking(self, truck_route, initial_drone_route, final_drone_route):
        initial_drone_route = {k: v for k, v in sorted(initial_drone_route.items(), key=lambda item: item[1])}
        final_drone_route = {k: v for k, v in sorted(final_drone_route.items(), key=lambda item: item[1])}
        current = copy.deepcopy(initial_drone_route)

        f_initial = self.sum_arr_time(truck_route, current)
        f_final = self.sum_arr_time(truck_route, final_drone_route)
        f_star = min(f_initial, f_final)
        # NOTE: must be an independent snapshot, not an alias -- `current` is
        # mutated in place for the rest of this function as the walk
        # proceeds, and `final_drone_route` is never copied by the caller
        # either. Aliasing here let x_star's *content* silently drift away
        # from the f_star value it's paired with by the time this returns.
        x_star = copy.deepcopy(current) if f_star == f_initial else copy.deepcopy(final_drone_route)

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
            current_sol = copy.deepcopy(current)
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

    # -------------------- main GRASP-PR(+EvPR) loop --------------------

    def solve(self) -> RunResult:
        cfg = self.config
        st = time.process_time()

        truck_route_best: Optional[Dict[int, List[int]]] = None
        drone_route_best: Optional[Dict[int, List[int]]] = None
        overall_obj_val = 9999.0

        elite_pool: List[Dict[int, List[int]]] = []
        it2evpr = cfg.evpr_freq
        iteration = 0
        # Convergence trace for time-to-target analysis, e.g. comparing how
        # fast EvPR-on vs EvPR-off reach a given objective threshold. Only
        # populated when explicitly requested -- zero overhead otherwise.
        self.history: List[tuple] = []

        while iteration < cfg.n_max:
            iteration += 1

            if cfg.time_limit is not None and (time.process_time() - st) > cfg.time_limit:
                iteration -= 1
                break

            truck_route, drone_route = self.construct_initial_solution()
            if not self.drone_capacity_verify(drone_route):
                continue
            satellites_demand = self.connect_demand(drone_route)
            if not self.capacity_verify(truck_route, satellites_demand):
                truck_route = self.repair(truck_route, drone_route)
            if not self.capacity_verify(truck_route, satellites_demand):
                continue

            obj_val = self.sum_arr_time(truck_route, drone_route)
            truck_route, drone_route, obj_val = self.local_search_descent(truck_route, drone_route, obj_val)

            if obj_val < (1 + cfg.mu) * overall_obj_val and drone_route not in elite_pool:
                if len(elite_pool) < cfg.maxps:
                    elite_pool.append(drone_route)
                elif len(elite_pool) == cfg.maxps:
                    del elite_pool[0]
                    elite_pool.append(drone_route)

            if elite_pool:
                target = elite_pool[int(np.random.randint(0, len(elite_pool)))]
                x_star_fwd, f_star_fwd = self.path_relinking(truck_route, drone_route, target)
                x_star_bwd, f_star_bwd = self.path_relinking(truck_route, target, drone_route)
                x_star, f_star = (x_star_fwd, f_star_fwd) if f_star_fwd < f_star_bwd else (x_star_bwd, f_star_bwd)
                if x_star not in elite_pool:
                    if len(elite_pool) < cfg.maxps:
                        elite_pool.append(x_star)
                    elif len(elite_pool) == cfg.maxps:
                        del elite_pool[0]
                        elite_pool.append(x_star)
                    # Path relinking only swaps which centroid two satellites
                    # serve, which can move enough demand onto a truck's
                    # existing route to break its capacity -- verify before
                    # adopting x_star as the working solution.
                    if f_star < obj_val and self.is_feasible(truck_route, x_star):
                        drone_route = x_star
                        obj_val = f_star

            if it2evpr < 1:
                if cfg.use_evpr:
                    elite_pool = self.evpr(truck_route, elite_pool)
                it2evpr = cfg.evpr_freq + 1
            it2evpr -= 1

            # Final safety net: never record a solution as the incumbent
            # without re-confirming feasibility, regardless of which step
            # above touched drone_route last.
            temp_obj_val = self.sum_arr_time(truck_route, drone_route)
            if temp_obj_val < overall_obj_val and self.is_feasible(truck_route, drone_route):
                overall_obj_val = temp_obj_val
                truck_route_best = truck_route
                drone_route_best = drone_route
                if cfg.record_history:
                    self.history.append((iteration, time.process_time() - st, overall_obj_val))

        runtime_s = time.process_time() - st
        objective = overall_obj_val if truck_route_best is not None else None

        return RunResult(
            instance=self.inst.name,
            objective=objective,
            runtime_s=runtime_s,
            iterations=iteration,
            truck_route=truck_route_best,
            drone_route=drone_route_best,
        )


# ============================================================
# CLI
# ============================================================
