"""
Simulated Annealing baseline for the 2E-CTVRP.

Single-trajectory search: build one feasible initial solution, then repeat --
pick one of the seven neighborhood operators uniformly at random, apply it,
and accept the candidate under the Metropolis criterion with a time-based
geometric cooling schedule:

    tau(t) = tau0 * (tau_min / tau0) ** (t / t_max),   t = elapsed CPU time

with tau0 = initial_temp_fraction * z0 and tau_min = min_temp_fraction * z0
(z0 = the initial solution's objective), so temperature reaches its floor
exactly at the time budget rather than after a fixed iteration count -- an
earlier per-iteration cooling schedule reached its floor within the first
second of a 30s run and then searched at zero temperature for the rest of
the budget; this fixes that.

Shares construction, repair, the objective and the neighborhood operators
with the other two heuristics via src/common.py, so all three search over
exactly the same solution space. Entry point: scripts/run_heuristic.py.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from common import ClusteredData, DistanceData, ECTVRPOperators, Instance, RunResult


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SAConfig:
    n_max: int = 100_000_000              # safety cap on iterations (never binding once time_limit is set)
    initial_temp_fraction: float = 0.10   # tau0 = fraction * initial solution's objective (paper: 0.10)
    min_temp_fraction: float = 1e-4       # tau_min = fraction * initial objective (paper: 1e-4)
    local_search_freq: int = 0            # 0 = pure SA; N>0 = local-search descent every N accepted moves
    time_limit: Optional[float] = 30.0    # t_max: primary stop and the cooling schedule's time horizon


# ============================================================
# SIMULATED ANNEALING SOLVER
# ============================================================

class SimulatedAnnealingSolver(ECTVRPOperators):
    def __init__(self, inst: Instance, clustered: ClusteredData, dist: DistanceData, config: SAConfig):
        super().__init__(inst, clustered, dist)
        self.config = config

    def _random_neighbor(self, truck_route, drone_route):
        """Uniformly pick one of the 6 truck-route neighborhoods or the
        drone-swap neighborhood and apply it once."""
        n_ops = len(self.NEIGHBORHOOD_OPERATORS)
        move = int(np.random.randint(0, n_ops + 1))
        if move < n_ops:
            candidate_truck = getattr(self, self.NEIGHBORHOOD_OPERATORS[move])(truck_route, drone_route)
            return candidate_truck, drone_route
        candidate_drone = self.bias_swap_drone(drone_route)
        return truck_route, candidate_drone

    def solve(self) -> RunResult:
        cfg = self.config
        if cfg.time_limit is None:
            raise ValueError("SA's time-based cooling schedule needs a time_limit (t_max)")
        st = time.process_time()

        truck_route, drone_route = self.construct_feasible_solution()
        demand = self.connect_demand(drone_route)
        if not (self.drone_capacity_verify(drone_route) and self.capacity_verify(truck_route, demand)):
            # construct_feasible_solution() exhausted its retries without
            # finding a capacity-feasible starting point (the instance is
            # likely infeasible under real truck/drone capacity) -- report a
            # clean failure instead of silently annealing an infeasible route.
            return RunResult(
                instance=self.inst.name, objective=None,
                runtime_s=time.process_time() - st, iterations=0,
                truck_route=None, drone_route=None,
            )

        current_obj = self.sum_arr_time(truck_route, drone_route)
        best_truck, best_drone, best_obj = truck_route, drone_route, current_obj
        t_best = time.process_time() - st

        temperature = max(cfg.initial_temp_fraction * current_obj, 1e-9)
        min_temp = max(cfg.min_temp_fraction * current_obj, 1e-12)
        initial_temp = temperature

        iteration = 0
        while iteration < cfg.n_max:
            iteration += 1

            elapsed = time.process_time() - st
            if cfg.time_limit is not None and elapsed > cfg.time_limit:
                iteration -= 1
                break
            # tau(t) = tau0 * (tau_min/tau0)**(t/t_max), t = CPU time since solve() started
            temperature = initial_temp * (min_temp / initial_temp) ** min(elapsed / cfg.time_limit, 1.0)

            cand_truck, cand_drone = self._random_neighbor(truck_route, drone_route)
            cand_obj = self.sum_arr_time(cand_truck, cand_drone)
            delta = cand_obj - current_obj

            # bias_swap_drone (unlike the truck-route operators) can change
            # which centroid's demand a satellite's truck carries without any
            # capacity check of its own, so a changed drone_route needs to be
            # verified before the move is allowed to become the new current
            # solution -- otherwise SA can "improve" by cheating on capacity.
            drone_changed = cand_drone is not drone_route
            candidate_ok = self.is_feasible(cand_truck, cand_drone) if drone_changed else True

            if candidate_ok and (delta <= 0 or np.random.random() < math.exp(-delta / temperature)):
                truck_route, drone_route, current_obj = cand_truck, cand_drone, cand_obj

                if cfg.local_search_freq and iteration % cfg.local_search_freq == 0:
                    truck_route, drone_route, current_obj = self.local_search_descent(
                        truck_route, drone_route, current_obj
                    )

                if current_obj < best_obj:
                    best_truck, best_drone, best_obj = truck_route, drone_route, current_obj
                    t_best = time.process_time() - st

        runtime_s = time.process_time() - st

        return RunResult(
            instance=self.inst.name,
            objective=best_obj,
            runtime_s=runtime_s,
            iterations=iteration,
            truck_route=best_truck,
            drone_route=best_drone,
            time_to_best=t_best,
        )


# ============================================================
# CLI
# ============================================================
