"""
Rolling-horizon execution simulator for the dynamic 2E-CTVRP.

A Plan is the full (whole-horizon) decision: one ordered list of satellites per
truck (depots implicit) plus the satellite -> centroid assignment sigma. The
simulator executes it against the TRUE road network and demand process:

    a_p = a_{p-1} + w_{p-1} + t_true(prev, cur, departure time)
    w_i = waves_i * 2 t'_{i sigma(i)},  waves_i = ceil(d_sigma(i)(a_i) / (U P))
    e_i = a_i + t'_{i sigma(i)}         (first drone delivery at the centroid)

`waves` generalises the paper's single parallel sortie (waves = 1 whenever
d <= U*P, which holds for every shipped instance) so that a demand surge above
U*P is served by repeated sorties instead of making the instance infeasible.

`commit(timeline, t_event)` freezes history (research guide Sec. 4.1): every
satellite a truck has reached by t_event is locked, a truck driving towards a
satellite is committed to finish that arc (the target is locked too), and its
virtual start becomes that satellite at time a + w. Trucks already heading back
to the depot are retired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynamic_env import DemandProcess, RoadNetwork


@dataclass
class Plan:
    routes: Dict[int, List[int]]       # truck k -> satellites in visiting order (no depots)
    sigma: Dict[int, int]              # satellite -> centroid (both 1-based)

    def copy(self) -> "Plan":
        return Plan({k: list(v) for k, v in self.routes.items()}, dict(self.sigma))


@dataclass
class SatRecord:
    truck: int
    position: int
    arrival: float
    wait: float
    delivery: float
    centroid: int
    demand_at_arrival: float
    delivered: float
    waves: int
    priority: float
    battery_use: float = 0.0     # sortie energy / usable battery (<= 1 means feasible)


@dataclass
class TruckRecord:
    truck: int
    legs: List[Tuple[int, int, float, float]]   # (from, to, depart, arrive)
    ret: float
    load: float


@dataclass
class Timeline:
    sats: Dict[int, SatRecord]
    trucks: Dict[int, TruckRecord]


@dataclass
class TruckState:
    k: int
    start_node: int          # 0 = depot, else satellite id
    avail_time: float        # time the truck is free to leave start_node
    prefix: List[int]        # locked satellites (in visiting order)
    active: bool             # False = already heading/returned to depot
    fixed_return: Optional[float] = None


@dataclass
class Snapshot:
    time: float
    trucks: Dict[int, TruckState]
    locked: Dict[int, SatRecord]          # locked satellites (visited or committed target)
    free_sats: List[int]
    free_centroids: List[int]
    traversed_arcs: List[Tuple[int, int]] = field(default_factory=list)


class Simulator:
    def __init__(self, n: int, Q: float, UP: float, drone_time: np.ndarray,
                 net: RoadNetwork, demand: DemandProcess, battery=None):
        self.n = n
        self.vend = n + 1
        self.Q = Q
        self.UP = UP
        self.tdm = drone_time          # (n satellites) x (n centroids), hours one-way
        self.net = net
        self.demand = demand
        self.battery = battery       # drone_energy.DroneBattery or None (no battery check)

    def waves(self, d: float) -> int:
        return max(1, int(math.ceil(d / self.UP - 1e-9)))

    def simulate(self, plan: Plan) -> Timeline:
        sats: Dict[int, SatRecord] = {}
        trucks: Dict[int, TruckRecord] = {}
        for k, route in sorted(plan.routes.items()):
            t, prev, load, legs = 0.0, 0, 0.0, []
            for pos, s in enumerate(route, start=1):
                dt = self.net.true_time(prev, s, t)
                legs.append((prev, s, t, t + dt))
                t += dt
                c = plan.sigma[s]
                d = float(self.demand.demand_at(t)[c - 1])
                w = float(self.demand.priority_at(t)[c - 1])
                fly = float(self.tdm[s - 1, c - 1])
                nw = self.waves(d)
                delivered = max(0.0, min(d, self.Q - load))
                load += delivered
                use = 0.0
                if self.battery is not None:
                    e = float(self.battery.model.sortie_energy(fly, self.battery.v_d, self.battery.load_frac(d)))
                    use = e / self.battery.model.usable
                sats[s] = SatRecord(k, pos, t, nw * 2 * fly, t + fly, c, d, delivered, nw, w, use)
                t += nw * 2 * fly
                prev = s
            dt = self.net.true_time(prev, self.vend, t)
            legs.append((prev, self.vend, t, t + dt))
            t += dt
            trucks[k] = TruckRecord(k, legs, t, load)
        return Timeline(sats, trucks)

    def initial_snapshot(self, trucks: List[int]) -> Snapshot:
        ts = {k: TruckState(k, 0, 0.0, [], True) for k in trucks}
        return Snapshot(0.0, ts, {}, list(range(1, self.n + 1)), list(range(1, self.n + 1)))

    def commit(self, plan: Plan, tl: Timeline, t_event: float) -> Snapshot:
        trucks: Dict[int, TruckState] = {}
        locked: Dict[int, SatRecord] = {}
        traversed: List[Tuple[int, int]] = []
        for k, route in sorted(plan.routes.items()):
            rec = tl.trucks[k]
            if t_event <= 0.0:
                trucks[k] = TruckState(k, 0, 0.0, [], True)
                continue
            prefix: List[int] = []
            state: Optional[TruckState] = None
            for (i, j, dep, arr) in rec.legs:
                if dep > t_event + 1e-12:
                    break                       # leg not started yet
                if arr <= t_event + 1e-12:
                    traversed.append((i, j))
                if j == self.vend:
                    state = TruckState(k, i, dep, list(prefix), False, fixed_return=rec.ret)
                    break
                prefix.append(j)                # reached, or committed to reach, satellite j
                locked[j] = tl.sats[j]
            if state is None:
                last = prefix[-1]
                srec = tl.sats[last]
                state = TruckState(k, last, srec.arrival + srec.wait, list(prefix), True)
            trucks[k] = state
        free_sats = [s for s in range(1, self.n + 1) if s not in locked]
        used_c = {plan.sigma[s] for s in locked}
        free_c = [c for c in range(1, self.n + 1) if c not in used_c]
        return Snapshot(t_event, trucks, locked, free_sats, free_c, traversed)


def realized_metrics(tl: Timeline, demand: DemandProcess, Q: float) -> dict:
    """Operational KPIs of an executed plan (after the last event).

    f1 = (1/D) sum_i w_sigma(i) * delivered_i * e_i   (priority-weighted delivery time)
    f2 = Z3 = max_i a_i - min_i a_i                    (arrival-time spread, equity)
    f3 = Z1 = sum_i a_i + sum_k return_k               (paper's objective)
    unmet = demand not delivered (incl. surges that hit an already-served cluster)."""
    final_d = demand.current_demand
    final_w = demand.current_priority
    D = float(final_d.sum())
    a = np.array([r.arrival for r in tl.sats.values()])
    e = np.array([r.delivery for r in tl.sats.values()])
    delivered_by_c = np.zeros(len(final_d))
    f1 = 0.0
    for r in tl.sats.values():
        delivered_by_c[r.centroid - 1] += r.delivered
        f1 += final_w[r.centroid - 1] * r.delivered * r.delivery
    unmet = float(np.clip(final_d - delivered_by_c, 0, None).sum())
    rets = np.array([t.ret for t in tl.trucks.values()])
    return dict(
        f1=f1 / D, f2=float(a.max() - a.min()), f3=float(a.sum() + rets.sum()),
        C_del=float(e.max()), C_ret=float(rets.max()), D=D,
        unmet=unmet, unmet_ratio=unmet / D,
        max_load_ratio=float(max(t.load for t in tl.trucks.values()) / Q),
        max_battery_use=float(max(r.battery_use for r in tl.sats.values())),
        battery_violations=int(sum(r.battery_use > 1 + 1e-9 for r in tl.sats.values())),
    )
