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

RECOURSE (re-delivery). A surge that reaches a cluster after its satellite was
served creates a residual-demand JOB (cluster, amount). A job is a route item
with id > n+1; it is executed at a HOST satellite chosen by the planner (any
satellite whose drones can reach the cluster with that load): the truck drives to
the host and flies extra drone wave(s). If the truck is still parked at the
cluster's own satellite when the surge arrives, inserting the job first with that
host costs no driving -- it is simply an extra wave ("surge during delivery").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynamic_env import DemandProcess, RoadNetwork


@dataclass
class Job:
    """Residual-demand job created by recourse: deliver `amount` to `cluster` by drone
    from satellite `host` (host chosen by the planner, default = cluster's own satellite)."""
    cluster: int
    amount: float
    host: int
    created: float = 0.0


@dataclass
class Plan:
    routes: Dict[int, List[int]]       # truck k -> items in visiting order (satellites 1..n, jobs > n+1)
    sigma: Dict[int, int]              # satellite -> centroid (both 1-based)
    jobs: Dict[int, Job] = field(default_factory=dict)

    def copy(self) -> "Plan":
        return Plan({k: list(v) for k, v in self.routes.items()}, dict(self.sigma),
                    {j: Job(**vars(x)) for j, x in self.jobs.items()})


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
    host: int = 0                # satellite the drones flew from (== item for a satellite)


@dataclass
class TruckRecord:
    truck: int
    legs: List[Tuple[int, int, float, float, int]]   # (from_loc, to_loc, depart, arrive, item)
    ret: float
    load: float


@dataclass
class Timeline:
    sats: Dict[int, SatRecord]
    trucks: Dict[int, TruckRecord]
    jobs: Dict[int, SatRecord] = field(default_factory=dict)


@dataclass
class TruckState:
    k: int
    start_node: int          # location: 0 = depot, else satellite id
    avail_time: float        # time the truck is free to leave start_node
    prefix: List[int]        # locked items (satellites / jobs) in visiting order
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
    locked_jobs: Dict[int, SatRecord] = field(default_factory=dict)


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

    def _battery_use(self, fly: float, d: float) -> float:
        if self.battery is None:
            return 0.0
        e = float(self.battery.model.sortie_energy(fly, self.battery.v_d, self.battery.load_frac(d)))
        return e / self.battery.model.usable

    def simulate(self, plan: Plan) -> Timeline:
        sats: Dict[int, SatRecord] = {}
        jobs: Dict[int, SatRecord] = {}
        trucks: Dict[int, TruckRecord] = {}
        for k, route in sorted(plan.routes.items()):
            t, prev, load, legs = 0.0, 0, 0.0, []
            for pos, item in enumerate(route, start=1):
                job = plan.jobs.get(item) if item > self.n else None
                loc = job.host if job else item
                dt = self.net.true_time(prev, loc, t)
                legs.append((prev, loc, t, t + dt, item))
                t += dt
                if job:
                    c, d = job.cluster, float(job.amount)
                else:
                    c = plan.sigma[item]
                    d = float(self.demand.demand_at(t)[c - 1])
                w = float(self.demand.priority_at(t)[c - 1])
                fly = float(self.tdm[loc - 1, c - 1])
                nw = self.waves(d)
                delivered = max(0.0, min(d, self.Q - load))
                load += delivered
                rec = SatRecord(k, pos, t, nw * 2 * fly, t + fly, c, d, delivered, nw, w,
                                self._battery_use(fly, d), loc)
                (jobs if job else sats)[item] = rec
                t += nw * 2 * fly
                prev = loc
            dt = self.net.true_time(prev, self.vend, t)
            legs.append((prev, self.vend, t, t + dt, self.vend))
            t += dt
            trucks[k] = TruckRecord(k, legs, t, load)
        return Timeline(sats, trucks, jobs)

    def initial_snapshot(self, trucks: List[int]) -> Snapshot:
        ts = {k: TruckState(k, 0, 0.0, [], True) for k in trucks}
        return Snapshot(0.0, ts, {}, list(range(1, self.n + 1)), list(range(1, self.n + 1)))

    def commit(self, plan: Plan, tl: Timeline, t_event: float) -> Snapshot:
        trucks: Dict[int, TruckState] = {}
        locked: Dict[int, SatRecord] = {}
        locked_jobs: Dict[int, SatRecord] = {}
        traversed: List[Tuple[int, int]] = []
        for k, route in sorted(plan.routes.items()):
            rec = tl.trucks[k]
            if t_event <= 0.0:
                trucks[k] = TruckState(k, 0, 0.0, [], True)
                continue
            prefix: List[int] = []
            state: Optional[TruckState] = None
            for (i, j, dep, arr, item) in rec.legs:
                if dep > t_event + 1e-12:
                    break                       # leg not started yet
                if arr <= t_event + 1e-12:
                    traversed.append((i, j))
                if item == self.vend:
                    state = TruckState(k, i, dep, list(prefix), False, fixed_return=rec.ret)
                    break
                prefix.append(item)             # reached, or committed to reach, this stop
                if item > self.n:
                    locked_jobs[item] = tl.jobs[item]
                else:
                    locked[item] = tl.sats[item]
            if state is None:
                last = prefix[-1]
                srec = tl.jobs[last] if last > self.n else tl.sats[last]
                state = TruckState(k, srec.host, srec.arrival + srec.wait, list(prefix), True)
            trucks[k] = state
        free_sats = [s for s in range(1, self.n + 1) if s not in locked]
        used_c = {plan.sigma[s] for s in locked}
        free_c = [c for c in range(1, self.n + 1) if c not in used_c]
        return Snapshot(t_event, trucks, locked, free_sats, free_c, traversed, locked_jobs)


def create_residual_jobs(plan: Plan, tl: Timeline, snap: Snapshot, demand: DemandProcess,
                         next_id: int) -> Tuple[Dict[int, Job], int]:
    """Recourse: after the demand update at snap.time, every cluster whose satellite has
    ALREADY been reached gets a residual job for

        r_c = d_c(now) - delivered_main_c - delivered_by_executed_jobs_c - planned_pending_jobs_c

    (a surge after or during delivery, or a shortfall because the truck ran out of goods).
    Clusters not yet reached need no job: their main visit delivers the demand at arrival."""
    d = demand.current_demand
    new: Dict[int, Job] = {}
    for s, rec in snap.locked.items():
        if rec.arrival > snap.time + 1e-12:
            continue                                    # committed target: delivers at arrival
        c = rec.centroid
        covered = rec.delivered
        for j, job in plan.jobs.items():
            if job.cluster != c:
                continue
            jr = tl.jobs.get(j)
            if j in snap.locked_jobs and jr is not None and jr.arrival <= snap.time + 1e-12:
                covered += jr.delivered                 # executed (possibly partially)
            else:
                covered += job.amount                   # still planned
        r = float(d[c - 1]) - covered
        if r > 1e-6:
            new[next_id] = Job(cluster=c, amount=r, host=s, created=snap.time)
            next_id += 1
    return new, next_id


def realized_metrics(tl: Timeline, demand: DemandProcess, Q: float) -> dict:
    """Operational KPIs of an executed plan (after the last event).

    f1 = (1/D) [ sum_stops w_c * delivered * e  +  sum_c w_c * unmet_c * C_ret ]
         priority-weighted delivery time incl. re-deliveries; demand never delivered is
         charged at the time the last truck is back (the earliest a new trip could start),
         so leaving demand undelivered can never look better than delivering it late.
    f2 = Z3 = max_i a_i - min_i a_i              (arrival-time spread over the satellites' main visits)
    f3 = Z1 = sum_i a_i + sum_k return_k         (paper's objective over the satellites' main visits;
         re-delivery stops are not added as extra terms, their cost shows up as later
         arrivals and returns)
    unmet = demand never delivered (only when the fleet ran out of goods / no recourse)."""
    final_d = demand.current_demand
    final_w = demand.current_priority
    D = float(final_d.sum())
    a = np.array([r.arrival for r in tl.sats.values()])
    stops = list(tl.sats.values()) + list(tl.jobs.values())
    e = np.array([r.delivery for r in stops])
    delivered_by_c = np.zeros(len(final_d))
    f1 = 0.0
    for r in stops:
        delivered_by_c[r.centroid - 1] += r.delivered
        f1 += final_w[r.centroid - 1] * r.delivered * r.delivery
    unmet_c = np.clip(final_d - delivered_by_c, 0, None)
    unmet = float(unmet_c.sum())
    rets = np.array([t.ret for t in tl.trucks.values()])
    f1 += float((final_w * unmet_c).sum()) * float(rets.max())
    return dict(
        f1=f1 / D, f2=float(a.max() - a.min()), f3=float(a.sum() + rets.sum()),
        C_del=float(e.max()), C_ret=float(rets.max()), D=D,
        unmet=unmet, unmet_ratio=unmet / D, n_redeliveries=len(tl.jobs),
        max_load_ratio=float(max(t.load for t in tl.trucks.values()) / Q),
        max_battery_use=float(max(r.battery_use for r in stops)),
        battery_violations=int(sum(r.battery_use > 1 + 1e-9 for r in stops)),
    )
