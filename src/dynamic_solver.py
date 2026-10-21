"""
Dynamic multi-objective re-optimizer for the rolling-horizon 2E-CTVRP.

At every decision epoch it solves the residual problem left by `Simulator.commit`:

* each ACTIVE truck k starts at its committed node v_{pi_k(0)}(t) at time
  a^k_{pi_k(0)}(t) with remaining capacity Q_k(t);
* the FREE satellites must be (re)distributed among active trucks and matched
  one-to-one to the FREE centroids (sigma_t);
* arc times are the planner's expected times E[t_ij(t)] (belief model);
* every drone sortie must be battery-feasible: its energy, which depends on the
  flight SPEED v_d and the carried LOAD (src/drone_energy.py), must not exceed the
  usable battery, e_ij(v_d, l_j) <= E (1 - reserve).

Objectives (research guide Sec. 3.4, all over the WHOLE horizon so values are
comparable across epochs; locked parts are constants):

    f1 = (1/D) sum_i w_sigma(i) d_sigma(i) e_i     priority-weighted delivery time
    f2 = max_i a_i - min_i a_i                      arrival spread (Z3, equity)
    f3 = sum_i a_i + sum_k ret_k                    cumulative fleet time (Z1, paper objective)

Search = the paper's GRASP-ILS-PR skeleton generalised to the dynamic, multi-
objective setting:

* single-move perturbations from the repo's neighbourhoods N1-N7 + a
  destroy-repair move, chosen by a pluggable OperatorSelector (uniform /
  adaptive roulette; the DRL agent of SR-DRL-DMOEA goes here);
* "segment-aware" focus: a move's primary truck is sampled with probability
  proportional to that route's disruption score c_k(t) (guide Sec. 4.4);
* periodic best-of-K local search and path relinking on sigma with an elite
  pool that SURVIVES epochs (memory);
* scalarisation by a cycle of weight vectors; every feasible candidate is
  offered to a Pareto archive, from which the plan to commit is chosen.

Warm start (strategy "warm"): the previous plan's unvisited suffix + projected
archive + elite pool. Cold start (strategy "restart"): randomized construction
from scratch, no memory.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from drone_energy import FixedEndurance
from dynamic_sim import Job, Plan, SatRecord, Snapshot
from mo_utils import ParetoArchive, simplex_weights

BIG = 1e6
EMPTY_TRUCK_PENALTY = 1e5
BATTERY_PENALTY = 1e5       # a satellite->centroid pair the drones cannot fly (C4)   # a depot-start truck must visit >=1 satellite (paper's MILP)

OPERATORS = (
    "relocate",        # N1 intra-route relocation
    "swap",            # N2 biased intra-route swap
    "two_opt",         # N3 intra-route 2-opt
    "inter_swap",      # N4 inter-route swap
    "two_opt_star",    # N5 inter-route 2-opt* (tail exchange)
    "inter_relocate",  # N6 inter-route relocation
    "drone_swap",      # N7 biased centroid swap (2nd echelon)
    "destroy_repair",  # ruin & recreate (spec action set, eq. 11)
    "job_host",        # recourse: move a re-delivery job to another battery-feasible host satellite
)


# ============================================================
# Residual problem at one epoch
# ============================================================

class ReplanProblem:
    def __init__(self, snap: Snapshot, T: np.ndarray, tdm: np.ndarray, demand: np.ndarray,
                 priority: np.ndarray, Q: float, UP: float, battery, n: int,
                 jobs: Optional[Dict[int, Job]] = None,
                 demand_sd: Optional[np.ndarray] = None, z: float = 0.0, service=None):
        self.snap = snap
        self.n = n
        self.vend = n + 1
        self.T = T.tolist()
        self.Tnp = T
        self.tdm_np = tdm
        self.tdm = tdm.tolist()
        self.d = [float(x) for x in demand]
        # demand belief (POMDP): capacity and battery use mean + z * sd (robust), objectives use the mean
        sd = np.zeros(len(self.d)) if demand_sd is None else np.asarray(demand_sd, dtype=float)
        self.d_cap = [float(m + z * s_) for m, s_ in zip(self.d, sd)]
        self.w = [float(x) for x in priority]
        self.Q = Q
        self.UP = UP
        # battery: drone_energy.DroneBattery (speed+load energy model) or a number (legacy fixed time budget)
        self.battery = FixedEndurance(battery) if isinstance(battery, (int, float)) else battery
        self.D = float(sum(self.d))

        self.free = list(snap.free_sats)
        self.free_c = list(snap.free_centroids)
        self.free_c_set = set(self.free_c)
        self.active = [k for k, ts in sorted(snap.trucks.items()) if ts.active]
        self.start = {k: snap.trucks[k].start_node for k in self.active}
        self.avail = {k: snap.trucks[k].avail_time for k in self.active}

        # second-echelon drone service model (drone_service.py): centroid (paper) or route (victim CVRP)
        if service is None:
            from drone_service import CentroidService
            service = CentroidService(tdm, self.battery, UP)
        self.service = service
        # dwell W_sc and delivery offset delta_sc at the planned (mean) demand; battery use and
        # feasibility at the robust demand d_cap (the load depends on the demand)
        W, OFF, USE, OK = service.tables(np.asarray(self.d, dtype=float), np.asarray(self.d_cap, dtype=float))
        self.Wl, self.OFFl = W.tolist(), OFF.tolist()
        self.energy = USE
        self.pair_ok = OK

        # ---- recourse jobs (residual demand) that still have to be delivered ----
        locked_jobs = getattr(snap, "locked_jobs", {}) or {}
        self.jobs: Dict[int, Tuple[int, float]] = {
            j: (job.cluster, float(job.amount)) for j, job in (jobs or {}).items() if j not in locked_jobs}
        self.free_jobs = sorted(self.jobs)
        self.job_hosts: Dict[int, List[int]] = {}
        self.job_svc: Dict[Tuple[int, int], Tuple[float, float]] = {}   # (job, host) -> (W, offset)
        for j, (c, a) in self.jobs.items():
            svc = {h: service.service(h, c, a, split=False) for h in range(1, n + 1)}
            hosts = [h for h, r in svc.items() if r.use <= 1.0 + 1e-12]
            # the cluster's own satellite flew this cluster before -- keep it as a last resort
            self.job_hosts[j] = hosts or [int(jobs[j].host)]
            for h, r in svc.items():
                self.job_svc[(j, h)] = (r.wait, r.offset)
        self.default_host = {j: (jobs[j].host if jobs[j].host in self.job_hosts[j] else self.job_hosts[j][0])
                             for j in self.jobs}

        # ---- constants from the locked part ----
        used = {k: 0.0 for k in self.active}
        a_list, f1num = [], 0.0
        for s, r in snap.locked.items():
            a_list.append(r.arrival)
            # delivered amount: fixed if already reached, current demand if only committed
            q = r.delivered if r.arrival <= snap.time + 1e-12 else self.d[r.centroid - 1]
            f1num += self.w[r.centroid - 1] * q * r.delivery
            if snap.trucks[r.truck].active:
                used[r.truck] += r.delivered if r.arrival <= snap.time + 1e-12 else self.d_cap[r.centroid - 1]
        for j, r in locked_jobs.items():   # executed / committed re-deliveries (not in f2/f3, only f1 + load)
            q = r.delivered if r.arrival <= snap.time + 1e-12 else r.demand_at_arrival
            f1num += self.w[r.centroid - 1] * q * r.delivery
            if snap.trucks[r.truck].active:
                used[r.truck] += q
        self.cap = {k: Q - used[k] for k in self.active}
        self.const_sum = sum(a_list) + sum(ts.fixed_return for ts in snap.trucks.values() if not ts.active)
        self.const_min = min(a_list) if a_list else math.inf
        self.const_max = max(a_list) if a_list else -math.inf
        self.const_f1 = f1num

    # ---------------- evaluation ----------------
    def evaluate(self, routes: Dict[int, List[int]], sigma: Dict[int, int],
                 hosts: Optional[Dict[int, int]] = None):
        """Returns (violation, (f1, f2, f3)). violation == 0 <=> feasible.
        Route items are free satellites (<= n) or re-delivery jobs (> n+1, served from hosts[j])."""
        n, jobs = self.n, self.jobs
        hosts = hosts or {}
        T, d, w, vend = self.T, self.d, self.w, self.vend
        W, OFF, job_svc = self.Wl, self.OFFl, self.job_svc
        s_arr, lo, hi, f1 = self.const_sum, self.const_min, self.const_max, self.const_f1
        viol = 0.0
        for k in self.active:
            r = routes[k]
            prev, t, load = self.start[k], self.avail[k], 0.0
            if prev == 0 and not r:
                viol += EMPTY_TRUCK_PENALTY
            Tp = None
            for s in r:
                if s > n:                                  # re-delivery job
                    c, amt = jobs[s]
                    h = hosts.get(s, self.default_host[s])
                    t += T[prev][h]                     # job stop: not counted in f2/f3 itself, but
                    wait, off = job_svc[(s, h)]            # it delays every later stop and the return
                    f1 += w[c - 1] * amt * (t + off)
                    t += wait
                    load += amt
                    prev = h
                    continue
                t += T[prev][s]
                s_arr += t
                if t < lo:
                    lo = t
                if t > hi:
                    hi = t
                c = sigma[s] - 1
                if not self.pair_ok[s - 1, c]:     # safety net for C4 (battery)
                    viol += BATTERY_PENALTY
                f1 += w[c] * d[c] * (t + OFF[s - 1][c])     # delivery time a_s + delta_sc
                t += W[s - 1][c]                           # truck waits for its drones
                load += self.d_cap[c]
                prev = s
            t += T[prev][vend]
            s_arr += t
            if load > self.cap[k]:
                viol += load - self.cap[k]
        spread = (hi - lo) if hi >= lo else 0.0
        return viol, (f1 / self.D, spread, s_arr)

    def item_demand(self, s: int, sigma: Dict[int, int]) -> float:
        return self.jobs[s][1] if s > self.n else self.d_cap[sigma[s] - 1]

    def loc(self, s: int, hosts: Dict[int, int]) -> int:
        """Physical location of a route item (a job sits at its host satellite)."""
        return hosts.get(s, self.default_host[s]) if s > self.n else s

    def route_load(self, r: List[int], sigma: Dict[int, int]) -> float:
        return sum(self.item_demand(s, sigma) for s in r)

    # ---------------- sigma helpers ----------------
    def assign_sigma_greedy(self, sats: List[int], centroids: List[int],
                            rng: np.random.Generator, base: Optional[Dict[int, int]] = None) -> Dict[int, int]:
        """Biased-random matching (repo's generate_drone_route: P ~ 1/t') restricted to
        battery-feasible pairs; falls back to an assignment-problem solve if the
        random pass dead-ends."""
        sigma = dict(base or {})
        left_c = [c for c in centroids if c not in sigma.values()]
        left_s = [s for s in sats if s not in sigma]
        order = list(left_s)
        rng.shuffle(order)
        ok = True
        for s in order:
            cand = [c for c in left_c if self.pair_ok[s - 1, c - 1]]
            if not cand:
                ok = False
                break
            inv = np.array([1.0 / max(self.tdm[s - 1][c - 1], 1e-9) for c in cand])
            c = cand[int(rng.choice(len(cand), p=inv / inv.sum()))]
            sigma[s] = c
            left_c.remove(c)
        if ok:
            return sigma
        # exact fallback: min total flight time subject to battery feasibility
        sigma = dict(base or {})
        cost = np.array([[self.tdm[s - 1][c - 1] if self.pair_ok[s - 1, c - 1] else BIG
                          for c in [c for c in centroids if c not in sigma.values()]] for s in left_s])
        cs = [c for c in centroids if c not in sigma.values()]
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            sigma[left_s[i]] = cs[j]
        return sigma

    def project_sigma(self, old: Dict[int, int], rng) -> Dict[int, int]:
        """Restrict an old (previous-epoch) assignment to the current free sets and
        complete it (battery-feasible) -- an elite sigma is NOT automatically valid after commitment
        because some of its centroids may now be locked to other satellites."""
        sigma, used = {}, set()
        for s in self.free:
            c = old.get(s)
            if c is not None and c in self.free_c_set and c not in used and self.pair_ok[s - 1, c - 1]:
                sigma[s] = c
                used.add(c)
        return self.assign_sigma_greedy(self.free, self.free_c, rng, base=sigma)

    def drone_ok(self, sigma: Dict[int, int]) -> bool:
        return all(self.pair_ok[s - 1, c - 1] for s, c in sigma.items())


# ============================================================
# Operator selection (DRL hook)
# ============================================================

class UniformSelector:
    name = "uniform"

    def __init__(self, rng):
        self.rng = rng

    def select(self, state=None) -> str:
        return OPERATORS[int(self.rng.integers(len(OPERATORS)))]

    def update(self, op: str, reward: float, next_state=None, done: bool = False) -> None:
        pass

    def new_epoch(self, state=None) -> None:
        pass

    def end_episode(self) -> None:
        pass


class AdaptiveSelector(UniformSelector):
    """ALNS-style adaptive roulette: score = sum of rewards / uses over a segment,
    weights smoothed with reaction factor r. Placeholder for the DRL controller:
    a DQN/PPO agent only has to implement select(state) / update(op, reward)."""
    name = "adaptive"

    def __init__(self, rng, segment: int = 100, reaction: float = 0.3, floor: float = 0.05):
        super().__init__(rng)
        self.weights = np.ones(len(OPERATORS))
        self.score = np.zeros(len(OPERATORS))
        self.uses = np.zeros(len(OPERATORS))
        self.segment, self.reaction, self.floor = segment, reaction, floor
        self.calls = 0
        self.idx = {op: i for i, op in enumerate(OPERATORS)}

    def select(self, state=None) -> str:
        p = self.weights / self.weights.sum()
        return OPERATORS[int(self.rng.choice(len(OPERATORS), p=p))]

    def update(self, op: str, reward: float, next_state=None, done: bool = False) -> None:
        i = self.idx[op]
        self.score[i] += reward
        self.uses[i] += 1
        self.calls += 1
        if self.calls % self.segment == 0:
            for j in range(len(OPERATORS)):
                if self.uses[j] > 0:
                    self.weights[j] = (1 - self.reaction) * self.weights[j] + self.reaction * self.score[j] / self.uses[j]
            self.weights = np.maximum(self.weights, self.floor * self.weights.max())
            self.score[:] = 0
            self.uses[:] = 0

    def new_epoch(self, state=None) -> None:
        # partial reset: keep what was learnt but let the new landscape re-shape it
        self.weights = 0.5 * self.weights + 0.5 * self.weights.mean()


def make_selector(name: str, rng, model_path: Optional[str] = None):
    """uniform | adaptive | dqn (trained Double-DQN policy, see drl_selector.py)."""
    if name == "dqn":
        from drl_selector import DQNSelector
        return DQNSelector.load(model_path or "models/dqn_selector.npz", rng, training=False)
    return AdaptiveSelector(rng) if name == "adaptive" else UniformSelector(rng)


# ============================================================
# Solver
# ============================================================

@dataclass
class ReoptConfig:
    time_limit: float = 5.0
    mode: str = "mo"                                   # "mo" (Pareto) or "so" (single weight)
    weights: Tuple[float, float, float] = (0.0, 0.0, 1.0)   # used when mode == "so"
    commit_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    k_candidates: int = 5
    local_search_freq: int = 30
    pr_freq: int = 10
    phase_patience: int = 400        # iterations w/o improvement before switching weight vector
    maxps: int = 20
    archive_size: int = 60
    destroy_frac: float = 0.25
    focus_mix: float = 0.5           # share of moves whose primary truck follows c_k(t)


@dataclass
class Sol:
    routes: Dict[int, List[int]]
    sigma: Dict[int, int]
    hosts: Dict[int, int] = field(default_factory=dict)     # re-delivery job -> host satellite

    def copy(self) -> "Sol":
        return Sol({k: list(v) for k, v in self.routes.items()}, dict(self.sigma), dict(self.hosts))


@dataclass
class ReoptResult:
    chosen: Sol
    chosen_objs: Tuple[float, float, float]
    chosen_viol: float
    archive: ParetoArchive
    elite: List[Dict[int, int]]
    iterations: int
    runtime_s: float
    op_weights: Optional[List[float]] = None


class DynamicReoptimizer:
    def __init__(self, prob: ReplanProblem, cfg: ReoptConfig, rng: np.random.Generator,
                 selector=None, focus: Optional[Dict[int, float]] = None):
        self.p = prob
        self.cfg = cfg
        self.rng = rng
        self.sel = selector or UniformSelector(rng)
        A = prob.active
        if focus and sum(focus.get(k, 0) for k in A) > 0:
            f = np.array([focus.get(k, 0.0) for k in A], dtype=float)
            f = f / f.sum()
            self.focus = (1 - cfg.focus_mix) / len(A) + cfg.focus_mix * f
        else:
            self.focus = np.full(len(A), 1.0 / max(1, len(A)))
        self.scale = np.ones(3)
        self.wvec = np.array(cfg.weights if cfg.mode == "so" else cfg.commit_weights, dtype=float)
        # epoch context for the DRL state (spec eq. 10)
        fv = np.array([focus.get(k, 0.0) for k in A], dtype=float) if focus else np.zeros(1)
        free_dem = sum(prob.d[c - 1] for c in prob.free_c) + sum(a for _, a in prob.jobs.values())
        cap = max(1.0, sum(prob.cap.values()))
        self.context = np.array([
            float(fv.max()) if fv.size else 0.0, float(fv.mean()) if fv.size else 0.0,
            len(prob.free) / max(1, prob.n), len(prob.free_jobs) / max(1, len(prob.free) + 1),
            min(1.0, free_dem / cap),
            float(np.mean([prob.d[c - 1] / prob.UP for c in prob.free_c])) if prob.free_c else 0.0,
            float(getattr(prob, "uncertain", 0.0)),
        ])

    def search_state(self, t_frac, since, infeasible, gap, acc_rate, imp_rate, last_op) -> np.ndarray:
        onehot = np.zeros(len(OPERATORS))
        if last_op is not None:
            onehot[last_op] = 1.0
        dyn = np.array([t_frac, min(1.0, since / max(1, self.cfg.phase_patience)), float(infeasible),
                        min(1.0, max(0.0, gap)), acc_rate, imp_rate,
                        len(self.archive) / max(1, self.archive.max_size)])
        return np.concatenate([self.context, dyn, onehot])

    # ---------- scalar ----------
    def score(self, objs) -> float:
        return float(np.dot(self.wvec, np.asarray(objs) / self.scale))

    def key(self, viol, objs):
        return (round(viol, 6), self.score(objs))

    # ---------- truck picking ----------
    def _pick(self, cond) -> Optional[int]:
        A = self.p.active
        idx = [i for i, k in enumerate(A) if cond(k)]
        if not idx:
            return None
        pr = self.focus[idx]
        pr = pr / pr.sum()
        return A[idx[int(self.rng.choice(len(idx), p=pr))]]

    def _other(self, k, cond=lambda k: True) -> Optional[int]:
        others = [j for j in self.p.active if j != k and cond(j)]
        return others[int(self.rng.integers(len(others)))] if others else None

    # ---------- construction ----------
    def _insert_cost(self, r, pos, s, k, hosts, at=None):
        """Detour of inserting item s (located at `at`) between positions pos-1 and pos."""
        T, p = self.p.T, self.p
        a = p.start[k] if pos == 0 else p.loc(r[pos - 1], hosts)
        b = p.vend if pos == len(r) else p.loc(r[pos], hosts)
        x = p.loc(s, hosts) if at is None else at
        return T[a][x] + T[x][b] - T[a][b]

    def cheapest_insert(self, sol: Sol, s: int, randomized: bool = True) -> None:
        """Repo's randomized cheapest insertion (P ~ 1/cost; zero-cost positions win),
        restricted to trucks with spare remaining capacity when possible.
        Under a degraded network the zero-detour Manhattan property disappears, so
        costs are taken from the (asymmetric) expected-time matrix.
        For a re-delivery job the host satellite is chosen jointly with the position
        (among battery-feasible hosts); a host equal to a neighbouring stop costs 0 extra
        driving, i.e. the job becomes an extra drone wave at that stop."""
        p = self.p
        ds = p.item_demand(s, sol.sigma)
        hosts_try = p.job_hosts[s] if s > p.n else [None]
        opts = []
        for k in p.active:
            r = sol.routes[k]
            fits = p.route_load(r, sol.sigma) + ds <= p.cap[k] + 1e-9
            for pos in range(len(r) + 1):
                for h in hosts_try:
                    opts.append((fits, self._insert_cost(r, pos, s, k, sol.hosts, at=h), k, pos, h))
        feas = [o for o in opts if o[0]] or opts
        costs = np.array([max(o[1], 0.0) for o in feas])
        if not randomized:
            o = feas[int(np.argmin(costs))]
        elif (costs <= 1e-9).any():
            z = np.flatnonzero(costs <= 1e-9)
            o = feas[int(self.rng.choice(z))]
        else:
            inv = 1.0 / costs
            o = feas[int(self.rng.choice(len(feas), p=inv / inv.sum()))]
        sol.routes[o[2]].insert(o[3], s)
        if o[4] is not None:
            sol.hosts[s] = o[4]

    def construct(self) -> Sol:
        p = self.p
        sigma = p.assign_sigma_greedy(p.free, p.free_c, self.rng)
        sol = Sol({k: [] for k in p.active}, sigma, {})
        todo = list(p.free)
        # seed depot-start trucks with mutually distant satellites (repo's create_init_tours)
        depot_trucks = [k for k in p.active if p.start[k] == 0]
        last = 0
        for k in depot_trucks:
            if not todo:
                break
            s = max(todo, key=lambda x: p.T[last][x])
            sol.routes[k].append(s)
            todo.remove(s)
            last = s
        self.rng.shuffle(todo)
        for s in todo + list(p.free_jobs):
            self.cheapest_insert(sol, s)
        return sol

    def _complete(self, sol: Sol) -> Sol:
        """Insert every free satellite / job that is missing from the routes."""
        placed = {s for r in sol.routes.values() for s in r}
        for s in list(self.p.free) + list(self.p.free_jobs):
            if s not in placed:
                self.cheapest_insert(sol, s, randomized=False)
        return sol

    def warm_start(self, prev_plan: Plan) -> Sol:
        p = self.p
        keep = set(p.free) | set(p.free_jobs)
        routes = {k: [s for s in prev_plan.routes.get(k, []) if s in keep] for k in p.active}
        sigma = {s: prev_plan.sigma[s] for s in p.free}
        if not (set(sigma.values()) == p.free_c_set and p.drone_ok(sigma)):
            sigma = p.project_sigma(sigma, self.rng)
        hosts = {j: prev_plan.jobs[j].host for j in p.free_jobs
                 if j in prev_plan.jobs and prev_plan.jobs[j].host in p.job_hosts[j]}
        return self._complete(Sol(routes, sigma, hosts))

    def project(self, old: Sol) -> Sol:
        p = self.p
        keep = set(p.free) | set(p.free_jobs)
        routes = {k: [s for s in old.routes.get(k, []) if s in keep] for k in p.active}
        hosts = {j: h for j, h in old.hosts.items() if j in p.jobs and h in p.job_hosts[j]}
        return self._complete(Sol(routes, p.project_sigma(old.sigma, self.rng), hosts))

    # ---------- neighbourhoods (single random move each) ----------
    def op_relocate(self, sol: Sol) -> Optional[Sol]:
        k = self._pick(lambda k: len(sol.routes[k]) >= 2)
        if k is None:
            return None
        new = sol.copy()
        r = new.routes[k]
        i = int(self.rng.integers(len(r)))
        s = r.pop(i)
        j = int(self.rng.integers(len(r) + 1))
        if j == i:
            j = (j + 1) % (len(r) + 1)
        r.insert(j, s)
        return new

    def op_swap(self, sol: Sol) -> Optional[Sol]:
        k = self._pick(lambda k: len(sol.routes[k]) >= 2)
        if k is None:
            return None
        new = sol.copy()
        r = new.routes[k]
        i = int(self.rng.integers(len(r)))
        others = [j for j in range(len(r)) if j != i]
        L = lambda x: self.p.loc(x, sol.hosts)
        inv = np.array([1.0 / max(self.p.T[L(r[i])][L(r[j])], 1e-9) for j in others])
        j = others[int(self.rng.choice(len(others), p=inv / inv.sum()))]
        r[i], r[j] = r[j], r[i]
        return new

    def op_two_opt(self, sol: Sol) -> Optional[Sol]:
        k = self._pick(lambda k: len(sol.routes[k]) >= 3)
        if k is None:
            return None
        new = sol.copy()
        r = new.routes[k]
        i, j = sorted(self.rng.choice(len(r), size=2, replace=False))
        r[i:j + 1] = r[i:j + 1][::-1]
        return new

    def op_inter_swap(self, sol: Sol) -> Optional[Sol]:
        k1 = self._pick(lambda k: len(sol.routes[k]) >= 1)
        if k1 is None:
            return None
        k2 = self._other(k1, lambda k: len(sol.routes[k]) >= 1)
        if k2 is None:
            return None
        new = sol.copy()
        r1, r2 = new.routes[k1], new.routes[k2]
        i, j = int(self.rng.integers(len(r1))), int(self.rng.integers(len(r2)))
        r1[i], r2[j] = r2[j], r1[i]
        return new

    def op_two_opt_star(self, sol: Sol) -> Optional[Sol]:
        k1 = self._pick(lambda k: True)
        k2 = self._other(k1) if k1 is not None else None
        if k2 is None:
            return None
        new = sol.copy()
        r1, r2 = new.routes[k1], new.routes[k2]
        i, j = int(self.rng.integers(len(r1) + 1)), int(self.rng.integers(len(r2) + 1))
        new.routes[k1], new.routes[k2] = r1[:i] + r2[j:], r2[:j] + r1[i:]
        return new

    def op_inter_relocate(self, sol: Sol) -> Optional[Sol]:
        k1 = self._pick(lambda k: len(sol.routes[k]) >= 1)
        if k1 is None:
            return None
        k2 = self._other(k1)
        if k2 is None:
            return None
        new = sol.copy()
        r1, r2 = new.routes[k1], new.routes[k2]
        s = r1.pop(int(self.rng.integers(len(r1))))
        r2.insert(int(self.rng.integers(len(r2) + 1)), s)
        return new

    def op_drone_swap(self, sol: Sol) -> Optional[Sol]:
        sats = list(sol.sigma.keys())
        if len(sats) < 2:
            return None
        new = sol.copy()
        s1 = sats[int(self.rng.integers(len(sats)))]
        others = [s for s in sats if s != s1]
        inv = np.array([1.0 / max(self.p.tdm[s1 - 1][s - 1], 1e-9) for s in others])
        s2 = others[int(self.rng.choice(len(others), p=inv / inv.sum()))]
        new.sigma[s1], new.sigma[s2] = new.sigma[s2], new.sigma[s1]
        if not (self.p.pair_ok[s1 - 1, new.sigma[s1] - 1] and self.p.pair_ok[s2 - 1, new.sigma[s2] - 1]):
            return None
        return new

    def op_job_host(self, sol: Sol) -> Optional[Sol]:
        """Recourse move: serve a re-delivery job from another battery-feasible satellite,
        biased toward hosts that are close to the job's route neighbours."""
        p = self.p
        jobs = [j for j in p.free_jobs if len(p.job_hosts[j]) > 1]
        if not jobs:
            return None
        j = jobs[int(self.rng.integers(len(jobs)))]
        k = next(k for k, r in sol.routes.items() if j in r)
        r = sol.routes[k]
        pos = r.index(j)
        cur = p.loc(j, sol.hosts)
        cand = [h for h in p.job_hosts[j] if h != cur]
        costs = np.array([max(self._insert_cost(r[:pos] + r[pos + 1:], pos, j, k, sol.hosts, at=h), 0.0)
                          for h in cand]) + 1e-3
        inv = 1.0 / costs
        new = sol.copy()
        new.hosts[j] = cand[int(self.rng.choice(len(cand), p=inv / inv.sum()))]
        return new

    def op_destroy_repair(self, sol: Sol) -> Optional[Sol]:
        allsats = [s for r in sol.routes.values() for s in r]
        if len(allsats) < 2:
            return None
        new = sol.copy()
        m = max(1, int(round(self.cfg.destroy_frac * len(allsats))))
        # half of the time ruin the focus segment (most disrupted route) first
        k = self._pick(lambda k: len(sol.routes[k]) >= 1)
        pool = list(new.routes[k]) if (k is not None and self.rng.random() < 0.5) else allsats
        removed = list(self.rng.choice(pool, size=min(m, len(pool)), replace=False))
        for s in removed:
            for r in new.routes.values():
                if s in r:
                    r.remove(s)
                    break
        for s in removed:
            self.cheapest_insert(new, int(s))
        return new

    def apply(self, op: str, sol: Sol) -> Optional[Sol]:
        return getattr(self, "op_" + op)(sol)

    # ---------- path relinking on sigma (repo Algorithm 5) ----------
    def path_relink(self, sol: Sol, target: Dict[int, int]):
        cur = dict(sol.sigma)
        best_key, best_sigma = None, None
        diff = [s for s in cur if cur[s] != target.get(s)]
        while diff:
            s = diff[0]
            want = target[s]
            other = next((x for x in cur if cur[x] == want), None)
            if other is None:
                break
            cur[s], cur[other] = cur[other], cur[s]
            diff = [x for x in cur if cur[x] != target.get(x)]
            if not self.p.drone_ok(cur):        # the walk may pass through infeasible states,
                continue                        # but only battery-feasible ones are evaluated
            v, o = self.p.evaluate(sol.routes, cur, sol.hosts)
            if v == 0:
                self.archive.add(o, Sol({k: list(r) for k, r in sol.routes.items()}, dict(cur), dict(sol.hosts)))
            kk = self.key(v, o)
            if best_key is None or kk < best_key:
                best_key, best_sigma = kk, dict(cur)
        return best_sigma, best_key

    # ---------- local search: best-of-K per operator ----------
    def local_search(self, sol: Sol, key):
        improved = True
        while improved:
            improved = False
            for op in OPERATORS:
                if op == "destroy_repair":
                    continue
                best = None
                for _ in range(self.cfg.k_candidates):
                    cand = self.apply(op, sol)
                    if cand is None:
                        continue
                    v, o = self.p.evaluate(cand.routes, cand.sigma, cand.hosts)
                    if v == 0:
                        self.archive.add(o, cand)
                    kk = self.key(v, o)
                    if kk < key and (best is None or kk < best[0]):
                        best = (kk, cand)
                if best is not None:
                    key, sol = best
                    improved = True
        return sol, key

    # ---------- main loop ----------
    def solve(self, init: Sol, archive_seed: Optional[List[Sol]] = None,
              elite: Optional[List[Dict[int, int]]] = None) -> ReoptResult:
        cfg, p = self.cfg, self.p
        st = time.process_time()
        self.archive = ParetoArchive(cfg.archive_size)
        elite = [p.project_sigma(e, self.rng) for e in (elite or [])]

        v0, o0 = p.evaluate(init.routes, init.sigma, init.hosts)
        self.scale = np.maximum(np.abs(np.array(o0)), 1e-6)
        if v0 == 0:
            self.archive.add(o0, init.copy())
        for s in archive_seed or []:
            v, o = p.evaluate(s.routes, s.sigma, s.hosts)
            if v == 0:
                self.archive.add(o, s)

        if cfg.mode == "so":
            schedule = [tuple(cfg.weights)]
        else:
            lattice = simplex_weights(3, 3)
            self.rng.shuffle(lattice)
            schedule = [tuple(cfg.commit_weights)] + lattice

        cur = init.copy()
        cur_key = self.key(v0, o0)
        cur, cur_key = self.local_search(cur, cur_key)
        if not elite:
            elite = [dict(cur.sigma)]

        it, since, phase = 0, 0, 0
        best_key = cur_key
        wants_state = getattr(self.sel, "name", "") == "dqn"
        op_index = {o: i for i, o in enumerate(OPERATORS)}
        acc_rate = imp_rate = 0.0
        last_op = None

        def _state():
            el = (time.process_time() - st) / max(cfg.time_limit, 1e-9)
            gap = (cur_key[1] - best_key[1]) / max(abs(best_key[1]), 1e-9) if cur_key[0] == best_key[0] else 1.0
            return self.search_state(min(1.0, el), since, cur_key[0] > 0, gap, acc_rate, imp_rate, last_op)

        hold = int(getattr(self.sel, "hold", 1)) if wants_state else 1
        seg_reward, seg_left, op = 0.0, 0, None
        while time.process_time() - st < cfg.time_limit:
            it += 1
            if wants_state:
                # DRL: one decision per segment of `hold` moves (macro action), reward summed
                if seg_left == 0:
                    op = self.sel.select(_state())
                    seg_left, seg_reward = hold, 0.0
            else:
                op = self.sel.select(None)
            cand = self.apply(op, cur)
            reward = 0.0
            if cand is not None:
                v, o = p.evaluate(cand.routes, cand.sigma, cand.hosts)
                if v == 0 and self.archive.add(o, cand):
                    reward += 1.0
                kk = self.key(v, o)
                if kk <= cur_key:
                    if kk < cur_key:
                        reward += 2.0
                    cur, cur_key = cand, kk
                if kk < best_key:
                    best_key = kk
                    reward += 5.0
                    since = 0
                else:
                    since += 1
            else:
                since += 1
            acc_rate = 0.98 * acc_rate + 0.02 * float(reward >= 2.0 or (cand is not None and kk <= cur_key))
            imp_rate = 0.98 * imp_rate + 0.02 * float(reward >= 5.0)
            last_op = op_index[op]
            if wants_state:
                seg_reward += reward
                seg_left -= 1
                if seg_left == 0 and getattr(self.sel, "training", False):
                    self.sel.update(op, seg_reward, _state(), done=time.process_time() - st >= cfg.time_limit)
                elif seg_left == 0:
                    self.sel.update(op, seg_reward)
            else:
                self.sel.update(op, reward)

            if it % cfg.local_search_freq == 0:
                cur, k2 = self.local_search(cur, cur_key)
                if k2 < best_key:
                    best_key, since = k2, 0
                cur_key = k2

            if it % cfg.pr_freq == 0 and elite:
                tgt = elite[int(self.rng.integers(len(elite)))]
                sig, kk = self.path_relink(cur, tgt)
                if sig is not None and kk < cur_key:
                    cur, cur_key = Sol(cur.routes, sig, cur.hosts), kk
                if cur_key[0] == 0 and cur.sigma not in elite:
                    elite.append(dict(cur.sigma))
                    if len(elite) > cfg.maxps:
                        del elite[0]

            # single-objective: GRASP-style restart when stuck (repo's restart_patience)
            if cfg.mode == "so" and since >= 3 * cfg.phase_patience:
                cur = self.construct()
                v, o = p.evaluate(cur.routes, cur.sigma, cur.hosts)
                cur, cur_key = self.local_search(cur, self.key(v, o))
                since = 0

            # weight-vector phase switch (spreads the Pareto archive)
            if cfg.mode == "mo" and since >= cfg.phase_patience:
                phase = (phase + 1) % len(schedule)
                self.wvec = np.array(schedule[phase])
                pick = self.archive.best_by_weights(self.wvec, self.scale)
                if pick is not None:
                    cur = pick[1].copy()
                v, o = p.evaluate(cur.routes, cur.sigma, cur.hosts)
                cur_key = best_key = self.key(v, o)
                since = 0

        self.sel.end_episode()
        # ---- choose the plan to commit ----
        if len(self.archive):
            cw = cfg.weights if cfg.mode == "so" else cfg.commit_weights
            objs, chosen = self.archive.best_by_weights(cw) if cfg.mode == "mo" else \
                min(self.archive.items, key=lambda it_: np.dot(cw, np.asarray(it_[0]) / self.scale))
            viol = 0.0
        else:
            chosen = cur
            viol, objs = p.evaluate(cur.routes, cur.sigma, cur.hosts)
        ow = list(map(float, getattr(self.sel, "weights", np.ones(len(OPERATORS)))))
        return ReoptResult(chosen.copy(), tuple(objs), viol, self.archive, elite, it,
                           time.process_time() - st, ow)


# ============================================================
# Disruption scores c_k(t) and DRL state (bai_toan_iii eqs. 9-10)
# ============================================================

def segment_disruption(prob: ReplanProblem, prev_plan: Optional[Plan], prev_T: Optional[np.ndarray],
                       prev_d: Optional[np.ndarray], omega=(0.5, 0.3, 0.2)) -> Dict[int, float]:
    """c_k(t) = norm(w1 r_road + w2 r_demand + w3 r_timing) for each active route of the
    previous plan's suffix: share of its arcs whose expected time grew, relative demand
    change of its centroids, and relative growth of its remaining duration."""
    if prev_plan is None or prev_T is None:
        return {k: 0.0 for k in prob.active}
    T, free = prob.Tnp, set(prob.free)
    raw = {}
    for k in prob.active:
        r = [s for s in prev_plan.routes.get(k, []) if s in free]
        path = [prob.start[k]] + r + [prob.vend]
        arcs = list(zip(path[:-1], path[1:]))
        r_road = np.mean([T[i, j] > prev_T[i, j] * 1.05 for i, j in arcs]) if arcs else 0.0
        old_dem = sum(prev_d[prev_plan.sigma[s] - 1] for s in r)
        new_dem = sum(prob.d[prev_plan.sigma[s] - 1] for s in r)
        r_dem = abs(new_dem - old_dem) / old_dem if old_dem > 0 else 0.0
        old_len = sum(prev_T[i, j] for i, j in arcs)
        new_len = sum(T[i, j] for i, j in arcs)
        r_time = max(0.0, (new_len - old_len) / old_len) if old_len > 0 else 0.0
        raw[k] = omega[0] * r_road + omega[1] * min(1.0, r_dem) + omega[2] * min(1.0, r_time)
    tot = sum(raw.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in raw.items()}


def drl_state(prob: ReplanProblem, c_scores: Dict[int, float], archive: Optional[ParetoArchive],
              uncertain_ratio: float, max_trucks: int) -> np.ndarray:
    """Fixed-length state s_t = [c_1..c_K (padded), |archive|/size, u_D, u_V, r_unmet, r_uncertain]."""
    c = [c_scores.get(k, 0.0) for k in prob.active][:max_trucks]
    c += [0.0] * (max_trucks - len(c))
    free_dem = sum(prob.d[c_ - 1] for c_ in prob.free_c)
    cap = sum(prob.cap.values()) or 1.0
    u_V = min(1.0, free_dem / cap)
    u_D = float(np.mean([prob.d[c_ - 1] / prob.UP for c_ in prob.free_c])) if prob.free_c else 0.0
    r_unmet = max(0.0, free_dem - cap) / max(prob.D, 1.0)
    h = len(archive) / archive.max_size if archive is not None else 0.0
    return np.array(c + [h, u_D, u_V, r_unmet, uncertain_ratio], dtype=float)
