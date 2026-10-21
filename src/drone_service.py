"""
Second-echelon (satellite -> victims) drone service models.

The truck parks at satellite s, its U drones serve cluster c, and the truck waits
until the last drone is back. Everything the first echelon needs from the drones
is captured by one call

    service(s, c, d) -> DroneServiceResult(wait, offset, use, trips)

  wait    W_sc(d)   truck dwell time at s (the drones' makespan, incl. setup times)
  offset  delta_sc(d)  delivery time measured from the truck's arrival a_s, used in
                    f1 = (1/D) sum w_c d_c (a_s + delta_sc)
  use     u_sc(d)   battery share of the most demanding trip (<= 1 is feasible)
  trips             number of drone trips (waves for the centroid model)

Two implementations share this interface, so the simulator and the re-optimizer do
not care which one is used:

CentroidService  (the paper's model, the default)
    All of cluster c's demand is delivered at its CENTROID. A wave sends the U drones
    out and back together, waves = ceil(d / (U P)):
        W = waves * (S_L + 2 t'_sc + S_R),  delta = S_L + t'_sc
    Battery: one out-(loaded)-and-back-(empty) sortie with load d / (U waves).
    With S_L = S_R = 0 this is exactly the model used before this file existed.

RouteService  (victim-level drone routing = a multi-trip CVRP per cluster)
    Every victim v of cluster c is visited. Route-first / cluster-second:
      1. a giant TSP tour over the cluster's victims (nearest neighbour + 2-opt),
         computed once per cluster and rotated to start at the victim nearest to s;
      2. an exact SPLIT (Prins 2004) of the tour into trips, minimizing total drone
         busy time, where a trip i..j is allowed iff
             load    sum_{k=i..j} q_k <= P
             energy  e_ij <= E (1 - reserve)
         and the energy follows the speed- and load-dependent power model of
         drone_energy.py LEG BY LEG: the drone takes off with the whole trip load and
         gets lighter at every drop (formula in `_trip_tables`);
      3. the trips are scheduled on the U drones (longest-processing-time first to the
         least busy drone, shortest-first inside each drone, so early deliveries are not
         delayed by long trips); the makespan is the truck's dwell W.
    delta = demand-weighted mean delivery time of the victims after the truck's arrival.
    A demand change d scales every victim's demand by d / d_c(0) (surges hit the whole
    shelter area); a re-delivery job of amount a is a delivery of a / d_c(0) of each
    victim's original demand. A victim heavier than P is split into several full loads.
    A pair (s, c) is battery-infeasible iff some victim cannot be served even alone.

Units: coordinates of the instance files are in tens of metres (x 10 / 1000 = km),
speeds in km/h, times in hours -- the same convention as common.compute_distances.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from drone_energy import DroneEnergyModel, FixedEndurance

KM_PER_UNIT = 10.0 / 1000.0


@dataclass
class DroneServiceResult:
    wait: float
    offset: float
    use: float
    trips: int


@dataclass
class SetupTimes:
    """Launch / recovery setup per drone trip (Kuo et al. 2022, S_L and S_R), hours.
    Recovery includes the battery swap before the next trip."""
    launch: float = 0.0
    recover: float = 0.0
    drop: float = 0.0          # hover time per victim drop (route model only)

    # defaults used by the route model / the "setup" experiments (minutes):
    # 1 min launch (load + take-off clearance), 2 min recovery (landing + battery swap),
    # 6 s hover per drop (parcel released on a tether / air-drop)
    DEFAULT_MIN = (1.0, 2.0, 0.1)

    @classmethod
    def from_minutes(cls, launch=0.0, recover=0.0, drop=0.0) -> "SetupTimes":
        return cls(launch / 60.0, recover / 60.0, drop / 60.0)


class _Base:
    name = "base"

    def tables(self, demand_mean, demand_cap):
        """(W, OFF, USE_cap, OK_cap) as (n x n) arrays: dwell and delivery offset at the
        planned (mean) demand, battery use and feasibility at the robust demand d_cap."""
        n = self.n
        W, OFF, USE = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for s in range(1, n + 1):
            for c in range(1, n + 1):
                r = self.service(s, c, float(demand_mean[c - 1]), split=False)
                W[s - 1, c - 1], OFF[s - 1, c - 1] = r.wait, r.offset
                if demand_cap[c - 1] != demand_mean[c - 1]:
                    r = self.service(s, c, float(demand_cap[c - 1]), split=False)
                USE[s - 1, c - 1] = r.use
        return W, OFF, USE, USE <= 1.0 + 1e-12

    def ok(self, s: int, c: int, d: float) -> bool:
        return self.service(s, c, d, split=False).use <= 1.0 + 1e-12


class CentroidService(_Base):
    name = "centroid"

    def __init__(self, tdm: np.ndarray, battery, UP: float, setup: Optional[SetupTimes] = None):
        self.tdm = np.asarray(tdm, dtype=float)
        self.n = self.tdm.shape[0]
        self.battery = FixedEndurance(battery) if isinstance(battery, (int, float)) else battery
        self.UP = float(UP)
        self.setup = setup or SetupTimes()

    def _waves(self, d: float) -> int:
        return max(1, int(math.ceil(d / self.UP - 1e-9)))

    def _use(self, fly: float, d: float, nw: int) -> float:
        b = self.battery
        if b is None:
            return 0.0
        if isinstance(b, FixedEndurance):
            return 2 * fly / b.t_max if math.isfinite(b.t_max) else 0.0
        lf = min(1.0, d / (b.U * nw * b.P))
        return float(b.model.sortie_energy(fly, b.v_d, lf)) / b.model.usable

    def service(self, s: int, c: int, d: float, split: bool = True) -> DroneServiceResult:
        """split=True (execution): if the per-drone load would exceed the battery, the
        operators fly extra, lighter waves instead of over-flying it."""
        fly = float(self.tdm[s - 1, c - 1])
        nw = self._waves(d)
        use = self._use(fly, d, nw)
        while split and use > 1.0 + 1e-12 and nw < 100:
            nw += 1
            use = self._use(fly, d, nw)
        st = self.setup
        return DroneServiceResult(nw * (st.launch + 2 * fly + st.recover), st.launch + fly, use, nw)

    def tables(self, demand_mean, demand_cap):
        # vectorised, bit-identical to the pre-service code when setup times are 0
        dm, dc = np.asarray(demand_mean, float), np.asarray(demand_cap, float)
        waves = np.maximum(1, np.ceil(dm / self.UP - 1e-9))
        st = self.setup
        W = waves[None, :] * (st.launch + 2 * self.tdm + st.recover)
        OFF = st.launch + self.tdm
        if self.battery is None:
            USE = np.zeros_like(self.tdm)
        else:
            USE = self.battery.energy_matrix(self.tdm, dc)
            if not isinstance(self.battery, FixedEndurance):
                USE = USE / self.battery.model.usable
        OK = self.battery.pair_ok(self.tdm, dc) if self.battery is not None else np.ones_like(self.tdm, bool)
        return W, OFF, USE, OK


class RouteService(_Base):
    name = "route"

    def __init__(self, inst, clustered, energy: DroneEnergyModel, setup: Optional[SetupTimes] = None,
                 cache_dir: Optional[str] = None):
        self.n = inst.n
        self.U, self.P, self.v = int(inst.no_u), float(inst.p), float(inst.v_d)
        self.model = energy
        self.setup = setup or SetupTimes()
        self.usable = energy.usable
        a = energy.weight_share / energy.t_empty_ref
        b = (1.0 - energy.weight_share) / energy.t_empty_ref
        # power p(l) = A + B * l / P  (fraction of battery per hour), linear in the load
        self.A = a + b * (self.v / energy.v_ref) ** 3 + energy.k_wind * energy.wind ** 2
        self.B = a * energy.beta_payload
        sat = np.column_stack([inst.x_n, inst.y_n]) * KM_PER_UNIT
        vic = np.column_stack([inst.x_m, inst.y_m]) * KM_PER_UNIT
        lab = np.asarray(clustered.label)
        self.sat = sat
        dem = np.asarray(inst.demand_m, float)
        # victims with zero demand need no visit
        self.members = [np.flatnonzero((lab == c) & (dem > 0)) for c in range(self.n)]
        self.vxy = [vic[m] for m in self.members]
        self.q0 = [np.asarray(inst.demand_m, float)[m] for m in self.members]
        self.d0 = np.array([q.sum() for q in self.q0])
        self._cache: Dict[Tuple[int, int, float], DroneServiceResult] = {}
        key = hashlib.md5(json.dumps([inst.name, self.U, self.P, self.v, energy.to_dict(),
                                      vars(self.setup), int(len(lab))], sort_keys=True).encode()).hexdigest()[:12]
        self._disk = os.path.join(cache_dir, f"{inst.name}-{key}.pkl") if cache_dir else None
        self.tours = self._load_or_build_tours()

    # ---------------- giant tours ----------------
    def _load_or_build_tours(self):
        if self._disk and os.path.exists(self._disk):
            with open(self._disk, "rb") as f:
                data = pickle.load(f)
            self._cache.update(data.get("cache", {}))
            return data["tours"]
        tours = [_tsp_tour(xy) for xy in self.vxy]
        self.tours = tours
        return tours

    def save_cache(self):
        if not self._disk:
            return
        os.makedirs(os.path.dirname(self._disk), exist_ok=True)
        base = {k: v for k, v in self._cache.items() if k[2] == 1.0}
        with open(self._disk, "wb") as f:
            pickle.dump(dict(tours=self.tours, cache=base), f)

    # ---------------- one (satellite, cluster, demand) service ----------------
    def service(self, s: int, c: int, d: float, split: bool = True) -> DroneServiceResult:
        scale = round(float(d) / self.d0[c - 1], 6) if self.d0[c - 1] > 0 else 0.0
        key = (s, c, scale)
        r = self._cache.get(key)
        if r is None:
            r = self._solve(s - 1, c - 1, scale)
            self._cache[key] = r
        return r

    def plan(self, s: int, c: int, d: float):
        """Full drone plan for inspection: list of (drone, start, [victim ids], load, energy share)."""
        scale = float(d) / self.d0[c - 1] if self.d0[c - 1] > 0 else 0.0
        return self._solve(s - 1, c - 1, scale, detail=True)

    def _solve(self, si: int, ci: int, scale: float, detail: bool = False):
        st = self.setup
        if scale <= 0 or len(self.tours[ci]) == 0:
            res = DroneServiceResult(0.0, 0.0, 0.0, 0)
            return (res, []) if detail else res
        order = self.tours[ci]
        xy = self.vxy[ci][order]
        dist_s = np.linalg.norm(xy - self.sat[si], axis=1)
        r0 = int(np.argmin(dist_s))                      # rotate: start next to the satellite
        idx = np.r_[r0:len(order), 0:r0]
        xy, dist_s = xy[idx], dist_s[idx]
        q = self.q0[ci][order][idx] * scale
        vid = self.members[ci][order][idx]
        # victims heavier than one drone load are split into full loads at the same point
        if (q > self.P + 1e-9).any():
            reps = np.ceil(q / self.P - 1e-9).astype(int)
            xy, dist_s, vid = np.repeat(xy, reps, 0), np.repeat(dist_s, reps), np.repeat(vid, reps)
            q = np.concatenate([[qq / r] * r for qq, r in zip(q, reps)])
        m = len(q)
        a = dist_s / self.v                                            # satellite <-> victim, h
        g = np.linalg.norm(np.diff(xy, axis=0), axis=1) / self.v if m > 1 else np.zeros(0)
        D = np.r_[0.0, np.cumsum(q)]                                   # D[k] = load of victims < k
        G = np.r_[0.0, np.cumsum(g)]                                   # G[k] = tour time v_0 -> v_k
        H = np.r_[0.0, np.cumsum(g * D[1:-1])] if m > 1 else np.zeros(1)  # H[k] = sum_{l<k} g_l D[l+1]
        E = np.r_[0.0, np.cumsum(D[:-1])]                              # E[k] = sum_{l<k} D[l]
        A, B, P, tau, usable = self.A, self.B, self.P, st.drop, self.usable

        # Split DP: V[j+1] = min_i V[i] + busy(i, j)
        V = np.full(m + 1, np.inf)
        V[0] = 0.0
        pred = np.zeros(m + 1, dtype=int)
        for i in range(m):
            if not math.isfinite(V[i]):
                continue
            jmax = max(i + 1, int(np.searchsorted(D, D[i] + P + 1e-9, side="right")) - 1)
            j = np.arange(i, min(m, jmax))
            L = D[j + 1] - D[i]
            nv = j - i + 1
            fly = a[i] + (G[j] - G[i]) + a[j]
            busy = st.launch + fly + nv * tau + st.recover
            energy = (a[i] * (A + B * L / P)                                     # take-off leg, full load
                      + A * (G[j] - G[i]) + B / P * (D[j + 1] * (G[j] - G[i]) - (H[j] - H[i]))  # inner legs
                      + a[j] * A                                                 # empty return
                      + tau * (nv * A + B / P * (nv * D[j + 1] - (E[j + 1] - E[i]))))  # hovering at drops
            ok = (L <= P + 1e-9) & (energy <= usable + 1e-12)
            ok[0] = True                     # a single victim is always tried (flags infeasible pairs)
            cand = V[i] + np.where(ok, busy, np.inf)
            better = cand < V[j + 1]
            V[j + 1] = np.where(better, cand, V[j + 1])
            pred[j + 1] = np.where(better, i, pred[j + 1])
        trips = []
        j = m
        while j > 0:
            i = pred[j]
            trips.append((i, j - 1))
            j = i
        trips.reverse()

        # trip data
        info = []
        for (i, j) in trips:
            L = D[j + 1] - D[i]
            nv = j - i + 1
            fly = a[i] + (G[j] - G[i]) + a[j]
            busy = st.launch + fly + nv * tau + st.recover
            energy = (a[i] * (A + B * L / P) + A * (G[j] - G[i])
                      + B / P * (D[j + 1] * (G[j] - G[i]) - (H[j] - H[i])) + a[j] * A
                      + tau * (nv * A + B / P * (nv * D[j + 1] - (E[j + 1] - E[i]))))
            # time from take-off to each drop: a_i + G[k] - G[i] + (k - i) tau
            k = np.arange(i, j + 1)
            t_drop = st.launch + a[i] + (G[k] - G[i]) + (k - i) * tau
            info.append(dict(i=i, j=j, busy=busy, energy=energy, load=L, t_drop=t_drop, q=q[i:j + 1],
                             vid=vid[i:j + 1]))
        # schedule: LPT assignment to U drones, shortest trip first inside each drone
        U = self.U
        drones = [[] for _ in range(U)]
        tot = np.zeros(U)
        for t in sorted(info, key=lambda x: -x["busy"]):
            u = int(np.argmin(tot))
            drones[u].append(t)
            tot[u] += t["busy"]
        num, den, sched = 0.0, 0.0, []
        for u in range(U):
            clock = 0.0
            for t in sorted(drones[u], key=lambda x: x["busy"]):
                num += float(((clock + t["t_drop"]) * t["q"]).sum())
                den += float(t["q"].sum())
                sched.append((u, clock, t))
                clock += t["busy"]
        use = max(t["energy"] for t in info) / usable
        res = DroneServiceResult(float(tot.max()), num / den if den > 0 else 0.0, float(use), len(info))
        if detail:
            return res, [dict(drone=u, start=float(cl), victims=[int(x) for x in t["vid"]],
                              load=float(t["load"]), energy=float(t["energy"] / usable), busy=float(t["busy"]))
                         for (u, cl, t) in sched]
        return res


def _tsp_tour(xy: np.ndarray) -> np.ndarray:
    """Nearest-neighbour tour + 2-opt (open path improvements on the closed tour)."""
    m = len(xy)
    if m <= 2:
        return np.arange(m)
    Dm = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    start = int(np.argmin(xy[:, 0] + xy[:, 1]))
    tour = [start]
    left = set(range(m)) - {start}
    while left:
        last = tour[-1]
        nxt = min(left, key=lambda j: Dm[last, j])
        tour.append(nxt)
        left.remove(nxt)
    tour = np.array(tour)
    improved = True
    it = 0
    while improved and it < 50:
        improved = False
        it += 1
        for i in range(1, m - 1):
            a, b = tour[i - 1], tour[i]
            c = tour[i + 1:]
            d = np.r_[tour[i + 2:], tour[0]]
            delta = Dm[a, c] + Dm[b, d] - Dm[a, b] - Dm[c, d]
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                j = i + 1 + k
                tour[i:j + 1] = tour[i:j + 1][::-1]
                improved = True
    return tour


def make_service(kind: str, inst, clustered, dist, battery, UP: float, energy: DroneEnergyModel,
                 setup: Optional[SetupTimes] = None, cache_dir: Optional[str] = "instances/.drone_cache"):
    """kind = 'centroid' (paper model) | 'route' (victim-level drone CVRP)."""
    if kind == "centroid":
        return CentroidService(dist.time_n_m, battery, UP, setup)
    if kind == "route":
        return RouteService(inst, clustered, energy, setup, cache_dir)
    raise ValueError(f"unknown drone model {kind!r}")
