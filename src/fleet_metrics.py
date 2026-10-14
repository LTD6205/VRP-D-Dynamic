"""
Fleet-performance metrics (paper Section 6) computed from a solved instance
and its best solution: per-satellite arrival/waiting/delivery times, per-truck
return time/load, and the derived Z1/C_ret/C_sat/C_del/Z2/Z3/Tbar_del/utilization
metrics. Entry point: scripts/compute_fleet_metrics.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

TOL = 1e-9

METRIC_COLS = ["instance", "n", "K", "fleet", "v_d", "D", "Z1", "C_ret", "C_sat", "C_del",
               "Z2_ret", "Z2_sat", "Z2_del", "Z3", "Z3_del", "Tbar_del",
               "m_min", "m_max", "util_mean", "util_max"]


def analyze_solution(inst, clustered, dist, truck_route: Dict, drone_route: Dict) -> Tuple[dict, dict, list]:
    """Recomputes every arrival/waiting/delivery/return time from a solution's
    routes and satellite->centroid assignment. Returns (sigma, satellites,
    trucks): sigma maps satellite -> centroid; satellites maps satellite ->
    a dict of its truck/position/arrival/flight/waiting/delivery/centroid/
    centroid_demand; trucks is a list of per-truck dicts (satellites visited,
    return time, load, satellite count)."""
    end = inst.n + 1
    sigma = {int(v[0]): int(v[1]) for v in drone_route.values()}
    dem = clustered.demand_n
    sats, trucks = {}, []
    for k, (_, route) in enumerate(sorted(((int(a), b) for a, b in truck_route.items())), 1):
        assert route[0] == 0 and route[-1] == end, f"malformed route {route}"
        t, pos, load, order, ret = 0.0, 0, 0.0, [], None
        for a, b in zip(route[:-1], route[1:]):
            t += dist.time_n_0[a, b]
            if b == end:
                ret = t
            else:
                pos += 1
                c = sigma[b]
                f = float(dist.time_n_m[b - 1, c - 1])
                sats[b] = dict(truck=k, position=pos, arrival=t, flight=f, waiting=2 * f, delivery=t + f,
                               centroid=c, centroid_demand=float(dem[c - 1]))
                load += dem[c - 1]
                order.append(b)
                t += 2 * f
        trucks.append(dict(truck=k, satellites=order, ret=ret, load=float(load), m=pos))
    return sigma, sats, trucks


def compute_metrics(inst, name: str, sigma: dict, sats: dict, trucks: list, D: float) -> dict:
    """Z1 = sum of arrival times at satellites and returns (the objective);
    C_ret/C_sat/C_del = completion time by return / last satellite arrival /
    last delivery; Z2_* = D / C_* (delivered demand per hour); Z3(_del) =
    spread of arrival (delivery) times; Tbar_del = demand-weighted mean
    delivery time; util_* = truck load / capacity."""
    a = np.array([s["arrival"] for s in sats.values()])
    e = np.array([s["delivery"] for s in sats.values()])
    r = np.array([t["ret"] for t in trucks])
    L = np.array([t["load"] for t in trucks])
    m = np.array([t["m"] for t in trucks])
    d_e = np.array([s["centroid_demand"] * s["delivery"] for s in sats.values()])
    C_ret, C_sat, C_del = r.max(), a.max(), e.max()
    return dict(instance=name, n=inst.n, K=inst.no_k, fleet=1 if inst.q == 12500 else 2, v_d=inst.v_d,
                D=int(round(D)), Z1=a.sum() + r.sum(), C_ret=C_ret, C_sat=C_sat, C_del=C_del,
                Z2_ret=D / C_ret, Z2_sat=D / C_sat, Z2_del=D / C_del,
                Z3=a.max() - a.min(), Z3_del=e.max() - e.min(), Tbar_del=d_e.sum() / D,
                m_min=int(m.min()), m_max=int(m.max()),
                util_mean=float((L / inst.q).mean()), util_max=float((L / inst.q).max()))


def check_solution(inst, D: float, sigma: dict, sats: dict, trucks: list) -> List[str]:
    """Structural/capacity consistency checks (empty list = all passed):
    every satellite visited exactly once, sigma a bijection, no empty truck,
    loads sum to D, no truck over capacity."""
    problems = []
    if sorted(sats) != list(range(1, inst.n + 1)):
        problems.append("a satellite is not visited exactly once")
    if sorted(sigma.keys()) != list(range(1, inst.n + 1)) or sorted(sigma.values()) != list(range(1, inst.n + 1)):
        problems.append("satellite->centroid assignment is not a bijection")
    if len(trucks) != inst.no_k or any(t["m"] < 1 for t in trucks):
        problems.append(f"{len(trucks)} routes / an empty truck (K={inst.no_k})")
    if abs(sum(t["load"] for t in trucks) - D) > 1e-6:
        problems.append("sum of truck loads does not equal total demand D")
    if any(t["load"] > inst.q + 1e-9 for t in trucks):
        problems.append("truck capacity violated")
    return problems
