"""Checks for the second-echelon drone service models (run: python3 tests/test_drone_service.py)."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from dynamic_runner import load_world  # noqa: E402
from drone_energy import DroneBattery  # noqa: E402
from drone_service import KM_PER_UNIT, CentroidService, RouteService, SetupTimes  # noqa: E402


def brute_energy(rs, sat_km, pts_km, loads, tau):
    """Leg-by-leg energy of one trip: take off with the whole load, drop at every victim."""
    v, A, B, P = rs.v, rs.A, rs.B, rs.P
    L = float(sum(loads))
    e, pos = 0.0, sat_km
    for p, q in zip(pts_km, loads):
        e += np.linalg.norm(p - pos) / v * (A + B * L / P)    # fly to the victim with load L
        e += tau * (A + B * L / P)                          # hover while dropping
        L -= q
        pos = p
    e += np.linalg.norm(sat_km - pos) / v * A               # empty return
    return e


def check_instance(name, sub, events):
    w = load_world(f"instances/{sub}/{name}.txt", f"{events}/{name}.events.json", "damage_30pct")
    inst, cl = w.inst, w.clustered
    rs = RouteService(inst, cl, w.scenario.energy, SetupTimes.from_minutes(1, 2, 0.1), cache_dir=None)
    vic = np.column_stack([inst.x_m, inst.y_m]) * KM_PER_UNIT
    rng = np.random.default_rng(0)
    for _ in range(6):
        s, c = int(rng.integers(1, inst.n + 1)), int(rng.integers(1, inst.n + 1))
        for scale in (1.0, 1.3):
            d = cl.demand_n[c - 1] * scale
            res, trips = rs.plan(s, c, d)
            served = sorted(v for t in trips for v in t["victims"])
            need = sorted(int(i) for i in np.flatnonzero((np.asarray(cl.label) == c - 1) & (inst.demand_m > 0)))
            assert sorted(set(served)) == need, "every victim with demand must be served"
            for t in trips:
                assert t["load"] <= rs.P + 1e-6, "drone payload exceeded"
                q = inst.demand_m[t["victims"]] * scale
                # victims split into several loads appear more than once with equal shares
                if len(set(t["victims"])) == len(t["victims"]):
                    e = brute_energy(rs, rs.sat[s - 1], vic[t["victims"]], q, rs.setup.drop)
                    assert abs(e / rs.usable - t["energy"]) < 1e-9, (e / rs.usable, t["energy"])
            if res.use <= 1:
                assert all(t["energy"] <= 1 + 1e-9 for t in trips)
            # makespan = max over drones of the sum of their trip busy times
            per = {}
            for t in trips:
                per[t["drone"]] = per.get(t["drone"], 0.0) + t["busy"]
            assert abs(max(per.values()) - res.wait) < 1e-9
    print(f"ok  {name}: route plans cover all victims, respect P, energy matches leg-by-leg physics")


def check_centroid_identity(name, sub, events):
    """With zero setup times the centroid service reproduces the paper's model exactly."""
    w = load_world(f"instances/{sub}/{name}.txt", f"{events}/{name}.events.json", "damage_30pct")
    inst, cl, dist = w.inst, w.clustered, w.dist
    bat = DroneBattery(w.scenario.energy, inst.v_d, inst.no_u, inst.p)
    UP = inst.no_u * inst.p
    cs = CentroidService(dist.time_n_m, bat, UP)
    d = np.array(cl.demand_n)
    W, OFF, USE, OK = cs.tables(d, d)
    waves = np.maximum(1, np.ceil(d / UP - 1e-9))
    assert np.array_equal(W, waves[None, :] * 2 * dist.time_n_m)
    assert np.array_equal(OFF, dist.time_n_m)
    assert np.array_equal(OK, bat.pair_ok(dist.time_n_m, d))
    print(f"ok  {name}: centroid service == paper model (dwell = waves * 2t', offset = t')")


if __name__ == "__main__":
    check_centroid_identity("M-09-09-60-1", "main", "instances/dynamic_events")
    check_instance("M-09-09-60-1", "main", "instances/dynamic_events")
    check_instance("S-R-12-1", "solomon", "instances/solomon_events")
    check_instance("S-C1-08-2", "solomon", "instances/solomon_events")
