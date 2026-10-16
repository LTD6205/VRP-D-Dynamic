#!/usr/bin/env python3
"""
Generate companion `<instance>.events.json` files for the dynamic 2E-CTVRP.

Based on the blueprint in dynamic_2e_ctvrp_research_guide_dataset_blueprint.md
(Sec. 5.3), with these corrections:

1. Trigger times are instance-relative. The guide's fixed 0.40 h / 0.85 h fall
   before the first satellite arrival on most instances (the depot sits ~20 km
   outside the ring, i.e. >0.5 h of driving), so nothing would be committed yet.
   Here t1 / t2 are the 30% / 65% quantiles of satellite arrival times of a
   deterministic baseline plan, so each event really splits served vs. unserved.
2. Damage is spatially clustered (bai_toan_iii Sec. 8.2 "theo cum khong gian"):
   a flood centre is placed on an inner-ring satellite and arcs are damaged in
   order of proximity to it; epoch 2 extends the flooded area (progression) and
   escalates part of epoch-1 damage. The 10/30/50% ratio is over ALL first-echelon
   arcs (depot->sat, sat->sat, sat->depot), as the spec states.
3. 30% of damaged arcs are directional (one-way), so t_ij != t_ji is exercised.
4. Return-to-depot arcs are encoded as (i, 0).
5. Demand surges hit the clusters closest to the flood (evacuation shelters),
   not random ones, and come with a priority raise.
6. Drone battery: a speed- and load-dependent energy model (src/drone_energy.py,
   block `drone_energy`). If no battery-feasible satellite<->centroid matching
   exists even at 1.3x demand, the battery capacity is raised until one does
   (logged in _meta.battery_note).
7. An extra partial-observability scenario (`damage_30pct_partial`, reveal_prob
   0.6) is emitted for the POMDP experiments.

Usage:
    python3 scripts/generate_dynamic_dataset.py                # the 12-instance subset
    python3 scripts/generate_dynamic_dataset.py --all-main     # every instance in instances/main
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from common import cluster_victims, compute_distances, parse_instance  # noqa: E402
from dynamic_env import DEFAULT_TRANSITION  # noqa: E402
from dynamic_sim import Simulator  # noqa: E402
from dynamic_env import DemandProcess, RoadNetwork  # noqa: E402
from dynamic_solver import DynamicReoptimizer, ReoptConfig, ReplanProblem  # noqa: E402
from dynamic_runner import merge  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from drone_energy import DroneBattery, DroneEnergyModel  # noqa: E402

SUBSET = [
    ("main", "M-08-05-60-1"), ("main", "M-08-05-60-2"),
    ("main", "M-09-09-60-1"), ("main", "M-09-09-60-2"),
    ("main", "M-12-11-60-1"), ("main", "M-12-11-60-2"),
    ("main", "M-15-13-60-1"), ("main", "M-15-13-60-2"),
    ("large", "M-20-20-60-1"), ("large", "M-20-20-60-2"),
    ("large", "M-50-50-60-1"), ("large", "M-50-50-60-2"),
]


def has_matching(ok: np.ndarray) -> bool:
    cost = np.where(ok, 0.0, 1.0)
    r, c = linear_sum_assignment(cost)
    return cost[r, c].sum() == 0


def calibrate_battery(inst, cl, dist, surge_margin: float = 1.3):
    """Default speed+load battery model; if no battery-feasible satellite<->centroid
    matching exists even with every demand inflated by `surge_margin`, raise the
    battery capacity (bisection) until one does."""
    model = DroneEnergyModel()
    dem = np.array(cl.demand_n) * surge_margin
    def ok_with(cap):
        m = DroneEnergyModel(**{**model.to_dict(), "battery": cap})
        return has_matching(DroneBattery(m, inst.v_d, inst.no_u, inst.p).pair_ok(dist.time_n_m, dem))
    if ok_with(model.battery):
        frac = float((~DroneBattery(model, inst.v_d, inst.no_u, inst.p).pair_ok(dist.time_n_m, np.array(cl.demand_n))).mean())
        return model, f"default battery; {100 * frac:.0f}% of satellite-centroid pairs battery-infeasible at base demand"
    lo, hi = model.battery, model.battery * 4
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok_with(mid) else (mid, hi)
    model.battery = round(hi * 1.02, 4)
    return model, f"battery raised to {model.battery} so a feasible matching exists at {surge_margin}x demand"


def baseline_arrivals(inst, cl, dist, seed, energy):
    """Deterministic baseline plan (construction + Z1 local search) -> arrival times."""
    n, Q, UP = inst.n, float(inst.q), float(inst.no_u * inst.p)
    net = RoadNetwork(dist.time_n_0)
    dem = DemandProcess(cl.demand_n, {})
    battery = DroneBattery(energy, inst.v_d, inst.no_u, inst.p)
    sim = Simulator(n, Q, UP, dist.time_n_m, net, dem)
    snap = sim.initial_snapshot(list(range(inst.no_k)))
    prob = ReplanProblem(snap, net.expected_time(), dist.time_n_m, dem.current_demand,
                         dem.current_priority, Q, UP, battery, n)
    rng = np.random.default_rng(seed)
    cfg = ReoptConfig(mode="so", weights=(0, 0, 1))
    s = DynamicReoptimizer(prob, cfg, rng)
    from mo_utils import ParetoArchive
    s.archive = ParetoArchive()
    sol = s.construct()
    v, o = prob.evaluate(sol.routes, sol.sigma)
    s.scale = np.maximum(np.array(o), 1e-6)
    sol, _ = s.local_search(sol, s.key(v, o))
    tl = sim.simulate(merge(snap, sol))
    return tl


def generate(inst_path: Path, out_dir: Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    inst = parse_instance(str(inst_path))
    cl = cluster_victims(inst, random_state=0)
    dist = compute_distances(inst, cl)
    n = inst.n
    xy = dist.data_v[1:-1]                      # satellite coordinates
    centre = xy.mean(axis=0)
    radius = np.linalg.norm(xy - centre, axis=1)

    energy, battery_note = calibrate_battery(inst, cl, dist)
    tl = baseline_arrivals(inst, cl, dist, seed, energy)
    arr = np.array(sorted(r.arrival for r in tl.sats.values()))
    t1 = round(float(np.quantile(arr, 0.30)), 3)
    t2 = round(float(np.quantile(arr, 0.65)), 3)
    if t2 <= t1:
        t2 = round(t1 + 0.05, 3)

    priorities = {str(j): round(float(rng.uniform(0.50, 0.85)), 2) for j in range(1, n + 1)}

    # all first-echelon arcs in events-file indexing (v == 0 means back to depot)
    arcs = [(0, j) for j in range(1, n + 1)] + [(i, 0) for i in range(1, n + 1)] + \
           [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
    depot = np.array([inst.x0, inst.y0])

    def pos(u):
        return depot if u == 0 else xy[u - 1]

    scenarios = {}
    variants = [("damage_10pct", 0.10, 1.0), ("damage_30pct", 0.30, 1.0), ("damage_50pct", 0.50, 1.0),
                ("damage_30pct_partial", 0.30, 0.6)]
    # satellites / centroids still unserved at t1 / t2 in the baseline plan: the flood
    # must hit the part of the network that is still ahead of the trucks, and surges
    # must hit clusters that are not served yet (guide Sec. 5.2), otherwise the event
    # changes nothing that can be re-planned.
    unserved_t1 = [s - 1 for s, rec in tl.sats.items() if rec.arrival > t1]
    unserved_c = {t: {rec.centroid for rec in tl.sats.values() if rec.arrival > t} for t in (t1, t2)}
    inner_unserved = [i for i in unserved_t1 if radius[i] <= np.median(radius)] or unserved_t1
    inner = inner_unserved or list(range(n))
    for name, ratio, reveal in variants:
        r = np.random.default_rng(seed + int(ratio * 100) + (7 if reveal < 1 else 0))
        flood_sat = int(r.choice(inner))
        fc = xy[flood_sat]
        # arc proximity = distance of the arc's closer endpoint (satellite) to the flood centre
        prox = np.array([min(np.linalg.norm(pos(u) - fc) if u else np.inf,
                             np.linalg.norm(pos(v) - fc) if v else np.inf) for u, v in arcs])
        prox += r.uniform(0, 1e-3, size=len(prox)) * prox.max()   # tie-break noise
        order = np.argsort(prox)
        n_total = int(round(ratio * len(arcs)))
        n_ep1 = max(1, n_total // 2)
        ep1_idx, ep2_idx = order[:n_ep1], order[n_ep1:n_total]
        dmax = prox[order[max(n_total - 1, 0)]] or 1.0

        def make_updates(idx, p_detour_base, already=None):
            ups, seen = [], set()
            for a in idx:
                u, v = arcs[a]
                if (u, v) in seen:
                    continue
                closeness = 1 - prox[a] / dmax
                st = "closed_detour" if r.random() < p_detour_base + 0.4 * closeness else "degraded"
                ups.append(dict(u=u, v=v, state=st))
                seen.add((u, v))
                # 70% of damage is bidirectional, 30% directional (one-way lanes)
                rev = (v, u)
                if u != 0 and v != 0 and r.random() < 0.7 and rev not in seen:
                    ups.append(dict(u=v, v=u, state=st))
                    seen.add(rev)
            return ups

        ep1 = make_updates(ep1_idx, 0.25)
        # epoch 2: flood spreads + 40% of epoch-1 'degraded' arcs escalate to detour
        ep2 = make_updates(ep2_idx, 0.15)
        for e in ep1:
            if e["state"] == "degraded" and r.random() < 0.4:
                ep2.append(dict(u=e["u"], v=e["v"], state="closed_detour"))

        # demand surges: clusters whose centroid is closest to the flood
        cdist = np.linalg.norm(cl.centers - fc, axis=1)
        k_s = max(1, int(round(0.15 * n)))
        near = [int(j) + 1 for j in np.argsort(cdist)]
        n1 = [c for c in near if c in unserved_c[t1]] or near
        s1 = n1[:k_s]
        n2 = [c for c in near if c in unserved_c[t2] and c not in s1] or n1
        s2 = n2[:k_s]
        scenarios[name] = dict(
            base_instance=inst_path.name, num_satellites=n,
            drone_energy=energy.to_dict(), reveal_prob=reveal,
            markov_transition=DEFAULT_TRANSITION.tolist(),
            flood_centre_satellite=flood_sat + 1,
            initial_priorities=priorities,
            epochs=[
                dict(epoch_id=0, trigger_time_h=0.0, description="Initial dispatch from depot",
                     edge_updates=[], demand_multipliers={}, priority_updates={}),
                dict(epoch_id=1, trigger_time_h=t1,
                     description="Flood onset around the flood centre; nearby shelters surge",
                     edge_updates=ep1,
                     demand_multipliers={str(c): round(float(r.uniform(1.10, 1.20)), 2) for c in s1},
                     priority_updates={str(c): 1.0 for c in s1}),
                dict(epoch_id=2, trigger_time_h=t2,
                     description="Flood progression + escalation; newly reported shelter demand",
                     edge_updates=ep2,
                     demand_multipliers={str(c): round(float(r.uniform(1.15, 1.30)), 2) for c in s2},
                     priority_updates={str(c): 0.95 for c in s2}),
            ],
        )
    scenarios["_meta"] = dict(seed=seed, baseline_arrival_quantiles=[t1, t2],
                              baseline_first_last_arrival=[float(arr[0]), float(arr[-1])],
                              battery_note=battery_note, n_arcs=len(arcs))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{inst_path.stem}.events.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-main", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="instances/dynamic_events")
    ap.add_argument("--instances", default=None,
                    help="comma list of sub/name, e.g. main/M-10-07-60-1,large/M-20-20-50-1 (overrides the subset)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)  # frozen clusters are looked up relative to the repo root
    targets = SUBSET
    if args.instances:
        targets = [tuple(x.split("/")) for x in args.instances.split(",")]
    elif args.all_main:
        targets = [("main", p.stem) for p in sorted((root / "instances/main").glob("*.txt"))]
    for sub, stem in targets:
        p = root / "instances" / sub / f"{stem}.txt"
        if p.exists():
            print("Generated:", generate(p, root / args.out, args.seed))


if __name__ == "__main__":
    main()
