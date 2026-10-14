"""
Closed-loop rolling-horizon driver: plan at t=0, execute, and at every event epoch
freeze history -> update road/demand/priority state -> re-optimize the residual
problem -> merge, until the last event; then execute the final plan to the end.

Strategies
  static   : solve once at t=0 and never re-plan (executes the stale plan under the
             real, disrupted network) -- the "static truck-drone heuristic" baseline.
  restart  : re-solve every epoch from a fresh randomized construction, no memory.
  warm     : warm start from the previous plan's suffix + projected Pareto archive +
             elite sigma pool + disruption-focused (segment-aware) operator moves.

All strategies of one run share the SAME epoch-0 plan (solved once), so differences
come only from how they react to events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from common import cluster_victims, compute_distances, parse_instance
from dynamic_env import DemandProcess, RoadNetwork, Scenario, load_scenarios
from dynamic_sim import Plan, Simulator, realized_metrics
from dynamic_solver import (DynamicReoptimizer, ReoptConfig, ReplanProblem, Sol, drl_state,
                            make_selector, segment_disruption)
from mo_utils import hypervolume, nondominated


def merge(snap, chosen: Sol) -> Plan:
    routes = {k: list(ts.prefix) + (list(chosen.routes.get(k, [])) if ts.active else [])
              for k, ts in snap.trucks.items()}
    sigma = {s: r.centroid for s, r in snap.locked.items()}
    sigma.update(chosen.sigma)
    return Plan(routes, sigma)


def paired_hv(F_a: np.ndarray, F_b: np.ndarray):
    """HV of two archives on the SAME residual problem, normalized by their union."""
    U = np.vstack([F_a, F_b])
    lo, hi = U.min(axis=0), U.max(axis=0)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    ref = np.full(3, 1.1)
    return (hypervolume((F_a - lo) / rng, ref), hypervolume((F_b - lo) / rng, ref))


def check_plan(plan: Plan, n: int, snap, sim: Simulator) -> None:
    """Invariants after a re-plan: every satellite visited exactly once, sigma a
    bijection, and the frozen history is untouched (locked satellites keep their
    truck, order, centroid and -- re-simulated -- their arrival time)."""
    visited = sorted(s for r in plan.routes.values() for s in r)
    assert visited == list(range(1, n + 1)), "satellite coverage broken"
    assert sorted(plan.sigma) == list(range(1, n + 1)) and sorted(plan.sigma.values()) == list(range(1, n + 1)), \
        "sigma is not a bijection"
    tl = sim.simulate(plan)
    for s, rec in snap.locked.items():
        new = tl.sats[s]
        assert new.truck == rec.truck and new.position == rec.position and new.centroid == rec.centroid, \
            f"locked satellite {s} was moved"
        assert abs(new.arrival - rec.arrival) < 1e-9, f"history of satellite {s} changed"


@dataclass
class World:
    inst: object
    clustered: object
    dist: object
    scenario: Scenario


def load_world(instance_path: str, events_path: str, scenario: str) -> World:
    inst = parse_instance(instance_path)
    clustered = cluster_victims(inst, random_state=0)
    dist = compute_distances(inst, clustered)
    sc = load_scenarios(events_path)[scenario]
    assert sc.num_satellites == inst.n, "events file does not match instance"
    return World(inst, clustered, dist, sc)


def run_closed_loop(world: World, strategies: List[str], cfg0: ReoptConfig, cfg: ReoptConfig,
                    seed: int = 0, selector: str = "adaptive", paired_hv_check: bool = False,
                    verbose: bool = False) -> Dict[str, dict]:
    inst, cl, dist, sc = world.inst, world.clustered, world.dist, world.scenario
    n, Q, UP = inst.n, float(inst.q), float(inst.no_u * inst.p)
    trucks = list(range(inst.no_k))
    epochs = sc.epochs

    def fresh_env():
        net = RoadNetwork(dist.time_n_0, sc.transition, sc.reveal_prob, rng=np.random.default_rng(seed + 7919))
        dem = DemandProcess(cl.demand_n, sc.initial_priorities)
        return net, dem, Simulator(n, Q, UP, dist.time_n_m, net, dem)

    # ---------------- shared epoch-0 solve ----------------
    net, dem, sim = fresh_env()
    ev0 = epochs[0]
    assert ev0.time == 0.0
    net.apply_edge_updates(0.0, ev0.edge_updates)
    dem.apply(0.0, ev0.demand_multipliers, ev0.priority_updates)
    snap0 = sim.initial_snapshot(trucks)
    prob0 = ReplanProblem(snap0, net.expected_time(), dist.time_n_m, dem.current_demand,
                          dem.current_priority, Q, UP, sc.drone_endurance_h, n)
    rng0 = np.random.default_rng(seed)
    solver0 = DynamicReoptimizer(prob0, cfg0, rng0, make_selector(selector, rng0))
    res0 = solver0.solve(solver0.construct())
    plan0 = merge(snap0, res0.chosen)
    epoch0_rec = dict(epoch=0, time=0.0, n_free=n, n_active=len(trucks), pred_objs=res0.chosen_objs,
                      viol=res0.chosen_viol, archive_size=len(res0.archive), iterations=res0.iterations,
                      runtime_s=res0.runtime_s)

    out = {}
    for si, strat in enumerate(strategies):
        net, dem, sim = fresh_env()
        net.apply_edge_updates(0.0, ev0.edge_updates)
        dem.apply(0.0, ev0.demand_multipliers, ev0.priority_updates)
        rng = np.random.default_rng(seed * 100 + si + 1)
        sel = make_selector(selector, rng)
        plan = plan0.copy()
        archive_items = list(res0.archive.items)
        elite = list(res0.elite)
        prev_T, prev_d = net.expected_time(), dem.current_demand.copy()
        recs = [dict(epoch0_rec)]
        replan_cpu = 0.0
        for ev in epochs[1:]:
            tl = sim.simulate(plan)
            snap = sim.commit(plan, tl, ev.time)
            net.observe_traversed(snap.traversed_arcs)
            net.apply_edge_updates(ev.time, ev.edge_updates)
            dem.apply(ev.time, ev.demand_multipliers, ev.priority_updates)
            T = net.expected_time()
            prob = ReplanProblem(snap, T, dist.time_n_m, dem.current_demand, dem.current_priority,
                                 Q, UP, sc.drone_endurance_h, n)
            rec = dict(epoch=ev.epoch_id, time=ev.time, n_free=len(prob.free), n_active=len(prob.active))
            if strat == "static" or not prob.free or not prob.active:
                v, o = prob.evaluate({k: [s for s in plan.routes[k] if s in set(prob.free)] for k in prob.active},
                                     {s: plan.sigma[s] for s in prob.free})
                rec.update(pred_objs=o, viol=v, archive_size=0, iterations=0, runtime_s=0.0)
                recs.append(rec)
                prev_T, prev_d = T, dem.current_demand.copy()
                continue

            focus = segment_disruption(prob, plan, prev_T, prev_d) if strat == "warm" else None
            sel.new_epoch()
            solver = DynamicReoptimizer(prob, cfg, rng, sel, focus)
            if strat == "warm":
                init = solver.warm_start(plan)
                seeds = [solver.project(s) for (_, s) in archive_items]
                res = solver.solve(init, seeds, elite)
            else:
                res = solver.solve(solver.construct())
            replan_cpu += res.runtime_s
            rec.update(pred_objs=res.chosen_objs, viol=res.chosen_viol, archive_size=len(res.archive),
                       iterations=res.iterations, runtime_s=res.runtime_s,
                       focus={int(k): round(v, 3) for k, v in (focus or {}).items()},
                       drl_state=drl_state(prob, focus or {}, res.archive, net.uncertain_ratio(),
                                           inst.no_k).round(4).tolist(),
                       op_weights=[round(x, 3) for x in (res.op_weights or [])])

            if paired_hv_check and strat == "warm":
                # same residual problem, same budget, solved cold -> HV comparison
                rng_b = np.random.default_rng(seed + 31 * ev.epoch_id)
                cold = DynamicReoptimizer(prob, cfg, rng_b, make_selector(selector, rng_b))
                res_b = cold.solve(cold.construct())
                if len(res.archive) and len(res_b.archive):
                    hv_w, hv_c = paired_hv(res.archive.objectives(), res_b.archive.objectives())
                    rec.update(hv_warm=hv_w, hv_cold=hv_c)

            new_plan = merge(snap, res.chosen)
            check_plan(new_plan, n, snap, sim)
            plan = new_plan
            archive_items = list(res.archive.items)
            elite = res.elite
            prev_T, prev_d = T, dem.current_demand.copy()
            recs.append(rec)
            if verbose:
                print(f"  [{strat}] epoch {ev.epoch_id} t={ev.time:.2f}h free={rec['n_free']} "
                      f"pred={tuple(round(x, 3) for x in rec['pred_objs'])} |A|={rec['archive_size']}")

        tl = sim.simulate(plan)
        m = realized_metrics(tl, dem, Q)
        m["replan_cpu_s"] = replan_cpu
        out[strat] = dict(metrics=m, epochs=recs,
                          plan=dict(routes={str(k): v for k, v in plan.routes.items()},
                                    sigma={str(k): v for k, v in plan.sigma.items()}))
        if verbose:
            print(f"  [{strat}] realized f1={m['f1']:.4f} f2={m['f2']:.4f} f3={m['f3']:.4f} "
                  f"unmet={m['unmet_ratio']:.3%}")
    return out
