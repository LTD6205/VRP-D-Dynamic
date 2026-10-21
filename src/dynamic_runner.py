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
  hybrid   : event-triggered repair (Han et al. 2023): first REPAIR the previous plan
             (keep its suffix, insert the new re-delivery jobs at their cheapest
             positions, re-match only drone pairings that became battery-infeasible).
             If the repaired plan is feasible and its predicted objective is at most
             `hybrid_tol` (relative, under the epoch's weights) worse than what the
             previous plan predicted, commit it without searching; otherwise run the
             full warm re-optimization. Saves CPU when an event barely matters.

All strategies of one run share the SAME epoch-0 plan (solved once), so differences
come only from how they react to events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from common import cluster_victims, compute_distances, parse_instance
from drone_energy import DroneBattery
from drone_service import SetupTimes, make_service
from dynamic_env import DemandProcess, RoadNetwork, Scenario, load_scenarios
from dynamic_sim import Job, Plan, Simulator, admissible_surges, create_residual_jobs, realized_metrics
from dynamic_solver import (DynamicReoptimizer, ReoptConfig, ReplanProblem, Sol, drl_state,
                            make_selector, segment_disruption)
from mo_utils import hypervolume, nondominated


def merge(snap, chosen: Sol, jobs: Optional[Dict[int, Job]] = None) -> Plan:
    routes = {k: list(ts.prefix) + (list(chosen.routes.get(k, [])) if ts.active else [])
              for k, ts in snap.trucks.items()}
    sigma = {s: r.centroid for s, r in snap.locked.items()}
    sigma.update(chosen.sigma)
    new_jobs = {}
    for j, job in (jobs or {}).items():                 # jobs keep cluster/amount; free ones get the chosen host
        new_jobs[j] = Job(job.cluster, job.amount, chosen.hosts.get(j, job.host), job.created)
    return Plan(routes, sigma, new_jobs)


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
    visited = sorted(s for r in plan.routes.values() for s in r if s <= n)
    assert visited == list(range(1, n + 1)), "satellite coverage broken"
    job_items = sorted(s for r in plan.routes.values() for s in r if s > n)
    assert job_items == sorted(plan.jobs), "every re-delivery job must be routed exactly once"
    assert sorted(plan.sigma) == list(range(1, n + 1)) and sorted(plan.sigma.values()) == list(range(1, n + 1)), \
        "sigma is not a bijection"
    tl = sim.simulate(plan)
    for s, rec in snap.locked.items():
        new = tl.sats[s]
        assert new.truck == rec.truck and new.position == rec.position and new.centroid == rec.centroid, \
            f"locked satellite {s} was moved"
        assert abs(new.arrival - rec.arrival) < 1e-9, f"history of satellite {s} changed"
    for j, rec in getattr(snap, "locked_jobs", {}).items():
        assert abs(tl.jobs[j].arrival - rec.arrival) < 1e-9, f"history of job {j} changed"


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
                    verbose: bool = False, recourse: bool = True, planner: str = "belief",
                    theta: float = 0.8, reveal_prob: Optional[float] = None,
                    obs_accuracy: Optional[float] = None, demand_noise: float = 0.0,
                    z_cap: float = 1.0, selector_model: Optional[str] = None,
                    paired_selectors: Optional[List[str]] = None, drone_model: str = "centroid",
                    setup: Optional[SetupTimes] = None, surge_scale: float = 1.0,
                    hybrid_tol: float = 0.02, road_events: bool = True) -> Dict[str, dict]:
    """recourse=True: re-planning strategies create re-delivery jobs for residual demand
    (surges after/during delivery, shortfalls); the 'static' strategy never re-plans and so
    never gets recourse.

    POMDP options: reveal_prob / obs_accuracy override the scenario's road observation model;
    demand_noise is the relative s.d. of reported surge demand; planner selects how the
    planner reads the observations (oracle | naive | belief | belief_cc, see RoadNetwork);
    z_cap: capacity/battery use  mean + z_cap * sd  of the demand belief.

    drone_model: 'centroid' (paper: drones fly to the cluster centroid) | 'route' (drones
    visit every victim, multi-trip CVRP per cluster); setup: drone launch/recovery/drop
    times (drone_service.SetupTimes); surge_scale (dynamic degree): every surge multiplier
    m becomes 1 + surge_scale (m - 1) (0 = no demand dynamics); hybrid_tol: repair
    threshold of the 'hybrid' strategy (a strategy name 'hybrid@0.05' sets it per strategy);
    road_events=False ignores the road-damage updates (isolates demand dynamics)."""
    def mk_sel(r):
        # a string names a selector type; an object (e.g. a DQN being trained) is shared as is
        return make_selector(selector, r, selector_model) if isinstance(selector, str) else selector

    inst, cl, dist, sc = world.inst, world.clustered, world.dist, world.scenario
    rp = sc.reveal_prob if reveal_prob is None else reveal_prob
    acc = 1.0 if obs_accuracy is None else obs_accuracy
    n, Q, UP = inst.n, float(inst.q), float(inst.no_u * inst.p)
    trucks = list(range(inst.no_k))
    battery = DroneBattery(sc.energy, inst.v_d, inst.no_u, inst.p)
    service = make_service(drone_model, inst, cl, dist, battery, UP, sc.energy, setup)
    epochs = sc.epochs

    def fresh_env():
        net = RoadNetwork(dist.time_n_0, sc.transition, rp, rng=np.random.default_rng(seed + 7919),
                          obs_accuracy=acc)
        dem = DemandProcess(cl.demand_n, sc.initial_priorities)
        return net, dem, Simulator(n, Q, UP, dist.time_n_m, net, dem, battery, service)

    # ---------------- shared epoch-0 solve ----------------
    net, dem, sim = fresh_env()
    ev0 = epochs[0]
    assert ev0.time == 0.0
    net.apply_edge_updates(0.0, ev0.edge_updates)
    dem.apply(0.0, ev0.demand_multipliers, ev0.priority_updates)
    snap0 = sim.initial_snapshot(trucks)
    prob0 = ReplanProblem(snap0, net.expected_time(), dist.time_n_m, dem.current_demand,
                          dem.current_priority, Q, UP, battery, n, service=service)
    if hasattr(service, "save_cache"):
        service.save_cache()                     # route model: persist the base-demand drone tables
    rng0 = np.random.default_rng(seed)
    solver0 = DynamicReoptimizer(prob0, cfg0, rng0, mk_sel(rng0))
    res0 = solver0.solve(solver0.construct())
    plan0 = merge(snap0, res0.chosen)
    epoch0_rec = dict(epoch=0, time=0.0, n_free=n, n_active=len(trucks), pred_objs=res0.chosen_objs,
                      viol=res0.chosen_viol, archive_size=len(res0.archive), iterations=res0.iterations,
                      runtime_s=res0.runtime_s)

    def paired_solve(prob, init_fn, cfg_, tag):
        """Solve the SAME residual problem with each selector (same seed, same start, same
        budget); returns {selector: (violation, scalar objective under cfg_ weights)}."""
        outp = {}
        for sname in paired_selectors or []:
            r_ = np.random.default_rng(seed * 7 + tag)
            sol_ = DynamicReoptimizer(prob, cfg_, r_, make_selector(sname, r_, selector_model),
                                      getattr(prob, "_focus", None))
            res_ = sol_.solve(init_fn(sol_))
            w_ = np.asarray(cfg_.weights if cfg_.mode == "so" else cfg_.commit_weights)
            outp[sname] = (float(res_.chosen_viol), float(np.dot(w_, res_.chosen_objs)), res_.iterations)
        return outp

    if paired_selectors:
        epoch0_rec["paired"] = paired_solve(prob0, lambda so: so.construct(), cfg0, 0)

    out = {}
    for si, strat_name in enumerate(strategies):
        strat, _, tol_s = strat_name.partition("@")
        tol = float(tol_s) if tol_s else hybrid_tol
        net, dem, sim = fresh_env()
        net.apply_edge_updates(0.0, ev0.edge_updates)
        dem.apply(0.0, ev0.demand_multipliers, ev0.priority_updates)
        rng = np.random.default_rng(seed * 100 + si + 1)
        sel = mk_sel(rng)
        plan = plan0.copy()
        archive_items = list(res0.archive.items)
        elite = list(res0.elite)
        prev_T, prev_d = net.expected_time(), dem.current_demand.copy()
        recs = [dict(epoch0_rec)]
        noise_rng = np.random.default_rng(seed + 104729)      # demand reports (same for all strategies)
        reported: Dict[int, float] = {}                       # cluster -> reported (believed) demand
        replan_cpu = 0.0
        n_repaired = 0
        surge_requested = surge_granted = 0.0
        prev_pred = res0.chosen_objs                   # predicted objectives of the plan in force
        next_job = n + 2                               # job ids start right after the end depot
        for ev in epochs[1:]:
            tl = sim.simulate(plan)
            snap = sim.commit(plan, tl, ev.time)
            net.observe_traversed(snap.traversed_arcs)
            net.apply_edge_updates(ev.time, ev.edge_updates if road_events else [])
            # demand rule: surges only before/during service and within the serving truck's capacity
            mult = {c: 1.0 + surge_scale * (float(m_) - 1.0) for c, m_ in ev.demand_multipliers.items()}
            incs, surge_stats = admissible_surges(plan, tl, ev.time, mult, dem.current_demand, Q)
            surge_requested += surge_stats["requested"]
            surge_granted += surge_stats["granted"]
            dem.apply_increments(ev.time, incs, ev.priority_updates)
            # demand observation: surged clusters are reported with noise until a truck reaches them
            for c in incs:
                eps = noise_rng.normal(0.0, demand_noise) if demand_noise > 0 else 0.0
                reported[c] = max(0.0, float(dem.current_demand[c - 1]) * (1.0 + eps))
            reached = {r.centroid for r in snap.locked.values() if r.arrival <= ev.time + 1e-12}
            for c in list(reported):
                if c in reached:
                    del reported[c]                            # truth revealed on arrival
            d_plan = dem.current_demand.copy()
            d_sd = np.zeros_like(d_plan)
            if planner != "oracle":
                for c, mu in reported.items():
                    d_plan[c - 1] = mu
                    if planner in ("belief", "belief_cc"):
                        d_sd[c - 1] = demand_noise * mu
            T = net.planning_time(planner, theta)
            new_jobs = {}
            if recourse and strat != "static":
                new_jobs, next_job = create_residual_jobs(plan, tl, snap, dem, next_job)
                plan.jobs.update(new_jobs)
            prob = ReplanProblem(snap, T, dist.time_n_m, d_plan, dem.current_priority,
                                 Q, UP, battery, n, jobs=plan.jobs, demand_sd=d_sd, z=z_cap, service=service)
            prob.uncertain = net.uncertain_ratio()
            rec = dict(epoch=ev.epoch_id, time=ev.time, n_free=len(prob.free), n_active=len(prob.active),
                       new_jobs=len(new_jobs), residual_demand=float(sum(j.amount for j in new_jobs.values())),
                       surge=surge_stats)
            if strat == "static" or not (prob.free or prob.free_jobs) or not prob.active:
                keep = set(prob.free) | set(prob.free_jobs)
                v, o = prob.evaluate({k: [s for s in plan.routes[k] if s in keep] for k in prob.active},
                                     {s: plan.sigma[s] for s in prob.free},
                                     {j: plan.jobs[j].host for j in prob.free_jobs})
                for j in new_jobs:                      # no active truck can take them: stays unmet
                    plan.jobs.pop(j, None)
                rec.update(pred_objs=o, viol=v, archive_size=0, iterations=0, runtime_s=0.0)
                prev_pred = o
                recs.append(rec)
                prev_T, prev_d = T, d_plan.copy()
                continue

            focus = segment_disruption(prob, plan, prev_T, prev_d) if strat in ("warm", "hybrid") else None
            sel.new_epoch()
            solver = DynamicReoptimizer(prob, cfg, rng, sel, focus)
            if strat == "hybrid":
                # (A) repair first; search only if the repair is infeasible or clearly worse
                import time as _time
                t_rep = _time.process_time()
                rep = solver.warm_start(plan)
                v_r, o_r = prob.evaluate(rep.routes, rep.sigma, rep.hosts)
                w_ = np.asarray(cfg.weights if cfg.mode == "so" else cfg.commit_weights, dtype=float)
                rel = float(np.dot(w_, (np.asarray(o_r) - np.asarray(prev_pred))
                                   / np.maximum(np.abs(prev_pred), 1e-9)) / max(w_.sum(), 1e-9))
                rec.update(repair_gap=rel, repair_feasible=bool(v_r == 0))
                if v_r == 0 and rel <= tol:
                    dt_rep = _time.process_time() - t_rep
                    replan_cpu += dt_rep
                    n_repaired += 1
                    rec.update(pred_objs=o_r, viol=0.0, archive_size=0, iterations=0, runtime_s=dt_rep,
                               repaired=True)
                    new_plan = merge(snap, rep, plan.jobs)
                    check_plan(new_plan, n, snap, sim)
                    plan = new_plan
                    prev_pred = o_r
                    prev_T, prev_d = T, d_plan.copy()
                    recs.append(rec)
                    continue
                rec["repaired"] = False
            if strat in ("warm", "hybrid"):
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

            if paired_selectors and strat == "warm":
                prob._focus = focus
                rec["paired"] = paired_solve(prob, lambda so: so.warm_start(plan), cfg, ev.epoch_id)

            if paired_hv_check and strat == "warm":
                # same residual problem, same budget, solved cold -> HV comparison
                rng_b = np.random.default_rng(seed + 31 * ev.epoch_id)
                cold = DynamicReoptimizer(prob, cfg, rng_b, mk_sel(rng_b))
                res_b = cold.solve(cold.construct())
                if len(res.archive) and len(res_b.archive):
                    hv_w, hv_c = paired_hv(res.archive.objectives(), res_b.archive.objectives())
                    rec.update(hv_warm=hv_w, hv_cold=hv_c)

            new_plan = merge(snap, res.chosen, plan.jobs)
            check_plan(new_plan, n, snap, sim)
            plan = new_plan
            prev_pred = res.chosen_objs
            archive_items = list(res.archive.items)
            elite = res.elite
            prev_T, prev_d = T, d_plan.copy()
            recs.append(rec)
            if verbose:
                print(f"  [{strat}] epoch {ev.epoch_id} t={ev.time:.2f}h free={rec['n_free']} "
                      f"pred={tuple(round(x, 3) for x in rec['pred_objs'])} |A|={rec['archive_size']}")

        tl = sim.simulate(plan)
        m = realized_metrics(tl, dem, Q)
        m["replan_cpu_s"] = replan_cpu
        m["n_repaired"] = n_repaired
        # (B) realized dynamic degree: share of the final demand that arrived as surges after t=0
        m["dynamic_degree"] = surge_granted / max(m["D"], 1e-9)
        m["surge_requested"] = surge_requested
        # (C) plan-vs-actual discrepancy: (predicted - realized) / realized, for the plan made at
        # t = 0 and for the last plan in force
        last_pred = recs[-1].get("pred_objs", res0.chosen_objs)
        for i_, f_ in ((0, "f1"), (2, "f3")):
            m[f"gap0_{f_}"] = (float(res0.chosen_objs[i_]) - m[f_]) / max(abs(m[f_]), 1e-9)
            m[f"gap_last_{f_}"] = (float(last_pred[i_]) - m[f_]) / max(abs(m[f_]), 1e-9)
        m["drone_model"] = service.name
        m["mean_wait"] = float(np.mean([r.wait for r in tl.sats.values()]))
        m["total_drone_trips"] = int(sum(r.waves for r in list(tl.sats.values()) + list(tl.jobs.values())))
        out[strat_name] = dict(metrics=m, epochs=recs,
                          plan=dict(routes={str(k): v for k, v in plan.routes.items()},
                                    sigma={str(k): v for k, v in plan.sigma.items()},
                                    jobs={str(j): vars(x) for j, x in plan.jobs.items()}))
        if verbose:
            print(f"  [{strat_name}] realized f1={m['f1']:.4f} f2={m['f2']:.4f} f3={m['f3']:.4f} "
                  f"unmet={m['unmet_ratio']:.3%}")
    return out
