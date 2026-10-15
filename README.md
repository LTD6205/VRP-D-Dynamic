# VRP-D-Dynamic: rolling-horizon dynamic 2E-CTVRP with drones

> Built on the static 2E-CTVRP code of Nguyen, Piplani & Gunawan,
> *"A two-echelon covering tour vehicle routing problem with drones for post-disaster relief"*
> (original repo: [andngdtudk/2E_CTVRP](https://github.com/andngdtudk/2E_CTVRP)).
> The original README — installation, static heuristics, MILP and how to reproduce the paper — is
> kept in [`README_STATIC.md`](README_STATIC.md).
>
> **Initial (proposal) formulation:** [`docs/INITIAL_FORMULATION.md`](docs/INITIAL_FORMULATION.md).
>
> **Formulation summary** (variables, formulas, constraints, dataset generation, changes vs. the first version): [`docs/FORMULATION.md`](docs/FORMULATION.md).

This extension turns the static 2E-CTVRP of Nguyen, Piplani & Gunawan (see [`README_STATIC.md`](README_STATIC.md)) into
the dynamic, event-driven truck–drone relief problem described in `bai_toan_iii_truck_drone.pdf`.
It follows the plan in `dynamic_2e_ctvrp_research_guide_dataset_blueprint.md` (called "the guide"
below), with the corrections listed in §5. **None of the original code or data was modified** (the original README was only renamed to `README_STATIC.md`). The
static solvers, the MILP and the shipped instances behave exactly as before.

## 1. What was added

| File | Role |
|---|---|
| `src/dynamic_env.py` | 3-state road model (Open / Degraded ×2.5 / Detour ×5.0), true state vs. planner **belief** with Markov prediction and partial observation (`reveal_prob`), demand and priority process, `.events.json` loader |
| `src/dynamic_sim.py` | Executes a plan on the *true* network (arc time fixed at departure), **freezes history** at an event (`commit`), and computes realized KPIs |
| `src/dynamic_solver.py` | Residual problem at epoch *t* plus the multi-objective re-optimizer: N1–N7 from the repo plus destroy-repair, segment-focused moves, best-of-K local search, path relinking with an elite σ-pool that survives epochs, Pareto archive, and the operator-selector hook (the slot for the DRL agent) |
| `src/dynamic_runner.py` | Closed loop: solve at t=0 → execute → event → freeze → re-plan → … for the strategies `static`, `restart` and `warm`, with invariant checks after every re-plan |
| `src/mo_utils.py` | Pareto archive (crowding truncation), exact 3-D hypervolume, Das-Dennis weights |
| `scripts/generate_dynamic_dataset.py` | Corrected version of the guide's generator → `instances/dynamic_events/*.events.json` |
| `scripts/run_dynamic.py` | Runs one instance × scenario |
| `scripts/run_dynamic_experiments.py`, `scripts/summarize_dynamic.py` | Run and summarize the benchmark → `results/dynamic_*.csv` |

Quick start (no Gurobi needed):

```bash
python3 scripts/generate_dynamic_dataset.py            # 12-instance subset, 4 scenarios each
python3 scripts/run_dynamic.py --instance instances/large/M-50-50-60-1.txt \
    --scenario damage_50pct --mode so --weights 0,0,1 --time 4 -v
python3 scripts/run_dynamic.py --instance instances/main/M-15-13-60-2.txt \
    --scenario damage_30pct --mode mo --paired-hv --time 2 -v --output out.json
```

## 2. Model implemented

**State at epoch t** (guide §3.1, spec §3.1). Every satellite reached by time t is *locked*. A truck
already driving toward a satellite is committed to that arc, and the target is locked too. Each
still-active truck k restarts from a virtual start node $v_{\pi_k(0)}(t)$ at time $a_k(t)$ (arrival
plus drone waiting) with remaining capacity $Q_k(t)=Q-\sum_{\text{locked}} d$. Trucks already
heading home are retired. The free satellites and free centroids form the residual bijection
$\sigma_t$.

**Road model** (guide §3.2): $E[t_{ij}(t)] = t^0_{ij}\sum_s \phi_s b^t_{ij}(s)$ with $\phi=(1,2.5,5)$.
The arc matrix may be asymmetric (one-way damage).
- With `reveal_prob=1`, the belief equals the true state (a dynamic MOOP).
- With `reveal_prob<1`, a changed arc is reported only with that probability. Unobserved arcs follow
  the Markov prediction $b\leftarrow bP$, and an arc a truck has driven reveals its true state.
  This is the minimal POMDP the spec §4.8 asks for.

The planner optimizes on the belief, while the simulator executes on the truth.

**Second echelon.** Drones fly out and back, $w_i = \lceil d_{\sigma(i)}/UP\rceil\cdot 2t'_{i\sigma(i)}$.
When $d \le UP$, which holds for every shipped instance, this is the paper's single parallel
sortie. A demand surge above $UP$ becomes extra sortie waves instead of an infeasibility.

**Drone battery** (`src/drone_energy.py`). The energy of each sortie depends on the flight
**speed** and the carried **load**. The power model is
$p(v,\ell)=a(1+\beta\ell/P)+b(v/v_{ref})^3$, and each sortie flies out loaded and back empty:
$e_{ij}=t'_{ij}[p(v_d,\ell_j)+p(v_d,0)]\le E(1-\rho)$.
- Range peaks near 60 km/h and shrinks with payload.
- A demand surge can make a satellite→cluster pair infeasible and force a re-assignment.
- The constraint is enforced on every σ move, and feasible pairs are recomputed each epoch from
  the current demand.
- Parameters are in each scenario's `drone_energy` block; details are in
  [`docs/FORMULATION.md`](docs/FORMULATION.md) §5.2b.

**Objectives** are computed over the whole horizon, with locked parts entering as constants so
values are comparable across epochs (guide §3.4):
- $f_1=\frac1D\sum_i w_{\sigma(i)}d_{\sigma(i)}e_i$, the priority-weighted delivery time;
- $f_2=\max_i a_i-\min_i a_i$, i.e. $Z_3$ (equity);
- $f_3=\sum_i a_i+\sum_k \text{ret}_k$, i.e. $Z_1$, the paper's objective.

`--mode so --weights 0,0,1` reduces to the paper's problem.

**Strategies compared**
- `static`: the t=0 plan, never re-planned. It is executed on the disrupted network.
- `restart`: cold randomized construction at each epoch, no memory.
- `warm`: the previous plan's suffix, the projected Pareto archive and elite σ-pool, the repo's
  3-stage repair idea (cheapest re-insertion), and *segment-aware* moves. In segment-aware moves the
  primary truck is sampled ∝ the disruption score $c_k(t)=\text{norm}(\omega_1 r^{road}_k+\omega_2
  r^{demand}_k+\omega_3 r^{timing}_k)$ (spec eq. 9).

All three strategies share the same t=0 plan, so differences come only from how they react.

**DRL hook.** `drl_state()` returns the spec's state vector (eq. 10): $[c_1..c_K, |A|/A_{max},
u_D, u_V, r_{unmet}, r_{uncertain}]$. The selector interface is `select(state) -> operator`,
`update(op, reward)`. The present `AdaptiveSelector` is an ALNS roulette and serves as the
"adaptive but no DRL" baseline (spec §8.3). A DQN only has to implement the same two methods. The
paired warm-vs-cold HV is a ready-made reward signal for spec eq. 12.

## 3. Verification

- **Static reproduction.** At t=0 in `so`/Z1 mode, the dynamic solver reaches the paper's
  GRASP-ILS-PR best values exactly in 5 s: M-08-05-60-1 12.6546, M-09-09-60-1 17.7304, M-09-09-60-2
  17.2726, M-12-11-60-1 23.4624, M-12-11-60-2 26.0638. Its objective is also identical to the repo's
  `ECTVRPOperators.sum_arr_time`, and its solutions pass the repo's `is_feasible`.
- **Invariants after every re-plan** (`dynamic_runner.check_plan`): each satellite is visited
  exactly once, σ is a bijection, and every locked satellite keeps its truck, position, centroid and
  re-simulated arrival time. Frozen history is never rewritten.

## 4. Benchmark results

Setup: 12 instances (guide §5.1) × 4 scenarios × 3 seeds × {static, restart, warm}.

**CPU budget per re-plan:**
- 1.5 s for n ≤ 15, 3 s for n = 20, 5 s for n = 50;
- the t=0 plan gets 2×.

All values are *realized*: the final plan is executed on the true disrupted network. The tables
show the mean change relative to the `static` baseline; negative is better.
Raw data: `results/dynamic_*.csv`. Regenerate the tables with `scripts/summarize_dynamic.py`.

### 4.1 Single objective Z1, the paper's objective (`--mode so --weights 0,0,1`)

| scale | 10% damage | 30% | 30% partial-obs | 50% |
|---|---|---|---|---|
| small (n ≤ 9), Z1 restart / warm | −0.06 / −0.06 | 0 / 0 | **+0.82 / +0.82** | −3.34 / −3.34 |
| medium (n 11–13) | −0.83 / −0.67 | −0.07 / −0.05 | −2.64 / −2.74 | −4.99 / −5.12 |
| large (n 20–50) | −2.04 / −1.98 | −0.66 / −0.79 | −1.90 / −1.81 | −4.47 / −3.97 |

Overall, re-planning lowers realized Z1 by about 1.7% on average and by up to 5% at 50% damage.
The gains grow with damage and with satellites per truck. Optimizing Z1 alone *worsens* the equity
spread f2 (+10% on average), which is the trade-off the MO formulation is for.

Under partial observability on small instances, re-planning on the *belief* is slightly worse than
not re-planning (+0.8%). The Markov prediction expects degradation everywhere, which pushes the
planner to avoid roads that are in fact open. This is a concrete argument for the spec's
observation / exploration design (spec §4.8).

**Tight re-planning deadline** (0.25 s per epoch, n = 20 and 50; `results/dynamic_so_z1_tight.csv`):

| | restart | warm |
|---|---|---|
| mean ΔZ1 vs static | −1.36% | **−2.35%** |
| runs better than static | 66.7% | **77.1%** |
| warm beats restart (paired) | — | 66.7% of runs (tie 2%) |

With a generous budget, cold restart and warm start end up equal: the residual problems are small
enough to re-solve from scratch, and warm beats restart in only 42% of runs. Warm start, memory
and segment focus pay off **when the re-planning deadline is tight**. That is the regime the
spec's $r_{late\_decision}$ penalty targets, and it is where a learned operator controller has
something to win.

### 4.2 Multi-objective (`--mode mo`, commit = equal-weight point of the archive)

| | Δf1 | Δf2 | Δf3 (Z1) |
|---|---|---|---|
| restart | −1.21% | −5.33% | −0.51% |
| warm | **−1.40%** | **−7.27%** | **−0.68%** |

**Paired hypervolume.** At each epoch, the warm strategy's residual problem is also solved cold
with the same budget. HV is normalized by the union of both archives, with reference point
1.1³:

| epoch | scale | HV warm | HV cold | warm better | tie |
|---|---|---|---|---|---|
| 1 | large | **0.851** | 0.805 | 54% | 2% |
| 1 | medium | **0.608** | 0.574 | 29% | 52% |
| 1 | small | 0.486 | 0.489 | 2% | 92% |
| 2 | large | **0.642** | 0.620 | 30% | 57% |
| 2 | medium / small | equal | equal | 0% | 100% |

Memory (the projected archive plus the elite pool) gives a better Pareto front right after a
change on large instances. On small residual problems both approaches reach the same front.

**Unmet demand.** Surges that land on a cluster already served cannot be recovered in this model.
They account for 0–4.7% unmet demand, and re-planning reduces it only slightly by reaching clusters
before the surge. See §6.1.

## 5. Review of the guide: what changed and why

1. **Trigger times (guide §5.3).** The fixed epochs of 0.40 h and 0.85 h fall before the first
   satellite arrival on most instances: the depot sits about 20 km outside the ring, which is more
   than 0.5 h of driving, so nothing would be committed. The generator now places t₁ and t₂ at the
   30% and 65% quantiles of satellite arrival times of a deterministic baseline plan.
2. **Damage pattern.** In the guide's script, damage is uniform-random over arcs. Return-to-depot
   arcs are missing, and `all_sat_arcs[:n]` only picks satellite 1's out-arcs. The new generator:
   - centres the flood on an *inner-ring satellite that is still unserved at t₁*;
   - damages arcs by proximity, with progression and escalation in epoch 2;
   - makes 30% of the damage one-way;
   - counts the 10/30/50% over all first-echelon arcs.
   An extra `damage_30pct_partial` scenario (`reveal_prob 0.6`) covers partial observability.
3. **Demand surges** now hit clusters *not yet served* in the baseline plan, near the flood (the
   guide also intends "unserved clusters"). A surge that still lands on an already-served cluster
   is reported as **unmet demand**. The model has no recourse for it, because each cluster is
   served once by one satellite. See §6.
4. **Elite pool across epochs (guide §4.3)** claims old σ's "remain valid permutations". That does
   not hold: after commitment some centroids are locked to other satellites, so an old σ restricted
   to the free satellites is generally not a bijection onto the free centroids. `project_sigma`
   repairs it by keeping compatible pairs and re-matching the rest under the battery constraint.
5. **2-opt asymmetry (guide §1.2.2).** No operator change is needed in this code base. Every move,
   in the original repo and here, is scored by *full* recomputation of the route times from the
   matrix, not by delta formulas, so asymmetric $t_{ij}\ne t_{ji}$ are handled automatically. Under
   the detour model, arcs are never infinite, so there is no directional infeasibility to check.
6. **Zero-insertion property (guide §1.2.1).** This is confirmed: under degradation the Manhattan
   zero-cost insertions vanish. The repo's insertion code already falls back to $P\propto 1/c$, so
   only the cost matrix had to change. Infinite costs never occur with detour multipliers.
7. **MILP (guide §1.1).** With the detour representation $\xi_{ij}\equiv1$, the dynamic MILP is the
   static one with: the expected-time matrix; locked $x$, $y$, $z$ variables fixed to the executed
   prefix; per-truck start nodes and times; and $H(t)$ recomputed from the new max arc time. This
   was **not implemented here** (no Gurobi licence in this environment). It is the natural next
   step for small-instance exact checks.
8. **Structural limit of re-planning in this model.** Each truck makes one trip, and with the small
   fleets (`-2`, K≈n/2) most trucks have only one or two satellites left at t₁. The residual
   problem is then tiny, and `static`, `restart` and `warm` often coincide (see §4). Re-planning
   value grows with satellites per truck (the `-1` fleets and n=50).

## 6. Suggested next steps (toward SR-DRL-DMOEA)

1. **Recourse for surges.** Allow split delivery across epochs (spec §1): residual demand at a
   served cluster becomes a new "virtual centroid" that a free satellite can cover (guide §1.3
   option 2, 1-to-many). This removes the unmet-demand artefact and gives the DRL agent a
   meaningful "drone-reassign" action.
2. **Deprivation objective.** Replace $f_1$ by the spec's $\sum_i w_i h_i[T_i-r_i]^+$ with
   $r_i$ = the event time for surge demand. The code already tracks the event times.
3. **DRL controller.** Replace `AdaptiveSelector` by a Double/Dueling DQN over `drl_state()`. Use
   the reward $\alpha\,\Delta HV - \beta\,\text{Eval}/\text{budget}$ (spec eq. 12), with the paired
   warm/cold HV as $HV_{base}$.
4. **MOEA population.** The archive and weight-cycling ILS here are a single-trajectory Pareto
   approximation. Swapping in NSGA-II or MOEA/D over the same `ReplanProblem.evaluate` (giant-tour
   plus σ encoding) gives the MOEA baselines of spec §8.3.
5. **Exact reference.** A dynamic MILP / ε-constraint on M-08-05 and M-09-09 to validate the
   empirical Pareto fronts (spec §8.4).
6. **Earlier and more frequent events** (more epochs, e.g. every 0.25 h) to stress the
   memory/warm-start components, which matter more when the residual problem is large.
