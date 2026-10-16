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
| `src/dynamic_env.py` | 3-state road model (Open / Degraded ×2.5 / Detour ×5.0), true state vs. planner **belief** (POMDP: Markov prediction, partial and **noisy** reports with Bayes update, chance-constrained planning view), demand and priority process, `.events.json` loader |
| `src/dynamic_sim.py` | Executes a plan on the *true* network (arc time fixed at departure), **freezes history** at an event (`commit`), applies the **demand rule** (`admissible_surges`), creates **re-delivery jobs** for residual demand (`create_residual_jobs`), and computes realized KPIs |
| `src/dynamic_solver.py` | Residual problem at epoch *t* plus the multi-objective re-optimizer: N1–N7 from the repo plus destroy-repair, segment-focused moves, best-of-K local search, path relinking with an elite σ-pool that survives epochs, Pareto archive, and the operator-selector hook (the slot for the DRL agent) |
| `src/dynamic_runner.py` | Closed loop: solve at t=0 → execute → event → freeze → re-plan → … for the strategies `static`, `restart` and `warm`, with invariant checks after every re-plan |
| `src/mo_utils.py` | Pareto archive (crowding truncation), exact 3-D hypervolume, Das-Dennis weights |
| `src/drone_energy.py` | Drone battery: sortie energy from speed, load and distance |
| `src/drl_selector.py` | **DRL operator controller**: Double DQN in plain numpy (no deep-learning library needed) |
| `scripts/train_drl_selector.py` | Trains the DQN on a separate training set (`instances/dynamic_events_train/`) → `models/dqn_selector.npz` |
| `scripts/run_pomdp_experiment.py` | POMDP experiment: oracle / naive / belief / belief + chance-constraint planners → `results/dynamic_pomdp.csv` |
| `scripts/run_selector_experiment.py` | Paired operator-selector experiment: uniform / adaptive / DQN → `results/dynamic_selectors.csv` |
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
- With `reveal_prob=1` and `obs_accuracy=1`, the belief equals the true state (a dynamic MOOP).
- Otherwise the problem is a **POMDP**:
  - each changed road is reported with probability `reveal_prob`;
  - a report is correct with probability `obs_accuracy`, otherwise it names a wrong state;
  - the planner predicts every road's belief one epoch forward with the Markov matrix, then
    applies **Bayes' rule** to each report;
  - a road a truck has driven reveals its true state.
- Surge demand is reported with relative noise `demand_noise`. The planner keeps a Gaussian belief
  until a truck reaches the cluster, and checks capacity and battery at mean + 1 sd.
- **How the planner reads the belief** (`planner=`):
  - `oracle`: knows the truth (upper bound);
  - `naive`: trusts every report;
  - `belief`: expected travel times;
  - `belief_cc`: belief plus the proposal's **chance constraint**. A road that is usable with
    probability below θ = 0.8 is planned as a detour.

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

**Demand rule** (`admissible_surges`). A cluster's demand may only grow **before or during** its
service:
- a requested surge is **dropped** if the cluster's truck has already left;
- it is **capped** at the spare capacity of the truck planned to serve that cluster;
- so the current plan always stays capacity-feasible and no new trips are ever needed.

**Recourse** (`create_residual_jobs`, `src/dynamic_sim.py`).
- When a surge arrives while the truck is still parked at the satellite (drones flying), or a
  truck ran short because the demand was observed with noise, the missing amount becomes a
  **re-delivery job**.
- A job is a route stop: a truck with goods left drives to a **host** satellite within battery
  range of the cluster and flies the extra wave(s). If the truck is still parked at that
  satellite, the job is just an extra wave there, with no extra driving.
- "All demand met" is a **soft** constraint: every unit is scheduled and counted in truck
  capacity. If the trucks don't carry enough, the least-overloaded plan is committed and the
  shortfall is reported as unmet.
- The `static` strategy never re-plans, so it gets no recourse. `--no-recourse` switches it off for
  the other strategies too. Details are in [`docs/FORMULATION.md`](docs/FORMULATION.md) §5.4b.

**Objectives** are computed over the whole horizon, with locked parts entering as constants so
values are comparable across epochs (guide §3.4):
- $f_1=\frac1D\big[\sum_{\text{stops}} w_c\,q\,e\big]$, the priority-weighted delivery time
  including re-deliveries. In realized results, undelivered demand is charged at $C_{ret}$, so
  leaving people unserved never looks better than serving them late;
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

**DRL operator controller** (`src/drl_selector.py`, the "DRL" of SR-DRL-DMOEA).
- A **Double DQN** picks which of the 9 search operators to apply, one decision per segment of 10
  moves.
- **State:** a 23-feature vector (spec eq. 10): route disruption $c_k$, fleet and drone load,
  share of uncertain roads, search progress, recent success rates, and the last operator.
- **Reward:** new best / improved / new Pareto point, the improvement part of spec eq. 12.
- **Training:** on 14 training instances that are not in the test benchmark
  (`scripts/train_drl_selector.py`); evaluated frozen.
- **Baselines** (spec §8.3): `uniform` (random operator) and `adaptive` (ALNS roulette).
- Select it with `--selector dqn`.

## 3. Verification

- **Static reproduction.** At t=0 in `so`/Z1 mode, the dynamic solver reaches the paper's
  GRASP-ILS-PR best values exactly in 5 s: M-08-05-60-1 12.6546, M-09-09-60-1 17.7304, M-09-09-60-2
  17.2726, M-12-11-60-1 23.4624, M-12-11-60-2 26.0638. Its objective is also identical to the repo's
  `ECTVRPOperators.sum_arr_time`, and its solutions pass the repo's `is_feasible`.
- **Invariants after every re-plan** (`dynamic_runner.check_plan`): each satellite is visited
  exactly once, σ is a bijection, and every locked satellite keeps its truck, position, centroid and
  re-simulated arrival time. Frozen history is never rewritten.

## 4. Benchmark results

All results below use the final model:
- speed- and load-dependent battery;
- the **demand rule** (surges only before or during service, capped by truck capacity);
- re-delivery recourse;
- 12 test instances (guide §5.1) × 4 scenarios × 3 seeds.

**CPU budget per re-plan:**
- 1.5 s for n ≤ 15, 3 s for n = 20, 5 s for n = 50;
- the t=0 plan gets 2×.

All values are *realized*: the final plan is executed on the true disrupted network.
- `static` never re-plans, so it has no recourse.
- f1 charges undelivered demand at $C_{ret}$.

Raw data: `results/dynamic_*.csv`; `scripts/summarize_dynamic.py` regenerates the benchmark
tables.

### 4.0 Unmet demand

| benchmark | static: runs with unmet demand | static: mean / worst | restart & warm: runs with unmet demand |
|---|---|---|---|
| single objective (144 runs) | 77 (53%) | 0.90% / 3.39% | **0** |
| tight deadline (48 runs) | 42 (88%) | 1.19% / 3.38% | **0** |
| multi-objective (144 runs) | 86 (60%) | 1.15% / 3.75% | **0** |

With the demand rule, surges only happen before or during service and always fit the serving
truck. Re-planning therefore delivers **all demand in every run** (672 of 672). The `static`
plan still misses the surges that arrive while its drones are already flying, because it never
schedules the extra wave.

### 4.1 Single objective Z1, the paper's objective (`--mode so --weights 0,0,1`)

Mean change vs. static, restart / warm (negative is better):

| scale | Δf1 | ΔZ1 10% damage | ΔZ1 30% | ΔZ1 30% partial-obs | ΔZ1 50% |
|---|---|---|---|---|---|
| small (n ≤ 9) | −3.0 / −3.0 | −1.94 / −1.94 | −3.32 / −3.32 | −2.49 / −2.49 | −2.21 / −2.21 |
| medium (n 11–13) | −1.2 / −1.1 | −2.25 / −2.12 | −5.01 / −4.63 | −0.49 / −0.47 | −4.15 / −3.88 |
| large (n 20–50) | −3.3 / −3.5 | −0.75 / −0.69 | −2.18 / −2.00 | −0.72 / −0.62 | −4.95 / −5.03 |

- **Overall:** re-planning lowers realized Z1 by about 2.5% (up to 5% at 50% damage) and f1 by
  about 2.5%, while delivering everything.
- **Equity cost:** optimizing Z1 alone worsens the equity spread f2 (+6–7%). This is the
  trade-off the MO formulation is for.
- **Partial observability:** re-planning now helps in every scale (−0.5% to −2.5%). The belief
  is no longer pessimistically "aged" at dispatch time, when the network has just been surveyed.

**Tight re-planning deadline** (0.25 s per epoch, n = 20 and 50; `results/dynamic_so_z1_tight.csv`):

| | restart | warm |
|---|---|---|
| mean Δf1 vs static | −3.29% | **−3.48%** |
| mean ΔZ1 vs static | −2.45% | **−2.65%** |
| mean Δf2 vs static | +0.2% | **−4.6%** |
| ΔZ1 at 50% damage | **−6.18%** | −5.93% |
| runs with Z1 better than static | 72.9% | **77.1%** |
| warm beats restart on Z1 (paired) | — | **64.6%** of runs (tie 8.3%) |

With a generous budget, cold restart and warm start mostly end up equal: warm is strictly better
in 9% of runs and tied in 76%. Warm start, memory and segment focus pay off **when the
re-planning deadline is tight**.

### 4.2 Multi-objective (`--mode mo`, commit = equal-weight point of the archive)

| | Δf1 | Δf2 | Δf3 (Z1) |
|---|---|---|---|
| restart | −3.93% | **−6.18%** | **−1.13%** |
| warm | **−4.07%** | −6.09% | −0.95% |

**Paired hypervolume** (warm vs. cold on identical residual problems, same budget):

| epoch | scale | HV warm | HV cold | warm better | tie |
|---|---|---|---|---|---|
| 1 | large | 0.884 | 0.886 | 54% | 2% |
| 1 | medium | **0.651** | 0.567 | 33% | 50% |
| 1 | small | 0.609 | 0.606 | 10% | 79% |
| 2 | large | **0.828** | 0.813 | 42% | 25% |
| 2 | medium | 0.691 | 0.695 | 4% | 83% |
| 2 | small | 0.968 | 0.969 | 2% | 94% |

Memory gives a better front on medium instances after the first event and on large instances
after the second. Otherwise warm and cold are level.

### 4.3 POMDP: using noisy, partial observations (`results/dynamic_pomdp.csv`)

**Setup** (`scripts/run_pomdp_experiment.py`):
- **Road reports:** a changed road is reported with probability 0.6, and a report is correct
  with probability 0.8.
- **Demand reports:** surge demand is reported with 15% noise.
- **Runs:** 12 instances × {30%, 50% damage} × 3 seeds, warm strategy, Z1 objective, same budget
  for every planner (288 runs).

Mean gap to the `oracle` planner, which knows the truth:

| planner | ΔZ1 | Δf1 | Δf2 | runs with unmet demand |
|---|---|---|---|---|
| naive (trusts reports) | +3.76% | +1.04% | +8.9% | 1 (0.25%) |
| **belief** (Bayes, expected times) | **+3.04%** | **−0.46%** | **+1.9%** | 0 |
| belief_cc (chance constraint θ = 0.8) | +3.65% | −0.07% | +5.0% | 0 |

- **Cost of imperfect information:** about 3–4% Z1 compared with full information.
- **Belief vs. naive:** the **Bayesian belief** is the best way to use the reports on average. It
  is much fairer (f2 +1.9% vs +8.9%), but on Z1 the difference from `naive` is not significant
  (Wilcoxon p = 0.87).
- **Where belief beats naive:** it gains most on small and medium instances (gap 1.4% / 3.3% vs
  3.4% / 4.4%). On large instances `naive` is slightly better (3.5% vs 4.4%), because the Markov
  prediction makes every unreported road look a little worse.
- **Demand noise:** the only run with unmet demand used the `naive` planner, which plans with the
  reported demand alone. The belief planners check capacity at mean + 1 sd and had none.
- **The chance constraint does not pay off** in this model. It is significantly worse than plain
  belief (+0.6% Z1, p = 0.009). Closed roads are modelled as 5× detours, not as blocked roads, so
  avoiding a merely *uncertain* road costs more than the risk it removes. It would matter if
  closures could cut a satellite off.

### 4.4 DRL operator controller (`results/dynamic_selectors.csv`)

**Setup** (`scripts/run_selector_experiment.py`):
- **Training:** the Double DQN was trained for 30 min (1.15 M decisions) on the 14 training
  instances.
- **Paired test:** on every residual problem of 12 test instances × 4 scenarios × 2 seeds × 3
  epochs, `uniform`, `adaptive` and `dqn` each solved the **same** problem. They started from the
  same solution with the same seed and the same CPU budget (576 paired problems per budget).

| budget per re-plan | DQN vs adaptive | DQN vs uniform | adaptive vs uniform |
|---|---|---|---|
| 0.25 s | win 21% / loss 24% / tie 55%; mean −0.03% (p = 0.11) | win 19% / loss 23%; −0.00% (p = 0.23) | win 17% / loss 22%; +0.04% (p = 0.37) |
| 1.0 s | win 16% / loss 24% / tie 60%; mean **+0.07%** (p = 0.025) | win 17% / loss 23%; +0.01% (p = 0.18) | win 16% / loss 21%; −0.06% (p = 0.64) |

**Honest conclusion: the learned controller does not beat the simple baselines yet.**
- At 0.25 s none of the three differs significantly from the others.
- At 1 s the DQN is very slightly worse than the adaptive roulette.
- Most residual problems end in a **tie** (55–62%): after an event they are small, so every
  operator mix finds the same solution. Operator choice can only matter on the larger ones
  (n ≥ 20), and even there the differences are within ±0.2%.
- This matches the proposal's own risk analysis: the value of DRL has to be shown, not assumed.
  The next steps in §6 list what would give it more room.

**Battery.** In all 288 POMDP runs the highest sortie energy was 99.9% of usable battery: no
sortie exceeded it.

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
   is handled by **re-delivery recourse** (§2).
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

1. **Give the DRL controller more room.** It is implemented and trained, but it does not yet beat
   the roulette (§4.4). Options:
   - larger problems and more frequent events, so residual problems are not tiny;
   - a richer action set, for example a destroy ratio as a continuous action (PPO/SAC, spec §7.3);
   - a reward on hypervolume change rather than on single moves (spec eq. 12);
   - a longer training run.
2. **Better belief model.** The Markov prediction makes every unreported road look worse, which
   hurts on large networks (§4.3). Learning the transition matrix from the reports would help, as
   would drone reconnaissance of uncertain roads.
3. **Blocked roads.** With true closures (a satellite can be cut off), the chance constraint
   becomes meaningful. This needs the 1-to-many satellite service of guide §1.3, option 2.
4. **Deprivation objective.** Replace $f_1$ by the spec's $\sum_i w_i h_i[T_i-r_i]^+$, with
   $r_i$ = the event time for surge demand.
5. **MOEA population.** Use NSGA-II or MOEA/D over the same `ReplanProblem.evaluate` as the spec
   §8.3 baselines.
6. **Exact reference.** A dynamic MILP / ε-constraint model on M-08-05 and M-09-09 to validate the
   empirical Pareto fronts (spec §8.4).
