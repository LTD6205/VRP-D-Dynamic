# Implementation guide: from the static 2E-CTVRP to the dynamic truck–drone relief model

This guide explains **everything implemented in the code**, starting from the original static
repository and ending with the latest additions:
- drone routes that visit each victim;
- the Solomon-based benchmark;
- the hybrid repair strategy;
- the dynamic-degree sweep;
- the plan-vs-actual gap;
- drone setup times.

It is written to be read top to bottom. Each part says
- **what** the code does;
- **where** it is (file, class, function);
- **why** it was done that way.

Companion documents:
- [`FORMULATION.md`](FORMULATION.md): every variable, formula and constraint as mathematics.
- [`INITIAL_FORMULATION.md`](INITIAL_FORMULATION.md): the proposal-level model.
- [`../README.md`](../README.md): results.
- [`../README_STATIC.md`](../README_STATIC.md): the original static repository.

---

## Contents

0. [Map of the code](#0-map-of-the-code)
1. [Starting point: the static 2E-CTVRP repository](#1-starting-point-the-static-2e-ctvrp-repository)
2. [The dynamic problem and the closed loop](#2-the-dynamic-problem-and-the-closed-loop)
3. [Data: instances, event files, the Solomon-based set](#3-data-instances-event-files-the-solomon-based-set)
4. [The environment: roads, observations, demand (`dynamic_env.py`)](#4-the-environment-roads-observations-demand)
5. [Drone battery (`drone_energy.py`)](#5-drone-battery)
6. [Second echelon: how drones serve a cluster (`drone_service.py`)](#6-second-echelon-how-drones-serve-a-cluster)
7. [The simulator: executing, freezing, demand rule, recourse (`dynamic_sim.py`)](#7-the-simulator)
8. [The re-optimizer (`dynamic_solver.py`)](#8-the-re-optimizer)
9. [The closed loop and the strategies (`dynamic_runner.py`)](#9-the-closed-loop-and-the-strategies)
10. [DRL operator controller (`drl_selector.py`)](#10-drl-operator-controller)
11. [Scripts: how to run everything](#11-scripts-how-to-run-everything)
12. [Verification and tests](#12-verification-and-tests)
13. [Symbol ↔ code glossary](#13-symbol--code-glossary)

---

## 0. Map of the code

```
src/
  common.py            STATIC (original): instance parser, k-means clustering, distances, Z1,
                       neighbourhoods N1-N7, construction/repair shared by the static heuristics
  grasp_ils_pr.py      STATIC: the paper's GRASP-ILS-PR
  grasp_evpr.py, sa.py STATIC: baselines;  milp.py: exact model (Gurobi)
  instance_generator.py, fleet_metrics.py   STATIC

  dynamic_env.py       NEW  roads (3 states), POMDP belief, demand/priority process, event files
  drone_energy.py      NEW  speed- and load-dependent drone battery
  drone_service.py     NEW  2nd echelon: CentroidService (paper) | RouteService (drones visit victims)
  dynamic_sim.py       NEW  plan execution on the true network, freezing, demand rule, recourse, KPIs
  dynamic_solver.py    NEW  residual problem + multi-objective re-optimizer + operator selectors
  dynamic_runner.py    NEW  closed loop, strategies static / restart / warm / hybrid
  mo_utils.py          NEW  Pareto archive, hypervolume, weight vectors
  drl_selector.py      NEW  Double DQN operator controller (numpy)

scripts/
  generate_dynamic_dataset.py   events files for an instance set
  generate_solomon_instances.py Solomon VRPTW -> 2-echelon instances
  run_dynamic.py                one instance x scenario
  run_dynamic_experiments.py    benchmark batches (all extension options)
  run_pomdp_experiment.py       planners under noisy/partial observations
  run_selector_experiment.py    uniform / adaptive / DQN operator selection
  train_drl_selector.py         trains the DQN
  summarize_dynamic.py          tables vs. static

instances/
  main/, large/                 original instances (unchanged)
  dynamic_events/               events for the 12 test instances
  dynamic_events_train/         events for 14 training instances (DQN only)
  solomon/raw/                  4 Solomon files (C1, C2, R, RC geometries)
  solomon/                      16 Solomon-based instances + frozen clusters
  solomon_events/               events for the Solomon-based instances
tests/test_drone_service.py     checks of the drone service models
```

Every static file was left as it was, except two minimal additions:
- `instances/solomon` was added to the list of folders searched for frozen clusters
  (`common._FROZEN_CLUSTER_DIRS`);
- the original README was renamed `README_STATIC.md`.

The static heuristics and the MILP behave exactly as published.

---

## 1. Starting point: the static 2E-CTVRP repository

**Problem (Nguyen, Piplani & Gunawan).** After a disaster, victims cannot be reached by road.
- Trucks leave the depot, visit **satellites**, and return.
- At each satellite the truck's **U drones** fly the supplies to one **cluster** of victims.
- The truck waits until the drones are back, then drives on.
- Every satellite is visited once, and every cluster is served by exactly one satellite.

**Instance file** (`instances/main/M-09-09-60-1.txt`, parsed by `common.parse_instance`):

```
n m K U Q P v_t v_d          9 satellites, 780 victims, 2 trucks, 3 drones/truck, Q=12500, P=1000, 40, 60 km/h
x0 y0                        depot
x y            (n lines)     satellites
x y demand     (m lines)     victims
```

Coordinates are in units of 10 m. `compute_distances` converts them to km (×10/1000) and to
hours with the vehicle speeds.

**Clustering.** `common.cluster_victims` groups the m victims into **n clusters** with k-means.
- The result is frozen in `<name>.clusters.json` so every script uses the same clusters.
- Each cluster has a **centroid** and a demand $d_c=\sum_{v\in c} q_v$.
- **Victims never appear again in the static model.** Drones fly to the centroid only.

**Times.**
- Truck: $t_{ij}$ on the **Manhattan** distance between depot and satellites (`time_n_0`).
- Drone: $t'_{sc}$, one-way on the **Euclidean** distance satellite → centroid (`time_n_m`).

**Objective Z1** (`ECTVRPOperators.sum_arr_time`): the sum of truck arrival times at
satellites, where each stop's dwell is the drone round trip $2t'_{s\sigma(s)}$, plus the return
times.

**Decisions.**
- Truck routes.
- $\sigma$: which cluster each satellite serves. It is a bijection.

**Method.** GRASP-ILS-PR:
- randomized construction;
- neighbourhoods N1–N7 (relocate, swap, 2-opt, inter-swap, 2-opt*, inter-relocate, centroid
  swap);
- iterated local search;
- path relinking on $\sigma$.

The MILP (`milp.py`) gives exact solutions for small instances.

> **What the dynamic code re-uses:** the parser, the clustering, the distance matrices and the
> neighbourhood ideas. At t = 0 with the paper's objective, the dynamic solver reproduces the
> paper's GRASP-ILS-PR best values (README §3).

---

## 2. The dynamic problem and the closed loop

The dynamic version (`bai_toan_iii_truck_drone.pdf`, the research guide) adds:
- **road damage** over time;
- **demand surges** and **priority changes**;
- **imperfect information** about roads and demand;
- **drone battery** limits that depend on speed and load;
- **re-planning** while trucks are already on the road.

Re-planning happens in a closed loop (rolling horizon):

```
t = 0 : read scenario epoch 0 -> build residual problem (= whole problem) -> solve -> plan P0
for each event epoch e (t1, t2):
    1. simulate the current plan on the TRUE network up to t_e      (dynamic_sim.Simulator.simulate)
    2. freeze history: everything reached by t_e is locked           (Simulator.commit)
    3. roads change; the planner's view is updated (belief / reports) (RoadNetwork)
    4. demand surges are admitted under the demand rule              (admissible_surges)
    5. demand already delivered but now short -> re-delivery jobs    (create_residual_jobs)
    6. build the residual problem from the snapshot                  (ReplanProblem)
    7. re-plan with a strategy (static / restart / warm / hybrid)    (DynamicReoptimizer)
    8. merge the frozen prefix + new suffix, check invariants        (merge, check_plan)
finally: execute the last plan to the end and measure realized KPIs (realized_metrics)
```

The **planner** works on its *belief*: expected travel times and reported demand. The
**simulator** executes on the *truth*. All reported results are realized values.

---

## 3. Data: instances, event files, the Solomon-based set

### 3.1 Event files (`scripts/generate_dynamic_dataset.py`)

For each instance, an `<name>.events.json` file holds several **scenarios**:
- `damage_10pct`, `damage_30pct`, `damage_50pct`;
- `damage_30pct_partial`, where only 60% of road changes are reported.

Each scenario has three epochs: t = 0, t₁ and t₂. How they are generated (`generate()`):

1. **Battery calibration** (`calibrate_battery`). Use the default energy model. If no
   battery-feasible satellite↔cluster matching exists at 1.3× demand, raise the battery
   capacity by bisection until one does. The result is stored in the `drone_energy` block.
2. **Event times.**
   - Solve a deterministic baseline plan (`baseline_arrivals`).
   - Set t₁ and t₂ to the **30% and 65% quantiles** of its satellite arrival times.
   - This way each event splits served from unserved satellites. The guide's fixed 0.40 h /
     0.85 h fell before the first arrival.
3. **Road damage.**
   - Pick a flood centre at an inner satellite that is still unserved at t₁.
   - Damage first-echelon arcs in order of closeness to it. Half of the arcs are damaged at t₁,
     the rest at t₂.
   - Each damaged arc becomes `degraded` (×2.5) or `closed_detour` (×5). Closer arcs are more
     likely to become detours.
   - 30% of the damage is one-way.
   - At t₂, 40% of degraded arcs escalate to detours.
4. **Demand surges.**
   - At each event, about 15% of clusters surge: the unserved clusters nearest the flood.
   - Multipliers are 1.10–1.20 at t₁ and 1.15–1.30 at t₂.
   - Surged clusters also get a priority raise to 1.0 / 0.95. Initial priorities are drawn from
     U(0.50, 0.85).
5. A `_meta` block records the seed, the baseline arrival quantiles and the battery note.

Sets:
- **Test set:** 12 instances (`SUBSET` in the script), in `instances/dynamic_events/`.
- **Training set** for the DQN only: 14 different instances, seed 7, in
  `instances/dynamic_events_train/`.

### 3.2 Solomon-based set (`scripts/generate_solomon_instances.py`, new)

This is a second benchmark built from the classic Solomon (1987) VRPTW instances. The 100-customer
files are vendored in `instances/solomon/raw/`, taken from the public copy
CervEdin/solomon-vrptw-benchmarks.

| Solomon | becomes |
|---|---|
| customer (x, y, demand) | a **victim**. Coordinates ×10, so 1 Solomon unit = 100 m and the area is 10 km × 10 km. Demand is kept. |
| depot | the truck depot |
| k-means groups of customers | n relief **clusters**, frozen in `<name>.clusters.json` |
| — | **satellites**: one per cluster, placed 1.0–2.5 km from its centroid in a random direction. The matching σ is a real decision. |
| time windows | **dropped** (see below) |
| service times | dropped; the drone drop time is a model parameter |

**Why the time windows are dropped.** Inside one geographic cluster the Solomon windows spread
over most of the day. In C101, each window is about 60 time units wide, but a cluster's windows
cover 690–1,110 of a 1,236-unit day. They were designed for one vehicle visiting customers one by
one, so a single truck stop at a satellite cannot satisfy them.

**Fleet.**
- K = 2 trucks for n = 8 clusters, K = 3 for n = 12.
- Q = ⌈1.25·D / K⌉. Solomon's own Q = 200 is about one cluster per truck.
- U = 3 drones, P = 60, so 2–4 victims per drone trip.
- Speeds 40 / 60 km/h.

**Instances.** There are only **4 distinct geometries**: C1, C2, R (R1 = R2) and RC (RC1 =
RC2). Within a class, the Solomon series differ only in their time windows. The set is
{C1, C2, R, RC} × n ∈ {8, 12} × satellite seed ∈ {1, 2} = **16 instances**, named
`S-<geom>-<n>-<seed>`. Their events (`instances/solomon_events/`) come from the same generator
as §3.1. The battery never binds in this compact 10 km area (0% infeasible pairs).

```bash
python3 scripts/generate_solomon_instances.py
python3 scripts/generate_dynamic_dataset.py --instances solomon/S-C1-08-1,...  --out instances/solomon_events
```

---

## 4. The environment: roads, observations, demand

`src/dynamic_env.py`

**Road states.** Every first-echelon arc (depot ↔ satellites, satellite ↔ satellite) is in one
of three states: Open, Degraded or Detour. The travel-time multipliers are
$\phi=(1, 2.5, 5)$ (`PHI`), so $t_{ij}(t)=t^0_{ij}\,\phi_{s_{ij}(t)}$.
- A closed road is modelled as a long detour, not removed. Every satellite stays reachable and
  σ stays a bijection.
- An arc's time is fixed at the **departure** time (`RoadNetwork.true_time`). A truck already on
  a road finishes it at the old speed.

**Truth vs. belief** (`RoadNetwork`). The network keeps two views:
- `true_state` (truth) and `belief[i, j, s]` (the planner's probability of each state);
- `naive_state`, what a planner that trusts every report would believe.

At an event (`apply_edge_updates`):
- With full observability (`reveal_prob = obs_accuracy = 1`) or at t = 0, the belief equals the
  truth.
- Otherwise:
  1. **prediction** for every arc: $b \leftarrow bP$, with the Markov matrix `DEFAULT_TRANSITION`;
  2. each changed arc is **reported** with probability `reveal_prob`;
  3. a report is correct with probability `obs_accuracy`, otherwise it names a wrong state;
  4. **Bayes update**: $b'(s)\propto O(o\mid s)\,b(s)$.
- `observe_traversed`: an arc a truck has driven is known exactly.

**How the planner reads it** (`planning_time(mode)`):

| mode | arc time used by the optimizer |
|---|---|
| `oracle` | truth (upper bound) |
| `naive` | reports at face value |
| `belief` | expected time $t^0\sum_s\phi_s b(s)$ |
| `belief_cc` | belief, but arcs with $P(\text{Open})+P(\text{Degraded})<\theta$ are planned as detours (chance constraint) |

**Demand process** (`DemandProcess`). Demand $d_c(t)$ and priority $w_c(t)$ per cluster are
piecewise constant. `apply_increments` adds the surge units allowed by the demand rule (§7.3).

**Scenario loading** (`load_scenarios`) reads the events JSON into `Scenario` / `EpochEvent`
objects. This includes the `drone_energy` block, as a `DroneEnergyModel`.

---

## 5. Drone battery

`src/drone_energy.py`

Battery use depends on **speed** v, **load** ℓ and **flight time**. Power is expressed as a
fraction of a full battery per hour:

$$p(v,\ell)=a\Big(1+\beta\frac{\ell}{P}\Big)+b\Big(\frac{v}{v_{ref}}\Big)^3+k_w\,\text{wind}^2,
\quad a=\frac{\text{weight\_share}}{t_{empty}},\; b=\frac{1-\text{weight\_share}}{t_{empty}}$$

Defaults:
- $v_{ref}$ = 60 km/h;
- $t_{empty}$ = 0.65 h, so an empty drone flies 39 min at $v_{ref}$;
- weight_share = 0.6;
- β = 0.8, i.e. +80% weight power at full payload;
- reserve ρ = 15%.

Usable battery is $E(1-\rho)$.

- The **weight term** grows with the payload.
- The **speed term** grows with $v^3$, while the time to fly a distance falls with $1/v$. Range
  therefore peaks near $v_{ref}$.

**Centroid model** (one out-and-back sortie with load ℓ out, empty back):
$e = t'\,[p(v,\ell)+p(v,0)] \le E(1-\rho)$ (`sortie_energy`, `DroneBattery.pair_ok`). The
per-drone load is $\ell=d/(U\cdot\text{waves})$.

The route model uses the same power function leg by leg (§6.2).

---

## 6. Second echelon: how drones serve a cluster

`src/drone_service.py` (new)

The truck parks at satellite s and its U drones serve cluster c with demand d. The truck waits
until the last drone is back. The first echelon (simulator and optimizer) only needs four
numbers, returned by **one call**:

```python
service(s, c, d) -> DroneServiceResult(wait, offset, use, trips)
```

| field | symbol | meaning |
|---|---|---|
| `wait` | $W_{sc}(d)$ | truck dwell at s, i.e. the drones' makespan |
| `offset` | $\delta_{sc}(d)$ | delivery time after the truck's arrival, used in $f_1$: $e_s=a_s+\delta_{sc}$ |
| `use` | $u_{sc}(d)$ | battery share of the most demanding trip; feasible iff $\le 1$ |
| `trips` | | number of drone trips (waves) |

`tables(d_mean, d_cap)` returns these as n×n matrices:
- wait and offset at the planned demand;
- battery use and feasibility at the robust demand, mean + z·sd (§8.1).

Two models implement the interface, so the rest of the code does not care which one runs. The
command-line option is `--drone-model centroid|route`.

### 6.1 `CentroidService`: the paper's model (default)

All of a cluster's demand is delivered at its **centroid**. A wave sends the U drones out and
back together:

$$\text{waves}=\Big\lceil\frac{d}{UP}\Big\rceil,\qquad
W=\text{waves}\,(S_L+2t'_{sc}+S_R),\qquad \delta=S_L+t'_{sc}$$

- With setup times $S_L=S_R=0$ this is **exactly** the earlier model, checked byte-for-byte
  (§12).
- `split=True` (used when executing): if a locked pairing became too heavy for the battery after
  a surge, the operators fly extra, lighter waves instead of over-flying the battery.

### 6.2 `RouteService`: drones visit every victim (new)

Here the second echelon is a **multi-trip CVRP** per cluster. Victims with zero demand are
skipped. Every victim with demand gets its parcels. The method is route-first, cluster-second:

1. **Giant tour** (`_tsp_tour`): one TSP tour over the cluster's victims (nearest neighbour +
   2-opt). It is computed once per cluster, then rotated so that it starts at the victim
   nearest to the satellite.
2. **Exact split** (Prins 2004) of the tour into consecutive trips $i..j$, minimizing total drone
   busy time. The DP is $V_{j+1}=\min_i V_i+\text{busy}(i,j)$. A trip is allowed iff
   - **payload:** $\sum_{k=i}^{j} q_k\le P$;
   - **battery:** $e_{ij}\le E(1-\rho)$, with the energy computed **leg by leg**. The drone takes
     off with the whole trip load and gets lighter after every drop:

   $$e_{ij}=a_i\,p(L_{ij})+\sum_{k=i}^{j-1} g_k\,p\Big(\sum_{l=k+1}^{j}q_l\Big)+a_j\,p(0)
   +\tau\sum_{k=i}^{j}p\Big(\sum_{l=k}^{j}q_l\Big)$$

   Here:
   - $a_k$ is satellite↔victim flight time;
   - $g_k$ is victim→next-victim time;
   - $\tau$ is hover time per drop;
   - $p(\ell)=A+B\ell/P$, since the power model is linear in load.

   With prefix sums of demand and tour time, every $e_{ij}$ for fixed i is computed in one
   vectorised step.
3. **Schedule on U drones.**
   - Trips are assigned longest-first to the least busy drone (LPT).
   - Inside each drone, trips run shortest-first, so early deliveries are not held up by long
     trips.
   - $W$ = the makespan.
   - $\delta$ = the demand-weighted mean drop time:
     $\delta=\sum_v q_v t_v/\sum_v q_v$.

**Busy time of a trip:** $S_L+(a_i+\text{tour}+a_j)+n_{ij}\tau+S_R$, where $n_{ij}$ is the
number of victims on it.

**Demand changes.**
- A surge to d scales every victim's demand by $d/d_c(0)$: the whole shelter area needs more.
- A re-delivery job of amount r delivers $r/d_c(0)$ of each victim's original demand.
- A victim heavier than P is split into several full loads.

**Feasibility.** A pair (s, c) is battery-infeasible iff some victim cannot be served even
alone.

**Speed.** Results are cached per (s, c, demand scale). The base-demand table is saved to
`instances/.drone_cache/` (git-ignored). A full n×n table takes 0.4 s for n = 9 and 15 s for
n = 50, once.

`RouteService.plan(s, c, d)` returns the full drone plan for inspection: drone, start time,
victims, load and energy of each trip.

### 6.3 Setup times (idea D, Kuo et al. 2022)

`SetupTimes(launch, recover, drop)` is applied per trip in both models.
- **launch** $S_L$: loading and take-off clearance.
- **recover** $S_R$: landing and battery swap.
- **drop** τ: hover time to release a parcel, route model only.

Defaults for the "+setup" experiments are `SetupTimes.DEFAULT_MIN = (1, 2, 0.1)` minutes, set with
`--setup 1,2,0.1`.

**How much the victim-level model changes things.** Median truck dwell over all
satellite–cluster pairs, at base demand:

| model (S_L, S_R, τ in minutes) | M-09-09-60-1 (≈80 victims/cluster): W (h) | trips | S-R-08-1 (≈12 victims/cluster): W (h) | trips |
|---|---|---|---|---|
| centroid (0, 0, 0) = paper | 0.373 | 1 | 0.122 | 1 |
| centroid (1, 2, 0) | 0.423 | 1 | 0.177 | 1 |
| route (0, 0, 0) | 0.476 | 2 | 0.177 | 3 |
| route (1, 2, 0) | 0.526 | 2 | 0.232 | 3 |
| **route (1, 2, 0.1)**, default | **0.556** | 3 | **0.238** | 3 |
| route (1, 2, 0.25) | 1.087 | 4 | 0.250 | 3 |
| route (1, 2, 0.5) | 1.148 | 6 | 0.269 | 3 |

The centroid shortcut underestimates the drones' time:
- **+28%** from flying to every victim instead of the centroid;
- **+49% (main) to +95% (Solomon)** with realistic setup times.

When a cluster has many victims, hover time per drop quickly becomes the dominant parameter:
battery-limited trips get shorter, and more of them are needed.

---

## 7. The simulator

`src/dynamic_sim.py`

### 7.1 Plan representation

- `Plan.routes[k]`: the ordered stops of truck k. A stop is a satellite (1..n) or a re-delivery
  job (id > n+1).
- `Plan.sigma[s] = c`: the satellite–cluster matching.
- `Plan.jobs[j] = Job(cluster, amount, host, created)`: re-delivery jobs.

### 7.2 Execution (`Simulator.simulate`)

For each truck, stop by stop:

$$a_p=a_{p-1}+w_{p-1}+t^{true}_{prev,cur}(\text{departure}),\qquad
(w,\delta,u,\text{trips})=\texttt{service}(loc,c,d_c(a_p))$$

- The truck delivers $\min(d,\ Q-\text{load})$.
- The record (`SatRecord`) stores arrival, wait, delivery time $a+\delta$, demand at arrival,
  delivered amount, trips, priority, battery use and host.
- A job stop is executed at its **host** satellite.

### 7.3 Freezing history (`Simulator.commit`)

At event time $t_e$:
- every stop reached by $t_e$ is **locked**;
- a truck driving towards a stop is committed to it, so that stop is locked too;
- a truck already heading to the depot is **retired**;
- each active truck restarts from its last locked stop at time arrival + wait.

The snapshot (`Snapshot`) holds:
- the locked stops;
- the free satellites and free clusters;
- the start node and time of each truck;
- the arcs driven so far, which reveal their true state.

### 7.4 Demand rule (`admissible_surges`)

A cluster's demand may only grow **before or during** its service, and never beyond the serving
truck's spare capacity:
- if the truck has **left** the cluster's satellite, the surge is dropped;
- if the satellite is not yet reached, or the truck is parked there with drones flying, the surge
  is allowed;
- the granted amount is capped at $\text{slack}_k=Q-(\text{delivered}+\text{planned})$.

So the plan in force always stays capacity-feasible.

### 7.5 Recourse (`create_residual_jobs`)

A cluster already reached but still short becomes a **re-delivery job** for the residual. This
happens after a surge during service, or after a shortfall caused by noisy demand reports. A
truck with goods left serves the job from any battery-feasible **host** satellite. If the truck
is still parked at the cluster's own satellite, the job is just extra drone trips there.

### 7.6 Realized KPIs (`realized_metrics`)

- $f_1=\frac1D[\sum_{stops}w_c\,q\,e+\sum_c w_c\,\text{unmet}_c\,C_{ret}]$: priority-weighted
  delivery time. Unmet demand is charged at the last truck's return time, so leaving people
  unserved never looks better than serving them late.
- $f_2=\max a-\min a$: arrival spread (equity).
- $f_3=Z_1=\sum a+\sum \text{ret}$: the paper's objective, over main satellite visits.
- Also reported:
  - `unmet_ratio`, `n_redeliveries`;
  - `max_battery_use`, `battery_violations`;
  - `C_del`, `C_ret`.

---

## 8. The re-optimizer

`src/dynamic_solver.py`

### 8.1 Residual problem (`ReplanProblem`)

Built from the snapshot at an epoch:
- **Free part:** the free satellites, the free clusters, the open jobs, and the active trucks
  with their start node, start time and remaining capacity $Q_k(t)$.
- **Arc times:** the planner's matrix $T$ from `planning_time`.
- **Demand:** the planned demand $d$ (reported or believed) and a robust demand
  $d^{cap}=\mu+z\sigma$. Capacity and battery are checked on $d^{cap}$.
- **Drone tables** from the service (§6):
  - `Wl`, `OFFl`: dwell and offset at $d$;
  - `pair_ok`: battery feasibility at $d^{cap}$.
- **Job tables:** for each job and each host, `job_svc[(j, h)] = (W, δ)`, plus `job_hosts[j]`,
  the battery-feasible hosts.
- **Locked part:** enters the objectives as **constants**. These are the arrival sum, min, max
  and $f_1$ numerator, so values are comparable across epochs.

**`evaluate(routes, sigma, hosts)`** walks each active truck and returns
`(violation, (f1, f2, f3))`:
- satellite stop: $t\mathrel{+}=T_{prev,s}$, add $a_s$ to $f_3$, $f_1\mathrel{+}=w_c d_c(t+\delta_{sc})$,
  then $t\mathrel{+}=W_{sc}$;
- job stop: drive to the host, add to $f_1$, wait $W$. It is not counted in $f_2$ or $f_3$
  itself, but it delays every later stop;
- violations:
  - capacity overload;
  - `BATTERY_PENALTY` for an infeasible pair;
  - `EMPTY_TRUCK_PENALTY` for a depot truck with no stop.

### 8.2 Search (`DynamicReoptimizer.solve`)

This is the paper's GRASP-ILS-PR skeleton, extended to the dynamic multi-objective case.

- **Start.**
  - `construct()`: randomized construction. σ is matched with $P\propto1/t'$ among
    battery-feasible pairs, with an assignment-problem fallback. Satellites and jobs are added by
    randomized cheapest insertion.
  - `warm_start(plan)`: keep the previous plan's suffix and σ (re-matched only where
    infeasible), keep job hosts, and insert anything missing.
- **Moves** (`OPERATORS`, one random move each):
  - `relocate`, `swap`, `two_opt` (N1–N3);
  - `inter_swap`, `two_opt_star`, `inter_relocate` (N4–N6);
  - `drone_swap` (N7, swaps two satellites' clusters);
  - `destroy_repair`;
  - `job_host` (move a re-delivery job to another feasible host).
- **Segment focus.** Under warm and hybrid, a move's truck is sampled ∝ the disruption score
  $c_k(t)$ (`segment_disruption`): road, demand and timing change of its route suffix.
- **Operator selector.**
  - `UniformSelector`;
  - `AdaptiveSelector` (ALNS roulette rewarded 5 / 2 / 1 for new best / improved / new archive
    point);
  - `DQNSelector` (§10).
- **Periodic steps:**
  - best-of-K **local search** every `local_search_freq` iterations;
  - **path relinking** on σ towards an elite σ pool (kept across epochs, projected with
    `project_sigma`) every `pr_freq` iterations.
- **Multi-objective.** Every feasible candidate is offered to a **Pareto archive**
  (`mo_utils.ParetoArchive`, crowding truncation). The scalarizing weight cycles through
  Das–Dennis vectors (`simplex_weights`) when stuck.
- **Single-objective** (`mode="so"`): weights fixed; GRASP-style restart when stuck.
- **Commit.** The archive point that is best under the commit weights.

---

## 9. The closed loop and the strategies

`src/dynamic_runner.py`, `run_closed_loop(world, strategies, cfg0, cfg, ...)`

Setup:
1. `load_world`: instance, frozen clusters, distances, scenario.
2. Build the `DroneBattery` and the **drone service**
   (`make_service(drone_model, ..., setup)`).
3. Solve epoch 0 once. All strategies share this t = 0 plan, so differences come only from how
   they react.

Per strategy and per event, the steps of §2 are carried out with these options:

| option | effect |
|---|---|
| `planner` | how observations are read (§4) |
| `reveal_prob`, `obs_accuracy`, `demand_noise`, `z_cap` | observation model and robust demand |
| `recourse` | create re-delivery jobs (re-planning strategies only) |
| `drone_model`, `setup` | second echelon (§6) |
| `surge_scale` (**B**, dynamic degree) | every surge multiplier $m\to1+s(m-1)$; 0 = no demand dynamics |
| `road_events=False` | ignore road damage (demand dynamics only) |
| `hybrid_tol` | repair threshold of `hybrid`; `hybrid@0.05` sets it per strategy |

**Strategies:**
- `static`: the t = 0 plan, never re-planned and with no recourse. It is executed on the
  disrupted network.
- `restart`: re-solve every epoch from a fresh construction, with no memory.
- `warm`: warm start, plus the projected Pareto archive, the elite σ pool and segment focus.
- `hybrid` (**A**, new; Han et al. 2023's "insert if it fits, otherwise re-optimize"):
  1. **Repair:** `warm_start` = keep the previous plan, insert the new jobs at their cheapest
     positions, and re-match only pairings that became battery-infeasible.
  2. Evaluate it under the new information:
     $\text{gap}=\sum_i w_i\,(o^{rep}_i-o^{prev}_i)/|o^{prev}_i|$. Here $o^{prev}$ is what the
     plan in force predicted before the event, so the gap is **the event's impact on the
     current plan**.
  3. If the repair is feasible and gap ≤ `tol` (default 2%), commit it with **no search**
     (milliseconds). Otherwise run the full warm re-optimization.

  The run records `repaired`, `repair_gap` and `repair_feasible` per epoch, and `n_repaired`.

After the last event the plan is executed to the end. Besides §7.6, the metrics include:
- `dynamic_degree` (**B**): granted surge demand ÷ final total demand. This is the realized
  share of demand that appeared after t = 0.
- **Plan-vs-actual gap** (**C**, Imran & Won 2024):
  - `gap0_f1`, `gap0_f3` = (predicted at t = 0 − realized) / realized;
  - `gap_last_*` = the same for the last plan in force.

  `gap0` measures how wrong the dispatch-time plan was about the day. `gap_last` measures the
  final plan; it is 0 under full information.
- `drone_model`, `mean_wait` (mean truck dwell), `total_drone_trips`, `replan_cpu_s`.

`check_plan` asserts after every re-plan:
- every satellite is visited once;
- σ is a bijection;
- every job is routed once;
- locked history is unchanged (re-simulated arrivals identical).

---

## 10. DRL operator controller

`src/drl_selector.py`, `scripts/train_drl_selector.py`

A Double DQN in plain numpy picks which operator to apply for the next 10 moves (a macro action).
- **State:** 23 features: epoch context, search progress, and the last operator as one-hot.
- **Reward:** the summed 5 / 2 / 1 signals.
- **Network:** MLP 64-64 with Adam, experience replay, a target network and the Double-DQN
  target.
- **Training** is on the training set; the trained model is `models/dqn_selector.npz`.
- **Result:** it does not beat the adaptive roulette (README §4.4).

The literature notes (`claude/vrpd_literature_notes.md` in the project) explain why. Chen,
Ulmer & Thomas use RL for *decisions with consequences*, such as which vehicle serves a request,
not for operator choice. That is suggested as the next step.

---

## 11. Scripts: how to run everything

```bash
# one run (prints realized f1/f2/f3, unmet, re-deliveries, drone trips, repaired epochs)
python3 scripts/run_dynamic.py --instance instances/main/M-09-09-60-1.txt --scenario damage_30pct \
    --mode so --weights 0,0,1 --time 1.5 --strategies static,warm,hybrid \
    --drone-model route --setup 1,2,0.1

# Solomon-based instance (events are found in instances/solomon_events automatically)
python3 scripts/run_dynamic.py --instance instances/solomon/S-RC-08-1.txt --drone-model route --setup 1,2,0.1

# benchmark batches (CSV, one row per strategy)
python3 scripts/run_dynamic_experiments.py --mode so --weights 0,0,1 --seeds 0,1 \
    --scenarios damage_30pct,damage_50pct --strategies static,restart,warm,hybrid,hybrid@0.05 \
    --drone-models centroid,centroid+setup,route+setup --workers 2 --output results/ext_drone_main.csv
#   add --events-dir instances/solomon_events for the Solomon set
#   dynamic-degree sweep: --no-road-events --surge-scales 0.5,1,2,4 --strategies static,warm,hybrid

python3 scripts/run_pomdp_experiment.py --scenarios damage_30pct --seeds 0,1 --output results/ext_pomdp_gap.csv
python3 scripts/run_selector_experiment.py --budgets 0.25,1.0 --output results/dynamic_selectors.csv
python3 scripts/train_drl_selector.py --minutes 25 --output models/dqn_selector.npz
python3 scripts/summarize_dynamic.py results/dynamic_so_z1.csv
python3 tests/test_drone_service.py
```

Result files:

| file | experiment |
|---|---|
| `dynamic_so_z1.csv`, `dynamic_so_z1_tight.csv`, `dynamic_mo.csv` | strategies, single objective / tight budget / multi-objective |
| `dynamic_pomdp.csv` | planners under partial information |
| `dynamic_selectors.csv` | operator selectors |
| `ext_drone_main.csv`, `ext_drone_solomon.csv` | drone model × strategy (incl. hybrid), main and Solomon sets |
| `ext_dod_main.csv`, `ext_dod_solomon.csv` | dynamic-degree sweep |
| `ext_pomdp_gap.csv` | plan-vs-actual gap per planner |

---

## 12. Verification and tests

- **Static reproduction:** at t = 0 with Z1, the dynamic solver matches the paper's best values
  (README §3).
- **Refactor identity.** After the drone-service layer was introduced, saved plans were
  re-simulated and re-evaluated with the old and new code. The output was **byte-identical**
  (centroid model, zero setup).
- **`tests/test_drone_service.py`:**
  - the centroid service equals the paper's formulas;
  - route plans serve every victim with demand exactly once;
  - every trip respects P;
  - each trip's energy equals a brute-force leg-by-leg recomputation (to 1e-9);
  - the makespan equals the busiest drone's total.
- **Invariants after every re-plan:** `check_plan` (§9).
- **Battery:** realized `battery_violations` is reported in every CSV.

---

## 13. Symbol ↔ code glossary

| symbol | meaning | code |
|---|---|---|
| $n$, $K$, $U$, $Q$, $P$ | satellites/clusters, trucks, drones per truck, truck and drone capacity | `inst.n`, `inst.no_k`, `inst.no_u`, `inst.q`, `inst.p` |
| $t^0_{ij}$, $t_{ij}(t)$ | base / current truck time | `dist.time_n_0`, `RoadNetwork.true_time` |
| $t'_{sc}$ | drone time satellite → centroid | `dist.time_n_m` |
| $\phi$, $b_{ij}(s)$ | state multipliers, belief | `PHI`, `RoadNetwork.belief` |
| $\sigma$ | satellite → cluster matching | `Plan.sigma`, `Sol.sigma` |
| $W_{sc}$, $\delta_{sc}$, $u_{sc}$ | dwell, delivery offset, battery use | `service().wait/.offset/.use`, `ReplanProblem.Wl/OFFl/energy` |
| $S_L, S_R, \tau$ | launch, recovery, drop time | `SetupTimes.launch/recover/drop` |
| $e_{ij}$ | trip energy | `RouteService._solve` (`energy`), `DroneEnergyModel.sortie_energy` |
| $d_c(t)$, $w_c(t)$ | demand, priority | `DemandProcess` |
| $d^{cap}$ | robust demand μ + zσ | `ReplanProblem.d_cap` |
| $c_k(t)$ | route disruption score | `segment_disruption` |
| $f_1, f_2, f_3$ | delivery time, spread, Z1 | `evaluate`, `realized_metrics` |
| DoD | realized dynamic degree | `metrics["dynamic_degree"]` |
| gap₀, gap_last | plan-vs-actual discrepancy | `metrics["gap0_f3"]`, `metrics["gap_last_f3"]` |
