# Dynamic Truck–Drone Relief Routing with Full Coverage

Oct 28, 2026 · @Dũng

## Abstract

We model post-disaster relief with trucks and drones as two linked capacitated routing problems that are re-planned while roads fail. In the first, trucks carry supplies from a depot to staging satellites over a road network whose links are open, damaged or closed. In the second, drones at each satellite make multi-drop trips to every assigned victim, with a fixed recharge interval between trips, while the truck waits.

The first contribution concerns the objective:

- **Full coverage is a hard rule.** Every victim must be served, enforced at every re-plan through an ordered set of fallbacks.
- **The objective follows the victims.** It is the priority-weighted average time victims wait for supplies, with a lateness penalty per priority class. Truck travel, truck waiting and drone energy come second.
- **A known problem class.** Once drone plans are fixed, the truck problem becomes a cumulative CVRP with service times. Each satellite is weighted by the total priority of its victims.

The dynamics:

- road states change at regular time steps;
- an edge a truck is driving on is locked until the truck leaves it;
- a satellite under service is locked until its drones return.

Re-planning is drone-aware: it can move unserved victims to another satellite in drone range, re-solve the drone plans that change, and extend the queued trips at a satellite still being served. Demand is fixed in this version. The paper sets out the model, its formulation and a solution method; experiments are planned for the next version.

## 1. Introduction

In the first days after a flood or an earthquake, relief must reach every affected person, and the most urgent first. Damaged roads make both goals hard. A plan made in the morning may route trucks over a bridge that is closed by noon.

Drones help because they ignore the road network. Their payload, though, is a few kilograms, and their range is a few kilometres. So in practice trucks bring supplies and drones to safe staging sites, and drones fly the last leg to people the trucks cannot reach.

**Setting.** We study this two-level system:

- **Trucks** leave a depot and visit staging **satellites**: schools, open ground, junctions on high ground.
- **At a satellite** the truck stops, and its drones deliver to the **victims** assigned there in multi-drop trips. Each drone recharges between trips. The truck leaves when the last drone is back.
- **Roads** between depot and satellites are open, damaged (slower) or closed (not usable). Their states change over time.
- **Victims** have a demand and a priority class, and **all of them must be served**.

**Why a static plan is not enough.** Road damage is revealed while trucks are already driving. A static plan can:

- route a truck over a closed road;
- serve high-priority victims late because of a slow detour;
- leave a satellite's victims cut off from any truck.

Re-planning must respect what is already under way: the road a truck is on, and the drone trips already in the air.

**Gap.** The literature (Section 2) offers two-echelon truck–drone models planned once, dynamic truck–drone models without satellites, and coverage handled by hard constraints at planning time or by penalties. None keeps every victim covered while re-planning a two-level system as roads fail.

**Contributions.**

1. **Model.** Two linked CVRPs: a truck CVRP over satellites, and a multi-trip drone CVRP with recharge at each satellite. They are coupled through the truck's waiting (dwell) time and the victims' delivery times. The road network has three states, open, damaged and closed, with closed roads infeasible for trucks.
2. **Victim-centred objective.** Priority-weighted average waiting time with class deadlines, under a hard full-coverage rule. Cost terms come second. We show that, given drone plans, the truck problem is a cumulative CVRP with service times, which brings its known moves and bounds.
3. **Dynamic re-planning with locks.** A time-stepped rolling horizon. Edges being driven and satellites under service are frozen. Re-planning is triggered by damage on the remaining plan.
4. **Drone-aware re-planning and coverage recourse.** Unserved victims can be moved between satellites in drone range, drone plans are re-solved, and queued trips are extended. A four-step fallback order keeps the coverage rule feasible.
5. **Solution method.** A decomposition: an exact split-based solver for each drone subproblem, inside a warm-started ALNS for the truck subproblem with O(1) move evaluation.

## 2. Related work

Our model joins four lines of work. Each solves part of the problem.

**Two-echelon truck–drone systems.**

- Nguyen et al. (2E-CTVRP) send trucks to satellites, and drones fly from each satellite to the centroid of one victim cluster.
- Yu et al. (2027) send drones from satellites to individual customers with load-dependent energy, but no truck waits at the satellite.
- Shi et al. (2026) feed satellites with large trucks and serve sites with small trucks that carry drones.

All three plan once. Most other truck–drone papers launch drones from customer nodes on the truck's route (Meng & Ma 2026; Luo et al. 2027; Yang et al. 2023), which suits parcel delivery more than relief.

**Multi-trip drones with waiting and recharge.**

- Meng & Ma (2026) make the vehicle wait while several multi-drop drones fly. Drone energy depends on mass × time, and batteries are swapped.
- Luo et al. (2027) allow multiple trips with payload-dependent energy.

We adopt multi-drop trips and the waiting truck, and simplify recharge to a fixed interval between trips.

**Coverage of every demand point.** Approaches differ in how strongly they guarantee service:

| Approach | Papers | Limitation for a dynamic disaster |
| --- | --- | --- |
| Hard constraint at planning time (every customer served, or within drone range of a stop) | Nguyen et al.; Li, Guo & Fang 2026; most VRP-D models | Holds for the t = 0 plan only; no repair after roads fail |
| Penalty for unserved demand (equity) | Peng et al. 2025 | Soft: a cheap enough plan may still leave people out |
| Outsourcing at high cost | Luo et al. 2027 | Always feasible, but only a cost, with no operational meaning |
| Drone-only sites for unreachable points | Peng et al. 2025; Shi et al. 2026; Du et al. 2026 | Which sites are drone-only is fixed in advance |
| Selective service (maximize profit) | Yang et al. 2023 | The opposite of relief needs |

We keep the hard rule and make it hold *during* execution. An ordered recourse (Section 5.5) combines re-routing, re-assignment within drone range, drone service from the depot and a reserve truck.

**Cumulative routing.** When the objective is the sum of arrival times rather than distance, the CVRP becomes the cumulative CVRP (Ngueveu et al. 2010). It is known to favour serving many or important customers early. Our truck subproblem is a weighted version of it with service times.

**Dynamic and robust truck–drone routing.**

- Peng et al. (2025) re-plan every 15 minutes with multi-agent reinforcement learning, but in a single echelon.
- Yang et al. (2023) and Shi et al. (2026) protect a single plan against uncertain travel times.

None of them locks partly executed decisions in a two-level system, or re-plans the drone level along with the trucks.

**Positioning.** Ours is, to our knowledge, the first model that has all of the following at once:

- two levels with a waiting truck;
- victim-level multi-trip drones with recharge;
- three road states, including closure;
- time-stepped re-planning with edge locks;
- a hard full-coverage rule kept by recourse.

## 3. Problem description

A relief operation runs over a horizon \[0, H\] from one depot with K trucks, each carrying U drones.

**Road network.** Trucks drive on a road graph G = (N, E). The nodes N are the depot 0, the satellites S and optional junctions. Each road e has an undamaged travel time τ⁰ₑ and, at time t, a state σₑ(t):

| State | Truck travel time | Meaning |
| --- | --- | --- |
| Open | τ⁰ₑ | Usable at normal speed |
| Damaged | μ · τ⁰ₑ, with μ > 1 (e.g. 2.5) | Usable but slow (debris, shallow water, one lane) |
| Closed | ∞ | Not usable by trucks |

Truck travel between two nodes follows the fastest path over usable roads at the current states. A satellite with no usable path from a truck's position is *cut off* for that truck.

**Trucks.** Each truck has capacity Q. It leaves the depot loaded, visits satellites, and returns to the depot. Each satellite is visited at most once, by one truck: there are no revisits and no second deliveries.

**Satellites and victims.**

- Victims V are the people, households or shelters to be served. Victim v has a location, a demand q\_v and a priority class.
- A satellite s *covers* victim v if a drone can fly from s to v and back on one battery with v's load.
- Every victim is assigned to exactly one covering satellite, and that satellite is visited by a truck carrying the victim's demand.
- At instance generation we require every victim to be covered by at least two satellites, so that a re-assignment is possible when a satellite is cut off.

**Drones.** Each truck carries U identical drones with payload P, battery capacity E and speed v\_d. At a satellite:

- **Trips.** A drone flies multi-drop trips, each starting and ending at the satellite, with total load ≤ P and energy within the battery. A trip has a launch time S\_L, a recovery time S\_R and a drop time τ\_d at each victim.
- **Recharge.** Between two consecutive trips of the same drone there is a fixed recharge interval t\_r. Batteries recharge on the truck while it drives, so no recharge is needed before the first trip at a satellite.
- **Dwell.** The truck waits until its last drone has landed. This waiting time is the satellite's dwell W\_s.

**Priorities and deadlines.** Each victim has a priority weight w\_v and a target time D\_v set by its class:

| Class | Weight w\_v | Target time D\_v |
| --- | --- | --- |
| Critical (injured, medical needs) | 5 | 2 h |
| High (children, elderly, isolated) | 2 | 4 h |
| Normal | 1 | 8 h |

These values are defaults for experiments. Being late is penalized, not forbidden, so a plan always exists.

**Full coverage.** Every victim must receive its demand within the horizon. When the main plan cannot achieve this, recourse actions are used (Section 5.5). They are costly, but they make sure no victim is left out.

**Dynamics.** Time advances in steps of length Δt (e.g. 5 minutes).

- **Road changes.** At each step, road states may change. Changes follow a Markov chain over {open, damaged, closed}, or a scenario file. They are known to the planner when they occur.
- **Edge lock.** When a truck starts driving along a road, that road is locked for that truck: its state, and so the truck's travel time on it, stay fixed until the truck reaches the end.
- **Service lock.** While a truck dwells at a satellite, drones already in the air and the victims assigned there cannot be withdrawn. Queued trips may still be changed (Section 5.3).
- **Re-planning.** After each step, the planner may re-plan everything not locked. Demand is fixed and known in this version.

**Assumptions.**

- One depot.
- Homogeneous trucks and drones.
- Drones are not affected by road states or weather.
- A drone returns to its own truck.
- Road travel times are deterministic given the state.

## 4. Mathematical formulation

This section states the problem at one decision time with known road states. Section 5 shows how it is re-solved over time. The model has two linked parts: a drone subproblem at each satellite (4.2) and a truck problem over satellites (4.3).

### 4.1 Notation

| Symbol | Meaning |
| --- | --- |
| 0, S, V, K | Depot, satellites, victims, trucks |
| U | Drones per truck |
| Q, P | Truck capacity, drone payload |
| q\_v, w\_v, D\_v | Demand, priority weight and target time of victim v |
| a\_vs | 1 if satellite s covers victim v (round trip within battery) |
| c\_ij(t) | Fastest truck time from i to j at time t over usable roads; ∞ if cut off |
| S\_L, S\_R, τ\_d, t\_r | Drone launch, recovery, drop and recharge times |
| p(ℓ) | Battery fraction used per minute when carrying load ℓ |
| r\_min | Battery reserve |
| π\_L | Lateness penalty weight |
| c\_T, c\_W, c\_D | Unit costs of truck driving, truck waiting and drone energy |
| M₁ ≪ M₂ | Penalties for the two recourse modes (Section 5.5) |

The main decision variables are:

- y\_vs ∈ {0, 1}: victim v is served from satellite s (allowed only if a\_vs = 1);
- z\_s ∈ {0, 1}: satellite s is used;
- x\_ijk ∈ {0, 1}: truck k drives from node i to node j;
- A\_s: truck arrival time at satellite s;
- W\_s: dwell at satellite s;
- T\_v: time victim v receives its supplies;
- fᵈ\_v, fʳ\_v ∈ {0, 1}: victim v is served by a recourse mode instead (direct drone from the depot, or a reserve truck).

### 4.2 Drone subproblem at a satellite

Given the victims V\_s = {v : y\_vs = 1} assigned to satellite s, the drones solve a multi-trip CVRP with recharge.

**Trips.** A trip ρ is an ordered set of victims (v₁, …, v\_m), flown from s and back. Its load, duration and energy are:

```latex
L_\rho = \sum_{i=1}^{m} q_{v_i} \le P, \qquad
d_\rho = S_L + \sum_{\text{legs } l} \frac{\text{len}_l}{v_d} + m\,\tau_d + S_R
```

```latex
\sum_{\text{legs } l} p(\ell_l)\, t_l \;+\; \sum_{i=1}^{m} p(\ell_{v_i})\,\tau_d \;\le\; 1 - r_{\min}
```

Here ℓ\_l is the load still on board along leg l, falling after each drop. Let o\_ρv be the time from launch until v's drop.

**Schedule with recharge.** Drone u flies its trips ρ\_u1, …, ρ\_un in order. Trip starts and victim offsets are:

```latex
b_{u1} = 0, \qquad b_{u,\ell+1} = b_{u\ell} + d_{\rho_{u\ell}} + t_r, \qquad \delta_v = b_{u\ell} + o_{\rho_{u\ell} v} \;\; (v \in \rho_{u\ell})
```

The dwell is the time until the last drone lands:

```latex
W_s = \max_{u} \big( b_{u n_u} + d_{\rho_{u n_u}} \big)
```

**Objective of the drone subproblem.** A drone plan affects two groups of victims:

- its own victims, through the offsets δ\_v;
- the victims of every satellite the truck visits afterwards, through the dwell W\_s.

Let Ω⁺\_s be the total priority weight served later on the same truck route. The drone plan minimizes:

```latex
\min \; J_D(s) = \sum_{v \in V_s} w_v\, \delta_v \;+\; \Omega^{+}_{s}\, W_s \;+\; c_D \sum_{\rho} e_\rho
```

This is how drone service enters re-planning. A satellite early in a long route (large Ω⁺) should get a short-makespan drone plan. A satellite last on its route should get a priority-first plan.

### 4.3 Truck problem

Given drone results (W\_s, δ\_v), victim v is served at:

```latex
T_v = \sum_{s} y_{vs}\, (A_s + \delta_v)
```

**Constraints.**

```latex
\sum_{s} a_{vs}\, y_{vs} + f^{d}_v + f^{r}_v = 1 \quad \forall v \qquad \text{(full coverage)}
```

```latex
y_{vs} \le z_s, \qquad z_s \le \sum_v y_{vs}, \qquad \sum_{k}\sum_{i} x_{isk} = z_s \quad \forall s \qquad \text{(a satellite is visited once iff used)}
```

```latex
\sum_{j} x_{0jk} \le 1, \qquad \sum_{i} x_{ihk} = \sum_{j} x_{hjk} \quad \forall h, k \qquad \text{(routes from and back to the depot)}
```

```latex
\sum_{s} \Big( \sum_{v} q_v\, y_{vs} \Big) \sum_{i} x_{isk} \le Q \quad \forall k \qquad \text{(truck capacity)}
```

```latex
A_j \ge A_i + W_i + c_{ij}(t) - M\,(1 - x_{ijk}), \qquad x_{ijk} = 0 \text{ if } c_{ij}(t) = \infty \qquad \text{(time; closed roads)}
```

Subtours are removed by the time constraints, as in a standard time-based CVRP.

**Objective.** The objective is lexicographic: minimize recourse use first, then service quality plus cost.

```latex
J_0 = \sum_{v} \big( M_1 f^{d}_v + M_2 f^{r}_v \big)
```

```latex
J_1 = \frac{1}{\sum_v w_v} \sum_{v} w_v \Big[ T_v + \pi_L\, (T_v - D_v)^{+} \Big]
```

```latex
J_2 = c_T \sum_{i,j,k} c_{ij}\, x_{ijk} \;+\; c_W \sum_{s} W_s \;+\; c_D \sum_{s}\sum_{\rho} e_\rho
```

```latex
\operatorname{lexmin}\; \big( J_0,\; J_1 + \lambda J_2 \big)
```

J₁ is the priority-weighted average time a victim waits for supplies, plus lateness against its class target. Alongside it we report each class's average and maximum T\_v, and the number of late victims, as fairness checks.

### 4.4 The truck problem is a cumulative CVRP

Let Ω\_s = Σ\_v w\_v y\_vs be the priority weight at satellite s. Write truck k's route as (π₁, …, π\_m), and let Ω\_{≥p} be the weight from position p to the end. Then:

```latex
\sum_v w_v T_v = \sum_{p=1}^{m} \Omega_{\ge p}\; c_{\pi_{p-1}\pi_p} \;+\; \sum_{p=1}^{m} \Omega_{> p}\; W_{\pi_p} \;+\; \sum_v w_v \delta_v
```

This has three consequences:

1. **Known problem class.** Each road time is weighted by the priority still on board, and each dwell by the priority still to come. This is a weighted cumulative CVRP with service times (Ngueveu et al. 2010), whose moves, lower bounds and metaheuristics apply.
2. **Ordering rule.** Ignoring travel, swapping two neighbouring satellites a, b improves the route exactly when Ω\_b / W\_b > Ω\_a / W\_a. So a satellite with heavy priority and a short dwell goes early (Smith's rule).
3. **Waiting cost.** With a fixed assignment y, the c\_W term is constant and cannot change the truck routes. It matters only when victims move between satellites, where it stops one satellite from being overloaded.

With fixed W and δ, the truck problem is NP-hard, as it contains the CVRP. The drone subproblem is small (5 to 20 victims) and is solved almost exactly (Section 6.1).

## 5. Dynamic re-planning

The plan is revised in a rolling horizon with a fixed time step Δt. At each step, road states are updated, locked decisions are kept, and the rest of the plan is checked. If needed, the residual problem of Section 4 is re-solved for unserved victims and unvisited satellites.

*(Figure: one time step, three re-planning levels. See the online doc.)*

Road changes flow into a check of the current plan. Only damage that cuts off a satellite or worsens the plan enough triggers the full re-optimization with drone re-planning.

### 5.1 State at a time step

The state at time t is:

```latex
X_t = \big( \sigma(t),\; \{\text{truck state}_k\}_{k \in K},\; V^{\text{done}}_t,\; y_t,\; \{\text{drone state}_s\}_{s \text{ active}} \big)
```

- σ(t): the state of every road.
- Vᵈᵒⁿᵉ\_t: victims already served.
- y\_t: the current assignment of unserved victims to satellites.

Each truck is in one of four modes. Each mode has a *free point*: the place, time and load from which its plan may be changed.

| Truck mode | Locked | Free point |
| --- | --- | --- |
| Driving on road (i, j) | That road and its travel time until arrival at j | Node j, at the locked arrival time |
| Dwelling at satellite s | Drones in the air; the victims they carry | Satellite s, when the last committed trip lands; queued trips stay open |
| Idle at a node | Nothing | Here, now |
| Back at the depot | Everything | None |

Only the current road is locked. A truck driving a multi-road path to its next satellite can still be re-routed at the next junction.

### 5.2 Road dynamics and travel times

At each step, every road that is not locked changes state according to a Markov matrix over {open, damaged, closed}, or follows a scenario file:

```latex
\Pr\big[\sigma_e(t+\Delta t) = s' \mid \sigma_e(t) = s\big] = P_{s s'}, \qquad s, s' \in \{O, D, C\}
```

A locked road keeps its state for the truck on it. Once the truck has left it, the road's true current state applies to everyone.

After the update, the fastest-path times c\_ij(t) are recomputed over usable roads, with Dijkstra from every free point and every unvisited satellite. A satellite with no path from any truck's free point is marked *cut off*.

### 5.3 Drone-aware residual problem

The residual problem P(t) is the model of Section 4 restricted to what is still open:

- **Trucks** start at their free points, with their remaining loads, and return to the depot.
- **Satellites** are the unvisited ones. An active satellite stays with its truck as the first stop.
- **Victims** are the unserved ones, which may be re-assigned among the satellites that cover them.

The drone level enters in three ways:

1. **Re-solved plans.** When a satellite's victim set changes, or its route position changes Ω⁺\_s, its drone plan is re-solved with J\_D (Section 4.2). The truck problem then sees the new W\_s and δ\_v.
2. **Extended service at an active satellite.** If the dwelling truck has spare load, unserved victims of a cut-off satellite that are covered by the active one can be added to its *queued* trips. The truck waits longer, but nothing is revisited.
3. **Recharge in the schedule.** Every re-solved plan includes the t\_r interval between trips. So adding a trip to a busy drone costs d\_ρ + t\_r, not just d\_ρ.

Arrival times now start from the free points:

```latex
A_j \ge \phi_k + c_{h_k j}(t) - M\,(1 - x_{h_k j k})
```

Here h\_k is truck k's free-point node and φ\_k its free time.

### 5.4 When to re-plan

Re-solving at every step is wasteful. Most steps change nothing that matters. After each update, the current plan is re-evaluated under the new travel times, and the cheapest action that keeps it good is taken:

| Level | Taken when | Action |
| --- | --- | --- |
| L0 — path update | No used road closed; ΔJ ≤ ε₁ J | Keep the satellite order; only the road paths between satellites change |
| L1 — repair | No satellite cut off; ΔJ ≤ ε₂ J | A few local moves (relocate, swap, 2-opt) from the current plan |
| L2 — re-optimize | A satellite is cut off, a used road is closed, or ΔJ > ε₂ J | Warm-started ALNS on P(t), with re-assignment and drone re-solves |

Here ΔJ = J(current plan | σ(t)) − J(current plan | σ(t − Δt)), measured on J₁ + λJ₂, and ε₁ < ε₂ (e.g. 1% and 5%).

### 5.5 Coverage recourse

The full-coverage rule is checked after every re-plan. Any victim the plan cannot serve, because its satellite is cut off or no truck has load for it, goes through these steps in order. Each step is used only if the earlier ones fail:

| Step | Action | Condition | Cost |
| --- | --- | --- | --- |
| R1 | Re-route and re-sequence trucks | A path to the satellite still exists | Part of J₁, J₂ |
| R2 | Re-assign the victim to another covering satellite (unvisited, or active with queued trips) | a\_vs' = 1 and its truck has spare load | Part of J₁, J₂ (longer dwell at s') |
| R3 | Direct drone delivery from the depot | The depot covers v; a depot drone pool exists | M₁ per victim |
| R4 | External resource (helicopter, boat or ground team), with lead time L\_ext | Always available | M₂ per victim |

R4 always exists, so the hard rule is always feasible. The number of R3 and R4 uses is a key result. A good plan keeps spare truck capacity (10–20%) and double coverage, so that R2 usually succeeds.

Closed roads can reopen under the Markov chain. For a victim with a soft target, the planner may choose to visit its satellite later in the hope of a reopening. We leave this to the probabilistic extension (Section 8).

## 6. Solution method

We solve the problem by decomposition. A fast, near-exact drone solver gives each satellite its dwell and victim offsets. An adaptive large neighbourhood search (ALNS) over truck routes and victim assignments calls it through a cache. The same search builds the first plan and is warm-started at every L2 re-plan.

### 6.1 Drone solver

For a satellite s with victims V\_s:

1. **Giant tour.** Build one tour through V\_s (nearest neighbour, then 2-opt). A priority-first variant sorts victims by w\_v first.
2. **Exact split.** Cut the tour into trips by dynamic programming (Prins 2004). A cut is allowed only if the trip meets the payload and battery limits of Section 4.2. This takes O(m²) time, with m ≤ 20.
3. **Schedule with recharge.** Assign trips to the U drones by list scheduling. Each drone is free again at the end of its trip plus t\_r. Two orders are tried:
   - longest trip first, which keeps the dwell W\_s short;
   - highest weight per minute first (Σw / d\_ρ, Smith's rule), which serves critical victims early.
4. **Menu.** Keep the non-dominated pairs (W\_s, Σ w\_v δ\_v) from both tours and both orders, usually two to four plans. When the truck search evaluates s at a route position with weight Ω⁺\_s still to come, it picks the plan that minimizes Σ w\_v δ\_v + Ω⁺\_s W\_s. This is J\_D, without re-running the solver.

Results are cached by (s, set of victims). Re-assignment moves touch only two satellites, so most calls are cache hits.

### 6.2 Truck ALNS

**Representation.** One ordered list of satellites per truck. An active satellite is pinned as the first stop of its truck. Victims carry their current satellite label.

**Destroy operators** (remove 10–30% of satellites):

- random removal;
- worst removal: the satellites with the largest weighted delay contribution, Ω\_s(A\_s − lower bound);
- related removal, by distance and by sharing damaged roads;
- **damage-related removal**: satellites whose paths use roads that changed state since the last plan;
- route removal.

**Repair operators:** greedy insertion and regret-2 insertion, both using the cumulative cost of 6.3. Satellites are inserted in decreasing Ω\_s, so heavy-priority satellites get first choice.

**Victim moves:**

- **re-assign** v from s to another covering satellite s′;
- **swap** two victims between satellites.

Both re-solve the two drone plans through the cache and check truck capacity.

**Local search** after each repair: relocate, swap, 2-opt\*, and Or-opt within and between routes.

**Acceptance and adaptation.** Simulated annealing on lexmin(J₀, J₁ + λJ₂). Operator weights are updated by recent success (Ropke & Pisinger 2006).

**Warm start and budget.** An L2 re-plan starts from the current plan with the new travel times, and stops after a time limit per step (e.g. 2 s for 30 satellites). The first plan at t = 0 gets a longer limit (e.g. 30 s).

### 6.3 Fast move evaluation

For each route we store prefix arrivals A\_p, dwell ends A\_p + W\_p, and suffix weights Ω\_{≥p}. Inserting satellite s between a and b, at position p, shifts every later arrival by the same amount:

```latex
\delta = c_{as} + W_s + c_{sb} - c_{ab}
```

The change in weighted waiting time is:

```latex
\Delta \sum_v w_v T_v = \Omega_s \big( A_a + W_a + c_{as} \big) + \delta\, \Omega_{\ge p} + \sum_{v \in V_s} w_v \delta_v
```

This takes O(1) time. The lateness term adds O(log m) with sorted suffix slacks. Removal and swap work the same way. For capacity, each route stores its load, so moves are checked in O(1).

### 6.4 Initial plan

1. **Assignment.** Each victim goes to its nearest covering satellite. Then a greedy rebalancing moves victims from overloaded satellites, measured by dwell or by truck capacity, to their second-nearest coverer.
2. **Drone plans.** Solve every satellite's drone menu (6.1).
3. **Trucks.** Regret-2 insertion of satellites in decreasing Ω\_s, followed by ALNS.
4. **Recourse.** Any victim still unserved goes through the recourse steps R2 to R4 of Section 5.5.

### 6.5 One time step

1. Advance trucks and drones to t + Δt, record deliveries, and release locks that have ended.
2. Update road states, keeping locked roads, and recompute fastest paths and cut-off satellites.
3. Evaluate the current plan, choose L0, L1 or L2 (Section 5.4), and run it.
4. Check coverage and apply recourse (Section 5.5).
5. Dispatch: trucks at a free point start their next road (which locks it), and active satellites launch their next queued trips.

## 7. Experimental design (planned)

No results are reported yet. The plan below is meant to test whether drone-aware re-planning keeps every victim served and lowers priority-weighted waiting time compared with a static plan, and at what computing cost.

**Instances.** Built from the existing 2E-CTVRP set and the Solomon-derived set:

- **Victims:** keep the coordinates and demands. Assign priority classes at random (20% critical, 30% high, 50% normal), or by distance from a damage centre.
- **Satellites:** keep them as candidate staging sites. Add sites until every victim is covered by at least two.
- **Road graph:** a sparse graph over the depot, the satellites and grid junctions, so that a closure forces a real detour. The complete graph is kept as a simpler variant.
- **Fleet:** truck capacity set to 1.15 × total demand / K. U = 3 drones per truck.
- **Drones:** energy as in Meng & Ma (2026), with S\_L = 1, S\_R = 2, τ\_d = 0.1 and t\_r = 10 minutes.

**Factors.**

- road damage intensity: low, medium, high (Markov matrices with stationary closed share 5%, 15%, 30%);
- time step Δt: 5, 10 or 15 minutes;
- recharge time t\_r: 5, 10 or 20 minutes;
- coverage redundancy: 1 or 2 covering satellites per victim.

**Strategies compared.**

| Strategy | Description |
| --- | --- |
| Static | Plan at t = 0. Trucks follow it and take the fastest path around closures, with no re-sequencing |
| Periodic | Full re-optimization (L2) at every step |
| Event-triggered | L2 only when a used road changes state |
| Tiered (ours) | L0 / L1 / L2 chosen by the rules of Section 5.4 |
| Tiered, no drone re-plan | As above, but drone plans and assignment frozen at t = 0 |
| Oracle | Knows the full road history in advance (lower bound) |

**Metrics.**

- recourse use: victims served by R3 and R4 (target 0);
- J₁ and its parts: priority-weighted average T\_v, lateness, and the number of late victims per class;
- average and maximum T\_v per priority class, as a fairness check;
- J₂ and its parts: truck driving time, truck waiting time and drone energy;
- re-plan statistics: how often each level was used, and CPU time per step;
- plan stability: satellites moved between trucks per re-plan.

**Questions.**

1. How much does re-planning reduce J₁ and recourse use compared with a static plan?
2. How much of the gain comes from re-planning the drone level (re-assignment, re-solved plans, extended service)?
3. Does the tiered trigger match periodic re-optimization at a fraction of the CPU time?
4. How do recharge time and coverage redundancy change the share of victims that need recourse?

## 8. Limitations and extensions

This version is kept simple on purpose. The extensions below are listed in the order we would add them.

- **Demand changes.** Demand is fixed. Earlier work on this project lets demand grow only before or during a satellite's service, within the truck's spare load, which fits the extended-service mechanism of Section 5.3 directly.
- **Partial information.** Road states are assumed known as soon as they change. A belief over states with noisy reports, and robust budgets derived from it, would replace c\_ij(t) with expected or worst-case times.
- **Waiting for reopening.** Closed roads are treated as closed for the rest of the plan. A probabilistic model could weigh a later visit against recourse.
- **Recharge model.** Recharge is one fixed interval t\_r. Battery swapping (zero interval, limited spare batteries) and charge-dependent times are natural variants.
- **Drone constraints.** Drones ignore weather and no-fly zones and always return to their own truck. Wind, restricted airspace and drone hand-over between trucks are left out.
- **One visit per satellite.** No revisits are allowed. Demand beyond the trucks' spare capacity goes to recourse. Multi-trip trucks would remove this, at the cost of a different problem class.
- **Exact bounds.** With fixed drone plans, the cumulative CVRP structure allows lower bounds from the literature. Using them to measure optimality gaps on small instances is future work.

## References

- Du, W., Wang, J., Liu, J., Wang, H., Zhou, J., Mei, C., & Pang, Y. (2026). Optimization of emergency supplies allocation for urban flood disasters. *Results in Engineering*, 32, 112619. [doi](https://doi.org/10.1016/j.rineng.2026.112619)
- Li, J., Guo, Q., & Fang, Y. (2026). Truck–drone cooperative delivery problem based on two-stage algorithm. *Symmetry*, 18, 1199. [doi](https://doi.org/10.3390/sym18071199)
- Luo, Y., Chen, Y., & Wahab, M. I. M. (2027). Multi-trip truck-drone routing problem with simultaneous pickup and delivery and temporal constraints. *Transportation Research Part E*, 217, 105235. [doi](https://doi.org/10.1016/j.tre.2026.105235)
- Meng, S., & Ma, H. (2026). Electric vehicle routing problem with multi-visit drones considering time windows and battery swapping stations. *Transportation Research Part E*, 215, 105107. [doi](https://doi.org/10.1016/j.tre.2026.105107)
- Ngueveu, S. U., Prins, C., & Wolfler Calvo, R. (2010). An effective memetic algorithm for the cumulative capacitated vehicle routing problem. *Computers & Operations Research*, 37(11), 1877–1885. [doi](https://doi.org/10.1016/j.cor.2009.06.014)
- Nguyen, D. V. A., Piplani, R., & Gunawan, A. A two-echelon covering tour vehicle routing problem with drones for post-disaster relief. *arXiv:2609.38227* (under review).
- Peng, W., Wang, D., Yin, Y., & Cheng, T. C. E. (2025). Multi-agent deep reinforcement learning-based truck-drone collaborative routing with dynamic emergency response. *Transportation Research Part E*, 195, 103974. [doi](https://doi.org/10.1016/j.tre.2025.103974)
- Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985–2002. [doi](<https://doi.org/10.1016/S0305-0548(03)00158-8>)
- Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455–472. [doi](https://doi.org/10.1287/trsc.1050.0135)
- Shi, Y., Guo, Q., Pan, P., Chen, M., Wu, Y., & Guo, H. (2026). Robust optimization for two-echelon truck and drone collaborative delivery in emergency relief. *Applied Soft Computing*, 201, 115738. [doi](https://doi.org/10.1016/j.asoc.2026.115738)
- Smith, W. E. (1956). Various optimizers for single-stage production. *Naval Research Logistics Quarterly*, 3(1–2), 59–66. [doi](https://doi.org/10.1002/nav.3800030106)
- Yang, Y., Yan, C., Cao, Y., & Roberti, R. (2023). Planning robust drone-truck delivery routes under road traffic uncertainty. *European Journal of Operational Research*, 309, 1145–1160. [doi](https://doi.org/10.1016/j.ejor.2023.02.031)
- Yu, F., Li, C., Chen, Q., Li, Y., Chen, Y., & Yang, X. (2027). Multi-objective two-echelon vehicle-drone routing with load-dependent drone under pickup delivery time windows. *Transportation Research Part E*, 217, 105195. [doi](https://doi.org/10.1016/j.tre.2026.105195)
