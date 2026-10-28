# Dynamic Truck–Drone Relief Routing with Full Coverage

Oct 28, 2026 · @Dũng

## Abstract

We model post-disaster relief with trucks and drones as two linked routing problems that are re-planned while roads degrade.

- **Trucks** make one or more trips from a depot to staging satellites over a road network whose roads are open, damaged (×2.5 travel time) or closed (×5, a long detour).
- **Drones** at each satellite deliver one parcel to every assigned victim in multi-drop trips, while the truck waits. Each trip's range is limited by a battery whose drain grows with the load carried, and batteries are swapped instantly.

**Objective.** We minimize the priority-weighted average time victims wait for supplies:

- every victim must be served;
- critical victims carry a deadline enforced by a dominant penalty;
- operating cost comes second.

With drone plans fixed, the truck problem is a multi-trip cumulative CVRP with service times.

**Dynamics.** Road states change hourly through a Markov chain, and the plan is re-optimized at each change. Roads a truck has driven or is driving are left unchanged. Re-planning covers both levels: it moves unserved victims between satellites in drone range and re-solves the affected drone plans.

**Method and data.**

- A decomposition: a split-based drone solver inside an adaptive large neighbourhood search (ALNS) for the trucks.
- A double deep Q-network (DQN) that learns how much re-optimization each step deserves and which search operators to use.
- A Solomon-based instance generator that guarantees every victim lies within single-trip drone range of at least two candidate satellites.

Experiments are planned for the next version.

### Overview

*(Figure: overview · the relief system and how it is planned. See the online doc.)*

Panel A shows trucks driving over open, damaged and closed roads to staging satellites, where drones fly multi-drop trips to victims while the truck waits. Panel B shows how the two levels are planned at every time step: drone plans give the truck search its dwell times, and the truck route tells each drone plan how much priority is waiting further along the route.

## 1. Introduction

In the first hours after a flood or an earthquake, relief must reach every affected person, and the most urgent first. Damaged roads make both goals hard: a route planned in the morning may cross a bridge that is closed by noon.

Drones help because they ignore the road network. But they carry only a few kilograms over a few kilometres. In practice, trucks bring supplies and drones to safe staging sites, and drones fly the last leg.

**Setting.** We study this two-level system:

- **Trucks** leave a depot loaded with parcels and visit staging **satellites**: schools, open ground, junctions on high ground. When empty, a truck may return to the depot, reload, and drive out again.
- **At a satellite** the truck stops. Its drones fly multi-drop trips that deliver one parcel to each **victim** assigned there. Batteries are swapped instantly between trips. The truck leaves when the last drone is back.
- **Roads** are open, damaged or closed. A closed road forces a long detour (×5 travel time). Road states change hourly.
- **Victims** each need one parcel and belong to a priority class. All of them must be served, and critical victims have a deadline.

**Why a static plan is not enough.** Road damage appears while trucks are already driving. A plan made at t = 0 can:

- send trucks over roads that have since become slow;
- serve critical victims late;
- keep using satellites that are now expensive to reach, when their victims could be served from elsewhere.

Re-planning must respect what is already under way: the road a truck is on, the roads it has passed, and the drone trips already in the air.

**Gap.** The literature (Section 2) offers:

- two-echelon truck–drone models that are planned once;
- dynamic truck–drone models without satellites;
- coverage guaranteed only at planning time.

None re-plans a two-level relief system hour by hour while keeping every victim served and meeting critical deadlines.

**Contributions.**

1. **Model.** Two linked CVRPs: a multi-trip truck CVRP over satellites, and a multi-trip drone CVRP at each satellite with payload-dependent battery limits and instant battery swaps. They are coupled through the truck's waiting (dwell) time and the victims' delivery times.
2. **Victim-centred objective.** A dominant penalty for missed critical deadlines, then the priority-weighted average waiting time, then cost. Given drone plans, the truck problem is a multi-trip cumulative CVRP with service times.
3. **Coverage guaranteed by construction.** Three things together make full service always achievable:
   - every victim is within single-trip drone range of at least two candidate satellites;
   - closed roads are slow, not impassable;
   - trucks can make several trips.
4. **Dynamic, drone-aware re-planning.** Hourly road changes, with driven and occupied roads frozen. A rolling-horizon loop re-optimizes truck routes, victim-to-satellite assignment and drone plans together.
5. **Learning-assisted solution method.** A split-based drone solver inside a warm-started ALNS. A double DQN chooses how much re-optimization each step deserves and which ALNS operators to apply. It does not build routes itself, so every plan remains feasible.
6. **Benchmark generator.** Solomon-based instances with equal demands, priority classes, satellite candidates placed by double coverage, a road graph, a flood zone and hourly damage scenarios.

## 2. Related work

Our model joins four lines of work. Each solves part of the problem.

**Two-echelon truck–drone systems.**

- Nguyen et al. (2E-CTVRP) send trucks to satellites, and drones fly from each satellite to the centroid of one victim cluster.
- Yu et al. (2027) send drones from satellites to individual customers with load-dependent energy, but no truck waits at the satellite.
- Shi et al. (2026) feed satellites with large trucks and serve sites with small trucks that carry drones.

All three plan once. Most other truck–drone papers launch drones from customer nodes on the truck's route (Meng & Ma 2026; Luo et al. 2027; Yang et al. 2023), which suits parcel delivery more than relief.

**Multi-trip drones, waiting trucks and batteries.**

- Meng & Ma (2026) make the vehicle wait while several multi-drop drones fly. Drone energy depends on mass × time, and batteries are swapped.
- Luo et al. (2027) allow multiple trips with payload-dependent energy.

We adopt multi-drop trips, payload-dependent energy and the waiting truck. Following Meng & Ma's battery swapping, we treat swaps as instant with unlimited spare batteries, so the battery limits each trip's range but adds no turnaround time.

**Coverage of every demand point.** Approaches differ in how strongly they guarantee service:

| Approach | Papers | Limitation for a dynamic disaster |
| --- | --- | --- |
| Hard constraint at planning time (every customer served, or within drone range of a stop) | Nguyen et al.; Li, Guo & Fang 2026; most VRP-D models | Holds for the t = 0 plan only; no repair after roads fail |
| Penalty for unserved demand (equity) | Peng et al. 2025 | Soft: a cheap enough plan may still leave people out |
| Outsourcing at high cost | Luo et al. 2027 | Always feasible, but only a cost, with no operational meaning |
| Drone-only sites for unreachable points | Peng et al. 2025; Shi et al. 2026; Du et al. 2026 | Which sites are drone-only is fixed in advance |
| Selective service (maximize profit) | Yang et al. 2023 | The opposite of relief needs |

We keep the hard rule and make it hold *by construction*, at planning time and during execution. Three properties combine to guarantee it:

- every victim is in single-trip drone range of at least two candidate satellites;
- closed roads are slow rather than impassable;
- trucks may make several trips.

Full service is therefore always feasible. The difficulty that remains is serving people *in time*, which the objective handles.

**Cumulative routing.** When the objective is the sum of arrival times rather than distance, the CVRP becomes the cumulative CVRP (Ngueveu et al. 2010). It is known to favour serving many or important customers early. Our truck subproblem is a weighted version of it with service times.

**Dynamic and robust truck–drone routing.**

- Peng et al. (2025) re-plan every 15 minutes with multi-agent reinforcement learning, but in a single echelon.
- Yang et al. (2023) and Shi et al. (2026) protect a single plan against uncertain travel times.

None of them locks partly executed decisions in a two-level system, or re-plans the drone level along with the trucks.

**Learning in dynamic routing.** Two lines of work use learning:

- **Learned routing policies.** Chen, Ulmer & Thomas (2022) learn with deep Q-learning which same-day requests to accept and whether a truck or a drone serves them. Peng et al. (2025) learn routing actions with multi-agent DRL. Li & Gonsalves (2026) learn to construct TSP-D and VRP-D routes with attention models.
- **Learned control of a search.** Ropke & Pisinger (2006) adapt operator weights from past success, and Duan et al. (2026) choose low-level heuristics with a hyper-heuristic.

We follow the second line: learning decides *how* to re-optimize, while the optimizer builds the routes and guarantees feasibility.

**Positioning.** To our knowledge, ours is the first model that combines:

- two levels with a waiting truck, and multi-trip trucks;
- victim-level multi-trip drones whose range depends on load;
- three road states with hourly Markov dynamics and frozen driven roads;
- full coverage guaranteed by construction, with dominant penalties on critical deadlines;
- re-planning of both levels, steered by a learned policy.

## 3. Problem description

A relief operation starts at time 0 from one depot with K trucks, each carrying U drones. It ends when every victim has been served and every truck is back at the depot.

**Road network and speeds.** Trucks drive on a road graph G = (N, E) over the depot, the satellites and junctions, at a constant speed vₜ = 40 km/h. Road e has a base travel time τ⁰ₑ and a state σₑ(t):

| State | Travel time | Meaning |
| --- | --- | --- |
| Open | τ⁰ₑ | Normal |
| Damaged | 2.5 · τ⁰ₑ | Usable but slow (debris, shallow water, one lane) |
| Closed | 5 · τ⁰ₑ | Blocked; the truck must take a long local detour |

Trucks follow the fastest path at the current states. Because no road is impassable, every satellite is always reachable.

**Trucks (multi-trip).**

- Each truck carries at most Q parcels.
- A trip leaves the depot, visits one or more satellites and returns.
- After returning, a truck may reload (time θ) and start another trip.
- Each satellite is visited **at most once over the whole operation**. There are no revisits and no second deliveries to the same satellite.
- Drones and batteries do not count toward Q.

**Victims and demand.**

- Every victim v needs exactly **one parcel of q kg**, with q ≤ the drone payload. So each victim is served by one drone drop and never revisited.
- Each victim has a priority class (below).
- A drone carries up to P parcels per trip (e.g. q = 1 kg, P = 3).

**Satellites and coverage.**

- Satellites are candidate staging sites. The plan decides which are used and which victims each one serves, so the number of victims per satellite is not fixed.
- Satellite s *covers* victim v if their distance is at most R₁, the guaranteed single-parcel round-trip range (Section 4.2). Victims may only be served from satellites that cover them.
- The generator ensures that every victim is covered by at least two candidates (Section 8).

**Drones and batteries.** Drones fly at a constant speed vₐ = 60 km/h.

- **Trips.** Each trip starts and ends at the satellite, carries at most P parcels, and must be battery-feasible. Battery drain per minute grows with the load on board, so a loaded drone's range is shorter. After each drop the drone is lighter and its range grows.
- **Times.** A trip has launch Sₗ, recovery Sᵣ and drop time κ per victim.
- **Batteries.** Batteries are swapped **instantly**, and spare batteries are unlimited, so a drone can launch again as soon as it lands.
- **Dwell.** The truck waits until its last drone has landed. This waiting time is the satellite's dwell Wₛ.

**Priority classes.**

| Class | Weight wᵥ | Target Dᵥ | Treatment |
| --- | --- | --- | --- |
| Critical (injured, medical needs) | 5 | 2 h | Deadline with a dominant penalty |
| High (children, elderly, isolated) | 2 | 4 h | Soft target, lateness penalized |
| Normal | 1 | 8 h | Soft target, lateness penalized |

These are default values. If a critical victim cannot be reached by Dᵥ even with a dedicated truck and drone, the generator extends its deadline (Section 8).

**Dynamics.** Time is divided into steps of Δt = 1 h (30 min as a variant).

- **Changes.** Road states change only at step boundaries. Roads inside a flood zone follow a Markov chain over {open, damaged, closed}, and closed roads may reopen. Roads outside the zone stay open.
- **Frozen roads.** A road that a truck has already driven, or is driving at the boundary, never changes again. Its state is fixed from that moment. Likewise, once a truck reaches a satellite, the victims served there and their drone trips are fixed until the last drone lands.
- **Information.** Changes are known to the planner as soon as they occur.
- **Re-planning.** It runs at every boundary, on everything not yet done or under way.

**Assumptions.** One depot; homogeneous trucks and drones; constant speeds; drones are unaffected by road states and weather; each drone returns to its own truck.

## 4. Mathematical formulation

This section states the problem at one decision time, with road states known. Section 5 shows how it is re-solved over time. There are two linked parts: a drone subproblem at each satellite (4.2) and a multi-trip truck problem over satellites (4.3).

### 4.1 Notation

| Symbol | Meaning |
| --- | --- |
| 0, S, V, K | Depot, candidate satellites, victims, trucks |
| Vᶜ ⊆ V | Critical victims |
| r = 1, …, R̄ | Trip index of a truck |
| U, Q | Drones per truck, truck capacity in parcels |
| q, P | Parcel weight (kg), drone payload (parcels) |
| wᵥ, Dᵥ | Priority weight and target time of victim v |
| aᵥₛ | 1 if satellite s covers victim v (distance ≤ R₁) |
| cᵢⱼ(t) | Fastest truck time from i to j at time t (always finite) |
| θ | Reload time at the depot between truck trips |
| Sₗ, Sᵣ, κ | Drone launch, recovery and drop times |
| η, w₀, E, rₘᵢₙ | Energy rate per kg-minute, empty drone mass, battery capacity, reserve |
| vₜ, vₐ | Truck speed (40 km/h), drone speed (60 km/h) |
| Π, πₗ | Critical-deadline penalty, lateness weight |
| cₜ, cₐ, λ | Costs of truck driving and drone energy, and their weight |

The main decision variables are:

- yᵥₛ ∈ {0, 1}: victim v is served from satellite s (allowed only if aᵥₛ = 1);
- zₛ ∈ {0, 1}: satellite s is used;
- xᵢⱼₖᵣ ∈ {0, 1}: truck k drives from i to j on its trip r;
- uₖᵣ ∈ {0, 1}: truck k makes a trip r;
- αₖᵣ, ωₖᵣ: start and return times of truck k's trip r;
- Aₛ, Wₛ: truck arrival and dwell at satellite s;
- Tᵥ: time victim v receives its parcel.

### 4.2 Drone subproblem at a satellite

Given the victims Vₛ = {v : yᵥₛ = 1}, the drones solve a multi-trip CVRP.

**Battery drain depends on load.** With n parcels on board, a drone uses this fraction of the battery per minute:

```latex
p(n) = \frac{\eta\,(w_0 + n\,q)}{E}
```

**Trips.** A trip ρ = (v₁, …, vₘ) visits m ≤ P victims and returns. Let nₗ be the number of parcels on board along leg l, which falls by one after each drop, and let eₗᵣ be the energy for launch and landing. The trip's duration dᵨ and battery constraint are:

```latex
d_\rho = S_l + \sum_{l} \frac{\text{len}_l}{v_a} + m\,\kappa + S_r,
\qquad
\sum_{l} p(n_l)\,\frac{\text{len}_l}{v_a} + \sum_{i=1}^{m} p(m - i + 1)\,\kappa + e_{lr} \;\le\; 1 - r_{\min}
```

A loaded drone drains faster, so the reachable distance shrinks with the load and grows after each drop.

**Guaranteed single-trip range.** A trip that carries one parcel to v and returns empty is feasible whenever dist(s, v) ≤ R₁:

```latex
R_1 = v_a \cdot \frac{1 - r_{\min} - e_{lr} - p(1)\,\kappa}{p(1) + p(0)}
```

With the drone of Meng & Ma (2026) (η = 0.667 Wh/(kg·min), w₀ = 6 kg, E = 180 Wh), q = 1 kg, rₘᵢₙ = 0.2 and vₐ = 60 km/h, we get p(1) ≈ 0.026 and p(0) ≈ 0.022 per minute, so **R₁ ≈ 15 km**. We keep the per-minute drain at its cruise value, a simplification. Since every victim is assigned only to a satellite that covers it, a trip to v alone is always feasible, so every Vₛ has a feasible drone plan. Coverage depends only on fixed locations and the battery model, so aᵥₛ is computed once when the instance is generated (one distance check per victim–satellite pair, e.g. 100 × 25 = 2,500 checks) and never changes during operations: road damage affects only the trucks.

**Schedule with instant battery swaps.** Drone u flies trips ρᵤ₁, …, ρᵤₙ back to back. With times measured from the truck's arrival:

```latex
b_{u1} = 0, \qquad b_{u,\ell+1} = b_{u\ell} + d_{\rho_{u\ell}}, \qquad \delta_v = b_{u\ell} + o_{\rho_{u\ell} v} \;\; (v \in \rho_{u\ell}), \qquad W_s = \max_u \big( b_{u n_u} + d_{\rho_{u n_u}} \big)
```

Here oᵨᵥ is the time from launch to v's drop.

**Objective of the drone subproblem.** A drone plan delays two groups: its own victims, through δᵥ, and every victim served later by the same truck, through Wₛ. Let Ω⁺ₛ be the priority weight served after s by that truck. The plan minimizes:

```latex
J_a(s) = \sum_{v \in V_s} w_v\, \delta_v + \Omega^{+}_{s}\, W_s + c_a \sum_\rho e_\rho
```

The solver keeps a small menu of plans. Short-makespan plans suit satellites early in a long route; critical-first plans suit satellites with critical victims or late in a route (Section 6.1).

### 4.3 Multi-trip truck problem

Victim v receives its parcel at:

```latex
T_v = \sum_s y_{vs}\,(A_s + \delta_v)
```

**Constraints.**

```latex
\sum_s a_{vs}\, y_{vs} = 1 \;\; \forall v \qquad y_{vs} \le z_s, \;\; z_s \le \sum_v y_{vs} \qquad \sum_{k,r,i} x_{iskr} = z_s \;\; \forall s
```

The first constraint is full coverage; the last makes each used satellite visited exactly once over all trucks and trips.

```latex
\sum_j x_{0jkr} = u_{kr}, \qquad \sum_i x_{ihkr} = \sum_j x_{hjkr} \;\; \forall h,k,r, \qquad u_{k,r+1} \le u_{kr}
```

Each trip leaves and returns to the depot, and trips are used in order.

```latex
\sum_{s} \Big( \sum_v y_{vs} \Big) \sum_i x_{iskr} \le Q \quad \forall k, r
```

This is the capacity per trip, in parcels.

```latex
A_j \ge \alpha_{kr} + c_{0j} - M(1 - x_{0jkr}), \qquad A_j \ge A_i + W_i + c_{ij} - M(1 - x_{ijkr})
```

```latex
\omega_{kr} \ge A_i + W_i + c_{i0} - M(1 - x_{i0kr}), \qquad \alpha_{k,r+1} \ge \omega_{kr} + \theta, \qquad \alpha_{k1} \ge 0
```

These time constraints also remove subtours.

**Objective.** One weighted sum, in which the deadline penalty dominates:

```latex
J_0 = \sum_{v \in V^c} \Big[ \mathbf 1(T_v > D_v) + \frac{(T_v - D_v)^+}{D_v} \Big]
```

```latex
J_1 = \frac{1}{\sum_v w_v} \sum_v w_v \big[ T_v + \pi_l (T_v - D_v)^+ \big], \qquad J_2 = c_t \sum_{i,j,k,r} c_{ij}\, x_{ijkr} + c_a \sum_{s,\rho} e_\rho
```

```latex
\min \; \Pi\, J_0 + J_1 + \lambda\, J_2, \qquad \Pi \gg \max (J_1 + \lambda J_2)
```

J₀ counts critical victims served after their deadline, plus how late they are. Because Π dominates, a plan that meets every critical deadline is always preferred, so the deadline behaves as a hard constraint whenever it can be met. When road damage makes that impossible, the plan misses as few deadlines, by as little, as it can, instead of failing. J₁ is the priority-weighted average waiting time, with lateness against the soft targets.

### 4.4 Structure and feasibility

**Cumulative form.** Write truck k's whole sequence, across all its trips, as stops π₁, …, πₘ, where a depot reload is a stop with dwell θ and weight 0. With Ωₛ = Σᵥ wᵥ yᵥₛ and Ω≥ₚ the weight from stop p onward:

```latex
\sum_v w_v T_v = \sum_{p} \Omega_{\ge p}\, c_{\pi_{p-1}\pi_p} + \sum_{p} \Omega_{> p}\, W_{\pi_p} + \sum_v w_v\, \delta_v
```

This has three consequences:

1. **Known problem class.** The truck problem is a weighted, multi-trip cumulative CVRP with service times (Ngueveu et al. 2010), so its moves and bounds apply.
2. **Ordering rules.** Ignoring travel, two neighbouring satellites i, j should swap when Ωⱼ / Wⱼ > Ωᵢ / Wᵢ (Smith 1956). For the same reason, a reload delays everything after it, so heavy-priority satellites belong on a truck's first trip.
3. **Dwell cost.** With the assignment y fixed, total dwell is constant. It matters only through later arrivals and when victims move between satellites.

**Proposition (full coverage is always feasible).** Suppose that:

- every victim is covered by some satellite;
- some covering assignment gives each used satellite at most Q victims;
- all road times are finite.

Then a plan serving every victim exists. Proof: serve each used satellite on its own truck trip. Each trip is within capacity, each drone plan is feasible because single trips are, and each trip ends in finite time. The generator verifies the first two conditions (Section 8), and closed roads at ×5 give the third. So coverage never needs a fallback; only timeliness is at stake.

## 5. Dynamic re-planning

The plan is re-optimized at every step boundary tₖ = k·Δt (Δt = 1 h), the only times road states change. Over an operation of 8–12 h this means 8–12 re-plans. Each one keeps what is done or under way and re-solves the rest of the problem of Section 4.

*(Figure: one time step · three re-planning levels. See the online doc.)*

Each step checks how much the road changes hurt the current plan, and only a large loss triggers the full re-optimization. Section 7 replaces this fixed rule with a learned policy.

### 5.1 Road dynamics

**Flood zone.**

- **Centre:** the location of a randomly chosen victim, jittered by up to 5% of the side and clipped to the map, so it always lies inside the area and where people are.
- **Radius:** it is set so that about 30% of the roads are in the zone. A road is in the zone if any part of it lies within the radius of the centre, so a road with only one end inside, or one that merely passes through, is affected too.
- Roads outside the zone stay open.

**Initial state.** At t = 0, each zone road's state is drawn independently from the long-run mix of the chosen intensity (the last column below), so the first plan already faces damage and the damage level stays stable on average.

**Hourly transitions.** At each boundary, every zone road that is not frozen moves according to a Markov matrix. Rows are "from" open, damaged, closed; columns are "to" the same states:

| Intensity | Hourly matrix P | Long-run open / damaged / closed |
| --- | --- | --- |
| Low | \[\[0.95, 0.04, 0.01\], \[0.30, 0.66, 0.04\], \[0.13, 0.13, 0.74\]\] | 83% / 12% / 6% |
| Medium | \[\[0.88, 0.08, 0.04\], \[0.28, 0.63, 0.09\], \[0.17, 0.13, 0.70\]\] | 65% / 20% / 15% |
| High | \[\[0.80, 0.11, 0.09\], \[0.23, 0.59, 0.18\], \[0.12, 0.13, 0.75\]\] | 45% / 23% / 32% |

A closure lasts about 3 hours on average, and closed roads can reopen. For the 30-minute variant, these matrices are replaced by matrices that give the same long-run shares.

**Frozen roads.** A road that any truck has driven, or is driving at a boundary, keeps its state from then on. Changes therefore never affect a truck mid-road, and a truck returning along a road it used keeps the travel time it saw.

After each update, fastest-path times cᵢⱼ(t) are recomputed with Dijkstra. Every road time is finite, so every satellite stays reachable.

### 5.2 State and locks at a boundary

The state at tₖ is:

- the road states;
- the victims served so far;
- the current assignment of unserved victims;
- each drone's ongoing trip;
- each truck's mode, remaining parcels and *free point*: where and when its plan may change.

| Truck mode at tₖ | Locked | Free point |
| --- | --- | --- |
| Driving road (i, j) | The road and its arrival time at j | Node j, at that arrival time |
| Dwelling at satellite s | The satellite's whole drone plan: its victims, trips and their order | s, when its last drone lands |
| Reloading at the depot | The reload | Depot, at the end of the reload |
| Waiting at the depot or finished | Nothing | Depot, now |

Only the current road is locked. A truck heading to a satellite along several roads can be re-routed at the next junction.

### 5.3 Residual problem

The residual problem P(tₖ) is the model of Section 4 restricted to what remains:

- **Trucks** start at their free points, with the parcels still on board. They may finish their current trip, return and reload for further trips.
- **Satellites** are the unvisited candidates, plus each active satellite, which stays first on its truck.
- **Victims** are those not yet served and not assigned to an active satellite. Any of them may move to another satellite that covers it.

Arrival times start from the free point (node hₖ, time φₖ):

```latex
A_j \ge \varphi_k + c_{h_k j}(t_k) - M\,(1 - x_{h_k j k r})
```

The drone level is re-planned with the trucks:

1. **Re-solved drone plans.** A satellite whose victim set changes, or whose route position changes Ω⁺ₛ, gets its drone plan re-solved. The truck problem then sees the new Wₛ and δᵥ.
2. **Frozen service at an active satellite.** Once a truck reaches a satellite, that satellite's victims and drone plan are fixed until its last drone lands. Re-planning never touches them; it only reshapes satellites that no truck has reached yet.
3. **Critical-first ordering.** Drone plans at satellites with critical victims are re-ordered so critical drops come first when their deadlines are tight.

### 5.4 One re-planning step

1. **Simulate** the current plan up to tₖ, and record every delivery time Tᵥ.
2. **Freeze** served victims, visited satellites, roads driven or occupied, and drone trips in the air. Compute each truck's free point.
3. **Update** the states of unfrozen zone roads, and recompute fastest paths.
4. **Evaluate** the current plan under the new times: ΔJ is the change in Π J₀ + J₁ + λJ₂.
5. **Choose** a re-planning level (rule below, or the learned policy of Section 7):

   | Level | Rule | Action |
   | --- | --- | --- |
   | L0 — path update | ΔJ ≤ 1% and no new critical deadline at risk | Keep the satellite order; only the road paths change |
   | L1 — repair | ΔJ ≤ 5% | A few local moves from the current plan |
   | L2 — re-optimize | ΔJ > 5% or a critical deadline at risk | Warm-started ALNS with victim re-assignment and drone re-solves |
6. **Commit** the new plan. Trucks at their free points start their next road, which freezes it.
7. Continue to tₖ₊₁.

**Coverage during execution.** By the Proposition of Section 4.4, every re-plan has a plan that serves everyone, so full coverage is never at risk. What changes from step to step is timeliness. If road damage makes a critical deadline impossible to meet, J₀ keeps the number and size of misses as small as possible.

## 6. Solution method

We solve each problem P(tₖ) by decomposition. A fast drone solver gives every satellite its dwell and victim offsets. An adaptive large neighbourhood search (ALNS) over truck trips and victim assignments calls it through a cache. The same search builds the first plan at t = 0, and is warm-started at every L2 re-plan.

*(Figure: solution framework · optimizer, simulator and the two learned policies. See the online doc.)*

After the first plan, each hour runs the same loop. Roads change; the re-planning policy picks a level; for L1 or L2 the optimizer rebuilds the plan, with the ALNS calling the drone solver for every satellite it changes; the plan is committed until the next hour. The two learned components, shaded, only choose levels and operators.

### 6.1 Drone solver

For a satellite s with victims Vₛ:

1. **Giant tour.** One tour through Vₛ (nearest neighbour, then 2-opt). A critical-first variant puts critical victims at the front.
2. **Exact split under the battery limit.** Dynamic programming cuts the tour into trips (Prins 2004). A cut is allowed only if the trip carries at most P parcels and satisfies the load-dependent energy constraint of Section 4.2, evaluated with the falling load leg by leg. This takes O(m²) time for m victims.
3. **Schedule on U drones.** Swaps are instant, so each drone flies its trips back to back. Two orders are tried:
   - longest trip first, for a short dwell Wₛ;
   - critical-first, or highest weight per minute (Σw / d\_ρ, Smith's rule), for early drops to urgent victims.
4. **Menu.** Keep the non-dominated pairs (Wₛ, Σ wᵥ δᵥ), usually two to four plans. The truck search picks the plan that minimizes Jₐ at the satellite's route position, without re-running the solver.

Results are cached by (satellite, victim set). A victim move changes only two entries, so most calls are cache hits.

### 6.2 Truck ALNS

**Representation.** Each truck has one sequence of stops across all its trips, with the depot as a reload separator: (0, s₁, s₂, 0, s₃, …, 0). An active satellite is pinned as the first stop. Each victim carries its current satellite label.

**Destroy operators** (remove 10–30% of the satellites):

- random removal;
- worst removal: the largest weighted-delay contribution Ωₛ · Aₛ, and satellites holding critical victims close to their deadline;
- related removal, by distance;
- **damage-related removal**: satellites whose paths use roads that changed state at this step;
- trip removal: a whole truck trip.

**Repair operators.** Greedy and regret-2 insertion with the cumulative cost of 6.3. A satellite may be inserted into an existing trip or open a new trip after a reload. Satellites are inserted in decreasing Ωₛ, with critical satellites first.

**Victim moves.** Re-assign a victim to another covering satellite, or swap two victims between satellites. Each move re-solves the two drone plans through the cache, and checks trip capacity.

**Local search** after each repair: relocate, swap, 2-opt\*, Or-opt, and moving a satellite between trips.

**Acceptance.** Simulated annealing on Π J₀ + J₁ + λJ₂. The destroy and repair operators are chosen either by adaptive weights (Ropke & Pisinger 2006) or by the learned selector of Section 7.2.

**Budgets.** About 30 s for the first plan, and 5–10 s per L2 re-plan.

### 6.3 Fast move evaluation

For each truck sequence we store prefix arrivals, dwell ends and suffix weights Ω≥ₚ. Inserting satellite s between stops a and b at position p shifts every later arrival by:

```latex
\delta = c_{as} + W_s + c_{sb} - c_{ab}
```

The change in weighted waiting time is then:

```latex
\Delta \sum_v w_v T_v = \Omega_s \big( A_a + W_a + c_{as} \big) + \delta\, \Omega_{\ge p} + \sum_{v \in V_s} w_v \delta_v
```

This takes O(1) time. Depot reloads are stops with weight 0, so the same formula covers multi-trip routes.

**Deadlines.** For each position we store the forward slack: how much later the remaining stops can be reached before any critical victim misses its deadline (Savelsbergh 1992). A move that shifts arrivals by δ ≤ slack changes J₀ by nothing, again in O(1). Otherwise J₀ is recomputed for the affected suffix only.

**Capacity.** Each trip stores its parcel count, so moves are checked in O(1).

### 6.4 Initial plan

1. **Assignment.** Each victim goes to its nearest covering satellite. A greedy pass then moves victims from heavily loaded satellites to their second-nearest coverer.
2. **Drone plans.** Solve every used satellite's menu.
3. **Truck trips.** Regret-2 insertion in decreasing Ωₛ, opening a new trip when capacity is reached, followed by ALNS.

## 7. Learning-assisted re-planning

We use deep reinforcement learning to decide **how** to re-optimize, not to build routes. Two small policies sit on top of the optimizer of Section 6. One chooses how much effort each hourly re-plan gets (7.1); the other chooses ALNS operators inside a re-plan (7.2).

**Why not learn the routes end to end?** There are three reasons:

- **Constraints.** Our constraints are hard and coupled: trip capacity, load-dependent battery limits, coverage, frozen roads, locked drone trips. A learned route builder needs masking for every one of them and still gives no guarantee.
- **Data.** Disaster data is scarce, and policies trained on one geometry transfer poorly.
- **Safety.** In our design, every action the policy can take produces a feasible plan built by the optimizer. A poor policy costs quality or computing time, never feasibility.

### 7.1 Re-planning policy

At each boundary tₖ the policy picks the re-planning level that the fixed rule of Section 5.4 picks today.

**State xₖ** (normalized features):

- the time tₖ relative to the expected end of operations;
- the share of victims still unserved, per priority class;
- the number of critical victims at risk, and the smallest critical-deadline slack;
- the number of unvisited satellites, parcels still on trucks, and trucks in each mode;
- the number of changed roads on the planned paths, and the share of planned driving time on damaged or closed roads;
- ΔJ after an L0 update and after an L1 repair (both cheap to compute);
- the share of closed roads in the flood zone, and the computing time used so far.

**Actions aₖ** ∈ {L0 path update, L1 repair, L2 with a 2 s budget, L2 with a 10 s budget}.

**Reward.** The negative of the realized objective accumulated during (tₖ, tₖ₊₁\], minus a computing-time penalty:

```latex
r_k = -\Big[ \Pi\, \Delta J_0^{\text{real}} + \Delta J_1^{\text{real}} + \lambda\, \Delta J_2^{\text{real}} \Big]_{(t_k,\, t_{k+1}]} - \lambda_{\text{cpu}}\, \mathrm{CPU}_k
```

Summing the rewards over an episode gives exactly the negative of the realized objective plus total computing cost. So maximizing the return optimizes what we report.

**Learning.**

- **Algorithm:** a double DQN (Mnih et al. 2015; van Hasselt et al. 2016), with experience replay, a target network and ε-greedy exploration, using a small network (two hidden layers of 64).
- **Episodes:** one episode is one simulated operation, on a training instance with a sampled road scenario.
- **Baseline:** the fixed rule of Section 5.4 is both the baseline and the warm-up policy that fills the replay buffer.

### 7.2 Operator selection inside the ALNS

Inside one L2 re-plan, each ALNS iteration chooses a (destroy, repair) pair: 5 × 2 = 10 actions.

**State:**

- the fraction of the iteration budget used;
- the annealing temperature;
- the improvement since the last new best;
- the last pair chosen;
- the gap between the current and best objectives;
- whether this re-plan was triggered by damage, by a deadline at risk, or both.

**Reward:** the relative improvement of the current objective, plus a bonus for a new best.

A DQN trained across instances replaces the adaptive weights of Ropke & Pisinger (2006). It can specialize: for example, it can prefer damage-related removal right after large road changes. Our earlier static study already includes a DQN operator selector that can be reused here.

### 7.3 Training and evaluation

- **Split:** policies are trained on instances and road seeds kept apart from the test set (Section 8).
- **Generalization:** we train at medium damage intensity and test at low, medium and high, and on unseen layouts.
- **Comparisons:** each policy is compared with the fixed rule, with always running L2, and with the full-information oracle (Section 9).

## 8. Benchmark generation

Instances are built from Solomon's (1987) VRPTW set. Every instance is guaranteed feasible, with every victim in single-trip drone range of at least two candidate satellites, and the guarantee is checked by a validation report.

*(Figure: data generation pipeline · eight steps, a check and the saved files. See the online doc.)*

The steps run left to right on the top row and right to left on the bottom row. Step 4, the double-coverage placement of satellites, is what guarantees every instance is feasible. The details of each step follow.

**Base data.**

- **Layouts:** Solomon's four distinct layouts, C1 (clustered), C2 (clustered, wider), R1 (random) and RC1 (mixed). R2 and RC2 share the coordinates of R1 and RC1.
- **Victims:** the first n = 25, 50 or 100 customers, as in Solomon's own subsets. The Solomon depot is our depot.
- **Dropped:** Solomon's demands (each victim needs one parcel), time windows and service times.

**Scale.** The map side L is set so that the guaranteed drone range is a fixed share of it: R₁ = αL, with α = 0.25 by default and 0.35 as a variant. With R₁ ≈ 15 km, L = 60 km (1 Solomon unit = 600 m), or about 43 km.

**Priorities.** Classes are drawn at random (20% critical, 30% high, 50% normal). In a variant, victims closer to the flood centre are more likely to be critical.

**Candidate satellites (double coverage).**

1. **Grid.** Candidates lie on a grid of spacing g = ρR₁ / √2, with ρ = 0.6. Every point then has the four corners of its grid cell within ρR₁, so each victim has at least four candidates in range.
2. **Selection.** We choose the fewest candidates such that every victim is within ρR₁ of at least two. This is a set multicover problem, solved with CP-SAT in seconds. Candidates inside an optional no-go core of the flood zone are excluded beforehand.
3. **Margin.** Using ρR₁ instead of R₁ leaves battery for multi-drop trips, so drones usually carry two or three parcels rather than one.

The resulting number of candidates on the actual Solomon files:

| Layout | n = 25 / 50 / 100, α = 0.25 | n = 25 / 50 / 100, α = 0.35 | Victims per candidate if each goes to its nearest (median, max) |
| --- | --- | --- | --- |
| C1 | 6 / 9 / 18 | 4 / 7 / 13 | 4–7, up to 18 |
| C2 | 8 / 13 / 20 | 6 / 9 / 14 | 3–7, up to 11 |
| R1 | 13 / 18 / 23 | 9 / 12 / 13 | 2–7, up to 15 |
| RC1 | 6 / 11 / 25 | 6 / 10 / 17 | 4–6, up to 10 |

The optimizer decides which candidates are used and how many victims each serves.

**Fleet.**

- **Trucks:** K = 2, 3 or 4 for n = 25, 50 or 100.
- **Capacity:** Q = ⌈1.15 n / K⌉ parcels by default. A tight variant, Q = ⌈0.6 n / K⌉, forces second trips.
- **Check:** the generator verifies that some covering assignment loads every satellite with at most Q victims, the condition of the Proposition in Section 4.4. The nearest assignment usually suffices; in the tight variant a small assignment model checks it.
- **Drones:** U = 3 drones per truck, each carrying P = 3 parcels.

**Road graph.** Roads follow the candidate grid: each grid node links to its four neighbours, and the depot links to its four nearest nodes. Base times are the Euclidean length × 1.2 (road circuity) / vₜ. A grid has no bridge edges, so every satellite has at least two separate paths to the depot.

**Flood zone and scenarios.**

- **Zone:** centred on a random victim, as in Section 5.1.
- **Road histories:** for each instance, intensity and seed, the full hourly road history is generated in advance and stored. Every strategy then faces the same realization.
- **Volume:** 10 seeds per instance and intensity, with 3 seeds of every instance kept apart for training the policies of Section 7.

**Deadline check.** For each critical victim, a lower bound on its service time is computed: the fastest drive at t = 0 to its nearest covering satellite, plus one direct drone trip. If this bound exceeds Dᵥ, the deadline becomes ⌈1.2 × bound⌉ and the change is logged. Each critical deadline is therefore reachable on its own.

**Parameters.**

| Parameter | Value |
| --- | --- |
| Truck speed vₜ, drone speed vₐ | 40 km/h, 60 km/h |
| Parcel q, drone payload P, drones per truck U | 1 kg, 3 parcels, 3 |
| Drone energy: η, w₀, E, rₘᵢₙ | 0.667 Wh/(kg·min), 6 kg, 180 Wh, 20% |
| Launch Sₗ, recovery Sᵣ, drop κ | 1, 2, 0.1 min |
| Battery swap | instant, unlimited spares |
| Truck reload θ | 15 min |
| Time step Δt | 1 h (variant 30 min) |
| Road multipliers | open 1, damaged 2.5, closed 5 |

**Size of the benchmark.** 4 layouts × 3 sizes × 2 values of α = 24 instances, each with 3 intensities × 10 seeds.

## 9. Experimental design (planned)

No results are reported yet. The study tests three claims:

- re-planning both levels serves victims sooner and misses fewer critical deadlines than a static plan;
- the learned policy matches always re-optimizing at a fraction of the computing time;
- each modelling choice matters, which the ablations measure.

**Instances.** The 24 instances of Section 8, each at 3 intensities with 10 road seeds. Three seeds per instance are used only for training.

**Strategies compared.**

| Strategy | Description |
| --- | --- |
| Static | Plan at t = 0; trucks follow it, taking the fastest path at current road states, but never re-sequence |
| Always L2 | Full re-optimization (10 s) at every step |
| Rule | Levels L0 / L1 / L2 chosen by the fixed rule of Section 5.4 |
| DRL policy | Levels chosen by the double DQN of Section 7.1 |
| DRL policy + DRL operators | As above, with the learned operator selector of Section 7.2 inside L2 |
| Oracle | Knows the full road history at t = 0 and plans once (a lower bound) |

**Ablations** (run with the Rule strategy unless stated):

| Factor | Values | Question |
| --- | --- | --- |
| Drones per truck U | 1, 2, 3, 4 | How much does dwell, and with it waiting time, fall per extra drone? |
| Multi-trip trucks | allowed / not allowed (tight Q) | What do second trips save when capacity is tight? |
| Drone-level re-planning | on / off (assignment and drone plans frozen at t = 0) | How much of the gain comes from re-planning the drones? |
| Critical deadline | dominant penalty / soft (same πₗ as other classes) | How many critical misses does the dominant penalty prevent, and at what cost to others? |
| Closed-road multiplier | ×5, ×10 | How sensitive are the results to closure severity? |
| Time step Δt | 1 h, 30 min | Does re-planning more often pay off? |
| Range share α | 0.25, 0.35 | How does drone range relative to the map change satellite use? |

**Metrics.**

- critical victims served late, and total critical lateness (J₀);
- priority-weighted average waiting time (J₁), and the average and maximum Tᵥ per class;
- makespan: the time the last victim is served;
- truck driving time, number of truck trips, total dwell and drone energy (J₂);
- satellites used, and victims per used satellite;
- re-planning: share of steps at each level, and CPU time per step;
- plan stability: satellites moved between trucks or trips per re-plan.

All comparisons are paired: the same instance and road seed for every strategy, tested with Wilcoxon signed-rank tests.

## 10. Limitations and extensions

This version is deliberately simple. We would add extensions in this order:

- **Demand changes.** Demand is fixed at one parcel per victim. Growing demand, or newly reported victims, could be handled by re-opening the not-yet-launched trips of an active satellite.
- **Partial information.** Road states are known as soon as they change. Noisy reports, a belief over road states and robust re-planning are a natural next step.
- **Battery logistics.** Swaps are instant and spares unlimited. A finite stock of batteries charged on the truck would add a resource constraint.
- **Drone physics.** The drain per minute is held at its cruise value, and wind, weather and no-fly zones are ignored. Drones always return to their own truck.
- **Satellites.** Candidate sites never fail. A satellite that floods could be handled by the same re-assignment moves.
- **Exact bounds.** Lower bounds for the cumulative CVRP would give optimality gaps on small instances; an exact model of the static problem in OR-Tools or Gurobi is planned for validation.

## References

- Chen, X., Ulmer, M. W., & Thomas, B. W. (2022). Deep Q-learning for same-day delivery with vehicles and drones. *European Journal of Operational Research*, 298(3), 939–952. Preprint: [arXiv:1910.11901](https://arxiv.org/abs/1910.11901)
- Du, W., Wang, J., Liu, J., Wang, H., Zhou, J., Mei, C., & Pang, Y. (2026). Optimization of emergency supplies allocation for urban flood disasters. *Results in Engineering*, 32, 112619. [doi](https://doi.org/10.1016/j.rineng.2026.112619)
- Duan, H., Li, X., Feng, Y., Zhang, G., Lu, Q., & Meng, X. (2026). Multi-objective hyper heuristic memetic algorithm for time-dependent vehicle routing problem with drone considering simultaneous pickup and delivery. *Expert Systems with Applications*, 297, 129196. [doi](https://doi.org/10.1016/j.eswa.2025.129196)
- Li, J., Guo, Q., & Fang, Y. (2026). Truck–drone cooperative delivery problem based on two-stage algorithm. *Symmetry*, 18, 1199. [doi](https://doi.org/10.3390/sym18071199)
- Li, Q., & Gonsalves, T. (2026). A reinforcement learning framework for traveling salesman and vehicle routing problem with drones. *Drones*, 10, 616. [doi](https://doi.org/10.3390/drones10080616)
- Luo, Y., Chen, Y., & Wahab, M. I. M. (2027). Multi-trip truck-drone routing problem with simultaneous pickup and delivery and temporal constraints. *Transportation Research Part E*, 217, 105235. [doi](https://doi.org/10.1016/j.tre.2026.105235)
- Meng, S., & Ma, H. (2026). Electric vehicle routing problem with multi-visit drones considering time windows and battery swapping stations. *Transportation Research Part E*, 215, 105107. [doi](https://doi.org/10.1016/j.tre.2026.105107)
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533. [doi](https://doi.org/10.1038/nature14236)
- Ngueveu, S. U., Prins, C., & Wolfler Calvo, R. (2010). An effective memetic algorithm for the cumulative capacitated vehicle routing problem. *Computers & Operations Research*, 37(11), 1877–1885. [doi](https://doi.org/10.1016/j.cor.2009.06.014)
- Nguyen, D. V. A., Piplani, R., & Gunawan, A. A two-echelon covering tour vehicle routing problem with drones for post-disaster relief. *arXiv:2609.38227* (under review).
- Peng, W., Wang, D., Yin, Y., & Cheng, T. C. E. (2025). Multi-agent deep reinforcement learning-based truck-drone collaborative routing with dynamic emergency response. *Transportation Research Part E*, 195, 103974. [doi](https://doi.org/10.1016/j.tre.2025.103974)
- Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985–2002. [doi](<https://doi.org/10.1016/S0305-0548(03)00158-8>)
- Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. *Transportation Science*, 40(4), 455–472. [doi](https://doi.org/10.1287/trsc.1050.0135)
- Savelsbergh, M. W. P. (1992). The vehicle routing problem with time windows: Minimizing route duration. *ORSA Journal on Computing*, 4(2), 146–154. [doi](https://doi.org/10.1287/ijoc.4.2.146)
- Shi, Y., Guo, Q., Pan, P., Chen, M., Wu, Y., & Guo, H. (2026). Robust optimization for two-echelon truck and drone collaborative delivery in emergency relief. *Applied Soft Computing*, 201, 115738. [doi](https://doi.org/10.1016/j.asoc.2026.115738)
- Smith, W. E. (1956). Various optimizers for single-stage production. *Naval Research Logistics Quarterly*, 3(1–2), 59–66. [doi](https://doi.org/10.1002/nav.3800030106)
- Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254–265. [doi](https://doi.org/10.1287/opre.35.2.254)
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1). [doi](https://doi.org/10.1609/aaai.v30i1.10295)
- Yang, Y., Yan, C., Cao, Y., & Roberti, R. (2023). Planning robust drone-truck delivery routes under road traffic uncertainty. *European Journal of Operational Research*, 309, 1145–1160. [doi](https://doi.org/10.1016/j.ejor.2023.02.031)
- Yu, F., Li, C., Chen, Q., Li, Y., Chen, Y., & Yang, X. (2027). Multi-objective two-echelon vehicle-drone routing with load-dependent drone under pickup delivery time windows. *Transportation Research Part E*, 217, 105195. [doi](https://doi.org/10.1016/j.tre.2026.105195)
