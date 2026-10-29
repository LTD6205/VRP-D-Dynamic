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
- An instance generator built on the Solomon and Gehring & Homberger benchmarks (25 to 1,000 victims) that guarantees every victim lies within single-trip drone range of at least two candidate satellites.

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
6. **Benchmark generator.** Instances from the Solomon and Gehring & Homberger benchmarks (25 to 1,000 victims) with equal demands, priority classes, satellite candidates placed by double coverage, a road graph, a flood zone and hourly damage scenarios.

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

- Each truck carries at most Q units per trip.
- A trip leaves the depot, visits one or more satellites and returns.
- After returning, a truck may reload (time θ) and start another trip.
- Each satellite is visited **at most once over the whole operation**. There are no revisits and no second deliveries to the same satellite.
- Drones and batteries do not count toward Q.

**Victims and demand.**

- Every victim v needs exactly **one unit (a parcel of q kg)**, with q ≤ the drone payload. So each victim is served by one drone drop and never revisited.
- Each victim has a priority class (below).
- Capacities are counted in units: each victim needs 1 unit, a drone carries up to P = 3 units per trip, and a truck carries up to Q = 500 units per trip. One unit weighs q kg (1 kg by default); only the battery model uses the weight.

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

**Battery figures.** At a constant 60 km/h, the drain per minute equals the drain per kilometre. The table gives it for each load, with the distance a drone could fly on the usable 80% of the battery while carrying that load:

| Units on board | Drain per minute (per km) | Distance on 80% at that load |
| --- | --- | --- |
| 0 | 2.22% | 36 km |
| 1 | 2.59% | 31 km |
| 2 | 2.96% | 27 km |
| 3 | 3.33% | 24 km |

Launch (1 min, loaded) and recovery (2 min, empty) together cost about 7–8% of the battery. Three reference figures follow. They are not limits that a trip must fall between: every trip is checked on its own energy, as described next.

- **Single-unit round trip:** out with one unit and back empty, up to R₁ ≈ 15.1 km. This is the coverage radius.
- **Full three-unit trip:** to a cluster of victims up to about 12.8 km away.
- **Worked example:** a trip that flies 6 km out, drops at three victims 3 km apart and returns 7 km takes 22.3 min and uses 61% of the battery, inside the 80% limit.

**How a trip is checked.** A candidate trip is a satellite s and an ordered list of victims (v₁, …, vₘ), with m ≤ 3. Each trip starts on a full battery, because batteries are swapped instantly. Distances are in km, which equal minutes at 60 km/h. The check has four steps:

1. **Units on board.** The drone leaves with m units. After the i-th drop it carries m − i, and it flies home empty.
2. **Energy of each part.** Launch: p(m) × 1 min. Each leg: p(units on board) × leg length. Each drop: p(units on board) × 0.1 min hover. Recovery: p(0) × 2 min.
3. **Total.** Add the parts:

```latex
e_\rho = p(m)\cdot S_l + \sum_{i=0}^{m} p(m-i)\,\frac{\ell_i}{v_a} + \sum_{i=1}^{m} p(m-i+1)\,\kappa + p(0)\cdot S_r
```

Here ℓ₀ is the leg from s to v₁, ℓᵢ the leg from vᵢ to vᵢ₊₁, and ℓₘ the leg from vₘ back to s.

4. **Decision.** The trip is feasible if eᵨ ≤ 1 − rₘᵢₙ = 0.80, so at least 20% of the battery is left on landing. Otherwise it is rejected.

**Worked example.** The first victim is 13 km from the satellite, the other two lie 1 km apart beyond it, and the way back is 13.5 km.

| Trip option | Energy used | Feasible |
| --- | --- | --- |
| All three victims in one trip | 87.6% | No |
| Two victims in one trip (13 km out, 2 km between, back) | 81.7% | No |
| One victim per trip (13 km out, 13 km back) | 69.9% | Yes |

These three victims therefore get three single-unit trips, flown by the three drones at the same time, so the dwell is one trip of about 29 minutes. Victims closer to the satellite share trips: the 22.3-minute three-drop trip above uses 61%.

**How the solver chooses trips.** The drone solver orders the satellite's victims into one tour, then the split step (Section 6.1) considers every way to cut that tour into consecutive trips of at most three victims. For the example tour (v₁, v₂, v₃) these are {v₁ v₂ v₃}, {v₁ v₂}{v₃}, {v₁}{v₂ v₃} and {v₁}{v₂}{v₃}. Every candidate trip goes through the four-step check, failing trips are discarded, and the best remaining split is kept. A feasible split always exists: every assigned victim is within R₁ ≈ 15.1 km of its satellite, and a single-unit trip to such a victim always passes. Far victims thus get trips of one or two units, near victims trips of three.

**Victims beyond the 9 km planning margin.** The generator gives every victim two candidates within 9 km, where three-unit trips usually fit. A victim may still be served from any satellite within 15.1 km, for example after a re-assignment; the check above then simply forms smaller trips for it.

**Tracking the battery during execution.**

- **In the plan:** each accepted trip stores its battery level after every leg and drop, so the plan shows the charge left at each delivery.
- **In the simulation:** every drone starts each trip at 100% and subtracts the same leg energies as it flies. Drones do not use roads and fly at a constant speed, so a trip that passed the check cannot run out mid-flight; road damage never changes drone feasibility.
- **In the results:** the simulator logs the lowest battery level reached on each trip, so the experiments can report how close trips come to the 20% reserve.

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

Instances are built from two classic VRPTW benchmarks: Solomon (1987), with 25 to 100 customers, and the extended set of Gehring & Homberger (1999), with 200 to 1,000 customers. Every instance is guaranteed feasible, with every victim in single-trip drone range of at least two candidate satellites, and the guarantee is checked by a validation report.

*(Figure: data generation pipeline · eight steps, a check and the saved files. See the online doc.)*

The steps run left to right on the top row and right to left on the bottom row. Step 4, the double-coverage placement of satellites, is what guarantees every instance is feasible. The details of each step follow.

**Base data.**

- **Solomon:** n = 25, 50 or 100 victims (the first n customers, as in Solomon's own subsets) on the four distinct layouts C1, C2, R1 and RC1. R2 and RC2 share the coordinates of R1 and RC1.
- **Gehring & Homberger:** n = 200, 400, 600, 800 or 1,000 victims (all customers) on the five distinct layouts C1, C2, R1, R2 and RC. RC1 and RC2 share coordinates, and the ten instances of a class differ only in time windows, so we use instance 1 of each class. The files come from the PyVRP instance repository.
- **Depot:** the benchmark depot.
- **Dropped:** demands (each victim needs one unit), time windows and service times.

**Scale.** Every map is scaled to the same 60 km × 60 km region: 1 unit = 60 km / side. That is 600 m for Solomon, and 430 m down to 120 m for Gehring & Homberger as the side grows from 140 to 500. With R₁ ≈ 15 km, the drone range is a quarter of the side (α = 0.25; a 43 km variant gives α = 0.35). Larger instances therefore mean more people in the same flooded region, not a larger region. We tested keeping 600 m per unit instead: the 1,000-victim maps stretched to 300 km, most satellites served only one or two victims, and the fleet estimate rose to 57 trucks.

**Priorities.** Classes are drawn at random (20% critical, 30% high, 50% normal). In a variant, victims closer to the flood centre are more likely to be critical.

**Candidate satellites (double coverage).**

1. **Grid.** Candidates lie on a grid of spacing g = ρR₁ / √2 ≈ 6.4 km, with ρ = 0.6. Every point then has the four corners of its grid cell within ρR₁ ≈ 9 km, so each victim has at least four candidates in range.
2. **Selection.** We choose the fewest candidates such that every victim is within ρR₁ of at least two. This is a set multicover problem, solved with CP-SAT in seconds. Candidates inside an optional no-go core of the flood zone are excluded beforehand.
3. **Margin.** A full three-unit trip can reach a cluster up to about 12.8 km away (Section 4.2), so a 9 km radius leaves battery for the detours between drops.

**Fleet.**

- **Units.** A victim needs 1 unit, a drone carries P = 3 units per trip, and a truck carries Q = 500 units per trip. One unit weighs q kg (1 kg by default); only the battery model uses the weight. Drones and batteries do not count toward Q.
- **Capacity in practice.** With Q = 500, capacity never binds: the fleet carries 3.5 to 40 times the total demand, as is realistic for light relief items. A **stress-test variant** sets Q = ⌈n / (1.5K)⌉, so total demand is about 1.5 times the fleet's capacity per trip and trucks must make second trips.
- **Drones per truck.** U = 3 in every instance, so the fleet differs only in the number of trucks. U ∈ {1, 2, 3, 4} is an ablation.
- **Trucks per instance.** K is sized by workload. The generator estimates the undamaged work: total drone dwell (each victim at its nearest covering satellite, three units per trip, three drones), plus one nearest-neighbour truck tour over the used satellites (×1.2 circuity, 40 km/h). It then sets K = max(2, ⌈work / 6 h⌉). An undamaged plan would finish in about 6 h, leaving room for damage within a 12-hour day.

Candidate satellites and trucks, from a prototype of the placement and sizing steps on the actual benchmark files:

| Set | Layout | Victims | Candidate satellites | Trucks K |
| --- | --- | --- | --- | --- |
| Solomon | C1 | 25 / 50 / 100 | 6 / 9 / 18 | 2 / 2 / 2 |
| Solomon | C2 | 25 / 50 / 100 | 8 / 13 / 20 | 2 / 2 / 3 |
| Solomon | R1 | 25 / 50 / 100 | 12 / 18 / 23 | 2 / 2 / 3 |
| Solomon | RC1 | 25 / 50 / 100 | 6 / 11 / 25 | 2 / 2 / 3 |
| G&H | C1 | 200 / 400 / 600 / 800 / 1,000 | 25 / 32 / 35 / 35 / 37 | 4 / 4 / 5 / 6 / 7 |
| G&H | C2 | 200 / 400 / 600 / 800 / 1,000 | 34 / 32 / 34 / 34 / 33 | 4 / 4 / 5 / 6 / 7 |
| G&H | R1 | 200 / 400 / 600 / 800 / 1,000 | 40 / 44 / 46 / 48 / 47 | 5 / 6 / 6 / 7 / 8 |
| G&H | R2 | 200 / 400 / 600 / 800 / 1,000 | 40 / 44 / 46 / 48 / 47 | 5 / 5 / 6 / 7 / 8 |
| G&H | RC | 200 / 400 / 600 / 800 / 1,000 | 37 / 40 / 44 / 44 / 48 | 4 / 5 / 7 / 7 / 8 |

With the region fixed, the number of candidates levels off at about 35 to 48, and the victims per satellite grow instead: a median of about 4 for Solomon, and 20 to 30 at 1,000 victims (up to 68). The optimizer decides which candidates are used and how many victims each serves.

**Road graph.** Roads follow the candidate grid (about 90 junctions on the 60 km map): each grid node links to its four neighbours, and the depot links to its four nearest nodes. Base times are the Euclidean length × 1.2 (road circuity) / vₜ. A grid has no bridge edges, so every satellite has at least two separate paths to the depot.

**Flood zone and scenarios.**

- **Zone:** centred on a random victim, as in Section 5.1.
- **Road histories:** for each instance, intensity and seed, the full hourly road history is generated in advance and stored. Every strategy then faces the same realization.
- **Volume:** 10 seeds per instance and intensity, with 3 seeds of every instance kept apart for training the policies of Section 7.

**Deadline check.** For each critical victim, a lower bound on its service time is computed: the fastest drive at t = 0 to its nearest covering satellite, plus one direct drone trip. If this bound exceeds Dᵥ, the deadline becomes ⌈1.2 × bound⌉ and the change is logged. Each critical deadline is therefore reachable on its own.

**Parameters.**

| Parameter | Value |
| --- | --- |
| Map | 60 km × 60 km |
| Truck speed vₜ, drone speed vₐ | 40 km/h, 60 km/h |
| Unit weight q | 1 kg |
| Drone payload P, truck capacity Q | 3 units, 500 units (stress test ⌈n / (1.5K)⌉) |
| Trucks K, drones per truck U | workload rule (2 to 8), 3 |
| Drone energy: η, w₀, E, rₘᵢₙ | 0.667 Wh/(kg·min), 6 kg, 180 Wh, 20% |
| Launch Sₗ, recovery Sᵣ, drop κ | 1, 2, 0.1 min |
| Battery swap | instant, unlimited spares |
| Truck reload θ | 15 min |
| Time step Δt | 1 h (variant 30 min) |
| Road multipliers | open 1, damaged 2.5, closed 5 |

**Size of the benchmark.** 12 Solomon instances (4 layouts × 3 sizes) and 25 Gehring & Homberger instances (5 layouts × 5 sizes), 37 in total, each with 3 intensities × 10 seeds: 1,110 scenarios.

## 9. Experimental design (planned)

No results are reported yet. The study tests three claims:

- re-planning both levels serves victims sooner and misses fewer critical deadlines than a static plan;
- the learned policy matches always re-optimizing at a fraction of the computing time;
- each modelling choice matters, which the ablations measure.

**Instances.** The 37 instances of Section 8 (12 Solomon, 25 Gehring & Homberger, from 25 to 1,000 victims), each at 3 intensities with 10 road seeds. Three seeds per instance are used only for training.

**Default settings and hyperparameters.** Instance and vehicle parameters are listed in Section 8. The table below lists the defaults of the method itself. Values marked "proposed" are starting points to be tuned in preliminary runs on the training seeds; the others are fixed by the model or taken from the cited source.

| Group | Setting | Default | Note |
| --- | --- | --- | --- |
| Objective | Priority weights wᵥ (critical / high / normal) | 5 / 2 / 1 | Section 3 |
| Objective | Targets Dᵥ (critical / high / normal) | 2 h / 4 h / 8 h | Critical: dominant penalty; others soft |
| Objective | Critical-deadline penalty Π | 10⁴ | Proposed; larger than any J₁ + λJ₂, so deadlines dominate |
| Objective | Lateness weight πₗ | 2 per hour late | Proposed |
| Objective | Cost weights cₜ, cₐ and λ | 1 per truck-hour, 0.5 per full battery, λ = 0.05 | Proposed; keeps cost secondary to waiting time |
| Re-planning | Time step Δt | 1 h (30 min variant) | Section 5 |
| Re-planning | Level thresholds ε₁, ε₂ | 1%, 5% of J | Rule of Section 5.4 |
| Re-planning | L2 time budget | 2 s (short) or 10 s (long) | Section 7.1 actions |
| Re-planning | First plan at t = 0 | 30 s | Section 6.2 |
| ALNS | Destroy size | 10–30% of used satellites, drawn uniformly | Section 6.2 |
| ALNS | Operator scores (new best / improving / accepted) | 33 / 9 / 13 | Ropke & Pisinger (2006) |
| ALNS | Weight update | Every 100 iterations, reaction factor 0.1 | Ropke & Pisinger (2006) |
| ALNS | Annealing start temperature | Accepts a 5% worse plan with probability 0.5 | Proposed |
| ALNS | Cooling | Geometric, down to accepting a 0.1% worse plan by the end of the budget | Proposed |
| ALNS | Repair | Greedy and regret-2 insertion | Section 6.2 |
| Drone solver | Units per trip | At most 3 (P) | Section 3 |
| Drone solver | Giant tour | Nearest neighbour, then 2-opt | Section 6.1 |
| Drone solver | Menu size | 2–4 non-dominated plans per satellite | Section 6.1 |
| Drone solver | Battery check | Trip energy ≤ 80% (rₘᵢₙ = 20%) | Section 4.2 |
| Generator | Coverage | ≥ 2 candidates within ρR₁, ρ = 0.6 (≈ 9 km) | Section 8 |
| Generator | Placement solver | CP-SAT, 30 s limit | Section 8 |
| Generator | Fleet sizing target | Undamaged workload ≤ 6 h per truck | Section 8 |
| Generator | Road circuity | ×1.2 | Section 8 |
| Generator | Flood zone | About 30% of roads | Section 5.1 |
| Generator | Priority mix | 20% / 30% / 50% | Section 8 |
| Generator | Deadline relaxation | ⌈1.2 × lower bound⌉ when unreachable | Section 8 |
| Generator | Seeds | 10 per instance and intensity, 3 for training | Section 8 |
| Re-planning policy (DQN) | Network | 2 hidden layers of 64, ReLU | Section 7.1 |
| Re-planning policy (DQN) | Learning rate, batch | 10⁻³ (Adam), 64 | Proposed |
| Re-planning policy (DQN) | Discount γ | 1.0 | Episodes are finite, and the return equals the realized objective |
| Re-planning policy (DQN) | Replay buffer, target update | 50,000 transitions, every 500 updates | Proposed |
| Re-planning policy (DQN) | Exploration ε | 1.0 → 0.05 over the first 2,000 episodes | Proposed |
| Re-planning policy (DQN) | Training | 5,000 episodes; first 200 filled by the fixed rule | Proposed |
| Re-planning policy (DQN) | Computing-time penalty λ\_cpu | 0.01 per CPU second | Proposed |
| Operator selector (DQN) | Network, actions | 2 hidden layers of 64, 10 (destroy, repair) pairs | Section 7.2 |
| Operator selector (DQN) | Learning rate, batch, γ | 10⁻³, 64, 0.9 | Proposed |
| Operator selector (DQN) | Replay buffer, exploration | 20,000 transitions, ε 0.3 → 0.05 | Proposed |
| Operator selector (DQN) | Reward bonus for a new best | +1 | Section 7.2 |

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
| Multi-trip trucks | Q = 500 (non-binding) vs. stress test Q = ⌈n / (1.5K)⌉, each with and without second trips | What do second trips save when capacity is tight? |
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
- Gehring, H., & Homberger, J. (1999). A parallel hybrid evolutionary metaheuristic for the vehicle routing problem with time windows. In *Proceedings of EUROGEN99*, 57–64. Instances: [SINTEF TOP](https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/), files via the [PyVRP instance repository](https://github.com/PyVRP/Instances).
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
