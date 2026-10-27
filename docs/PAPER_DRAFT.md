# Dynamic Two-Echelon Truck–Drone Relief Routing

Oct 26, 2026 · @Dũng

## Abstract

We propose a **robust, information-aware re-planning method** for post-disaster relief with trucks and drones. Trucks carry supplies from a depot to satellites. At each satellite the truck waits while its drones deliver to every victim of one cluster. Roads degrade, demand surges and priorities change while the trucks are on the road, and the planner learns about these changes only through partial, noisy reports.

The method combines seven elements:

- a rolling-horizon model that freezes executed history and re-plans the rest;
- a Bayesian belief over road states;
- a robust re-planning layer whose uncertainty budgets are built from that belief (after Bertsimas & Sim, Yang et al. 2023 and Shi et al. 2026);
- a demand rule that keeps every satellite visited exactly once;
- multi-drop drone trips with payload-dependent energy, calibrated to Meng & Ma (2026) and Luo et al. (2027);
- optional drone reconnaissance flights that turn uncertain roads into observed ones (after Tang et al. 2025 and Chen et al. 2026);
- a learned policy that decides at each event whether to repair, re-optimize or re-optimize robustly (after Peng et al. 2025).

Soft cluster deadlines, a degree-of-dynamism measure and a Monte Carlo feasibility metric come from the same literature. Experiments are left for a later version.

## 1. Introduction

This work plans post-disaster relief with trucks and drones *while the disaster is still unfolding*, and re-plans each time roads fail or needs grow. No published model combines that with a two-echelon truck–drone network.

**Setting.** After a flood or earthquake, many victims cannot be reached by road. A common response is two-echelon:

- trucks carry supplies from a depot to **satellites** (open, reachable points near the affected area);
- at each satellite the truck stops and launches **drones**, which fly the supplies to the victims of one cluster;
- the truck waits for its drones to return, then drives on.

Nguyen, Piplani & Gunawan formulated this as the two-echelon covering tour VRP with drones (2E-CTVRP) and solved it with GRASP-ILS-PR. It minimizes the sum of truck arrival times.

**Why a static plan is not enough.** The 2E-CTVRP, like most truck–drone models, is solved once with full information. In a disaster, the information keeps changing *after* the trucks leave:

- **roads** degrade or close as water spreads, so truck travel times rise and some routes become detours;
- **demand** at shelters grows as people are evacuated, and **priorities** change;
- **information is incomplete and noisy**: damage reports are partial and sometimes wrong, and surge figures are estimates until a truck arrives.

A plan made at dispatch underestimates the realized operation time. In our experiments, the t = 0 plan's own forecast of its total arrival time (Z1) is 8–17% too low under 30–50% road damage. A plan that is never revised also leaves surge demand unserved.

**Research gap.** Recent work covers each ingredient separately (Section 2):

- **two-echelon truck–drone relief, but planned once** with robust uncertainty sets (Shi et al., 2026);
- **dynamic relief with re-planning, but single-echelon**, one drone per truck, and drone energy proportional to distance (Peng et al., 2025);
- **load-dependent, multi-drop drone energy, but static** (Meng & Ma, 2026; Yu et al., 2027; Luo et al., 2027).

No study combines a two-echelon truck–drone network, events revealed during execution, partial observation of those events, and drones whose range depends on payload and speed. No study compares robust one-shot planning with closed-loop re-planning either.

**Contributions.**

1. **Problem.** A dynamic, partially observed 2E-CTVRP with soft cluster deadlines. Road states and demand change at event epochs while trucks are on the road. Executed history is frozen, and the rest is re-planned each epoch.
2. **Information model.** Roads have three states (open / degraded / detour) with a Bayesian belief: Markov prediction, noisy reports, exact observation of traversed roads. The transition matrix can be estimated from flood scenarios (after Du et al. 2026). Reported surges carry a Gaussian belief.
3. **Belief-derived robust re-planning.** At each epoch the belief is turned into Bertsimas–Sim uncertainty budgets on truck travel times and on demand. This joins the one-shot robust planning of Shi et al. (2026) and Yang et al. (2023) with closed-loop re-planning: the budget shrinks as roads are observed.
4. **Demand dynamics without revisits.** Demand may grow only before or during a cluster's service, capped by the serving truck's spare capacity. A surge during service lengthens the truck's dwell (extra drone trips at the same stop), so every satellite is visited exactly once.
5. **Drone layer.** The second echelon is a multi-trip CVRP over individual victims. Energy is payload- and time-dependent, calibrated to Meng & Ma (2026) and Luo et al. (2027), and includes setup times. The drones' makespan is the truck's dwell.
6. **Reconnaissance sorties.** At a satellite, a drone may scout nearby uncertain roads before the truck leaves. This trades dwell time for information (after Tang et al. 2025 and Chen et al. 2026).
7. **Solution method.** A rolling-horizon, multi-objective re-optimizer (warm start, Pareto archive, elite memory, disruption-focused moves, segment-based evaluation after Luo et al. 2027). A learned meta-policy chooses, at each event, between repair, re-optimization and robust re-optimization (after Peng et al. 2025).
8. **Benchmarks and metrics.** Event generators for the 2E-CTVRP instances and a Solomon-derived set. Degree of dynamism and a Monte Carlo feasibility rate are reported (after Larsen et al. 2002 and Shi et al. 2026).

## 2. Related work

Truck–drone routing has moved from one truck with one drone to fleets, multi-drop drones and realistic energy. Almost all of it still plans once, before dispatch, on a single-echelon network.

### 2.1 Static truck–drone routing

- **Stodola & Kutěj (2024): multi-depot VRP with drones.** Each truck carries one UAV that serves one customer per flight. Adaptive ant colony optimization is tested on the Sacramento VRP-D set and on Cordeau-based multi-depot instances (up to 360 customers). UAVs save 24.7% time and 17.7% cost vs. trucks alone. *Limitations:* one UAV per truck, fixed endurance, deterministic.
- **Li, Guo & Fang (2026): contactless delivery.** Trucks stop only at temporary points chosen by K-means, and drones serve every customer within soft time windows. Simulated annealing with variable neighbourhoods matches Gurobi on small Solomon instances and saves 21.8% vs. trucks in a Chongqing case. *Limitations:* fixed drone range, no setup times, routing done after clustering is fixed.
- **Tang, Sun & Zou (2025): delivery plus time-windowed monitoring.** One truck and one drone; a two-stage greedy genetic algorithm. *Limitations:* single vehicle pair, one parcel per flight, random data, no exact benchmark.
- **Yanpirat, Silva & Smith (2023): deliveries and returns.** One truck and a multi-parcel drone (FSTSP-DR-MP); MILP with CPLEX plus VNS up to 100 customers. Service time is up to 36% shorter than truck-only. *Limitation:* one truck and one drone.

### 2.2 Multi-drop drones and load-dependent energy

- **Meng & Ma (2026): electric vans with several drones.** Each drone serves several customers per trip. Battery swapping, time windows and drone energy proportional to (empty weight + payload) × time. The van **waits while its drones fly** and may delay launch to avoid hovering. A hybrid ALNS matches Gurobi on small cases and is about 30% cheaper than vans alone. *Limitation:* static.
- **Luo, Chen & Wahab (2027): multi-trip truck–drone pairs.** Pickup and delivery, release dates and time windows. Payload × time energy; VNS with segment-based feasibility checks (87% faster). *Limitations:* deterministic truck times, linear energy.
- **Yu et al. (2027): two-echelon vehicle–drone routing.** Depots → satellites → drones that pick up or deliver within time windows, with a rotor power model that depends on real-time payload. Solved by a constrained multi-objective evolutionary algorithm (DSDE). *Limitations:* static; drones only in the second echelon, so no truck waits at satellites.
- **Duan et al. (2026): time-dependent VRP with drones.** Multi-visit drone trips, simultaneous pickup and delivery, hover energy, and road speeds that vary by period. A multi-objective hyper-heuristic memetic algorithm chooses among 12 low-level heuristics. *Limitation:* the speed profile is known in advance.

### 2.3 Planning under uncertainty: robust and time-varying

- **Yang, Yan, Cao & Roberti (2023): robust profit plan for one truck and one drone.** Uncertain truck travel times are modelled with budgeted uncertainty sets and solved exactly by branch-and-price, up to 40 requests. The truck waits at the launch site. The robust plan has up to 58% lower variance and up to 90% higher on-time feasibility than a deterministic plan.
- **Shi et al. (2026): robust two-echelon relief.** Large trucks supply chosen satellites; small trucks with one drone each serve affected sites, some reachable only by drone. Uncertainty in demand and travel time is budgeted. Two objectives (completion time, cost) are solved with NSGA-II plus tabu search, and the model is applied to a Sichuan earthquake. The robust plan is about 45 percentage points more often feasible, at +12% time and +7% cost.
- **Liu & Cao (2025): weather as time-windowed no-fly zones.** Solved with K-means and small neural networks on 20 Solomon customers.

**Limitation shared by all three:** the plan is fixed before execution. When delays or new demand are actually observed, there is no recourse.

### 2.4 Dynamic and learning-based routing

- **Peng, Wang, Yin & Cheng (2025): dynamic relief with truck–drone tandems.** Every 15 minutes new affected areas appear and demands change; some areas are reachable only by drone. The problem is cast as an event-driven Markov game and solved with multi-agent deep RL (MADDPG with action masking and prioritized replay). It beats a dynamic heuristic by about 11.5% and a rolling-horizon LNS by about 15%, on 900 instances and a Wenchuan case. *Limitations stated by the authors:*
  - single echelon;
  - one drone per truck, one area per flight;
  - drone energy proportional to distance;
  - zero launch and recovery times;
  - black-box policy. In addition (our reading), new information is assumed observed exactly when it appears, and truck travel times are not affected by worsening roads.
- **Li & Gonsalves (2026): attention policy for TSP-D and VRP-D.** Trained by policy gradient; answers in under 0.5 s and beats OR-Tools on random instances. *Limitation:* static.

### 2.5 Disaster logistics around the routing problem

- **Du et al. (2026): flood-scenario siting.** Chooses supply depots under five hydraulic flood scenarios (Zhengzhou 2021) and derives rules for when to use trucks, drones, or truck-launched drones.
- **Chen et al. (2026): truck–UAV area search.** Plans search of a large area by a truck carrying several UAVs.

Both supply inputs to relief routing (damage scenarios, reconnaissance) but do not route deliveries dynamically.

### 2.6 Positioning

| Work | Two echelons | Events during execution | Partial / noisy information | Load- and speed-dependent drone energy | Multi-drop drones | Truck waits for drones | Method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nguyen et al. (2E-CTVRP) | Yes | No | No | No (fixed) | No | Yes | GRASP-ILS-PR, MILP |
| Shi et al. 2026 | Yes | No (robust) | No | No | No | No | NSGA-II + tabu |
| Yu et al. 2027 | Yes | No | No | Payload | Yes | No trucks at satellites | Constrained MOEA |
| Peng et al. 2025 | No | **Yes** | No | No (distance) | No | No | Multi-agent DRL |
| Meng & Ma 2026 | No | No | No | Payload | Yes | Yes | ALNS + SA |
| Luo et al. 2027 | No | No | No | Payload | Yes | No | VNS |
| Yang et al. 2023 | No | No (robust) | No | No | No | Yes | Branch-and-price |
| Duan et al. 2026 | No | No (known profile) | No | Hover only | Yes | Mutual wait | Hyper-heuristic |
| **This work** | **Yes** | **Yes** (roads + demand) | **Yes** (Bayesian belief) | **Payload and speed, leg by leg** | **Yes** | **Yes** (dwell = drone makespan) | Rolling-horizon multi-objective re-optimizer |

### 2.7 What we take from the literature

No surveyed paper solves our problem, but several solve one piece of it well. The table lists what we adopt, from where, and how it changes once it sits inside a closed loop with partial information. Section 5 gives the formulations.

| Component | Source | Original use | Our adaptation | Section |
| --- | --- | --- | --- | --- |
| Budgeted uncertainty set (Γ-robustness) | Bertsimas & Sim 2004; Yang et al. 2023; Shi et al. 2026 | One robust plan, fixed before departure | Budgets rebuilt from the belief at every epoch; they shrink as roads are observed | 5.1 |
| Monte Carlo feasibility rate | Shi et al. 2026 | Test a robust plan on sampled scenarios | Score a plan by sampling the current belief | 5.1, 6 |
| Soft deadlines with lateness penalty | Peng et al. 2025; Li et al. 2026 (Symmetry); Shi et al. 2026 | Customer or rescue-point time windows | One deadline per cluster, scaled by priority; penalized lateness and unserved demand | 5.2 |
| Payload- and time-dependent drone energy | Meng & Ma 2026; Luo et al. 2027; Yu et al. 2027 | Multi-visit drone trips in (E)VRP-D | Calibrates the leg-by-leg energy of the victim-level drone CVRP | 5.3 |
| Segment-based route evaluation | Luo et al. 2027 | Fast move evaluation in multi-trip routing | Prefix/suffix arrival and slack tables for the residual truck routes | 5.4 |
| Drone reconnaissance (monitoring tasks) | Tang et al. 2025; Chen et al. 2026 | Drones observe targets or search areas | Optional scouting sortie at a satellite that reveals nearby road states | 5.5 |
| Learned re-planning decisions | Peng et al. 2025; Chen, Ulmer & Thomas 2021; Li & Gonsalves 2026 | DRL routing or assignment policies | A meta-policy picks repair / re-optimize / robust re-optimize per event | 5.6 |
| Hyper-heuristic operator control | Duan et al. 2026 | Choose among low-level heuristics | Baseline for the DQN operator selector | 5.6 |
| Degree of dynamism | Larsen et al. 2002 (used by Peng et al. 2025) | Share of requests revealed online | Share of final demand granted as surges | 5.7 |
| Scenario-based flood damage | Du et al. 2026; Liu & Cao 2025 | Flood scenarios and weather layers | Estimate the road-state Markov matrix P from scenario depth data | 5.7 |

**Not adopted.** Some features of the surveyed models are left out on purpose:

- **Satellite or launch-site location** (Shi et al. 2026; Stodola & Kutěj 2024). The satellite–cluster bijection σ is given by the 2E-CTVRP, and choosing locations mid-operation is not realistic.
- **Cross-vehicle drone recovery** (Luo et al. 2027; Li et al. 2026). A drone must return to the truck it left from, because that truck is waiting.
- **Pickups and returns** (Duan et al. 2026; Yanpirat et al. 2023; Yu et al. 2027). Relief delivery is one-way.
- **EV battery swapping and public transit** (Meng & Ma 2026; Liu & Cao 2025). Out of scope for diesel relief trucks.
- **Multi-trip trucks and revisits.** Ruled out by the no-revisit rule; see the conclusion for future work.

## 3. Problem description

The problem is the 2E-CTVRP of Nguyen et al., played out over time. Trucks are already on the road when roads fail and demand grows, and the planner sees these changes only partly.

**Network.**

- A depot v₀ (also the end depot vₙ₊₁).
- n satellites, connected by a road network with baseline truck times t⁰ᵢⱼ.
- n victim clusters, obtained by grouping the victims with k-means. Victim v has a location and demand qᵥ; cluster c has demand d\_c = Σ qᵥ and priority w\_c.

**Fleet.**

- K trucks of capacity Q, each making one trip from the depot.
- At a satellite the truck launches U drones of payload P and speed v\_d.
- The drones' battery use depends on speed and on the load carried on each leg.

**Service rule.** Each satellite is visited **exactly once**, by one truck. Its drones serve **exactly one** cluster: the satellite→cluster matching σ is a bijection. The truck **waits** at the satellite until its drones are back, then drives on.

**Events.** At epochs 0 = t₀ < t₁ < t₂ < … the world changes while trucks are travelling:

- **Roads.** Each arc is in one of three states: *open*, *degraded* (time ×2.5) or *detour* (time ×5; a closed road is modelled as a long detour, so every satellite stays reachable). Damage spreads from a flood centre, can be one-way, and can escalate.
- **Demand and priority.** Clusters near the damage surge, and their priorities rise.

**Information.**

- The **true** state drives execution.
- The **planner** sees only reports. A changed road is reported with probability ρ\_obs, and a report is correct with probability κ.
- Roads a truck has driven are known exactly.
- Surge sizes are reported with relative noise σ\_d until a truck reaches the cluster.

**Demand rule.** A cluster's demand may grow only **before or during** its service, never after its truck has left. Growth is capped by the remaining capacity of the truck assigned to it. A surge that arrives while the truck is parked adds drone trips at that same stop.

**Decisions at each epoch.** For the satellites not yet reached:

- the order in which each active truck visits them;
- the matching σₜ of these satellites to the free clusters.

Everything already executed, and the satellite each truck is currently driving to, are fixed.

**Objectives** (whole horizon, executed part counted as constants):

- f₁: priority-weighted delivery time (efficiency for victims);
- f₂ = max aᵢ − min aᵢ: spread of satellite arrival times (equity);
- f₃ = Z₁ = Σ aᵢ + Σ retₖ: the 2E-CTVRP objective (fleet time).

The single-objective version with weights (0, 0, 1) reduces to the original 2E-CTVRP at t = 0.

## 4. Proposed method

The method is a closed loop. At each event it:

1. freezes what has been executed;
2. updates its belief about roads and demand;
3. admits only feasible demand growth;
4. re-optimizes the remaining two-echelon plan on that belief.

The plan is then executed on the true network until the next event.

### 4.1 Epoch state and commitment

At event time tₑ the current plan is simulated on the true network up to tₑ.

- Every satellite a truck has reached is **locked**, together with its truck, its position in the route, its cluster, its arrival time and its delivery time.
- A truck driving toward a satellite is committed to that arc, and the target is locked too.
- A truck already heading home is retired.

Each active truck k restarts from its last locked stop at its available time αₖ (arrival + dwell), with remaining capacity Qₖ = Q − load already handed out. The free satellites F(t) and the free clusters M\_F(t) define the residual problem. Locked arrivals enter every objective as constants, so values are comparable across epochs.

### 4.2 Information model (POMDP layer)

Each arc (i, j) carries a belief bᵢⱼ over the states {open, degraded, detour}, with multipliers φ = (1, 2.5, 5).

**Between epochs** the belief is predicted with a Markov matrix 𝒫:

```latex
b_{ij}^{t} \leftarrow b_{ij}^{t-1}\,\mathcal{P}
```

**A report o** on a changed arc (sent with probability ρ\_obs, correct with probability κ) updates the belief by Bayes' rule:

```latex
b_{ij}(s) \propto O(o \mid s)\, b_{ij}(s), \qquad O(o\mid s)=\begin{cases}\kappa & o=s\\ (1-\kappa)/2 & o\neq s\end{cases}
```

**A traversed arc** becomes known exactly.

**Travel time used by the planner:**

```latex
\mathbb{E}[t_{ij}] = t^{0}_{ij}\sum_{s}\phi_s\, b_{ij}(s)
```

Three alternative planners are compared:

- *oracle*: knows the truth;
- *naive*: trusts every report;
- *belief with a chance constraint*: an arc with P(open) + P(degraded) < θ is planned as a detour.

**Demand belief.** A reported surge is believed to be Gaussian with mean μ\_c and standard deviation σ\_dμ\_c until a truck reaches cluster c. Objectives use μ\_c. Capacity and battery are checked at the robust value:

```latex
d^{cap}_c = \mu_c + z\,\sigma_d\,\mu_c
```

### 4.3 Demand rule: growth only before or during service

A requested surge Δ\_c on cluster c is admitted only if its truck has not yet left the satellite. It is capped by that truck's spare capacity:

```latex
\Delta^{grant}_c = \min\big(\Delta_c,\ \mathrm{slack}_k\big),\qquad \mathrm{slack}_k = Q - \Big(\sum_{\text{visited}}\text{delivered} + \sum_{\text{planned}} d\Big)
```

The slack is reduced as surges are granted.

- A surge on a cluster that is **not yet reached** is delivered at arrival.
- A surge on a cluster whose truck is **parked** is delivered by extra drone trips at the same stop, which lengthens the dwell.

So the plan in force is always capacity-feasible, every satellite is visited once, and no truck has to go back.

### 4.4 Second echelon: drone service at a satellite

The first echelon sees the drones only through three functions of satellite i, cluster c and demand d:

- Wᵢ\_c(d): the truck's dwell;
- δᵢ\_c(d): the delivery offset after arrival;
- uᵢ\_c(d): the battery use of the most demanding trip (feasible if ≤ 1).

**Drone routes.** The drones deliver to every victim of the cluster: a multi-trip CVRP from the satellite.

1. A giant tour over the victims (nearest neighbour + 2-opt) is rotated to start next to the satellite.
2. An exact split cuts it into trips that minimize total drone busy time.
3. The trips are scheduled on the U drones: longest first to the least busy drone, shortest first within each drone.

A trip T = (v₁, …, vₘ) must satisfy the payload limit and the battery limit. The drone takes off with the whole trip load and gets lighter after each drop:

```latex
\sum_{l=1}^{m} q_{v_l} \le P, \qquad e_T = \frac{\lVert i\,v_1\rVert}{v_d}\,p\Big(\sum_{l\ge 1} q_{v_l}\Big) + \sum_{l=1}^{m-1}\frac{\lVert v_l\,v_{l+1}\rVert}{v_d}\,p\Big(\sum_{r>l} q_{v_r}\Big) + \frac{\lVert v_m\,i\rVert}{v_d}\,p(0) + \tau\sum_{l=1}^{m} p\Big(\sum_{r\ge l} q_{v_r}\Big) \le E(1-\rho)
```

The power p (share of a full battery per hour) depends on payload ℓ and speed v:

```latex
p(\ell) = a\Big(1+\beta\,\frac{\ell}{P}\Big) + b\Big(\frac{v_d}{v_{ref}}\Big)^{3}
```

**Dwell and delivery offset.** Each trip has busy time

```latex
\mathrm{busy}(T) = S_L + \text{flight}(T) + m\,\tau + S_R
```

with launch time S\_L, recovery and battery-swap time S\_R, and hover time τ per drop. The dwell is the drones' makespan, and the offset is the demand-weighted mean drop time:

```latex
W_{ic} = \max_{u=1..U}\sum_{T\in u}\mathrm{busy}(T), \qquad \delta_{ic} = \frac{\sum_v q_v\, t_v}{\sum_v q_v}
```

**Battery feasibility.** A satellite–cluster pair is battery-infeasible if some victim cannot be served even alone. Such pairs are excluded from the matching σ.

**Centroid model.** The original 2E-CTVRP is the special case where drones fly to the cluster centroid in ⌈d/(UP)⌉ waves:

```latex
W = \Big\lceil \tfrac{d}{UP}\Big\rceil\,(S_L + 2t' + S_R), \qquad \delta = S_L + t'
```

### 4.5 Timeline and objectives

For active truck k with route (πₖ(1), …, πₖ(mₖ)), starting at αₖ:

```latex
a_{\pi_k(p)} = a_{\pi_k(p-1)} + W_{\pi_k(p-1)\,\sigma(\pi_k(p-1))} + \mathbb{E}\big[t_{\pi_k(p-1)\,\pi_k(p)}\big], \qquad e_i = a_i + \delta_{i\,\sigma(i)}
```

```latex
f_1 = \frac{1}{D}\sum_{i} w_{\sigma(i)}\, d_{\sigma(i)}\, e_i,\qquad f_2 = \max_i a_i - \min_i a_i,\qquad f_3 = \sum_i a_i + \sum_k \mathrm{ret}_k
```

**Constraints, each with how it is enforced:**

- every free satellite in exactly one route: by the representation;
- σₜ a bijection onto the free clusters: by the representation;
- truck capacity Qₖ checked at d^cap: penalty;
- battery feasibility uᵢ\_c ≤ 1 per trip: infeasible pairs excluded, plus a penalty;
- a depot-start truck visits at least one satellite: penalty;
- locked history unchanged: verified after every re-plan.

Candidates are compared feasibility first, then by the scalarized objective.

### 4.6 Rolling-horizon multi-objective re-optimizer

**Representation.** As in the original GRASP-ILS-PR: an ordered satellite list per truck plus the matching σ.

**Moves.**

- The original neighbourhoods N1–N7: relocate, swap, 2-opt, inter-route swap, 2-opt\*, inter-route relocate, cluster swap.
- Plus destroy-and-repair.
- Cheapest insertion uses the planner's expected (possibly asymmetric) times, since damage removes the zero-detour property of the Manhattan grid.

**Disruption-focused search.** A move's truck is sampled in proportion to a route disruption score:

```latex
c_k(t) = \mathrm{norm}\big(\omega_1 r^{road}_k + \omega_2 r^{demand}_k + \omega_3 r^{timing}_k\big)
```

The three terms are the share of the route's arcs whose expected time grew, the relative demand change of its clusters, and the growth of its remaining duration.

**Memory across epochs (warm start).** The new search starts from:

- the previous plan's unvisited suffix;
- the previous Pareto archive, projected onto the residual problem;
- an elite pool of matchings σ, used for path relinking.

An old σ is not automatically valid after commitment, because some clusters are now locked to other satellites. It is projected: compatible pairs are kept, and the rest are re-matched under the battery constraint.

**Multi-objective search.** Every feasible candidate is offered to a Pareto archive with crowding truncation. The scalarizing weights cycle through a Das–Dennis lattice when the search stalls. The plan committed is the archive point that is best under the commit weights.

**Re-planning strategies.**

- *static*: never re-plan (baseline).
- *restart*: solve each residual problem from scratch.
- *warm*: memory plus disruption focus.
- *hybrid* (repair first): keep the previous plan, re-match only infeasible pairs, and measure the event's impact Δ on the plan's predicted objectives. The search runs only if the repair is infeasible or the impact exceeds a tolerance:

```latex
\Delta = \frac{\sum_m \lambda_m\,(o^{rep}_m - o^{prev}_m)/|o^{prev}_m|}{\sum_m \lambda_m}, \qquad \text{commit the repair} \iff \mathrm{viol}=0 \;\wedge\; \Delta \le \mathrm{tol}
```

### 4.7 Learning component

The re-optimizer exposes an operator-selection hook. A Double DQN picks the next operator for blocks of 10 moves.

- **State (23 features):** epoch context, including route disruption, fleet and drone load and the share of uncertain roads; search progress; the last operator used.
- **Reward:** new best, improvement, or new Pareto point.
- **Training:** on instances separate from the test set.

Following the survey (Peng et al., 2025, and Chen, Ulmer & Thomas), the next step moves learning to **decisions with lasting consequences**:

- when to re-plan (learning the hybrid trigger);
- how to absorb a surge (wait for extra drone trips vs. re-match the cluster).

In both cases the re-optimizer stays as the inner solver.

### 4.8 Benchmark generation

**Event files** come from the original 2E-CTVRP instances. Event times are the 30% and 65% quantiles of baseline arrival times, so each event splits served from unserved satellites. Each scenario:

- damages 10/30/50% of first-echelon arcs around a flood centre, with progression, escalation and one-way damage;
- surges the unserved clusters nearest the flood and raises their priority;
- optionally reveals only part of the road changes.

**A second set** is derived from the Solomon geometries C1, C2, R and RC. Customers become victims, k-means clusters become relief clusters, and a satellite is placed 1–2.5 km from each cluster. Solomon's time windows are dropped, because one cluster's windows span most of the day.

### 4.9 Framework overview

&#91;embedded content: closed-loop rolling-horizon method · one pass per event\]

The plan is optimized on the belief and executed on the truth. Each event freezes the past, updates the belief, admits only feasible demand growth, and then either keeps a repaired plan or runs the warm multi-objective re-optimizer.

## 5. Components adopted from the literature

This section formulates the pieces listed in Table 2.7. Each one plugs into the closed loop of Section 4 without changing its structure. The no-revisit rule, the satellite–cluster bijection σ and the demand rule of Section 4.3 still hold.

### 5.1 Belief-derived robust re-planning

**Idea.** Shi et al. (2026) and Yang et al. (2023) protect one plan against a budget Γ of worst-case deviations (Bertsimas & Sim 2004). They fix Γ by hand and plan once. We rebuild the uncertainty set from the current belief at every epoch. Observed arcs carry no deviation, so the set shrinks as trucks drive and the price of robustness falls over time.

**Nominal time and deviation.** Let τ⁰ᵢⱼ be the undamaged travel time of arc (i, j), mₛ the multiplier of state s ∈ {open, degraded, detour}, and bᵢⱼ the current belief. The nominal time is the belief mean. The deviation is the gap to the worst state that still has belief at least ε:

```latex
\bar t_{ij} = \sum_{s} b_{ij}(s)\, m_s\, \tau^0_{ij},
\qquad
\tilde t_{ij} = \max_{s:\, b_{ij}(s) \ge \varepsilon} m_s\, \tau^0_{ij} \;-\; \bar t_{ij}
```

An observed arc has a one-hot belief, so its deviation is 0.

**Travel-time uncertainty set.** For the residual route Rₖ of truck k:

```latex
\mathcal U_T(\Gamma_k) = \Big\{ t :\; t_{ij} = \bar t_{ij} + \xi_{ij}\,\tilde t_{ij},\; 0 \le \xi_{ij} \le 1,\; \sum_{(i,j)\in R_k} \xi_{ij} \le \Gamma_k \Big\}
```

**Budget from the belief.** Let πᵢⱼ = 1 − bᵢⱼ(open) be the chance that an arc is damaged. The number of damaged arcs on Rₖ is a sum of Bernoulli variables. The budget is its normal-approximation (1 − εΓ) quantile:

```latex
\Gamma_k = \min\Big\{ |R_k|,\; \Big\lceil \sum_{(i,j)\in R_k} \pi_{ij} + z_{1-\varepsilon_\Gamma} \sqrt{\sum_{(i,j)\in R_k} \pi_{ij}(1-\pi_{ij})} \Big\rceil \Big\}
```

The planner sets one parameter, the risk level εΓ, instead of a raw Γ.

**Worst-case arrival.** Arrival times are prefix sums along the route, so the worst case at position p adds the Γₖ largest deviations among the first p arcs (Bertsimas & Sim 2004). Dwell times W are added unchanged:

```latex
a^{\Gamma}_{p} = a^{\mathrm{nom}}_{p} + \max_{S \subseteq \{1..p\},\, |S| \le \Gamma_k} \sum_{q \in S} \tilde t_{q}
```

The sum is kept with a size-Γₖ min-heap while scanning the route, so one route costs O(p log Γₖ).

**Demand uncertainty set.** A reported but unconfirmed surge at cluster c has mean μ\_c and spread σ\_d μ\_c (Section 4.2). With deviation d̂\_c = z σ\_d μ\_c, the robust capacity check for truck k is:

```latex
\sum_{c \in C_k} \bar d_c \;+\; \max_{S \subseteq C_k,\, |S| \le \Gamma_D} \sum_{c \in S} \hat d_c \;\le\; Q - \mathrm{delivered}_k
```

The old chance-constrained check puts every cluster at its quantile at once. This version allows at most Γ\_D surges to hit their worst case together, so it is less conservative. Battery feasibility per drone trip is checked the same way, with the cluster's robust demand.

**Robust re-optimization.** In robust mode the re-optimizer of Section 4.6 evaluates f₁ and Z₁ at a^Γ instead of a^nom and enforces the robust capacity check. The rest of the search is unchanged. The price of robustness at an epoch is f(robust plan, nominal) / f(nominal plan, nominal) − 1.

**Monte Carlo feasibility (after Shi et al. 2026).** Sample M scenarios ω from the current belief (road states and surges). A plan is feasible in ω if no truck runs short of capacity, every drone trip is within battery limits, and no cluster is later than D\_c + L\_max:

```latex
\mathrm{FR} = \frac{1}{M} \sum_{m=1}^{M} \mathbf 1\big[\,\text{plan feasible in } \omega_m\,\big]
```

FR is reported for every strategy, and the meta-policy (5.6) also uses it as an input.

### 5.2 Soft cluster deadlines

**Idea.** Peng et al. (2025), Shi et al. (2026) and Li et al. (2026) give each demand point a time window. Relief agencies set target times by priority, but a hard window can make a dynamic instance infeasible after a road closes. We use soft deadlines with a lateness penalty instead.

**Deadline generation (Solomon-style).** Let ê⁰\_c be the cluster's service time in the baseline static plan, w\_c its priority weight and H the horizon:

```latex
D_c = \hat e^{0}_c + \theta(w_c)\,(H - \hat e^{0}_c), \qquad \theta(w) \in \{0.1, 0.25, 0.5\} \text{ for high, medium, low priority}
```

Tight deadlines go to urgent clusters, as in Solomon's tight and loose window classes.

**Revised f₁.** Let e\_c be the delivery completion time at cluster c, d\_c its final demand, and u\_c = Δ\_c − Δᵍʳ\_c the part of a surge that was not granted:

```latex
f_1 = \frac{1}{\sum_c w_c d_c} \sum_{c} w_c \Big[ d_c\, e_c + \pi_L\, d_c\, (e_c - D_c)^{+} \Big] \;+\; \pi_U \sum_c w_c\, u_c
```

The unserved term rewards plans that keep spare capacity on trucks headed to clusters likely to surge. Under the demand rule, slack is the only way to serve a surge. With π\_L = π\_U = 0 we recover the f₁ of Section 4.5.

### 5.3 Calibrated payload-dependent drone energy

**Idea.** Meng & Ma (2026) and Luo et al. (2027) model drone energy as linear in total mass times flight time, and Yu et al. (2027) also make it load-dependent. Our leg power p(ℓ) = A + Bℓ/P (Section 4.4) has the same form, so their parameters can calibrate it directly.

**Model.** Let η be the energy rate per unit mass, w₀ the empty drone mass, P the payload capacity, E the battery capacity and ℓ the load carried on a leg. The battery fraction used per minute is:

```latex
p(\ell) = \frac{\eta\,(w_0 + \ell)}{E} = \underbrace{\frac{\eta\, w_0}{E}}_{A} + \underbrace{\frac{\eta\, P}{E}}_{B}\cdot\frac{\ell}{P}
```

With the shared values of both papers (η = 0.667 Wh/(kg·min), w₀ = 6 kg, P = 3 kg, E = 180 Wh), A ≈ 0.0222 and B ≈ 0.0111 per minute. An empty drone can fly about 45 minutes and a fully loaded one about 30.

**Trip feasibility.** A trip of legs l = 1..L, with leg times tₗ, loads ℓₗ and a hover of τ at each drop, is battery-feasible if it leaves a reserve r\_min:

```latex
\sum_{l=1}^{L} p(\ell_l)\, t_l \;+\; \sum_{\text{drops } v} p(\ell_v)\,\tau \;\le\; 1 - r_{\min}
```

The speed-based form A = a + b(v/v\_ref)³ stays available for sensitivity runs on cruise speed.

### 5.4 Segment-based evaluation

**Idea.** Luo et al. (2027) evaluate multi-trip moves on cached route segments and report an 87% speed-up. Our residual truck routes have no hard time windows, so a move shifts every later arrival by the same amount. That makes segment caching exact.

**Cached data.** For residual route πₖ = (s₁, …, sₘ) we store prefix arrivals a\_p, and suffix weights and lateness slacks:

```latex
\Omega_p = \sum_{q \ge p} w_q d_q, \qquad \lambda_q = D_q - e_q
```

**Move cost.** A move that delays position p by δ changes f₁ by:

```latex
\Delta f_1 = \frac{1}{\sum_c w_c d_c}\Big[\delta\,\Omega_p + \pi_L \sum_{q \ge p} w_q d_q \big((\delta - \lambda_q)^{+} - (-\lambda_q)^{+}\big)\Big]
```

With the suffix slacks kept sorted, the second term costs O(log m). Insertion, removal and swap therefore cost O(log m) instead of a full O(m) re-simulation. In robust mode the worst-case term of 5.1 is recomputed only from p onward.

### 5.5 Drone reconnaissance sorties

**Idea.** Tang et al. (2025) give drones monitoring tasks alongside deliveries, and Chen et al. (2026) use UAVs launched from trucks to search an area. In our setting, information about roads is what the planner lacks most. A short scouting flight from a satellite can reveal the arcs the truck will drive next.

**Decision.** For each satellite i on a residual route, let y\_i ∈ {0, 1} say whether a scouting sortie is flown while the truck dwells there. The sortie observes the arc set:

```latex
\mathcal A(i, \rho) = \{ (u, v) : \mathrm{dist}(i, \mathrm{mid}(u, v)) \le \rho \}
```

Its flight time is s\_rec(i) = 2ρ/v\_d + τ\_scan, and it is battery-checked with p(0).

**Effect on the dwell.** If a drone is idle during the delivery waves (U drones and fewer than U trips in the last wave), the sortie runs in parallel. Otherwise it is added at the end:

```latex
W'_i = \begin{cases} \max(W_i,\, s_{\mathrm{rec}}(i)) & \text{if a drone is idle} \\ W_i + s_{\mathrm{rec}}(i) & \text{otherwise} \end{cases}
```

**Observation.** Observed arcs get a one-hot belief (or accuracy κ\_rec < 1 through the Bayes update of Section 4.2). The new information is a decision epoch.

**When to fly.** The sortie pays off if the expected cost after re-planning on what it reveals beats the cost of the extra dwell, which delays every later cluster on route k:

```latex
y_i = 1 \iff \mathbb E_{o \sim b}\big[f(\text{plan}\mid b) - f(\text{replan}\mid b \oplus o)\big] \;>\; \frac{(W'_i - W_i)\,\Omega_{p(i)+1}}{\sum_c w_c d_c}
```

The expectation is estimated from K samples of the belief on 𝒜(i, ρ), each re-planned with the cheap repair move. The test is run only at satellites with residual-route arcs inside the radius whose belief entropy is above a threshold.

### 5.6 Learned re-planning policy

**Idea.** Peng et al. (2025) learn routing decisions with multi-agent DRL. Chen, Ulmer & Thomas (2021) and Li & Gonsalves (2026) learn assignment and construction policies. We do not replace the optimizer. A small policy decides how much optimization each event deserves, a choice our Δ-threshold rule (Section 4.6) currently makes by hand.

**Formulation as an MDP.**

- **Epoch e:** each road, demand, priority or reconnaissance event.
- **State x\_e:** time fraction t/H; number of free satellites; total belief entropy on residual arcs; the budgets Γₖ; total spare capacity; the cost change after repair, Δ\_rep; the share of damaged arcs on residual routes; the size of the surge; and FR of the current plan.
- **Action a\_e** ∈ {keep, repair, warm re-optimize, robust re-optimize (εΓ ∈ {0.1, 0.3})}.
- **Reward:** the realized cost added before the next epoch, minus a penalty on computing time:

```latex
r_e = -\big(Z^{\mathrm{real}}_{e+1} - Z^{\mathrm{real}}_{e}\big) \;-\; \lambda_{\mathrm{cpu}}\,\mathrm{CPU}_e \;-\; \pi_U \sum_c w_c\, u_c^{(e)}
```

**Training.** A double DQN over generated event files, as in Chen, Ulmer & Thomas (2021). One central agent is enough because trucks report to the depot. The multi-agent decomposition of Peng et al. (2025) targets decentralized vehicles.

**Baselines.** The fixed Δ-threshold rule (Section 4.6), and a choice-function hyper-heuristic in the style of Duan et al. (2026) for the operator selector. The latter scores each move h by its recent improvement and the time since it was last used:

```latex
\mathrm{CF}(h) = \alpha\, \bar I_h + \beta\, \bar I_{h' \to h} + \gamma\, \Delta t_h
```

### 5.7 Measuring dynamism and estimating road dynamics

**Degree of dynamism.** Following Larsen et al. (2002), as used by Peng et al. (2025), we report the share of final demand that arrived as granted surges, and an effective version weighted by how late each surge appeared:

```latex
\mathrm{DoD} = \frac{\sum_c \Delta^{gr}_c}{\sum_c d_c}, \qquad \mathrm{eDoD} = \frac{\sum_c \Delta^{gr}_c\, t^{\mathrm{rev}}_c / H}{\sum_c d_c}
```

**Flood-driven road dynamics.** Du et al. (2026) build urban-flood scenarios from depth data, and Liu & Cao (2025) add weather layers. Given scenario depth series hᵢⱼ(t) on time slices of length Δt, map depth to state with thresholds h₁ < h₂ (open below h₁, detour above h₂). Then estimate the Markov matrix of Section 4.2 by counting transitions, with Laplace smoothing:

```latex
\hat P_{ss'} = \frac{N_{ss'} + 1}{\sum_{s''} N_{ss''} + 3}
```

The state multipliers mₛ can likewise come from a speed–depth curve instead of the fixed ×2.5 and ×5. This ties the belief's prediction step to data rather than a hand-set matrix.

## 6. Experimental design (planned)

Results are not reported in this version. The planned study is below.

**Instances.** The 2E-CTVRP main and large sets, and the Solomon-derived set (C1, R1, RC1 classes), with event files at two epochs (30% and 65% of baseline arrivals). Soft deadlines follow 5.2.

**Factors.**

- road damage share (10%, 30%, 50%) and report accuracy κ (0.6, 0.8, 1.0);
- surge scale (0.5 to 4), giving a range of DoD values;
- drone model: centroid vs. victim-level routes with calibrated energy (5.3);
- reconnaissance on or off, with radius ρ.

**Strategies compared.**

| Strategy | Description | Source of the idea |
| --- | --- | --- |
| Static | Plan once at t = 0, execute with recourse only | Nguyen et al. |
| Robust once | One Γ-robust plan at t = 0, no re-planning | Shi et al. 2026; Yang et al. 2023 |
| Restart / warm | Re-optimize at each epoch, cold or warm start | Section 4.6 |
| Hybrid | Repair first, re-optimize if Δ > tolerance | Section 4.6 |
| Robust warm | Warm re-planning with belief-derived budgets | 5.1 |
| Learned meta-policy | DQN chooses the action per event | 5.6 |
| Oracle | Knows the truth at each epoch (bound) | — |

**Metrics.**

- realized f₁, f₂ and Z₁, with paired relative change against static and Wilcoxon tests;
- lateness, the number of late clusters and unserved demand (5.2);
- Monte Carlo feasibility rate FR and price of robustness (5.1);
- plan-vs-actual gaps gap₀ and gap\_last;
- DoD and eDoD (5.7);
- re-planning CPU per epoch and the share of events handled by repair;
- for reconnaissance: sorties flown, added dwell, and entropy removed.

**Key questions.**

1. Does belief-derived Γ beat a fixed Γ and a non-robust warm start on FR, at what cost in Z₁?
2. Does robustness still pay once re-planning is allowed? This is the main test of closed-loop vs. one-shot robust planning.
3. When does a reconnaissance sortie pay for its dwell?
4. Does the learned policy match the best fixed strategy while using less CPU?

## 7. Conclusion and limitations

We set out a dynamic, partially observed version of the two-echelon truck–drone relief problem, together with a solution method built around a closed loop. The core is our own: rolling-horizon re-planning on a belief, a demand rule that never revisits a satellite, and a victim-level drone layer whose makespan is the truck's dwell. Around it we placed components that the surveyed papers have tested in simpler settings: budgeted robustness, soft deadlines, calibrated drone energy, segment-based evaluation, reconnaissance flights, a learned decision layer and standard dynamism measures. The main new element is that robustness budgets come from the belief and shrink as information arrives, joining robust and dynamic routing, two lines of work that have so far been studied apart.

**Limitations and future work.**

- Each truck serves its satellites in one trip and never revisits. Demand beyond a truck's spare capacity is lost. A second-wave truck (multi-trip, revisits allowed) would remove this, at the cost of leaving the VRP structure.
- σ is fixed. Re-assigning a cluster to another satellite after a road closes is not modelled.
- The belief treats arcs as independent. Spatially correlated flooding would need a joint model, for example one driven by the scenario sets of Du et al. (2026).
- The meta-policy is trained on synthetic events. Its transfer to real disaster logs is untested.
- No exact method gives optimality gaps on small dynamic instances. A branch-and-price bound in the style of Yang et al. (2023) for the robust sub-problem is a natural next step.

## References

- Bertsimas, D., & Sim, M. (2004). The price of robustness. *Operations Research*, 52(1), 35–53. [doi](https://doi.org/10.1287/opre.1030.0065)
- Chen, X., Ulmer, M. W., & Thomas, B. W. (2021). Deep Q-learning for same-day delivery with vehicles and drones. *arXiv:1910.11901*.
- Chen, Y., Hong, F., Ye, Q., & Wu, G. (2026). Path planning for large-area search by a collaborative truck–UAVs system. *Expert Systems with Applications*, 331, 133385. [doi](https://doi.org/10.1016/j.eswa.2026.133385)
- Du, W., Wang, J., Liu, J., Wang, H., Zhou, J., Mei, C., & Pang, Y. (2026). Optimization of emergency supplies allocation for urban flood disasters. *Results in Engineering*, 32, 112619. [doi](https://doi.org/10.1016/j.rineng.2026.112619)
- Duan, H., Li, X., Feng, Y., Zhang, G., Lu, Q., & Meng, X. (2026). Multi-objective hyper heuristic memetic algorithm for time-dependent vehicle routing problem with drone considering simultaneous pickup and delivery. *Expert Systems with Applications*, 297, 129196. [doi](https://doi.org/10.1016/j.eswa.2025.129196)
- Li, J., Guo, Q., & Fang, Y. (2026). Truck–drone cooperative delivery problem based on two-stage algorithm. *Symmetry*, 18, 1199. [doi](https://doi.org/10.3390/sym18071199)
- Li, Q., & Gonsalves, T. (2026). A reinforcement learning framework for traveling salesman and vehicle routing problem with drones. *Drones*, 10, 616. [doi](https://doi.org/10.3390/drones10080616)
- Larsen, A., Madsen, O., & Solomon, M. (2002). Partially dynamic vehicle routing — models and algorithms. *Journal of the Operational Research Society*, 53(6), 637–646. [doi](https://doi.org/10.1057/palgrave.jors.2601352)
- Liu, F., & Cao, H. (2025). Neural network-based air–ground collaborative logistics delivery path planning with dynamic weather adaptation. *Mathematics*, 13, 2798. [doi](https://doi.org/10.3390/math13172798)
- Luo, Y., Chen, Y., & Wahab, M. I. M. (2027). Multi-trip truck-drone routing problem with simultaneous pickup and delivery and temporal constraints. *Transportation Research Part E*, 217, 105235. [doi](https://doi.org/10.1016/j.tre.2026.105235)
- Meng, S., & Ma, H. (2026). Electric vehicle routing problem with multi-visit drones considering time windows and battery swapping stations. *Transportation Research Part E*, 215, 105107. [doi](https://doi.org/10.1016/j.tre.2026.105107)
- Nguyen, D. V. A., Piplani, R., & Gunawan, A. A two-echelon covering tour vehicle routing problem with drones for post-disaster relief. *arXiv:2609.38227* (under review).
- Peng, W., Wang, D., Yin, Y., & Cheng, T. C. E. (2025). Multi-agent deep reinforcement learning-based truck-drone collaborative routing with dynamic emergency response. *Transportation Research Part E*, 195, 103974. [doi](https://doi.org/10.1016/j.tre.2025.103974)
- Shi, Y., Guo, Q., Pan, P., Chen, M., Wu, Y., & Guo, H. (2026). Robust optimization for two-echelon truck and drone collaborative delivery in emergency relief. *Applied Soft Computing*, 201, 115738. [doi](https://doi.org/10.1016/j.asoc.2026.115738)
- Stodola, P., & Kutěj, L. (2024). Multi-depot vehicle routing problem with drones: Mathematical formulation, solution algorithm and experiments. *Expert Systems with Applications*, 241, 122483. [doi](https://doi.org/10.1016/j.eswa.2023.122483)
- Tang, M., Sun, J., & Zou, R. (2025). A two-stage greedy genetic algorithm for simultaneous delivery and monitoring tasks with time windows. *Drones*, 9, 50. [doi](https://doi.org/10.3390/drones9010050)
- Yang, Y., Yan, C., Cao, Y., & Roberti, R. (2023). Planning robust drone-truck delivery routes under road traffic uncertainty. *European Journal of Operational Research*, 309, 1145–1160. [doi](https://doi.org/10.1016/j.ejor.2023.02.031)
- Yanpirat, N., Silva, D. F., & Smith, A. E. (2023). Sustainable last mile parcel delivery and return service using drones. *Engineering Applications of Artificial Intelligence*, 124, 106631. [doi](https://doi.org/10.1016/j.engappai.2023.106631)
- Yu, F., Li, C., Chen, Q., Li, Y., Chen, Y., & Yang, X. (2027). Multi-objective two-echelon vehicle-drone routing with load-dependent drone under pickup delivery time windows. *Transportation Research Part E*, 217, 105195. [doi](https://doi.org/10.1016/j.tre.2026.105195)
