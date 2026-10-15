# VRP-D for post-disaster relief: initial dynamic formulation

*Transcribed from `Prj3_proposal.pptx` (Le Trung Dung, 20235489), Section 2 "Problem formulation"
(slides 6–15) and Section 3 "Proposed solution" (slide 16). This is the **first version** of the
dynamic problem. The simplified version that is implemented in the code is described in
[`FORMULATION.md`](FORMULATION.md).*

---

## 1. Introduction to the problem

- Severe natural and man-made disasters have devastating impacts on communities and human lives.
  Delivering essential items such as food, drinking water and medicines to affected areas is
  crucial to reduce human suffering and risk.
- Humanitarian logistics plays a pivotal role in disaster relief by ensuring timely and effective
  responses to the needs of affected communities.
- Ground vehicles (trucks, vans) struggle to reach remote locations, because roads after a disaster
  are often damaged or congested by unplanned panic evacuations.
- Drones need no driver on board and are not restricted by terrain, road networks or traffic.
  However, they have small capacity and inaccurate GPS drop-offs, and they must return to a
  station to recharge or swap batteries.

**→ Combine the speed of drones with the capacity of trucks.**

---

## 2. Problem formulation

### 2.1 Sets and indices

| Symbol | Meaning |
|---|---|
| $\mathcal V_R$ | Ground nodes: depots, safe intersections, launch/recovery hubs |
| $\mathcal V_D$ | Demand nodes: isolated victim clusters or shelters |
| $\mathcal V=\mathcal V_R\cup\mathcal V_D$ | Complete graph of all nodes |
| $v\in\mathcal V_{veh}$ | Ground trucks |
| $d\in\mathcal D_v$ | Drones carried by truck $v$ |
| $t$ | The current planning epoch or event horizon |

### 2.2 Decision variables

**Routing & delivery**

| Variable | Meaning |
|---|---|
| $x_{ijv}(t)\in\{0,1\}$ | 1 if truck $v$ travels directly from ground node $i$ to node $j$ during epoch $t$ |
| $z^{vd}_{mn}(t)\in\{0,1\}$ | 1 if drone $d$ launched from truck $v$ flies directly from node $m$ to node $n$ (enables multi-drop sorties across several victims) |
| $y_{iv}(t),\ y_{id}(t)\ge 0$ | Quantity of relief goods delivered to node $i$ by truck $v$ or by drone $d$, respectively |

**Health & failures (uncertainty)**

| Variable | Meaning |
|---|---|
| $\gamma_v(t)\in\{0,1\}$ | Health state of truck $v$: 1 if operational, 0 if immobilized by disaster or debris |
| $\chi_d(t)\in\{0,1\}$ | Health state of drone $d$: 1 if operational, 0 if crashed or lost mid-flight |

**State & timing**

| Variable | Meaning |
|---|---|
| $T_{iv}(t),\ T_{md}(t)$ | Arrival times of truck $v$ and drone $d$ at their respective nodes |
| $v^d_{mn}$ | Chosen flight speed of drone $d$ on edge $(m,n)$ |
| $L_v(t)$ | Remaining total payload on truck $v$ |
| $L_d(m)$ | Actual payload weight on drone $d$ when departing node $m$ (decreases after each drop) |
| $E_d(n)$ | Remaining battery energy of drone $d$ on arrival at node $n$ |

### 2.3 Predictive environment variables

| Symbol | Meaning |
|---|---|
| $\tilde\eta_{ij}(t)\in[0,1]$ | Hidden, continuous physical capacity of edge $(i,j)$: 1 = Open, 0 = Closed, intermediate = Degraded |
| $b_t(\eta_{ij})$ | Continuous probability density: the algorithm's belief about the true state of $\tilde\eta_{ij}(t)$ |
| $\mathcal P_{ij}(s'\mid s,t)$ | Markov state transition matrix: how the edge state evolves from epoch $t$ to $t+1$, from historical environmental data |
| $b_{t+1\mid t}(\eta_{ij})$ | *A priori* predicted belief of the road state for the next epoch, before new observations are made |
| $\tilde h_i(t)$ | Hidden, continuous true demand at node $i$, estimated by the belief $b_t(h_i)\sim\mathcal N\big(\mu_i(t),\sigma_i(t)^2\big)$ |

### 2.4 Constraints

#### Flow, health and predictive routing

**Truck operational flow.** A truck must conserve flow and can only route if it is operational.

$$\sum_j x_{ijv}-\sum_j x_{jiv}=0\le\gamma_v(t)\qquad\forall i,v$$

**Drone operational flow.**

$$\sum_m z^{vd}_{ml}-\sum_n z^{vd}_{ln}=0\le\chi_d(t)\qquad\forall l,v,d$$

**Predictive safety threshold.** Trucks can only route through an edge if the *predicted*
probability of that road remaining viable in the next epoch exceeds the safety threshold $\theta$.

$$\sum_{s\in\{\text{Open},\ \text{Degraded}\}} b_{t+1\mid t}(\eta_{ij}=s)\ \ge\ \theta$$

**Expected travel time.** $l_{ij}$ is the edge length and $v_g$ the truck speed on an open road.

$$\mathbb E\big[\tilde t^{\,v}_{ij}(t+1)\big]=\int_0^1\left(\frac{l_{ij}}{v_g\cdot\eta_{ij}}\right)b_{t+1\mid t}(\eta_{ij})\,d\eta_{ij}$$

#### Capacity and multi-drop logistics

**Truck capacity.**

$$\sum_i y_{iv}\le Q_v\qquad\forall v$$

**Drone multi-drop capacity.**

$$\sum_{l\in\text{Sortie}} y_{ld}\le q^{\max}_d\qquad\forall d$$

**Demand fulfillment.** Unmet demand is updated from continuous observations and delivered
payloads. $\lambda_i(t)$ is newly arriving demand. If a vehicle fails mid-route
($\chi_d(t)=0$ or $\gamma_v(t)=0$), its payload is lost and the demand resets.

$$h_i(t^+)=\Big[\mathbb E\big[\tilde h_i(t)\big]+\lambda_i(t)-y_{iv}(t)-y_{id}(t)\Big]^+$$

#### Physics-based energy and weight dynamics

**Speed- and weight-dependent power.** Drone energy depends on:
- the tare weight $W_d$;
- the current payload $L_d(m)$;
- the chosen speed $v^d_{mn}$;
- the stochastic wind resistance $\tilde w_{mn}$.

$d_{mn}$ is the flight distance and $k_1,k_2,k_3$ are coefficients.

$$e^d_{mn}=\Big(k_1\big(W_d+L_d(m)\big)+k_2\big(v^d_{mn}\big)^3+k_3\,\tilde w_{mn}(t)^2\Big)\frac{d_{mn}}{v^d_{mn}}\cdot z^{vd}_{mn}$$

**Sequential battery drain.** $M$ is a big-M constant.

$$E_d(n)\le E_d(m)-e^d_{mn}+M\big(1-z^{vd}_{mn}\big)$$

**Probabilistic safety reserve.** The drone must have a high probability of returning with
emergency reserve power.

$$P\big(E_d(\text{recovery\_node})\ge E^{res}_d\ \big|\ \chi_d(t)=1\big)\ \ge\ 1-\alpha$$

#### Synchronization and deadlock prevention

**Launch linkage.** Drones only launch from nodes currently occupied by their parent truck.

$$\sum_n z^{vd}_{in}\le\sum_j x_{ijv}$$

**Fault-tolerant rendezvous.** The truck must wait for the drone to arrive. If the drone crashes
($\chi_d(t)=0$), its expected arrival time is zeroed out, so the truck does not wait indefinitely.

$$T^{depart}_{recovery\_node,\,v}\ \ge\ \max\Big(T^{arrive}_{recovery\_node,\,v},\ \chi_d(t)\cdot T^{arrive}_{recovery\_node,\,d}\Big)$$

#### Catastrophic failure probabilities

$\zeta^R_{ij}$ and $\zeta^A_{mn}$ are the per-unit failure hazard rates on road edges and on air
edges.

**Truck survival.**

$$P\big(\gamma_v(t_{arrive})=1\ \big|\ \gamma_v(t_{depart})=1\big)=\exp\big(-\zeta^R_{ij}\cdot l_{ij}\big)$$

**Drone survival.**

$$P\big(\chi_d(t_{arrive})=1\ \big|\ \chi_d(t_{launch})=1\big)=\exp\!\left(-\zeta^A_{mn}\cdot\frac{d_{mn}}{v^d_{mn}}\right)$$

### 2.5 Objective functions

The model optimizes three competing objectives, evaluated over the expected values
($\mathbb E$) of the belief state $b(t)$.

**$f_1$: Expected deprivation cost (efficiency).** Minimizes the human suffering caused by waiting
times, weighted by node priority. $T_i(t)$ is the service time of node $i$, $r_i$ is the time its
demand was recorded, and $[\cdot]^+=\max(\cdot,0)$.

$$f_1=\mathbb E_{b(t)}\left[\sum_{i\in\mathcal V_D} w_i(t)\,\tilde h_i(t)\,\big[T_i(t)-r_i\big]^+\right]$$

**$f_2$: Worst unmet demand ratio (equity).** Ensures no individual cluster is systematically
neglected in favour of global efficiency.

$$f_2=\max_{i\in\mathcal V_D}\left\{w_i(t)\,\frac{\mathbb E\big[\tilde h_i(t)\big]}{h_i^{init}+\epsilon}\right\}$$

**$f_3$: Expected operational energy and catastrophic loss (cost).** Minimizes energy consumption
while heavily penalizing the loss of physical assets (trucks and drones). The terms are:
- $c_g L^{travel}_v$: truck travel cost;
- $c_a\tilde E^{flight}_d$: drone flight energy;
- $c_{hov}\tilde T^{wait}_d$: drone hovering / waiting;
- $C^{truck}_{loss}$, $C^{drone}_{loss}$: asset-loss penalties.

$$f_3=\mathbb E_{b(t)}\left[\sum_v c_g\,L^{travel}_v+\sum_d\Big(c_a\,\tilde E^{flight}_d+c_{hov}\,\tilde T^{wait}_d\Big)+\sum_v C^{truck}_{loss}\big(1-\gamma_v(t)\big)+\sum_d C^{drone}_{loss}\big(1-\chi_d(t)\big)\right]$$

The full problem at epoch $t$ is a time-varying multi-objective problem under partial observation:

$$\min_{x(t),\,z(t),\,y(t),\,v^d(t)}\ \big(f_1,\ f_2,\ f_3\big)\quad\text{s.t. the constraints of §2.4}$$

---

## 3. Proposed solution: SR-DRL-DMOEA

The problem is solved dynamically, epoch by epoch, with the **Segmented Route-aware
Deep-RL-assisted Dynamic MOEA (SR-DRL-DMOEA)**:

- **Multi-Objective Evolutionary Algorithm (MOEA).** It maintains a diverse population of feasible
  truck–drone routes that tracks the Pareto front between deprivation, equity and operational cost.
- **Deep Reinforcement Learning (DRL) controller.** It monitors the continuous environmental state
  vector $s_t$, including uncertainty variance and vehicle health. When new observations collapse
  the predictive belief $b_{t+1\mid t}$, or a vehicle is lost, the agent selects route repair
  operators (e.g. destroy-repair, 2-opt, drone-reassign) to fix the MOEA's broken paths.
- **Rolling horizon.** The algorithm evaluates the infinite horizon but commits only to the routes
  needed for the immediate epoch $t$. It recalculates the POMDP variables at $t+1$ as reality
  unfolds.
