# Dynamic 2E-CTVRP: formulation summary (as implemented)

This is a compact reference for the model that the code in `src/dynamic_*.py` actually solves. It
covers:
- the sets, parameters and decision variables;
- every formula and constraint, with its meaning and where it lives in the code;
- how the dynamic dataset is built from the original static instances;
- what changed relative to the first version of the dynamic problem, i.e. the formulation in
  `dynamic_2e_ctvrp_research_guide_dataset_blueprint.md` (§3) and the spec
  `bai_toan_iii_truck_drone.pdf`.

---

## 1. Problem in one paragraph

A depot sends **K trucks** along road routes to **n satellites** on the edge of a disaster area.
At each satellite the truck stops, and its drones fly supplies to **one victim cluster centroid**
and back. Each satellite serves exactly one cluster: this is the satellite→cluster assignment σ.
After dispatch, roads degrade or close, demand surges and priorities change. These changes are
revealed at **event epochs** $t_1, t_2, \dots$. At each epoch, everything already executed is
frozen and only the **residual problem** is re-optimized: the satellites not yet reached, starting
from each truck's current position. A cluster's demand can only grow **before or during** its
service, and never beyond what its truck can still carry (§5.4a). A surge that arrives while the
drones are still flying is delivered as an **extra drone wave** (§5.4b).

The planner does not see the world perfectly. Road reports are partial and may be wrong, and
surge sizes are reported with noise. It plans on a **belief** that it updates with Bayes' rule,
which makes the problem a POMDP (§5.1, §5.4c). An operator-selection policy learned with deep
reinforcement learning drives the search (§5.8). Three objectives are traded off:
priority-weighted delivery time, equity, and total fleet time.

---

## 2. Sets and indices

| Symbol | Meaning | In code |
|---|---|---|
| $v_0$, $v_{n+1}$ | start / end depot (same location) | node `0`, node `n+1` |
| $V_s=\{1..n\}$ | satellites | `1..n` |
| $M=\{1..n\}$ | victim cluster centroids (K-Means, frozen) | `clustered.centers` |
| $\mathcal K=\{0..K-1\}$ | trucks | `inst.no_k` |
| $\mathcal S=\{\text{Open},\text{Degraded},\text{Detour}\}$ | road states | `STATES` |
| $e=0,1,2,\dots$ | decision epochs, at trigger times $t_e$ ($t_0=0$) | `Scenario.epochs` |
| $L(t)\subseteq V_s$ | **locked** satellites at time $t$ (reached, or committed target) | `Snapshot.locked` |
| $F(t)=V_s\setminus L(t)$ | **free** satellites | `Snapshot.free_sats` |
| $M_F(t)$ | **free** centroids (not assigned to a locked satellite) | `Snapshot.free_centroids` |
| $\mathcal K_A(t)$ | **active** trucks (not yet heading home) | `ReplanProblem.active` |
| $\mathcal J(t)$ | **re-delivery jobs** still to be served (residual demand of already-served clusters) | `ReplanProblem.free_jobs`, `Plan.jobs` |

## 3. Parameters

| Symbol | Meaning | Value / source |
|---|---|---|
| $Q$ | truck capacity | instance (12 500 or 3 500) |
| $U, P$ | drones per satellite, drone payload | instance; $UP$ = parallel drone capacity |
| $v_g, v_d$ | truck speed, drone speed | 40 km/h; 40–80 km/h |
| $t^0_{ij}$ | baseline truck time = Manhattan distance / $v_g$ | `dist.time_n_0` |
| $t'_{ij}$ | one-way drone time satellite $i$ → centroid $j$ (Euclidean / $v_d$) | `dist.time_n_m` |
| $\phi_s$ | travel-time multiplier of state $s$ | $(1.0,\ 2.5,\ 5.0)$ (`PHI`) |
| $\mathcal P(s'\mid s)$ | per-epoch Markov transition of road states (belief prediction) | `DEFAULT_TRANSITION` |
| $\rho_{obs}$ | probability that a changed road is reported (`reveal_prob`) | 1.0 (full) or 0.6 (partial) |
| $s_{ij}(t)$ | **true** state of arc $(i,j)$ (hidden from the planner if $\rho_{obs}<1$) | `RoadNetwork.true_state` |
| $\kappa$ | accuracy of a road report | 1.0 (benchmark), 0.8 (POMDP experiment) |
| $\theta$ | chance-constraint threshold of the `belief_cc` planner | 0.8 |
| $\sigma_d$, $z$ | relative noise of reported surge demand; robustness factor for capacity/battery | 0 (benchmark), 0.15 (POMDP experiment); $z=1$ |
| $b_{ij}^t(s)$ | planner's **belief** over the state of arc $(i,j)$ | `RoadNetwork.belief` |
| $d_j(t)$ | demand of cluster $j$ (changes with surges) | `DemandProcess` |
| $w_j(t)\in[0,1]$ | priority of cluster $j$ | `initial_priorities` + updates |
| $E$, $\rho$ | battery capacity (normalized to 1) and landing reserve | $E=1$, $\rho=0.15$ (`drone_energy` block) |
| $T_0$, $v_{ref}$ | full-battery flight time of an empty drone at the reference speed | 0.65 h (39 min) at 60 km/h |
| $\omega$ | share of empty-drone power due to weight at $v_{ref}$ (the rest is speed-dependent) | 0.6 |
| $\beta$ | extra weight-term power at full payload | 0.8 (+80%) |
| $k_w$, $\bar w$ | optional wind term coefficient and wind speed | 0 (off by default) |

## 4. Decision variables (at epoch $t$)

The code uses a **route + assignment** encoding (the static paper's representation), not binary
MILP variables.

| Variable | Meaning | In code |
|---|---|---|
| $r_k(t)=(\pi_k(1),\dots,\pi_k(m_k))$ | ordered list of stops truck $k\in\mathcal K_A$ will make next: free satellites **and** re-delivery jobs | `Sol.routes[k]` |
| $\sigma_t: F(t)\to M_F(t)$ | bijection: which centroid each free satellite's drones serve | `Sol.sigma` |
| $h_j\in H_j$ | **host satellite** of re-delivery job $j$: where the truck stops and launches the extra drone wave(s) | `Sol.hosts[j]` |

The following quantities are **derived** from the decisions, not chosen freely:

| Quantity | Meaning |
|---|---|
| $a_i$ | truck arrival time at satellite $i$ |
| $w_i$ | time the truck waits at $i$ while drones fly |
| $e_i$ | time supplies reach the centroid served from $i$ (delivery time) |
| $\text{ret}_k$ | truck $k$'s return time to the depot |
| $L_k$ | load truck $k$ hands to its drones |

The equivalent MILP variables, for reference, are $x^k_{ij}=1$ if $j$ directly follows $i$ in
$r_k$, $y^k_i=1$ if $i\in r_k$, and $z_{ij}=1$ if $\sigma_t(i)=j$.

**Fixed by history (not decision variables):** for every truck, its start node
$v_{\pi_k(0)}(t)$, its available time $\alpha_k(t)$ and its remaining capacity $Q_k(t)$. For every
locked satellite, its truck, position, centroid, arrival and delivery time.

---

## 5. Formulas

### 5.1 Road model and expected travel time (`dynamic_env.py`)

$$t_{ij}^{\text{true}}(\tau)=t^0_{ij}\,\phi_{s_{ij}(\tau)}$$

This is the real time a truck needs on arc $(i,j)$ if it **departs** at time $\tau$. The state at
departure decides the time, so a truck already on a road finishes it at the old speed. The matrix
can be asymmetric ($t_{ij}\neq t_{ji}$) because some damage is one-way.

$$\mathbb E[t_{ij}(t)]=t^0_{ij}\sum_{s\in\mathcal S}\phi_s\,b^t_{ij}(s)$$

This is the travel time the **planner** uses: the baseline time scaled by the expected multiplier
under its belief. A "closed" road is a 5× detour, never an infinite cost, so every satellite stays
reachable.

**Observation model and belief update (POMDP, `RoadNetwork`).** At each event:

1. **Prediction.** Every arc's belief moves one epoch forward, $b\leftarrow b\,\mathcal P$, since
   roads may have degraded unseen.
2. **Reports.** Each arc whose true state changed is reported with probability $\rho_{obs}$. A
   report $o$ is correct with probability $\kappa$ (the observation accuracy), otherwise it names
   one of the two other states:
   $$O(o\mid s)=\begin{cases}\kappa & o=s\\ (1-\kappa)/2 & o\neq s\end{cases}$$
3. **Bayes update** of every reported arc: $b'(s)\propto O(o\mid s)\,b(s)$.
4. **Traversal.** A truck that has driven an arc observes its true state exactly.

At $t=0$ the network is surveyed, so the belief equals the truth. With $\rho_{obs}=\kappa=1$ the
belief is the truth at every event, which gives the fully observed dynamic problem.

**How the planner reads the belief** (`--planner`, `RoadNetwork.planning_time`):

| mode | planning travel time | meaning |
|---|---|---|
| `oracle` | $t^0_{ij}\,\phi_{s_{ij}(t)}$ | knows the true states (upper bound) |
| `naive` | $t^0_{ij}\,\phi_{o_{ij}}$ | trusts every report; unreported roads assumed unchanged |
| `belief` | $\mathbb E[t_{ij}]=t^0_{ij}\sum_s\phi_s b_{ij}(s)$ | expected time under the Bayesian belief |
| `belief_cc` | $t^0_{ij}\,\phi_{Detour}$ if $b_{ij}(Open)+b_{ij}(Degraded)<\theta$, else $\mathbb E[t_{ij}]$ | **chance constraint** of the proposal (slide 10): a road that is usable with probability below $\theta$ (0.8) is planned as a detour, so routes avoid unreliable roads |

### 5.2 Drone service at a satellite (`dynamic_sim.py`, `dynamic_solver.py`)

$$n^{\text{wave}}_i=\left\lceil \frac{d_{\sigma(i)}}{UP}\right\rceil,\qquad
w_i=n^{\text{wave}}_i\cdot 2\,t'_{i\sigma(i)}$$

The number of drone waves needed is the demand divided by the drones' combined payload, rounded
up. The truck waits for all waves, each one a round trip. When $d\le UP$ there is one wave, which
is exactly the static paper's model.

### 5.2b Drone battery: energy depends on speed and load (`drone_energy.py`)

**Flight power** while carrying payload $\ell$ at speed $v$, as a share of a full battery per hour:

$$p(v,\ell)=\underbrace{a\Big(1+\beta\,\frac{\ell}{P}\Big)}_{\text{weight term}}+\underbrace{b\Big(\frac{v}{v_{ref}}\Big)^{3}}_{\text{speed term}}+\underbrace{k_w\,\bar w^{2}}_{\text{wind (optional)}},
\qquad a=\frac{\omega}{T_0},\ \ b=\frac{1-\omega}{T_0}$$

Heavier loads and higher speeds both draw more power. This is the proposal's power model
$k_1(W_d+L_d)+k_2 v^3+k_3\tilde w^2$ (slide 12), calibrated so that an empty drone at $v_{ref}$
flies $T_0$ hours on a full battery.

**Per-drone load of one wave.** The cluster's demand is split evenly over the $U$ parallel
drones of each wave:

$$\ell_{j}=\frac{d_j}{U\,n^{\text{wave}}_j}\ \le P$$

**Energy of one sortie** from satellite $i$ to centroid $j$. The drone flies out loaded and back
empty, each leg lasting $t'_{ij}=\text{dist}_{ij}/v_d$:

$$e_{ij}=t'_{ij}\Big[p(v_d,\ell_j)+p(v_d,0)\Big]$$

**Where flight distance enters.** $t'_{ij}$ is the one-way flight time, i.e. the straight-line
(Euclidean) distance $\delta_{ij}$ from satellite $i$ to centroid $j$ divided by the drone speed
(`dist.time_n_m` in `src/common.py`). Written with distance explicitly:

$$t'_{ij}=\frac{\delta_{ij}}{v_d}\quad\Longrightarrow\quad
e_{ij}=\delta_{ij}\cdot\frac{p(v_d,\ell_j)+p(v_d,0)}{v_d}$$

So energy grows **linearly with distance**: the fraction is the battery used per kilometre of
satellite→cluster distance (flown out loaded and back empty). At 60 km/h this is 5.1%/km empty
and 6.4%/km at full payload, giving the 16.6 km and 13.4 km ranges below. Take-off, landing and
hovering are not modelled, so there is no fixed per-sortie overhead.

**Battery constraint (C4):** a sortie must leave the safety reserve in the battery,

$$e_{i\sigma(i)}\ \le\ E\,(1-\rho)\qquad\forall i .$$

**Consequences:**

- **Speed.** Flight time falls as $1/v$ but the speed term grows as $v^3$, so range peaks near
  $v_{ref}$. With the defaults, the one-way range of a fully loaded drone is 11.5 km at 40 km/h,
  13.4 km at 60 km/h and 12.4 km at 80 km/h. A fixed time budget would wrongly give a faster drone
  more range.
- **Load.** At 60 km/h the range drops from 16.6 km empty to 13.4 km at full payload.
- **Dynamics.** Demand surges raise $\ell_j$. A satellite→cluster pair that was feasible before a
  surge can become infeasible, which forces a re-assignment of σ at the next epoch. The feasible
  pairs are recomputed at every epoch from the current demand.
- **Instances.** On the 12 benchmark instances, 13–32% of satellite–centroid pairs are
  battery-infeasible at base demand. At t=0 the paper's optimal plan becomes infeasible on 3 of 5
  checked instances, raising the optimal $Z_1$ by 0.1–0.5%.

Each extra wave is assumed to start with a fresh (swapped) battery; the swap takes no time.

**Execution safeguard: lighter waves.** A surge can raise a cluster's demand after its
satellite→cluster pairing is already locked, for example while the truck is driving to that
satellite. At execution the drones then fly **more, lighter waves**: the smallest
$n\ge\lceil d/UP\rceil$ with $e_{ij}(v_d,\,d/(U n))\le E(1-\rho)$. This costs extra waiting time
instead of overflying the battery. The planner itself never selects a battery-infeasible pair
(C4).

### 5.3 Timeline recursion (truck $k$, from its committed start)

$$a_{\pi_k(1)}=\alpha_k(t)+\mathbb E[t_{\pi_k(0)\pi_k(1)}],\qquad
a_{\pi_k(p)}=a_{\pi_k(p-1)}+w_{\pi_k(p-1)}+\mathbb E[t_{\pi_k(p-1)\pi_k(p)}]$$

Each arrival equals the previous arrival, plus the drone waiting time there, plus the driving time.
The first leg starts at the truck's committed position and available time.

$$e_i=a_i+t'_{i\sigma(i)}$$

The first delivery happens one drone flight after the truck arrives.

$$\text{ret}_k=a_{\pi_k(m_k)}+w_{\pi_k(m_k)}+\mathbb E[t_{\pi_k(m_k),\,n+1}]$$

The return time adds the final drive back to the depot.

### 5.4 Commitment (freezing history) at event time $t_e$ (`Simulator.commit`)

1. Satellites with $a_i\le t_e$ are **locked**, together with their centroid $\sigma(i)$.
2. A truck driving toward satellite $b$ at $t_e$ finishes that arc, so $b$ is locked too. Its
   virtual start becomes $v_{\pi_k(0)}(t)=b$ and its available time becomes
   $\alpha_k(t)=a_b+w_b$.
3. A truck waiting at satellite $i$ has $v_{\pi_k(0)}(t)=i$ and $\alpha_k(t)=a_i+w_i$.
4. A truck already heading to the depot is retired: it becomes inactive and its return time is
   fixed.

Remaining capacity is

$$Q_k(t)=Q-\sum_{i\in L(t)\cap r_k}q_i ,$$

where $q_i$ is the delivered amount if $i$ was already reached, or the current demand if $i$ is
only a committed target.

### 5.4a Demand rule: surges only before or during service (`admissible_surges`)

An event file gives a *requested* surge multiplier $m_c$ for some clusters. The environment
applies it only if it is admissible for the plan being executed:

| status of cluster $c$'s satellite at $t_e$ | surge |
|---|---|
| truck has **left** ($a_s+w_s\le t_e$) | **dropped**: demand cannot change after service |
| truck **parked**, drones flying ($a_s\le t_e<a_s+w_s$) | allowed; served as an extra drone wave (§5.4b) |
| not reached yet | allowed; delivered at arrival |

The increase is capped by the remaining capacity of the truck $k$ currently planned to serve $c$:

$$\Delta d_c=\Big\lfloor\min\big(d_c(t_e)\,(m_c-1),\ \text{slack}_k\big)\Big\rfloor,\qquad
\text{slack}_k=Q-\Big(\sum_{\text{visited}}q+\sum_{\text{planned}}d+\sum_{\text{jobs}}r_j\Big)$$

$\text{slack}_k$ is reduced as surges are granted, so the current plan always stays
capacity-feasible. Because admissibility depends on where the trucks are, two strategies can see
slightly different realized surges at $t_2$. Both the requested and the granted amounts are
logged (`surge` in each epoch record).

### 5.4b Recourse: re-delivery jobs (`create_residual_jobs`, `dynamic_sim.py`)

Right after the demand update at event $t_e$, every cluster $c$ whose satellite has **already been
reached** gets a re-delivery job if it is still short:

$$r_c(t_e)=d_c(t_e)-q^{main}_c-\sum_{j\in\mathcal J^{done}_c}q_j-\sum_{j\in\mathcal J^{plan}_c}r_j\ >\ 0
\ \Rightarrow\ \text{new job }j=(c,\ r_c)$$

- $q^{main}_c$ is what the cluster's own satellite delivered;
- $\mathcal J^{done}_c$ are its jobs already executed (counted at what they actually delivered);
- $\mathcal J^{plan}_c$ are its jobs still planned.

Under the demand rule (§5.4a) this covers two cases: a surge **during** delivery (truck still
parked, drones flying), and a **shortfall** when the truck carried less than the true demand,
which can only happen when demand is observed with noise (§5.4c). Clusters not yet reached need
no job, because their satellite delivers the demand valid at arrival. Surges after the truck left
no longer exist.

**Serving a job.** A job is a route stop like a satellite. The truck drives to the job's host
$h_j$, waits for $n^{wave}_j=\lceil r_j/UP\rceil$ waves of $2t'_{h_jc}$, and the delivery time is
$e_j=a_j+t'_{h_jc}$. The host can be any satellite whose drones reach the cluster with that load:

$$H_j=\{h\in V_s:\ e_{hc}(v_d,\ell_j)\le E(1-\rho)\},\qquad \ell_j=\frac{r_j}{U\,n^{wave}_j}$$

If a job is placed right next to a stop at the same satellite (for example first in the route of
a truck still parked at the cluster's own satellite), it costs **no extra driving**: it is simply
an extra drone wave there. The planner chooses the host together with the insertion position,
and the `job_host` operator moves a job to another host.

The `static` strategy never re-plans, so it never creates jobs. `--no-recourse` switches recourse
off for re-planning strategies too (for ablation).

### 5.4c Demand belief (POMDP)

The true size of an admissible surge is reported with noise,
$\hat d_c=d_c\,(1+\varepsilon)$ with $\varepsilon\sim\mathcal N(0,\sigma_d^2)$ (`demand_noise`).
The planner's belief is $\mathcal N(\mu_c,\ (\sigma_d\mu_c)^2)$ with $\mu_c=\hat d_c$, until a
truck reaches the cluster and observes the true demand.

- **Objectives** use the mean $\mu_c$.
- **Truck capacity and the drone battery** use a robust value
  $\mu_c+z\,\sigma_d\mu_c$ with $z=1$ (`z_cap`); the `naive` planner uses $\hat d_c$ alone.
- **If the truth is larger than planned**, the truck delivers what it carries and the shortfall
  becomes a re-delivery job at the next event (§5.4b).

### 5.5 Objectives (minimize; over the **whole horizon**, locked parts are constants)

$$f_1=\frac{1}{D(t)}\Big[\sum_{i\in V_s} w_{\sigma(i)}(t)\;q_{i}\;e_i+\sum_{j\in\mathcal J} w_{c_j}(t)\;q_j\;e_j\Big]
\qquad\text{(priority-weighted delivery time — efficiency)}$$

This is the average time supplies reach victims, weighted by how much is delivered and how urgent
the cluster is, **including re-deliveries**. $D(t)=\sum_j d_j(t)$, and $q$ is the amount
delivered by a stop: the planned amount for future stops, the actual delivered amount for executed
ones.

$$f_2=Z_3=\max_{i\in V_s}a_i-\min_{i\in V_s}a_i\qquad\text{(arrival spread — equity)}$$

This is the gap between the first and the last community to be reached. Smaller means fairer.

$$f_3=Z_1=\sum_{i\in V_s}a_i+\sum_{k\in\mathcal K}\text{ret}_k\qquad\text{(cumulative fleet time — operational cost)}$$

This is the static paper's objective, the sum of all arrival and return times. In code,
`--mode so --weights 0,0,1` optimizes $f_3$ alone and reproduces the paper.

$f_2$ and $f_3$ sum over the satellites' **main** visits only. A re-delivery stop is not added as
an extra term; its cost appears as later arrivals for the stops after it and a later return. This
keeps $f_3$ comparable with the paper and between strategies: otherwise each extra stop serving
people would add its whole arrival time and make serving more demand look worse.

**Scalarization used inside the search:**
$\;g(S)=\sum_m \lambda_m f_m(S)/\hat f_m$, where $\hat f$ are the objectives of the epoch's
starting solution. The weights $\lambda$ cycle through a Das-Dennis lattice to spread the Pareto
archive.

**Plan to commit:** the archive member that minimizes
$\sum_m \lambda^{c}_m\,(f_m-f_m^{\min})/(f_m^{\max}-f_m^{\min})$, with commit weights
$\lambda^{c}=(1/3,1/3,1/3)$ by default.

### 5.6 Constraints

| # | Constraint | Meaning | How enforced |
|---|---|---|---|
| C1 | $\bigcup_k r_k=F(t)\cup\mathcal J(t)$, each free satellite and each job in exactly one route | every remaining satellite is visited once and every residual demand is scheduled | representation (operators only move or swap stops); checked by `check_plan` |
| C2 | $\sigma_t$ is a bijection $F(t)\leftrightarrow M_F(t)$ | each cluster served by exactly one satellite | representation (operators only swap centroids) |
| C3 | $\sum_{i\in r_k} d_{\sigma(i)}(t)\le Q_k(t)\ \ \forall k\in\mathcal K_A$ | truck capacity, minus what was already handed out | penalty $\max(0,L_k-Q_k)$; feasible-first comparison |
| C4 | $e_{i\sigma(i)}(v_d,\ell_{\sigma(i)})\le E(1-\rho)\ \ \forall i$ (§5.2b) | each sortie's energy, which depends on speed and load, leaves the safety reserve | infeasible pairs are excluded from σ moves, construction, repair and path relinking; pairs are recomputed each epoch from current demand; any remaining infeasible pair gets a $10^5$ penalty; at execution, extra lighter waves absorb surges on locked pairs |
| C5 | $d_{\sigma(i)}\le UP$, **relaxed** to waves (§5.2) | drone payload | never infeasible; costs extra waiting |
| C6 | $\lvert r_k\rvert\ge 1$ if truck $k$ starts at the depot ($t=0$ only) | no idle truck, as in the paper's MILP | penalty $10^5$ |
| C7 | locked part of the plan unchanged | cannot undo the past | the residual problem contains only free satellites; checked by `check_plan` after every re-plan |
| C8 | each route starts at $v_{\pi_k(0)}(t)$ at $\alpha_k(t)$ and ends at $v_{n+1}$ | trucks continue from where they are | built into the timeline recursion |
| C9 | $h_j\in H_j$ for every job $j$ (§5.4b) | the extra sortie is battery-feasible from its host | hosts chosen only from $H_j$ in insertion and in the `job_host` move |
| C10 | all demand covered: $q^{main}_c+\sum_{j\in\mathcal J_c}q_j=d_c(t)\ \forall c$ | no unmet demand | **soft**: C1 schedules every unit, and C3 counts it in truck load. If the remaining fleet capacity is too small, no plan is feasible; the plan with the smallest overload (C3 penalty) is committed, the simulator delivers what fits, and the rest is reported as unmet and charged in $f_1$ (§5.9) |

Candidate solutions are compared **lexicographically**: first by total violation, then by $g(S)$.
Only feasible solutions (violation 0) enter the Pareto archive.

C10 is deliberately not a hard constraint. A surge larger than what the trucks still carry would
make a hard version infeasible, and the optimizer would return no plan at the moment one is most
needed. With the soft version, unmet demand is zero whenever delivering everything is physically
possible, and is minimized and reported when it is not.

### 5.7 Disruption score per route (segment-aware search; spec eq. 9)

$$c_k(t)=\operatorname{norm}\big(0.5\,r^{road}_k+0.3\,r^{demand}_k+0.2\,r^{timing}_k\big)$$

The score is a normalized blend of three parts:
- $r^{road}_k$: share of truck $k$'s planned arcs whose expected time rose by more than 5%;
- $r^{demand}_k$: relative change in the demand of its clusters;
- $r^{timing}_k$: relative growth of its remaining driving time.

Each move picks its "primary" truck with probability $0.5/\lvert\mathcal K_A\rvert+0.5\,c_k$, so
the most disrupted routes are searched more.

### 5.8 DRL operator controller (SR-DRL; spec eqs. 10–12; `drl_selector.py`)

A **Double DQN** chooses which neighbourhood operator the search applies next. This is the
spec's "DRL selects operators" design (eq. 11), and the action space is the 9 operators of the
re-optimizer.

**State** $s_t\in\mathbb R^{23}$ (spec eq. 10):

| group | features |
|---|---|
| epoch context (7) | $\max_k c_k$, mean $c_k$; share of free satellites; re-delivery jobs per free satellite; $u_V$ = free demand / remaining fleet capacity; $u_D$ = mean $d/UP$; $r_{uncertain}$ = share of road arcs whose belief is not certain |
| search state (7) | share of the time budget used; steps since the last improvement (scaled); current solution infeasible; gap current vs. best; recent acceptance rate; recent improvement rate; archive fill $\lvert A\rvert/A_{\max}$ |
| last action (9) | one-hot of the previous operator |

**Action:** an operator. It is applied for a segment of 10 moves (a macro action), so the
network runs once per 10 moves and costs almost no search time.

**Reward:** per move, +5 for a new best solution, +2 for an improved current solution, +1 for a
new Pareto-archive member; summed over the segment and rescaled. This is the improvement part of
spec eq. 12. The budget term $-\beta\,\text{Eval}/\text{budget}$ is implicit, because every run
has the same CPU budget.

**Learning:**
- **Network and update.** MLP 23–64–64–9 (ReLU) with Double-DQN targets
  $y=r+\gamma\,Q_{target}\big(s',\arg\max_a Q(s',a)\big)$, $\gamma=0.9$.
- **Training mechanics.** Huber loss, Adam ($10^{-3}$), replay buffer of 50 000, target sync
  every 1000 decisions, and $\varepsilon$-greedy decaying from 1.0 to 0.05.

**Training data:** `scripts/train_drl_selector.py`, on a **separate** set of 14 instances
(`instances/dynamic_events_train/`, none of them in the test benchmark) with random scenarios,
seeds and budgets. At test time the policy is frozen ($\varepsilon=0.05$).

`uniform` (random operator) and `adaptive` (ALNS roulette) are the non-learning baselines
(spec §8.3, "adaptive weight but no DRL").

### 5.9 Realized evaluation (after the last event; `realized_metrics`)

The final plan is **executed on the true network** (§5.1). The same $f_1,f_2,f_3$ are reported
using actual times. In the realized $f_1$, demand that was never delivered is charged as if
delivered at $C_{ret}$, the time the last truck is back (the earliest a new trip could start):

$$f_1^{real}=\frac{1}{D}\Big[\sum_{\text{stops}}w_c\,q\,e+\sum_c w_c\,u_c\,C_{ret}\Big],\qquad u_c=\max\big(0,\ d_c^{final}-\text{delivered}_c\big)$$

Without that term, leaving demand undelivered would lower $f_1$ and look better than delivering
it late. Also reported:

| Metric | Formula | Meaning |
|---|---|---|
| $C_{del}$ | $\max_i e_i$ | time the last cluster receives supplies |
| $C_{ret}$ | $\max_k \text{ret}_k$ | time the last truck is back at the depot |
| unmet | $\sum_c u_c/D$ | share of demand never delivered. With recourse this is non-zero only if the fleet ran out of goods, or for the `static` strategy, which never re-plans |
| re-deliveries | $\lvert\mathcal J\rvert$ executed | number of re-delivery stops made |

---

## 6. Generating the dynamic dataset from the static instances

The original files (`instances/main|large/*.txt` and `*.clusters.json`) are **not modified**. A
companion `instances/dynamic_events/<instance>.events.json` is created for each instance.

```bash
python3 scripts/generate_dynamic_dataset.py              # 12-instance benchmark subset (v_d = 60)
python3 scripts/generate_dynamic_dataset.py --all-main   # every instance in instances/main
python3 scripts/generate_dynamic_dataset.py --seed 7 --out instances/dynamic_events_s7   # another replica
```

**Steps for each static instance:**

1. **Load the static instance**: coordinates, demands and the frozen clusters. Compute $t^0$ and
   $t'$.
2. **Build a baseline plan.** This is a deterministic construction plus Z1 local search. Simulate
   it to obtain the satellite arrival times.
3. **Set the event times.** $t_1$ and $t_2$ are the 30% and 65% quantiles of those arrival times,
   so each event really splits served from unserved satellites.
4. **Calibrate the battery.** Use the default speed+load energy model (§5.2b). If no
   battery-feasible satellite↔centroid matching exists even with all demand at 1.3× (surge
   margin), raise the battery capacity $E$ by bisection until one does. The result is written to
   `_meta.battery_note`. None of the 12 benchmark instances needed this.
5. **Set initial priorities.** $w_j\sim U(0.50,0.85)$.
6. **Generate one scenario per damage level (10/30/50%) and observability setting.**
   - **Flood centre:** an inner-ring satellite that is still unserved at $t_1$.
   - **Ranking:** all first-echelon arcs are ranked by distance to the flood. These are depot→sat,
     sat→sat and sat→depot arcs; the last are written as `(i, 0)` in the file.
   - **Epoch 1:** the closest half of the target share is damaged. The state is Detour with
     probability $0.25+0.4\cdot$closeness, otherwise Degraded.
   - **Epoch 2:** the flood spreads to the next half, and 40% of epoch-1 Degraded arcs escalate to
     Detour.
   - **Direction:** 70% of damaged sat↔sat arcs are damaged in both directions; 30% are one-way.
   - **Demand surges:** about 15% of clusters, chosen as the unserved ones nearest the flood, surge
     ×1.10–1.20 at $t_1$ (priority set to 1.0) and ×1.15–1.30 at $t_2$ (priority 0.95).
7. **Partial-observability variant.** `damage_30pct_partial` additionally sets
   `reveal_prob = 0.6`. Report accuracy and demand noise are run-time options
   (`obs_accuracy`, `demand_noise`), so any scenario can be run as a POMDP.
8. **Requested vs. realized surges.** The multipliers in the file are *requested* surges. At run
   time the demand rule (§5.4a) drops those that arrive after the truck has left and caps the rest
   to the serving truck's spare capacity.

**Training set for the DRL controller.** The same generator, with a different seed, builds event
files for 14 *other* instances. These are M-10-07-60, M-10-8-60 and M-17-15-60 (both fleets),
and M-20-20 and M-50-50 at 50 and 70 km/h (both fleets). None of them is in the 12-instance test
benchmark:

```bash
python3 scripts/generate_dynamic_dataset.py --seed 7 --out instances/dynamic_events_train \
    --instances main/M-10-07-60-1,main/M-10-07-60-2,...,large/M-50-50-70-2
```

**File format** (one file per instance; each top-level key is one scenario):

```json
{
  "damage_30pct": {
    "base_instance": "M-09-09-60-1.txt", "num_satellites": 9,
    "drone_energy": {"v_ref": 60.0, "t_empty_ref": 0.65, "weight_share": 0.6, "beta_payload": 0.8,
                     "reserve": 0.15, "battery": 1.0, "wind": 0.0, "k_wind": 0.0},
    "reveal_prob": 1.0,
    "markov_transition": [[0.92,0.06,0.02],[0.15,0.75,0.10],[0.05,0.15,0.80]],
    "initial_priorities": {"1": 0.73, "...": 0.6},
    "epochs": [
      {"epoch_id": 0, "trigger_time_h": 0.0, "edge_updates": [], "demand_multipliers": {}, "priority_updates": {}},
      {"epoch_id": 1, "trigger_time_h": 1.03,
       "edge_updates": [{"u": 3, "v": 5, "state": "closed_detour"}, {"u": 4, "v": 0, "state": "degraded"}],
       "demand_multipliers": {"3": 1.13}, "priority_updates": {"3": 1.0}},
      {"epoch_id": 2, "trigger_time_h": 1.656, "...": "..."}
    ]
  },
  "_meta": {"seed": 42, "baseline_arrival_quantiles": [0.807, 1.435], "battery_note": "..."}
}
```

**Writing events by hand.** You can write your own events file in the same format:
- `u`, `v` are in `0..n`, and `v = 0` means "back to the depot";
- `state` is one of `open`, `degraded` or `closed_detour`. A `travel_time_multiplier` field from
  the guide's original format is also accepted.

Then run:

```bash
python3 scripts/run_dynamic.py --instance instances/main/M-09-09-60-1.txt \
    --events my.events.json --scenario damage_30pct -v
```

---

## 7. Changes compared with the first version of the dynamic problem

The first version is the guide's §3 formulation and §5.3 generator, together with the
`bai_toan_iii` spec.

| Topic | First version | Implemented version | Why |
|---|---|---|---|
| Event times | fixed 0.40 h / 0.85 h | 30% / 65% quantiles of baseline arrival times, per instance | the depot is more than 0.5 h from the ring, so fixed times fired before any satellite was reached and nothing was committed |
| Damage pattern | uniform-random arcs; `all_sat_arcs[:n]` bug; no return arcs | spatially clustered flood on an unserved inner-ring satellite, with progression and escalation; return arcs `(i,0)`; ratio over all arcs | the spec (§8.2) asks for spatial clusters; the old version could damage only roads already used |
| Arc direction | symmetric | 30% one-way damage, so $t_{ij}\ne t_{ji}$ | exercises the asymmetry concern of guide §1.2.2 |
| Demand surges | random clusters | unserved clusters nearest the flood, with a priority raise | a surge on a served cluster cannot be re-planned |
| Drone capacity $d\le UP$ | hard constraint | relaxed to **waves** $\lceil d/UP\rceil$ (more waiting) | a surge above $UP$ would otherwise make the instance infeasible |
| Closed roads $\xi_{ij}=0$ / $x\le\xi$ | arc removed | Detour multiplier ×5 (guide §1.3 option 1) | keeps every satellite reachable and σ a bijection |
| Planning vs. execution | single time matrix | planner uses $\mathbb E[t]$ from the **belief**; simulator uses the **true** departure-time-dependent times | needed for partial observability and for honest realized metrics |
| Partial observability | mentioned (POMDP) only | implemented: `reveal_prob`, Markov prediction $b\leftarrow b\mathcal P$, reveal on traversal | spec §4.8 |
| Commitment rule | "last reached or currently targeted satellite" | same, made precise: the en-route target and its σ are locked; trucks heading home are retired | — |
| Objectives | $f_1$ with $d$, $f_2=Z_3$, $f_3=Z_1$ | same three, but over the **whole horizon** (locked parts constant); $f_1$ uses delivered amounts for executed stops | makes values comparable across epochs and with the static result |
| Demand coverage after surges | not addressed (guide: random surges; proposal: $h_i(t^+)=[\,\cdot\,]^+$ unmet update) | **re-delivery recourse**: residual demand becomes a job served from any battery-feasible host satellite, including an extra wave while still parked; "all demand met" as a soft constraint with a capacity-overload penalty | removes unmet demand whenever the fleet still has goods, without making the problem infeasible when it does not |
| Unmet demand in $f_1$ | — | undelivered demand charged at $C_{ret}$ | otherwise leaving people unserved would improve $f_1$ |
| Elite pool across epochs | "old σ remain valid permutations" | old σ are **projected and repaired** (`project_sigma`) | after commitment some centroids are locked to other satellites, so an old σ is not a bijection on the free sets |
| 2-opt asymmetry fix | operators must check direction | no change needed | every move is scored by full re-evaluation, never by delta formulas |
| Empty truck at t=0 | not discussed | penalty: every depot-start truck visits ≥1 satellite | matches the static MILP, so t=0 reproduces the paper |
| Search operators | N1–N7 | N1–N7 plus **destroy-repair**; moves focused on routes with high $c_k(t)$ | spec action set (eq. 11) and segment idea (eq. 9) |
| Drone battery | proposal: energy $\propto k_1(W+L)+k_2v^3+k_3\tilde w^2$, sequential drain, probabilistic reserve; guide: fixed time budget $2t'\le T_d^{\max}$ | **energy model depending on speed and load** (same form as the proposal, wind optional), deterministic reserve $\rho$ | keeps the proposal's physics while staying an O(1) check per sortie; the fixed time budget never bound on these instances and gave faster drones more range |
| Vehicle crashes, variable drone speed, multi-drop sorties | in the proposal slides | **not modelled** (vehicles always operational, fixed $v_d$ per instance, out-and-back sorties) | simplifications recommended by guide §2 |
| Dynamic MILP | $x\le\xi$, recomputed $H(t)$ | **not implemented** (no Gurobi licence here) | listed as future work |
| When demand may change | proposal: $h_i(t^+)=[\mathbb E\tilde h_i+\lambda_i-y]^+$ at any time | **only before or during service**, capped by the serving truck's remaining capacity (§5.4a) | a surge after the truck has left would require new trips and makes the model much harder |
| Partial observability | proposal: continuous capacity $\tilde\eta_{ij}\in[0,1]$, belief density $b_t(\eta)$, integral for $\mathbb E[t]$ | **3 discrete states**, Markov prediction, **noisy reports with Bayes update**, exact observation on traversal; demand belief $\mathcal N(\mu,\sigma^2)$ (§5.1, §5.4c) | the integral diverges near $\eta=0$ (guide §2); the discrete belief keeps the POMDP but stays exact and cheap |
| Safety threshold | proposal: route only if $\sum_{Open,Degraded}b_{t+1\mid t}\ge\theta$ | `belief_cc` planner: roads below $\theta$ are planned as detours (soft) | keeps every satellite reachable while still steering routes away from unreliable roads |
| DRL controller | proposal: DRL selects repair operators | **Double DQN** over a 23-feature state, trained on separate instances (§5.8) | spec eqs. 10–12 |

**Remaining limitations.**
- A truck that is already driving back to the depot is retired and is not used for re-deliveries,
  even if it still carries goods.
- Under demand noise, a truck can arrive with less than the true demand. The shortfall is
  re-delivered at the next event, but after the last event it stays unmet: there is no second
  trip from the depot.
- Vehicle crashes, variable drone speed, multi-drop sorties and the MILP are still not modelled.
