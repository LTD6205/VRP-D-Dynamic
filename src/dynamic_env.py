"""
Dynamic environment for the rolling-horizon 2E-CTVRP (D-2E-CTVRP).

Implements the "discrete predictive road model" of the research guide (Sec. 3.2):
every first-echelon arc (i, j) is in one of three states

    s in {Open, Degraded, Detour}   with travel-time multipliers phi = (1.0, 2.5, 5.0)

so the truck travel time is t_ij(t) = t0_ij * phi[s_ij(t)]. A "closed" road is
modelled as a long detour (multiplier 5.0) instead of removing the arc, which keeps
every satellite reachable and preserves the satellite<->centroid bijection of the
static model (guide Sec. 1.3, option 1).

Two views of the road network are kept:

* the TRUE state (used by the simulator to execute the plan), and
* the planner's BELIEF b_ij(t) over the 3 states (used by the optimizer through
  E[t_ij(t)] = t0_ij * sum_s phi_s b_ij(s)).

With reveal_prob = 1 the planner is told every change (full observability -> a
dynamic MOOP). With reveal_prob < 1 a changed arc is only reported with that
probability; unreported arcs keep a Markov-predicted belief b <- b P until a truck
actually drives the arc (which reveals its true state). This is the minimal
POMDP-style observation model asked for in bai_toan_iii Sec. 4.8.

Arc time realized by a truck is fixed by the true state at its DEPARTURE time
(a truck already on a road finishes it at the old speed).

Node indexing follows src/common.py: 0 = start depot, 1..n = satellites,
n+1 = end depot (same location as 0). In .events.json files arcs are written with
u, v in 0..n; v == 0 means "back to the depot" and is mapped to column n+1.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

STATES = ("open", "degraded", "closed_detour")
STATE_INDEX = {"open": 0, "degraded": 1, "closed_detour": 2, "detour": 2, "closed": 2}
PHI = np.array([1.0, 2.5, 5.0])

#: Default per-epoch Markov transition matrix P(s' | s) used for belief prediction
#: of arcs the planner has no fresh observation of (rows: from, cols: to).
DEFAULT_TRANSITION = np.array([
    [0.92, 0.06, 0.02],
    [0.15, 0.75, 0.10],
    [0.05, 0.15, 0.80],
])


# ============================================================
# Scenario / event file
# ============================================================

@dataclass
class EdgeUpdate:
    u: int
    v: int
    state: int  # index into STATES


@dataclass
class EpochEvent:
    epoch_id: int
    time: float                      # trigger time (h)
    description: str = ""
    edge_updates: List[EdgeUpdate] = field(default_factory=list)
    demand_multipliers: Dict[int, float] = field(default_factory=dict)   # centroid (1-based) -> multiplier
    priority_updates: Dict[int, float] = field(default_factory=dict)     # centroid (1-based) -> new weight


@dataclass
class Scenario:
    name: str
    base_instance: str
    num_satellites: int
    drone_endurance_h: float
    initial_priorities: Dict[int, float]
    epochs: List[EpochEvent]
    transition: np.ndarray = field(default_factory=lambda: DEFAULT_TRANSITION.copy())
    reveal_prob: float = 1.0


def _parse_epoch(e: dict) -> EpochEvent:
    edges = []
    for x in e.get("edge_updates", []):
        st = x.get("state")
        if st is None:  # backward compatible with the guide's original format
            mult = float(x.get("travel_time_multiplier", 1.0))
            st = "open" if mult <= 1.0 else ("degraded" if mult < 4.0 else "closed_detour")
        edges.append(EdgeUpdate(int(x["u"]), int(x["v"]), STATE_INDEX[st]))
    return EpochEvent(
        epoch_id=int(e["epoch_id"]),
        time=float(e.get("trigger_time_h", 0.0)),
        description=e.get("description", ""),
        edge_updates=edges,
        demand_multipliers={int(k): float(v) for k, v in e.get("demand_multipliers", {}).items()},
        priority_updates={int(k): float(v) for k, v in e.get("priority_updates", {}).items()},
    )


def load_scenarios(path: str) -> Dict[str, Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, sc in raw.items():
        if name.startswith("_"):
            continue  # metadata block
        out[name] = Scenario(
            name=name,
            base_instance=sc["base_instance"],
            num_satellites=int(sc["num_satellites"]),
            drone_endurance_h=float(sc.get("drone_endurance_max_h", math.inf)),
            initial_priorities={int(k): float(v) for k, v in sc["initial_priorities"].items()},
            epochs=sorted((_parse_epoch(e) for e in sc["epochs"]), key=lambda e: e.time),
            transition=np.array(sc.get("markov_transition", DEFAULT_TRANSITION.tolist()), dtype=float),
            reveal_prob=float(sc.get("reveal_prob", 1.0)),
        )
    return out


# ============================================================
# Road network: true state + planner belief
# ============================================================

class RoadNetwork:
    def __init__(self, base_time: np.ndarray, transition: np.ndarray = DEFAULT_TRANSITION,
                 reveal_prob: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.t0 = np.asarray(base_time, dtype=float)
        self.N = self.t0.shape[0]
        self.n = self.N - 2
        self.P = np.asarray(transition, dtype=float)
        self.reveal_prob = float(reveal_prob)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.true_state = np.zeros((self.N, self.N), dtype=int)
        self.belief = np.zeros((self.N, self.N, 3))
        self.belief[:, :, 0] = 1.0
        # arcs whose current true state the planner knows exactly
        self.known = np.ones((self.N, self.N), dtype=bool)
        # (time, multiplier matrix) history of the true network, for departure-time lookups
        self.history: List[Tuple[float, np.ndarray]] = [(0.0, np.ones((self.N, self.N)))]

    # ---- index mapping for event files ----
    def _matrix_arcs(self, u: int, v: int) -> List[Tuple[int, int]]:
        """(u, v) from an events file -> matrix arcs. v == 0 means 'to depot' (column n+1);
        u == 0 is the start depot row."""
        col = self.n + 1 if v == 0 else v
        return [(u, col)]

    # ---- events ----
    def apply_edge_updates(self, time: float, updates: List[EdgeUpdate]) -> List[Tuple[int, int]]:
        """Apply true-state changes at `time`; return the matrix arcs that changed."""
        changed = []
        for up in updates:
            for (i, j) in self._matrix_arcs(up.u, up.v):
                if self.true_state[i, j] != up.state:
                    self.true_state[i, j] = up.state
                    changed.append((i, j))
        self.history.append((float(time), PHI[self.true_state].copy()))

        if self.reveal_prob >= 1.0:
            # full observability: belief == truth everywhere
            self.belief[:] = 0.0
            ii, jj = np.indices(self.true_state.shape)
            self.belief[ii, jj, self.true_state] = 1.0
            self.known[:] = True
            return changed

        # partial observability: unknown arcs drift by the Markov prediction ...
        self.known[:] = False
        unk = ~self.known
        self.belief[unk] = self.belief[unk] @ self.P
        # ... and each changed arc is reported with probability reveal_prob
        for (i, j) in changed:
            if self.rng.random() < self.reveal_prob:
                self._set_known(i, j)
        return changed

    def _set_known(self, i: int, j: int) -> None:
        self.belief[i, j] = 0.0
        self.belief[i, j, self.true_state[i, j]] = 1.0
        self.known[i, j] = True

    def observe_traversed(self, arcs: List[Tuple[int, int]]) -> None:
        """A truck that drove (i, j) reveals its current true state."""
        for (i, j) in arcs:
            self._set_known(i, j)

    # ---- times ----
    def expected_multiplier(self) -> np.ndarray:
        return self.belief @ PHI

    def expected_time(self) -> np.ndarray:
        return self.t0 * self.expected_multiplier()

    def true_time(self, i: int, j: int, t_depart: float) -> float:
        mult = self.history[0][1]
        for (t, m) in self.history:
            if t <= t_depart + 1e-12:
                mult = m
            else:
                break
        return float(self.t0[i, j] * mult[i, j])

    def uncertain_ratio(self) -> float:
        """Share of satellite/depot arcs whose state the planner is not sure about."""
        ent = -(self.belief * np.log(np.clip(self.belief, 1e-12, 1))).sum(axis=2)
        mask = ~np.eye(self.N, dtype=bool)
        return float((ent[mask] > 1e-6).mean())


# ============================================================
# Demand + priority process
# ============================================================

class DemandProcess:
    """Centroid demand d_j(t) and priority w_j(t), piecewise constant between events."""

    def __init__(self, base_demand: List[float], priorities: Dict[int, float]):
        self.n = len(base_demand)
        self.history: List[Tuple[float, np.ndarray, np.ndarray]] = []
        d = np.array(base_demand, dtype=float)
        w = np.array([priorities.get(j + 1, 1.0) for j in range(self.n)], dtype=float)
        self.history.append((0.0, d, w))

    def apply(self, time: float, demand_mult: Dict[int, float], prio: Dict[int, float]) -> None:
        _, d, w = self.history[-1]
        d, w = d.copy(), w.copy()
        for j, m in demand_mult.items():
            d[j - 1] = round(d[j - 1] * m)
        for j, p in prio.items():
            w[j - 1] = p
        self.history.append((float(time), d, w))

    def _at(self, t: float):
        cur = self.history[0]
        for h in self.history:
            if h[0] <= t + 1e-12:
                cur = h
            else:
                break
        return cur

    def demand_at(self, t: float) -> np.ndarray:
        return self._at(t)[1]

    def priority_at(self, t: float) -> np.ndarray:
        return self._at(t)[2]

    @property
    def current_demand(self) -> np.ndarray:
        return self.history[-1][1]

    @property
    def current_priority(self) -> np.ndarray:
        return self.history[-1][2]
