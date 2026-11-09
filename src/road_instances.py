"""
Road-network instances for the dynamic truck-multi-drone relief problem
(proposal Section 7): Solomon (1987) customers on a generated road graph with
flood zones and roads whose states change in continuous time.

Pipeline for one instance
  1. nodes      depot + the first n Solomon customers, 0.4 km per Solomon unit;
                demand -> 1..3 parcels by tercile of the instance's demands
  2. roads      Delaunay triangulation (or symmetric k nearest neighbours); very long
                hull edges are pruned while the graph stays connected and every node
                keeps >= 2 roads; length = Euclidean x detour factor U[1.3, 1.5]
  3. zones      m flood discs centred on random customers, common radius set by
                bisection so that a target share of victims lies inside; the depot
                is never inside a zone
  4. states     each flood zone has a water level (low / medium / high) that follows a
                continuous-time Markov chain whose generator is the matrix logarithm
                of the hourly matrices of the main paper (low / medium / high
                intensity), so exp(60 Q) reproduces them exactly. A road that touches
                a zone gets a depth in [0, 1] (1 = through the centre); its state
                (open / slow / closed) follows from the zone's level and its depth
                (LEVEL_RULE). Flooding is therefore spatially correlated: when the
                water rises, the inner roads close together and the victims there
                lose every road at once. Roads outside the zones stay open.
                road_model="independent" instead gives every zone road its own
                chain (closures are then uncorrelated and rarely isolate a victim).
                A share of zone victims are drone-only for good (rooftops, islands):
                their roads are closed and never reopen.
  5. check      every victim that can ever lose its last road must lie within the
                single-parcel drone radius R1 of a node that dry roads always reach;
                otherwise the zones are redrawn
  6. victims    priority class (critical / high / normal, 20 / 30 / 50 %), reveal
                times (Poisson with degree of dynamism delta, or Solomon ready times),
                soft due date = reveal + dd(class) x estimated service time

Closed roads are impassable here (unlike the main paper, where they were x5 slow):
a victim with no open or slow road left is drone-only until a road reopens.

A *scenario* (one seed) fixes the reveal times and the full road-state history,
so every method faces the same realisation.
"""
from __future__ import annotations

import heapq
import zlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import logm
from scipy.spatial import Delaunay

# ----------------------------------------------------------------------------- constants

KM_PER_UNIT = 0.4
TRUCK_KMH = 40.0
DRONE_KMH = 60.0
OPEN, SLOW, CLOSED = 0, 1, 2
STATE_NAMES = ("open", "slow", "closed")
SLOW_FACTOR = 2.5                      # travel-time multiplier on a slow road
R1_KM = 16.3                           # single-parcel drone radius (main paper, 20% reserve)

# hourly matrices of the main paper (rows: from open / damaged / closed)
HOURLY = {
    "low":    [[0.95, 0.04, 0.01], [0.30, 0.66, 0.04], [0.13, 0.13, 0.74]],
    "medium": [[0.88, 0.08, 0.04], [0.28, 0.63, 0.09], [0.17, 0.13, 0.70]],
    "high":   [[0.80, 0.11, 0.09], [0.23, 0.59, 0.18], [0.12, 0.13, 0.75]],
}

CLASSES = ("critical", "high", "normal")
CLASS_SHARE = (0.2, 0.3, 0.5)
CLASS_WEIGHT = {"critical": 5, "high": 2, "normal": 1}
DUE_FACTOR = {"critical": 1.5, "high": 3.0, "normal": 6.0}   # dd(class), DFJSS-style
SERVICE_MIN = 5.0                                          # hand-over time at a victim

# zone level -> (depth above which a road is closed, depth above which it is slow)
LEVEL_RULE = {0: (math.inf, 0.5),      # low water: only the inner roads are slow
              1: (0.5, 0.0),           # medium: inner roads closed, the rest slow
              2: (0.1, 0.0)}           # high: almost every zone road closed

GEOMETRIES = {"C1": "c101.txt", "C2": "c201.txt", "R": "r101.txt", "RC": "rc101.txt"}
DEFAULT_TRUCKS = {25: 2, 50: 2, 100: 3}


# ----------------------------------------------------------------------------- generator matrix

def generator_matrix(intensity: str) -> np.ndarray:
    """Continuous-time generator Q (per minute) with exp(60 Q) ~ the hourly matrix.

    logm can give small negative off-diagonal entries (the embedding problem);
    those are clipped to 0 and the diagonal is reset so rows sum to zero.
    """
    P = np.array(HOURLY[intensity], dtype=float)
    Q = np.real(logm(P)) / 60.0
    off = ~np.eye(3, dtype=bool)
    Q[off] = np.clip(Q[off], 0.0, None)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def stationary(Q: np.ndarray) -> np.ndarray:
    A = np.vstack([Q.T, np.ones(3)])
    b = np.array([0.0, 0.0, 0.0, 1.0])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


# ----------------------------------------------------------------------------- data classes

@dataclass
class Params:
    geometry: str = "R"
    n: int = 50
    graph: str = "delaunay"            # "delaunay" | "knn"
    k: int = 5                         # for knn
    prune_factor: float = 2.5          # prune edges longer than this x median (Delaunay)
    detour: tuple = (1.3, 1.5)
    zones: int | None = None           # default: 1 / 2 / 3 for n = 25 / 50 / 100
    zone_share: float = 0.2            # target share of victims inside a flood zone
    permanent_share: float = 0.2       # share of zone victims drone-only for good
    intensity: str = "medium"
    road_model: str = "zone"           # "zone" (correlated water level) | "independent"
    trucks: int | None = None          # K, default 2 / 2 / 3
    drones: int = 2                    # N per truck
    delta: float = 0.5                 # degree of dynamism: share revealed after t = 0
    utilisation: float = 0.8
    reveal: str = "poisson"            # "poisson" | "solomon"
    horizon_min: float = 720.0         # road histories are sampled over [0, horizon]
    seed: int = 1


@dataclass
class Instance:
    name: str
    params: dict
    depot: int
    xy_km: list                        # node coordinates (km), node 0 = depot
    parcels: list                      # per node, depot 0
    ready_time: list                   # Solomon ready times (min), depot 0
    edges: list                        # [u, v, length_km, zone_road (0/1), permanent (0/1)]
    edge_depth: list                   # per edge, depth in each zone (-1 = not in that zone)
    zones: list                        # [cx, cy, r] in km
    in_zone: list                      # per node 0/1
    permanent_drone_only: list         # node ids
    anchors: list                      # nodes always reachable on dry roads
    anchor_dist_km: list               # per node, distance to nearest anchor
    generator: list                    # Q (per minute)
    feasible: bool
    tries: int
    victims: list = field(default_factory=list)   # filled per scenario


# ----------------------------------------------------------------------------- geometry helpers

def read_solomon(path: Path, n: int):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 7 and parts[0].isdigit():
                rows.append([float(p) for p in parts])
    rows = np.array(rows)[: n + 1]
    return rows[:, 1:3], rows[:, 3], rows[:, 4]


def seg_point_dist(p, a, b) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(a + t * ab - p))


def connected(n_nodes: int, edges) -> bool:
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n_nodes


def build_roads(xy: np.ndarray, p: Params, rng) -> list[tuple[int, int]]:
    n_nodes = len(xy)
    if p.graph == "knn":
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        E = set()
        for i in range(n_nodes):
            for j in np.argsort(d[i])[1 : p.k + 1]:
                E.add((min(i, int(j)), max(i, int(j))))
        E = sorted(E)
        if not connected(n_nodes, E):          # join components through nearest pairs
            E = _connect(xy, E)
        return E
    jitter = rng.normal(0, 1e-6, xy.shape)     # Solomon has collinear points
    tri = Delaunay(xy + jitter)
    E = set()
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
            E.add((int(min(a, b)), int(max(a, b))))
    E = sorted(E)
    L = {e: float(np.linalg.norm(xy[e[0]] - xy[e[1]])) for e in E}
    cut = p.prune_factor * float(np.median(list(L.values())))
    deg = np.zeros(n_nodes, int)
    for u, v in E:
        deg[u] += 1
        deg[v] += 1
    for e in sorted(E, key=lambda e: -L[e]):
        if L[e] <= cut:
            break
        u, v = e
        if deg[u] <= 2 or deg[v] <= 2:
            continue
        trial = [x for x in E if x != e]
        if connected(n_nodes, trial):
            E = trial
            deg[u] -= 1
            deg[v] -= 1
    return E


def _connect(xy, E):
    E = list(E)
    n_nodes = len(xy)
    while not connected(n_nodes, E):
        comp = _component(n_nodes, E)
        inside = [i for i in range(n_nodes) if comp[i]]
        outside = [i for i in range(n_nodes) if not comp[i]]
        d = np.linalg.norm(xy[inside][:, None] - xy[outside][None], axis=2)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        u, v = inside[a], outside[b]
        E.append((min(u, v), max(u, v)))
    return sorted(set(E))


def _component(n_nodes, E):
    adj = [[] for _ in range(n_nodes)]
    for u, v in E:
        adj[u].append(v)
        adj[v].append(u)
    comp = [False] * n_nodes
    comp[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not comp[v]:
                comp[v] = True
                stack.append(v)
    return comp


def place_zones(xy, m, share, rng):
    """m discs centred on random customers, common radius by bisection on the share inside."""
    n_cust = len(xy) - 1
    centers = xy[1 + rng.choice(n_cust, size=m, replace=False)]

    def inside(r):
        d = np.min(np.linalg.norm(xy[1:, None] - centers[None], axis=2), axis=1)
        return float(np.mean(d <= r))

    lo, hi = 0.0, float(np.ptp(xy, axis=0).max())
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inside(mid) < share else (lo, mid)
    r = hi
    if np.min(np.linalg.norm(centers - xy[0], axis=1)) <= r:
        return None                                     # depot would be flooded
    return [(float(c[0]), float(c[1]), r) for c in centers]


def dijkstra(n_nodes, adj, src, weight):
    dist = [math.inf] * n_nodes
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, ei in adj[u]:
            w = weight(ei)
            if w is None:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ----------------------------------------------------------------------------- instance

def build_instance(p: Params, solomon_dir: Path, max_tries: int = 50) -> Instance:
    rng = np.random.default_rng(p.seed)
    xy_u, demand, ready = read_solomon(solomon_dir / GEOMETRIES[p.geometry], p.n)
    xy = xy_u * KM_PER_UNIT
    n_nodes = len(xy)
    q = np.quantile(demand[1:], [1 / 3, 2 / 3])
    parcels = [0] + [int(1 + (d > q[0]) + (d > q[1])) for d in demand[1:]]

    E = build_roads(xy, p, rng)
    length = [float(np.linalg.norm(xy[u] - xy[v]) * rng.uniform(*p.detour)) for u, v in E]
    m = p.zones or {25: 1, 50: 2, 100: 3}.get(p.n, max(1, p.n // 33))

    for tries in range(1, max_tries + 1):
        zones = place_zones(xy, m, p.zone_share, rng)
        if zones is None:
            continue
        C = np.array([z[:2] for z in zones])
        r = zones[0][2]
        in_zone = [0] + [int(np.min(np.linalg.norm(C - xy[i], axis=1)) <= r) for i in range(1, n_nodes)]
        depth = [[round(1 - seg_point_dist(c, xy[u], xy[v]) / r, 4)
                  if seg_point_dist(c, xy[u], xy[v]) <= r else -1.0 for c in C] for u, v in E]
        zone_road = [int(max(d) >= 0) for d in depth]
        zone_vics = [i for i in range(1, n_nodes) if in_zone[i]]
        k_perm = int(round(p.permanent_share * len(zone_vics)))
        perm = []
        for x in rng.permutation(zone_vics):              # a rooftop must not cut others off
            if len(perm) == k_perm:
                break
            cut = set(perm) | {int(x)}
            rest = [i for i in range(n_nodes) if i not in cut]
            idx = {v: j for j, v in enumerate(rest)}
            sub = [(idx[u], idx[v]) for u, v in E if u not in cut and v not in cut]
            if connected(len(rest), sub):
                perm.append(int(x))
        perm.sort()
        permanent = [int(u in perm or v in perm) for u, v in E]

        adj = [[] for _ in range(n_nodes)]
        for ei, (u, v) in enumerate(E):
            adj[u].append((v, ei))
            adj[v].append((u, ei))
        dry = dijkstra(n_nodes, adj, 0, lambda ei: None if zone_road[ei] else length[ei])
        anchors = [i for i in range(n_nodes) if dry[i] < math.inf]
        A = xy[anchors]
        anchor_d = [float(np.min(np.linalg.norm(A - xy[i], axis=1))) for i in range(n_nodes)]
        if max(anchor_d) <= R1_KM:
            break
    else:
        tries = max_tries

    name = f"RD-{p.geometry}-{p.n:03d}-{p.intensity[0].upper()}-s{p.seed}"
    return Instance(
        name=name, params=asdict(p), depot=0,
        xy_km=xy.round(4).tolist(), parcels=parcels, ready_time=ready.tolist(),
        edges=[[u, v, round(L, 4), zr, pm] for (u, v), L, zr, pm in zip(E, length, zone_road, permanent)],
        edge_depth=depth,
        zones=[[round(a, 4), round(b, 4), round(c, 4)] for a, b, c in zones],
        in_zone=in_zone, permanent_drone_only=perm, anchors=anchors,
        anchor_dist_km=[round(d, 4) for d in anchor_d],
        generator=generator_matrix(p.intensity).round(8).tolist(),
        feasible=bool(max(anchor_d) <= R1_KM), tries=tries,
    )


# ----------------------------------------------------------------------------- scenario

def _gillespie(Q, s, horizon, rng):
    """One CTMC path from state s: list of (time, new state)."""
    out, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / -Q[s, s])
        if t >= horizon:
            return out
        w = Q[s].copy()
        w[s] = 0.0
        s = int(rng.choice(3, p=w / w.sum()))
        out.append((t, s))


def road_state(depths, levels) -> int:
    """State of a zone road from the levels of the zones it touches (worst one wins)."""
    st = OPEN
    for d, L in zip(depths, levels):
        if d < 0:
            continue
        closed_above, slow_above = LEVEL_RULE[L]
        st = max(st, CLOSED if d > closed_above else SLOW if d > slow_above else OPEN)
    return st


def sample_road_history(inst: Instance, horizon: float, rng):
    """Sample the road-state history. Returns (initial road states, road events,
    zone-level events); road events are (time, edge, new state)."""
    Q = np.array(inst.generator)
    pi = stationary(Q)
    model = inst.params.get("road_model", "zone")
    E = inst.edges
    if model == "independent":
        init, events = [], []
        for ei, (_, _, _, zr, pm) in enumerate(E):
            if pm or not zr:
                init.append(CLOSED if pm else OPEN)
                continue
            s = int(rng.choice(3, p=pi))
            init.append(s)
            events += [(round(t, 3), ei, ns) for t, ns in _gillespie(Q, s, horizon, rng)]
        events.sort()
        return init, events, []

    m = len(inst.zones)
    levels = [int(rng.choice(3, p=pi)) for _ in range(m)]
    zone_ev = sorted((round(t, 3), z, L) for z in range(m)
                     for t, L in _gillespie(Q, levels[z], horizon, rng))

    def states_for(lv):
        return [CLOSED if e[4] else (road_state(inst.edge_depth[i], lv) if e[3] else OPEN)
                for i, e in enumerate(E)]

    cur = states_for(levels)
    init, events = list(cur), []
    lv = list(levels)
    for t, z, L in zone_ev:
        lv[z] = L
        new = states_for(lv)
        events += [(t, i, s) for i, (a, s) in enumerate(zip(cur, new)) if a != s]
        cur = new
    return init, events, [[t, z, L] for t, z, L in zone_ev]


def truck_times_from_depot(inst: Instance, states) -> list[float]:
    n_nodes = len(inst.xy_km)
    adj = [[] for _ in range(n_nodes)]
    for ei, (u, v, *_r) in enumerate(inst.edges):
        adj[u].append((v, ei))
        adj[v].append((u, ei))

    def w(ei):
        s = states[ei]
        if s == CLOSED:
            return None
        return inst.edges[ei][2] / TRUCK_KMH * 60.0 * (SLOW_FACTOR if s == SLOW else 1.0)

    return dijkstra(n_nodes, adj, 0, w)


def estimated_service(inst: Instance, states) -> list[float]:
    """p_hat: truck time from the depot if reachable, else truck time to the nearest
    reachable node + a single drone out-and-back; plus the hand-over time."""
    tt = truck_times_from_depot(inst, states)
    xy = np.array(inst.xy_km)
    reach = [i for i, t in enumerate(tt) if t < math.inf]
    out = [0.0]
    for i in range(1, len(xy)):
        if tt[i] < math.inf:
            out.append(tt[i] + SERVICE_MIN)
            continue
        best = min(tt[j] + 2 * np.linalg.norm(xy[i] - xy[j]) / DRONE_KMH * 60 for j in reach)
        out.append(float(best) + SERVICE_MIN)
    return out


def sample_scenario(inst: Instance, seed: int) -> dict:
    p = Params(**{k: (tuple(v) if k == "detour" else v) for k, v in inst.params.items()})
    rng = np.random.default_rng([seed, zlib.crc32(inst.name.encode())])
    init, events, zone_events = sample_road_history(inst, p.horizon_min, rng)
    p_hat = estimated_service(inst, init)
    n = len(inst.xy_km) - 1
    K = p.trucks or DEFAULT_TRUCKS.get(p.n, 3)

    # reveal times
    if p.reveal == "solomon":
        horizon_ready = max(inst.ready_time) or 1.0
        reveal = [0.0] + [480.0 * r / horizon_ready for r in inst.ready_time[1:]]
    else:
        known = rng.random(n) >= p.delta
        channels = K * (1 + p.drones)
        rate = p.utilisation * channels / float(np.mean(p_hat[1:]))    # victims per minute
        late = int((~known).sum())
        gaps = rng.exponential(1.0 / rate, size=late)
        times = iter(np.cumsum(gaps))
        order = rng.permutation(n)
        reveal = [0.0] * (n + 1)
        for i in order:
            reveal[i + 1] = 0.0 if known[i] else float(next(times))

    classes = rng.choice(len(CLASSES), size=n, p=CLASS_SHARE)
    victims = []
    for i in range(1, n + 1):
        c = CLASSES[classes[i - 1]]
        victims.append({
            "id": i, "parcels": inst.parcels[i], "class": c, "weight": CLASS_WEIGHT[c],
            "reveal": round(reveal[i], 3), "p_hat": round(p_hat[i], 3),
            "due": round(reveal[i] + DUE_FACTOR[c] * p_hat[i], 3),
            "permanent_drone_only": int(i in inst.permanent_drone_only),
        })
    return {"instance": inst.name, "seed": seed, "trucks": K, "drones_per_truck": p.drones,
            "initial_road_states": init, "road_events": events, "zone_level_events": zone_events,
            "victims": victims}


# ----------------------------------------------------------------------------- statistics

def drone_only_profile(inst: Instance, scen: dict, step: float = 10.0):
    """Share of victims with no open or slow road path from the depot, every `step` minutes."""
    states = list(scen["initial_road_states"])
    ev = scen["road_events"]
    horizon = inst.params["horizon_min"]
    n = len(inst.xy_km) - 1
    shares, ever = [], set()
    k = 0
    t = 0.0
    while t <= horizon:
        while k < len(ev) and ev[k][0] <= t:
            states[ev[k][1]] = ev[k][2]
            k += 1
        tt = truck_times_from_depot(inst, states)
        cut = [i for i in range(1, n + 1) if tt[i] == math.inf]
        ever.update(cut)
        shares.append(len(cut) / n)
        t += step
    return shares, ever


def summarise(inst: Instance, scen: dict) -> dict:
    shares, ever = drone_only_profile(inst, scen)
    n = len(inst.xy_km) - 1
    deg = np.zeros(n + 1, int)
    for u, v, *_ in inst.edges:
        deg[u] += 1
        deg[v] += 1
    vic = scen["victims"]
    return {
        "instance": inst.name, "seed": scen["seed"], "n": n,
        "roads": len(inst.edges), "mean_degree": round(float(deg.mean()), 2),
        "zone_roads_pct": round(100 * np.mean([e[3] for e in inst.edges]), 1),
        "victims_in_zone_pct": round(100 * np.mean(inst.in_zone[1:]), 1),
        "permanent_drone_only": len(inst.permanent_drone_only),
        "drone_only_t0_pct": round(100 * shares[0], 1),
        "drone_only_mean_pct": round(100 * float(np.mean(shares)), 1),
        "drone_only_max_pct": round(100 * float(np.max(shares)), 1),
        "ever_drone_only_pct": round(100 * len(ever) / n, 1),
        "road_events": len(scen["road_events"]),
        "zone_level_changes": len(scen.get("zone_level_events", [])),
        "max_anchor_km": round(max(inst.anchor_dist_km), 2),
        "feasible": inst.feasible,
        "revealed_at_0_pct": round(100 * np.mean([v["reveal"] == 0 for v in vic]), 1),
        "last_reveal_min": round(max(v["reveal"] for v in vic), 1),
        "mean_p_hat_min": round(float(np.mean([v["p_hat"] for v in vic])), 1),
    }


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj if isinstance(obj, dict) else asdict(obj), f, separators=(",", ":"))
