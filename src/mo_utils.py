"""Small multi-objective helpers: dominance, a bounded Pareto archive, and exact
hypervolume for 2-3 objectives (minimization). No external MO library needed."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    better = False
    for x, y in zip(a, b):
        if x > y + EPS:
            return False
        if x < y - EPS:
            better = True
    return better


class ParetoArchive:
    """Non-dominated set of (objectives, payload). When full, the most crowded
    member (smallest crowding distance) is dropped to keep the front spread."""

    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        self.items: List[Tuple[Tuple[float, ...], Any]] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, objs: Sequence[float], payload: Any) -> bool:
        objs = tuple(float(x) for x in objs)
        for o, _ in self.items:
            if dominates(o, objs) or all(abs(a - b) <= 1e-9 for a, b in zip(o, objs)):
                return False
        self.items = [(o, p) for (o, p) in self.items if not dominates(objs, o)]
        self.items.append((objs, payload))
        if len(self.items) > self.max_size:
            self._truncate()
        return True

    def _truncate(self) -> None:
        F = np.array([o for o, _ in self.items])
        cd = crowding_distance(F)
        drop = int(np.argmin(cd))
        del self.items[drop]

    def objectives(self) -> np.ndarray:
        return np.array([o for o, _ in self.items]) if self.items else np.zeros((0, 3))

    def best_by_weights(self, w: Sequence[float], scale: Optional[Sequence[float]] = None):
        """Archive member minimizing the weighted sum of range-normalized objectives."""
        if not self.items:
            return None
        F = self.objectives()
        lo, hi = F.min(axis=0), F.max(axis=0)
        if scale is None:
            rng = np.where(hi - lo > EPS, hi - lo, 1.0)
            G = (F - lo) / rng
        else:
            G = F / np.asarray(scale)
        idx = int(np.argmin(G @ np.asarray(w)))
        return self.items[idx]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    cd = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k])
        span = F[order[-1], k] - F[order[0], k]
        cd[order[0]] = cd[order[-1]] = np.inf
        if span <= EPS:
            continue
        cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return cd


def nondominated(F: np.ndarray) -> np.ndarray:
    keep = []
    for i in range(len(F)):
        if not any(dominates(F[j], F[i]) for j in range(len(F)) if j != i):
            keep.append(i)
    return F[keep]


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hypervolume(F: np.ndarray, ref: Sequence[float]) -> float:
    """Exact HV (minimization) for 2 or 3 objectives by slicing on the last axis."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return _hv2d(F, ref)
    F = F[np.argsort(F[:, 2])]
    hv = 0.0
    for i in range(len(F)):
        z_lo = F[i, 2]
        z_hi = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        if z_hi - z_lo <= 0:
            continue
        hv += _hv2d(F[: i + 1, :2], ref[:2]) * (z_hi - z_lo)
    return hv


def simplex_weights(m: int = 3, h: int = 3) -> List[Tuple[float, ...]]:
    """Das-Dennis lattice of weight vectors (m objectives, h divisions)."""
    out = []

    def rec(prefix, left, depth):
        if depth == m - 1:
            out.append(tuple(prefix + [left / h]))
            return
        for i in range(left + 1):
            rec(prefix + [i / h], left - i, depth + 1)

    rec([], h, 0)
    return out
