"""
DRL operator controller for the dynamic re-optimizer (the "DRL" of SR-DRL-DMOEA).

A Double DQN, written in plain numpy (no deep-learning framework needed), chooses at
every search step which neighbourhood operator to apply (spec eq. 11: operator
selection instead of drawing routes directly).

State s_t (DQNSelector.N_FEATURES = 23), built by DynamicReoptimizer.search_state():
  epoch context  (7): max / mean route disruption c_k(t), share of free satellites,
                      re-delivery jobs per free satellite, fleet load u_V, drone load u_D,
                      share of uncertain road arcs (POMDP belief entropy > 0)
  search state   (7): elapsed share of the time budget, steps since last improvement
                      (scaled), current solution infeasible, gap current-vs-best,
                      recent acceptance rate, recent improvement rate, archive fill
  last operator  (9): one-hot
Action a_t: one of the 9 operators in dynamic_solver.OPERATORS.
Reward r_t: per move +5 new best, +2 current improved, +1 new Pareto-archive member (the
            signal the adaptive roulette uses), summed over the segment and rescaled.
Decision rate: the agent picks an operator for a segment of `hold` = 10 moves (macro action),
            so network inference costs ~1/10 of a forward pass per move.

Training: experience replay, target network, Double-DQN target
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a)),
Huber loss, Adam. Each re-optimization call is one episode (done at the time limit).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dynamic_solver import OPERATORS


class _MLP:
    """Tiny fully-connected network with ReLU hidden layers and Adam."""

    def __init__(self, sizes, rng: np.random.Generator, lr: float = 1e-3):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), size=(a, b)))
            self.b.append(np.zeros(b))
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h, acts

    def __call__(self, x):
        return self.forward(x)[0]

    def step(self, x, grad_out):
        """Backprop grad_out (dL/dQ) through the net and apply one Adam step."""
        _, acts = self.forward(x)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g / len(x)
            gb[i] = g.mean(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self.W + self.b
        grads = gW + gb
        for k, (p, gr) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other: "_MLP"):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


class DQNSelector:
    name = "dqn"
    N_FEATURES = 23
    hold = 10          # moves per decision (macro action): keeps inference overhead ~1/10 per move

    def __init__(self, rng: np.random.Generator, hidden=(64, 64), gamma: float = 0.9,
                 lr: float = 1e-3, buffer: int = 50_000, batch: int = 64, train_every: int = 4,
                 target_sync: int = 1000, eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay_steps: int = 150_000, training: bool = True):
        self.rng = rng
        sizes = [self.N_FEATURES, *hidden, len(OPERATORS)]
        self.q = _MLP(sizes, rng, lr)
        self.q_target = _MLP(sizes, rng, lr)
        self.q_target.copy_from(self.q)
        self.gamma, self.batch, self.train_every, self.target_sync = gamma, batch, train_every, target_sync
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.training = training
        self.cap = buffer
        self.S = np.zeros((buffer, self.N_FEATURES))
        self.A = np.zeros(buffer, dtype=int)
        self.R = np.zeros(buffer)
        self.S2 = np.zeros((buffer, self.N_FEATURES))
        self.D = np.zeros(buffer)
        self.ptr = self.size = 0
        self.steps = 0
        self.idx = {op: i for i, op in enumerate(OPERATORS)}
        self._last_state: Optional[np.ndarray] = None
        self.losses = []

    # ---- policy ----
    @property
    def epsilon(self) -> float:
        if not self.training:
            return self.eps_end
        f = min(1.0, self.steps / max(1, self.eps_decay))
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def select(self, state=None) -> str:
        self._last_state = None if state is None else np.asarray(state, dtype=float)
        if state is None or self.rng.random() < self.epsilon:
            return OPERATORS[int(self.rng.integers(len(OPERATORS)))]
        qv = self.q(self._last_state[None, :])[0]
        return OPERATORS[int(np.argmax(qv))]

    # ---- learning ----
    def update(self, op: str, reward: float, next_state=None, done: bool = False) -> None:
        self.steps += 1
        if not self.training or self._last_state is None or next_state is None:
            return
        i = self.ptr
        # segment reward (sum over `hold` moves of 5/2/1 signals) rescaled to ~[0, 1+]
        self.S[i], self.A[i], self.R[i] = self._last_state, self.idx[op], reward / (5.0 * self.hold) * 3.0
        self.S2[i], self.D[i] = next_state, float(done)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        if self.size >= 1000 and self.steps % self.train_every == 0:
            self._train()
        if self.steps % self.target_sync == 0:
            self.q_target.copy_from(self.q)

    def _train(self):
        b = self.rng.integers(0, self.size, size=self.batch)
        s, a, r, s2, d = self.S[b], self.A[b], self.R[b], self.S2[b], self.D[b]
        a2 = np.argmax(self.q(s2), axis=1)                                 # Double DQN
        y = r + self.gamma * (1.0 - d) * self.q_target(s2)[np.arange(len(b)), a2]
        qs = self.q(s)
        err = qs[np.arange(len(b)), a] - y
        g = np.zeros_like(qs)
        g[np.arange(len(b)), a] = np.clip(err, -1.0, 1.0)                  # Huber gradient
        self.q.step(s, g)
        self.losses.append(float(np.mean(np.minimum(err ** 2, np.abs(err)))))

    def new_epoch(self, state=None) -> None:
        pass

    def end_episode(self) -> None:
        self._last_state = None

    # ---- persistence ----
    def save(self, path: str) -> None:
        arrs = {f"W{i}": w for i, w in enumerate(self.q.W)}
        arrs.update({f"b{i}": b for i, b in enumerate(self.q.b)})
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path: str, rng: np.random.Generator, training: bool = False) -> "DQNSelector":
        z = np.load(path)
        sel = cls(rng, training=training)
        n = len(sel.q.W)
        sel.q.W = [z[f"W{i}"] for i in range(n)]
        sel.q.b = [z[f"b{i}"] for i in range(n)]
        sel.q_target.copy_from(sel.q)
        return sel
