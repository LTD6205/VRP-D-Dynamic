#!/usr/bin/env python3
"""
Train the Double-DQN operator controller (src/drl_selector.py) on the TRAINING event set
(instances/dynamic_events_train, instances disjoint from the 12-instance test benchmark).

Each training episode = one closed-loop run (warm strategy) on a random training instance /
scenario / seed / time budget; every re-optimization call inside it is one RL episode.

    python3 scripts/train_drl_selector.py --minutes 25 --output models/dqn_selector.npz
"""
import argparse
import glob
import os
import sys
import time
from dataclasses import replace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from drl_selector import DQNSelector  # noqa: E402
from dynamic_env import load_scenarios  # noqa: E402
from dynamic_runner import load_world, run_closed_loop  # noqa: E402
from dynamic_solver import ReoptConfig  # noqa: E402


def find_instance(name):
    for sub in ("main", "large"):
        p = os.path.join(ROOT, "instances", sub, f"{name}.txt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", default="instances/dynamic_events_train")
    ap.add_argument("--minutes", type=float, default=25.0, help="wall-clock training budget")
    ap.add_argument("--eps-decay-steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="models/dqn_selector.npz")
    args = ap.parse_args()
    os.chdir(ROOT)

    rng = np.random.default_rng(args.seed)
    dqn = DQNSelector(np.random.default_rng(args.seed + 1), eps_decay_steps=args.eps_decay_steps, training=True)
    files = sorted(glob.glob(os.path.join(args.events_dir, "*.events.json")))
    worlds = {}
    t_end = time.time() + 60 * args.minutes
    ep = 0
    while time.time() < t_end:
        f = files[int(rng.integers(len(files)))]
        name = os.path.basename(f).replace(".events.json", "")
        scs = [k for k in load_scenarios(f)]
        sc = scs[int(rng.integers(len(scs)))]
        key = (name, sc)
        if key not in worlds:
            worlds[key] = load_world(find_instance(name), f, sc)
        t = float(rng.choice([0.25, 0.5, 1.0]))
        base = ReoptConfig(mode="so", weights=(0, 0, 1))
        run_closed_loop(worlds[key], ["warm"], replace(base, time_limit=2 * t), replace(base, time_limit=t),
                        seed=int(rng.integers(1_000_000)), selector=dqn)
        ep += 1
        if ep % 10 == 0:
            loss = np.mean(dqn.losses[-500:]) if dqn.losses else float("nan")
            print(f"[{ep}] steps={dqn.steps} eps={dqn.epsilon:.3f} buffer={dqn.size} loss={loss:.4f}", flush=True)
            dqn.save(args.output)
    dqn.save(args.output)
    print(f"done: {ep} closed-loop runs, {dqn.steps} decisions; saved {args.output}")


if __name__ == "__main__":
    main()
