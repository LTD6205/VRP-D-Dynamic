import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import road_instances as ri  # noqa: E402

RAW = ROOT / "instances/solomon/raw"


def test_generator_reproduces_hourly_matrices():
    for k, P in ri.HOURLY.items():
        Q = ri.generator_matrix(k)
        assert np.allclose(Q.sum(axis=1), 0, atol=1e-12)
        assert (Q[~np.eye(3, dtype=bool)] >= 0).all()
        assert np.allclose(expm(60 * Q), P, atol=5e-3)


def test_instance_and_scenario_are_consistent():
    p = ri.Params(geometry="RC", n=50, intensity="high", seed=3)
    inst = ri.build_instance(p, RAW)
    assert inst.feasible and max(inst.anchor_dist_km) <= ri.R1_KM
    assert 0 not in inst.anchors[1:] and inst.anchors[0] == 0
    assert inst.in_zone[0] == 0                                   # depot is dry
    assert abs(np.mean(inst.in_zone[1:]) - p.zone_share) <= 0.05
    # rooftop victims never cut anyone else off
    scen = ri.sample_scenario(inst, 1)
    states = [ri.CLOSED if e[4] else ri.OPEN for e in inst.edges]
    tt = ri.truck_times_from_depot(inst, states)
    cut = {i for i, t in enumerate(tt) if t == math.inf}
    assert cut == set(inst.permanent_drone_only)
    # road events replay to valid states, permanent roads never reopen
    st = list(scen["initial_road_states"])
    for t, ei, s in scen["road_events"]:
        assert 0 <= t < p.horizon_min and s in (0, 1, 2)
        assert not inst.edges[ei][4]
        st[ei] = s
    for v in scen["victims"]:
        assert v["due"] > v["reveal"] >= 0 and 1 <= v["parcels"] <= 3
    # same seed -> same scenario
    assert ri.sample_scenario(inst, 1) == scen
