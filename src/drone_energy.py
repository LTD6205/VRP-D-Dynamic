"""
Drone battery model: energy of an out-and-back sortie as a function of flight
SPEED and carried LOAD (simplified from the proposal's slide 12 power model).

Power drawn while flying, expressed as "fraction of a full battery per hour":

    p(v, l) = a * (1 + beta * l / P)  +  b * (v / v_ref)^3  +  k_w * wind^2
              \\_____ weight term ____/   \\__ speed term __/   \\ wind (opt.) /

with  a = weight_share / t_empty_ref,  b = (1 - weight_share) / t_empty_ref,
so that an EMPTY drone at the reference speed v_ref hovers/flies t_empty_ref
hours on a full battery. l / P in [0, 1] is the payload as a share of the drone's
capacity P.

A sortie from satellite i to centroid j flies out loaded and back empty, each leg
lasting t'_ij = dist_ij / v_d:

    e_ij = t'_ij * [ p(v_d, l_ij) + p(v_d, 0) ]

and is battery-feasible iff   e_ij <= battery * (1 - reserve).

The per-drone load of a wave is l = d_j / (U * waves_j), waves_j = ceil(d_j / (U P)):
demand is split evenly over the U parallel drones of each wave.

Consequences (default parameters, v_ref = 60 km/h):
  * heavier load  -> more power -> shorter feasible distance;
  * the speed term grows with v^3 while flight time falls with 1/v, so the
    maximum range has an optimum around v_ref and drops at both 40 and 80 km/h
    (unlike a fixed time budget, where a faster drone always flies further);
  * a demand surge raises the load and can make a previously feasible
    satellite -> centroid pair infeasible, forcing a re-assignment.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DroneEnergyModel:
    v_ref: float = 60.0          # km/h, speed of the calibration point
    t_empty_ref: float = 0.65    # h (39 min), full-battery flight time of an empty drone at v_ref
    weight_share: float = 0.6    # share of power due to (tare) weight at v_ref, empty
    beta_payload: float = 0.8    # +80% weight-term power at full payload
    reserve: float = 0.15        # safety reserve left in the battery on landing
    battery: float = 1.0         # usable battery capacity (1 = nominal battery)
    wind: float = 0.0            # wind speed (m/s), optional
    k_wind: float = 0.0          # power per (m/s)^2 of wind, fraction of battery / h

    @classmethod
    def from_dict(cls, d: dict | None) -> "DroneEnergyModel":
        d = dict(d or {})
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- physics ----
    def power(self, v: float, load_frac) -> np.ndarray:
        a = self.weight_share / self.t_empty_ref
        b = (1.0 - self.weight_share) / self.t_empty_ref
        return a * (1.0 + self.beta_payload * np.asarray(load_frac)) + b * (v / self.v_ref) ** 3 \
            + self.k_wind * self.wind ** 2

    def sortie_energy(self, t_oneway, v: float, load_frac) -> np.ndarray:
        """Battery share used by one out-(loaded)-and-back-(empty) sortie."""
        return np.asarray(t_oneway) * (self.power(v, load_frac) + self.power(v, 0.0))

    @property
    def usable(self) -> float:
        return self.battery * (1.0 - self.reserve)

    def max_oneway_time(self, v: float, load_frac: float) -> float:
        return float(self.usable / (self.power(v, load_frac) + self.power(v, 0.0)))

    def max_range_km(self, v: float, load_frac: float) -> float:
        return v * self.max_oneway_time(v, load_frac)


class DroneBattery:
    """DroneEnergyModel bound to one instance's drone speed, fleet U and payload P."""

    def __init__(self, model: DroneEnergyModel, v_d: float, U: int, P: float):
        self.model, self.v_d, self.U, self.P = model, float(v_d), int(U), float(P)

    def waves(self, d) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        return np.maximum(1, np.ceil(d / (self.U * self.P) - 1e-9))

    def load_frac(self, d) -> np.ndarray:
        """Per-drone payload share l/P in one wave serving demand d."""
        d = np.asarray(d, dtype=float)
        return np.clip(d / (self.U * self.waves(d) * self.P), 0.0, 1.0)

    def energy_matrix(self, tdm: np.ndarray, demand) -> np.ndarray:
        """e[s, c]: battery share of a sortie from satellite s to centroid c (demand of c)."""
        return self.model.sortie_energy(tdm, self.v_d, self.load_frac(demand)[None, :])

    def pair_ok(self, tdm: np.ndarray, demand) -> np.ndarray:
        return self.energy_matrix(tdm, demand) <= self.model.usable + 1e-12


class FixedEndurance:
    """Legacy fixed time budget 2 t' <= T_max (used only when T_max is given as a number)."""

    def __init__(self, t_max: float):
        self.t_max = t_max

    def pair_ok(self, tdm: np.ndarray, demand) -> np.ndarray:
        return (2 * tdm) <= self.t_max + 1e-9

    def energy_matrix(self, tdm: np.ndarray, demand) -> np.ndarray:
        return 2 * tdm / self.t_max if math.isfinite(self.t_max) else np.zeros_like(tdm)
