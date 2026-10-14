# Instances

`main/` — 90 instances used in the paper's main experiments (Table 8 and appendix): nine
satellite counts n ∈ {3, 4, 5, 7, 8, 9, 11, 13, 15}, two truck fleets, five drone speeds.

`large/` — 20 larger instances (n = 20 and n = 50) used for the scalability results
(Table 11 and appendix), same fleet/speed structure.

## Naming

`M-<n_c>-<n>-<v_d>-<f>.txt`, e.g. `M-15-13-60-2`:

| Field | Meaning |
|---|---|
| `n_c` | number of victim-cluster centers generated (see Section 5.1) |
| `n`   | number of satellites |
| `v_d` | drone speed, km/h (40, 50, 60, 70 or 80) |
| `f`   | fleet: `1` = large trucks (Q = 12,500), `2` = small trucks (Q = 3,500) |

Every `(n_c, n, f)` combination has 5 files, one per drone speed, that share the same
depot/satellite/victim coordinates, demand, truck count K and drone count U -- only `v_d`
and the header's last field differ.

## File format

Plain text, all coordinates in distance units (100 units = 1 km); all derived times
(in the code, not in this file) are in hours.

```
n m K U Q P v_t v_d
x0 y0
x_1 y_1
...
x_n y_n
x_1' y_1' d_1
...
x_m' y_m' d_m
```

- Line 1: `n` satellites, `m` victims, `K` trucks, `U` drones per satellite, truck
  capacity `Q`, drone capacity `P`, truck speed `v_t` (km/h, always 40), drone speed `v_d`.
- Line 2: depot coordinates `(x0, y0)`.
- Next `n` lines: satellite coordinates.
- Next `m` lines: victim coordinates and demand `d_i`.

## Frozen clustering

Each instance ships with a companion `<name>.clusters.json`: the K-Means clustering of
the `m` victims into `n` centroids (one per satellite), computed once and read by every
script (`src/common.py`, `src/milp.py`) instead of being recomputed. This is necessary
for exact reproducibility: K-Means cluster assignment is not perfectly reproducible
across machines or thread counts (float tie-breaks in the underlying implementation can
differ at the ~1e-13 level), which is enough to change which local optimum a heuristic
search converges to on the harder instances. `scripts/freeze_clustering.py` regenerates
these files if you add new instances; it is not needed to reproduce the paper's results.

```json
{"instance": "M-15-13-60-2",
 "centers": [[x, y], ...],
 "label":   [0, 2, 1, ...],
 "demand_n": [d_1, ..., d_n]}
```

`centers` are the n centroid coordinates, `label` says which centroid each of the m
victims is assigned to, and `demand_n` is the total demand of each centroid.

## Internal-name mapping

`name_mapping.csv` lists, for every file, the internal name used during development
(underscore-separated, e.g. `M_15_13_60_2`) alongside the paper name used here
(hyphen-separated). The two differ only in the separator character.
