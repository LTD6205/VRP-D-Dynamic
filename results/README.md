# Results

Final CSV files behind the paper's tables. Column meanings are in each script's docstring
(`src/`, `scripts/`) unless noted below. Instance names use the paper's hyphen convention.

| File | Content | Paper reference |
|---|---|---|
| `MILP_results.csv` | Gurobi solve of every main instance (status, incumbent, best bound, MIP gap, runtime) | Table 8 and A.1-A.2 |
| `MILP_large.csv` | Same, for the 20 large instances | Table 11 and A.3 |
| `GRASP_EvPR_main.csv` | GRASP-EvPR, 10 reps x t_max=30s, one row per (instance, rep) | Table 8 and A.1-A.2 |
| `GRASP_EvPR_large.csv` | GRASP-EvPR, 10 reps x t_max=120s | Table 11 and A.3 |
| `GRASP_ILS_PR_main.csv` | GRASP-ILS-PR (the paper's method), 10 reps x t_max=30s | Table 8 and A.1-A.2 |
| `GRASP_ILS_PR_large.csv` | GRASP-ILS-PR, 10 reps x t_max=120s | Table 11 and A.3 |
| `SA_main.csv` | SA (time-based cooling), 10 reps x t_max=30s | Table 8 and A.1-A.2 |
| `SA_large.csv` | SA, 10 reps x t_max=120s | Table 11 and A.3 |
| `SA_iterations.csv` | SA iteration counts within t_max=30s, 6 representative instances | Table 9 |
| `sensitivity.csv` | One-factor-at-a-time sensitivity of GRASP-ILS-PR's parameters (EvPR off) | Table 10 |
| `verification.csv` | Fixed-route MILP check of every solution reported as best-known or better than the Gurobi incumbent | Section 5.2 |
| `fleet_metrics.csv` | Per-instance fleet-performance metrics (Z1, C_ret, C_sat, C_del, Z2_*, Z3, Z3_del, Tbar_del, utilization) from GRASP-ILS-PR's best main-set solutions | Table 13 and Figure 8 |

**The table numbers above are our best mapping from context, not copied from the manuscript
PDF -- please check them against the actual submitted Table/Figure numbers before publishing.**

## Regenerating these files

Each is reproducible from the instances and scripts in this repository; see the main
`README.md`'s "Reproducing the experiments" section for the exact commands. `verification.csv`
and `fleet_metrics.csv` are derived from the other files' solutions (via
`scripts/verify_solution.py` and `scripts/compute_fleet_metrics.py`) rather than from a
fresh heuristic run.

## Per-replication columns (`*_main.csv`, `*_large.csv`)

`instance, rep, seed, objective, feasible, objective_check, iterations, runtime_s` --
`feasible`/`objective_check` are computed independently from the returned routes (not the
solver's own bookkeeping): `feasible` checks every satellite is visited once, the
satellite-to-centroid assignment is a bijection, and truck/drone capacities hold;
`objective_check` recomputes the objective from the routes and compares it to the value
the solver reported.
