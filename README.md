# 2E-CTVRP: a two-echelon covering tour vehicle routing problem with drones

Code and data for:

> Dang Viet Anh Nguyen, Rajesh Piplani, Aldy Gunawan. "A two-echelon covering tour vehicle
> routing problem with drones for post-disaster relief." *Computers & Industrial
> Engineering* (under review, manuscript CAIE-D-25-06482).

## The problem

After a disaster, victims are aggregated at demand points a fleet of trucks cannot reach
directly. Trucks route between a depot and a set of *satellites*; from each satellite,
drones fly out to deliver supplies to the victims assigned to it. The problem is to choose
truck routes, satellite-to-victim-cluster assignments and drone dispatch so as to minimize
the total time victims wait for supplies, subject to truck and drone capacity. We propose
**GRASP-ILS-PR**, a hybrid of Iterated Local Search and GRASP with Path Relinking, and
compare it against an exact MILP formulation (small instances), a GRASP-EvPR baseline, and
a Simulated Annealing baseline.

## Citation

```bibtex
@article{nguyen2026twoechelon,
  title   = {A two-echelon covering tour vehicle routing problem with drones for post-disaster relief},
  author  = {Nguyen, Dang Viet Anh and Piplani, Rajesh and Gunawan, Aldy},
  journal = {Computers \& Industrial Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## Installation

Requires Python 3.10+. The MILP solver (`scripts/run_milp.py`) and the fixed-route
verification (`scripts/verify_solution.py`) require a Gurobi license (a free academic
license is enough); the three heuristics do not need Gurobi at all.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Solve one small instance with GRASP-ILS-PR in 30 seconds and print its objective:

```bash
python3 scripts/run_heuristic.py --method grasp-ils-pr \
    --instance instances/main/M-05-03-60-1.txt --time 30 --seed 0
```

## Reproducing the experiments

Hardware used for the paper's runtimes: AMD Ryzen Threadripper 3960X (24 cores / 48
threads), 125 GB RAM, Ubuntu 22.04.3 LTS. Every heuristic run is single-threaded; batches
run 14 replications in parallel (`--workers 14`). Runtimes for time-limited runs, and the
exact solution found, will differ on different hardware, thread counts, or Gurobi
versions -- see `instances/README.md` for why the shipped instances include frozen K-Means
clustering rather than recomputing it.

```bash
# MILP, main set (Table 8 / A.1-A.2)
python3 scripts/run_milp.py --folder instances/main --time-limit 3600 \
    --output results/MILP_results.csv

# GRASP-ILS-PR, main set, 10 reps x 30s (Table 8 / A.1-A.2)
python3 scripts/run_experiments.py --method grasp-ils-pr --folder instances/main \
    --reps 10 --time 30 --workers 14 --output results/GRASP_ILS_PR_main.csv \
    --solutions-dir solutions/grasp_ils_pr_main

# SA and GRASP-EvPR, main set: same command with --method sa / --method grasp-evpr

# large set: --folder instances/large --time 120

# fixed-route MILP verification of a solution batch (Section 5.2)
python3 scripts/verify_solution.py --instances-dir instances/main \
    --solutions-dir solutions/grasp_ils_pr_main --output results/verification_check.csv

# fleet metrics from GRASP-ILS-PR's main-set solutions (Section 6)
python3 scripts/compute_fleet_metrics.py --instances-dir instances/main \
    --solutions-dir solutions/grasp_ils_pr_main --output results/fleet_metrics.csv
```

`results/README.md` maps every shipped CSV to the paper's tables and figures.

## Repository structure

```
2E_CTVRP/
  README.md
  LICENSE
  requirements.txt
  src/                    importable modules
    common.py               instance parsing, K-Means (frozen-cluster aware), construction,
                             repair, objective, neighborhood operators -- shared by all
                             three heuristics
    milp.py                 exact MILP formulation (Gurobi); self-contained (does not
                             import common.py) so it also serves as an independent check
    grasp_ils_pr.py          GRASP-ILS-PR (the paper's method)
    grasp_evpr.py            GRASP-EvPR baseline
    sa.py                    Simulated Annealing baseline (time-based cooling)
    instance_generator.py    the instance generator (Section 5.1)
    fleet_metrics.py         fleet-performance metric definitions (Section 6)
  instances/
    main/                    90 instances (see instances/README.md)
    large/                   20 larger instances (n=20, n=50)
  results/                 CSV files behind the paper's tables (see results/README.md)
  scripts/                 command-line entry points (see docstring of each for usage)
    run_milp.py
    run_heuristic.py
    run_experiments.py
    verify_solution.py
    compute_fleet_metrics.py
    freeze_clustering.py
    generate_instances.py
```

## Instance naming

`M-<n_c>-<n>-<v_d>-<f>`, e.g. `M-15-13-60-2`: `n_c` generated victim-cluster centers,
`n` satellites, drone speed `v_d` km/h, fleet `f` (`1` = large trucks, Q = 12,500;
`2` = small trucks, Q = 3,500). See `instances/README.md` for the full format.

## License

MIT -- see `LICENSE`.

## Contact

andng@dtu.dk
