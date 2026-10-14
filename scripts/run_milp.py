#!/usr/bin/env python3
"""
Solve one or more 2E-CTVRP instances exactly with Gurobi (paper Section 3/5).

Usage:
    # one instance
    python3 scripts/run_milp.py --instance instances/main/M-05-03-60-1.txt --time-limit 3600

    # a whole folder (writes one CSV row per instance)
    python3 scripts/run_milp.py --folder instances/main --pattern "M-*.txt" \\
        --time-limit 3600 --output results/MILP_results.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from milp import run_batch, solve_instance  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", help="Path to a single instance .txt file.")
    ap.add_argument("--folder", help="Folder of instances to batch-solve.")
    ap.add_argument("--pattern", default="M-*.txt", help="Glob pattern used with --folder.")
    ap.add_argument("--output", default="results.csv", help="Output CSV path (batch mode).")
    ap.add_argument("--time-limit", type=float, default=3600.0, help="Gurobi time limit (s) per instance.")
    ap.add_argument("--mip-gap", type=float, default=0.0, help="Gurobi MIPGap.")
    ap.add_argument("--seed", type=int, default=0, help="Gurobi seed.")
    ap.add_argument("--threads", type=int, default=None, help="Gurobi thread count (default: unrestricted).")
    ap.add_argument("--verbose", action="store_true", help="Show the Gurobi solver log.")
    ap.add_argument("--report-dir", default=None, help="If set, write a per-instance route report here.")
    args = ap.parse_args(argv)

    if args.instance:
        row = solve_instance(args.instance, mip_gap=args.mip_gap, time_limit=args.time_limit, seed=args.seed,
                              threads=args.threads, output_flag=1 if args.verbose else 0, report_dir=args.report_dir)
        print(row)
        return

    if args.folder:
        run_batch(folder=args.folder, pattern=args.pattern, output_csv=args.output, mip_gap=args.mip_gap,
                  time_limit=args.time_limit, seed=args.seed, threads=args.threads,
                  output_flag=1 if args.verbose else 0, report_dir=args.report_dir)
        return

    ap.error("give either --instance or --folder")


if __name__ == "__main__":
    main()
