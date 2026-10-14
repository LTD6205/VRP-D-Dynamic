#!/usr/bin/env python3
"""
Generates a family of 2E-CTVRP instances (both fleets x five drone speeds =
10 files) at a given size, following the spatial/demand model of paper
Section 5.1 (see src/instance_generator.py for the two documented departures
from the paper's prose). Output files still need clustering frozen -- run
scripts/freeze_clustering.py afterward.

Usage:
    python3 scripts/generate_instances.py --n-c 20 --n 20 --n1 10 --n2 10 \\
        --seed 42 --out-dir instances/large --name-prefix M-20-20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from instance_generator import generate_family  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-c", type=int, required=True, help="Number of generated victim-cluster centers.")
    ap.add_argument("--n", type=int, required=True, help="Number of satellites (n1 + n2).")
    ap.add_argument("--n1", type=int, required=True, help="Satellites in group 1 (radius 1000).")
    ap.add_argument("--n2", type=int, required=True, help="Satellites in group 2 (radius 1200).")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name-prefix", required=True, help='e.g. "M-20-20" -> M-20-20-{40..80}-{1,2}.txt')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = generate_family(n_c=args.n_c, n=args.n, n1=args.n1, n2=args.n2, seed=args.seed,
                              out_dir=args.out_dir, name_prefix=args.name_prefix)
    print(summary)


if __name__ == "__main__":
    main()
