#!/usr/bin/env python3
"""
Computes K-Means clustering for every instance in a folder and writes it to
a `<name>.clusters.json` companion file next to the instance -- this is the
frozen clustering already shipped with every instance in instances/main and
instances/large. Re-run this only if you add new instances; it is NOT needed
to reproduce the paper's results, since the frozen files are already present.

K-Means cluster assignment is not perfectly reproducible across machines or
thread counts (float tie-breaks in the underlying C implementation can differ
at the ~1e-13 level, enough to change which local optimum a heuristic search
converges to on harder instances), so results depend on reading the SAME
frozen clustering rather than each script re-clustering independently.

Usage:
    python3 scripts/freeze_clustering.py --folder instances/main
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
from common import parse_instance, cluster_victims  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="M-*.txt")
    ap.add_argument("--force", action="store_true", help="Recompute even if a .clusters.json already exists.")
    args = ap.parse_args()

    n_written = n_skipped = 0
    for fp in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
        name = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(args.folder, f"{name}.clusters.json")
        if os.path.exists(out):
            if not args.force:
                n_skipped += 1
                continue
            os.remove(out)  # cluster_victims() prefers an existing frozen file, so it must go before recomputing
        inst = parse_instance(fp)
        c = cluster_victims(inst, random_state=0)
        with open(out, "w") as f:
            json.dump(dict(instance=name, centers=np.asarray(c.centers).tolist(),
                           label=np.asarray(c.label).tolist(), demand_n=list(c.demand_n)), f)
        n_written += 1
    print(f"Wrote {n_written} new frozen-cluster files, skipped {n_skipped} already present.")


if __name__ == "__main__":
    main()
