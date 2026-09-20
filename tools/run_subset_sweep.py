#!/usr/bin/env python3
"""Run the fixed-reserve (8192) subset policy sweep via the opt-in
`policies-subset` entry point of tools/evict_sim.py.

Verifies (asserts, prints nothing extra) that the pre-existing certified
snapshots are byte-identical before and after the run:
  data/evict_sim.csv
  data/eviction_policies.csv
  data/exposure_check.csv
Prints the entry point's stdout (the 5-row table + n_sessions + gating line).
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = [ROOT / "data" / "evict_sim.csv",
       ROOT / "data" / "eviction_policies.csv",
       ROOT / "data" / "exposure_check.csv"]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    before = {str(p): md5(p) for p in OLD}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "evict_sim.py"), "policies-subset"],
        cwd=ROOT,
    )
    after = {str(p): md5(p) for p in OLD}
    if before != after:
        print("ERROR: existing snapshot changed: %s" % [
            k for k in before if before[k] != after[k]], file=sys.stderr)
        proc.returncode = 1
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
