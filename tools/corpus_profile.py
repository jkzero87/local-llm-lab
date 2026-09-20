#!/usr/bin/env python3
"""Corpus viability profile for the exposure metric.

For every session in the corpus (one file per session via pick_files), compute:
  - n_paths   = number of distinct file paths in ground_truth(path)
  - n_reads   = number of tool/call events whose name is "read" or "grep"
  - first_ts  = "time" of the first event carrying it (epoch ms -> ISO UTC)

Writes data/corpus_profile.csv and prints only the aggregates requested.
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meter_check import pick_files, load_events
from reread_v2 import ground_truth, uuid_of

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus_profile.csv"


def to_iso(ms):
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def profile(path):
    p = Path(path)
    spath = str(p)
    n_paths = len(ground_truth(spath))
    n_reads = 0
    first_ts = None
    for e in load_events(p):
        t = e.get("time")
        if first_ts is None and isinstance(t, (int, float)):
            first_ts = t
        if e.get("type") == "tool/call":
            name = (e.get("data") or {}).get("name")
            if name in ("read", "grep"):
                n_reads += 1
    return uuid_of(spath), n_paths, n_reads, first_ts


def main():
    files = pick_files()
    rows = []
    for p in files:
        try:
            rows.append(profile(p))
        except Exception:
            rows.append((uuid_of(str(p)), None, None, None))
    rows.sort(key=lambda r: r[0])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "n_paths", "n_reads", "first_ts_iso"])
        for uuid, n_paths, n_reads, first_ts in rows:
            w.writerow([uuid, n_paths, n_reads, to_iso(first_ts) if first_ts is not None else ""])

    total = len(rows)
    valid = [r for r in rows if r[1] is not None]
    n3 = [r for r in valid if r[1] >= 3]
    n8 = [r for r in valid if r[1] >= 8]

    print(f"total sessions: {total}")
    print(f"n_paths >= 3: {len(n3)}")
    print(f"n_paths >= 8: {len(n8)}")
    if n3:
        ts = sorted(to_iso(r[3]) for r in n3 if r[3] is not None)
        print(f"n_paths >= 3 count: {len(n3)}  min first_ts_iso: {ts[0]}  max first_ts_iso: {ts[-1]}")
    else:
        print("n_paths >= 3 count: 0")


if __name__ == "__main__":
    main()
