#!/usr/bin/env python3
"""Diagnose what compaction actually shadows in a DSH session file.

Usage: python3 shadow_types.py <session.jsonl.zstd>

Streams the file exactly once via `zstd -dc` and builds:
  - seq -> event type, for every event that has a top-level `seq`
  - the list of compaction/summary events (events carrying
    `data.shadowedSeqs` — in this file: compaction/summary and
    compaction/prune) and their shadowedSeqs lists

Prints exactly five aggregate items and nothing else. No decompressed
session content is ever printed: only counts, event-type names, and the
seq integers explicitly requested in item 5.
"""
import json
import subprocess
import sys
from collections import Counter


def main(path: str) -> None:
    p = subprocess.Popen(
        ["zstd", "-dc", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    seq_to_type: dict = {}   # seq -> event type, every event that has a seq
    shadow_lists = []        # data.shadowedSeqs of each compaction/summary event, file order
    first_shadow_list = None  # shadowedSeqs of the FIRST such event

    for line in p.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        if isinstance(ev.get("seq"), int):
            seq_to_type.setdefault(ev["seq"], ev.get("type"))
        data = ev.get("data")
        if isinstance(data, dict) and "shadowedSeqs" in data:
            s = data["shadowedSeqs"]
            if isinstance(s, list):
                shadow_lists.append(s)
                if first_shadow_list is None:
                    first_shadow_list = s
    p.wait()

    # 1. number of compaction/summary events
    print(f"1. compaction/summary events: {len(shadow_lists)}")

    # 2. total shadowed seq references (with multiplicity)
    total_refs = sum(len(s) for s in shadow_lists)
    print(f"2. total shadowed seq references: {total_refs}")

    # 3 + 4. distinct shadowed seqs: missing vs existing (per event type)
    distinct = set()
    for s in shadow_lists:
        distinct.update(x for x in s if isinstance(x, int))
    missing = [x for x in distinct if x not in seq_to_type]
    print(f"3. shadowed seqs NOT present as any event seq: {len(missing)}")

    hist = Counter(seq_to_type[x] for x in distinct if x in seq_to_type)
    print("4. histogram of shadowed seqs that DO exist, per event type:")
    for t, c in sorted(hist.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"   {t}: {c}")

    # 5. FIRST compaction only: first 12 shadowed seqs as (seq, type) pairs
    print("5. FIRST compaction, first 12 shadowed seqs as (seq, type):")
    for x in (first_shadow_list or [])[:12]:
        t = seq_to_type.get(x, "MISSING (no event with this seq)")
        print(f"   ({x}, {t})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: shadow_types.py <session.jsonl.zstd>")
    main(sys.argv[1])
