#!/usr/bin/env python3
"""Compaction failure census over the whole session corpus.

Counts compaction/end events grouped by error string (absent error -> "ok")
and reports, for each distinct error, its count and the number of distinct
sessions in which it appears. Prints only the aggregates.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meter_check import pick_files, read_lines
from reread_v2 import uuid_of


def main():
    counts = Counter()
    sessions = {}  # error -> set of uuids, lazily created
    for path in pick_files():
        u = uuid_of(str(path))
        for line in read_lines(path):
            try:
                e = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(e, dict) or e.get("type") != "compaction/end":
                continue
            err = (e.get("data") or {}).get("error")
            key = "ok" if err is None else str(err)
            counts[key] += 1
            sessions.setdefault(key, set()).add(u)

    for key, n in counts.most_common():
        print(f"{key}\tcount={n}\tsessions={len(sessions[key])}")


if __name__ == "__main__":
    main()
