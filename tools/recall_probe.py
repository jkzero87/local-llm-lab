#!/usr/bin/env python3
"""
recall_probe.py — can shadowed events be re-derived from the raw session log?

For each viable session (n_paths >= 3 in data/corpus_profile.csv) we stream the
raw zstd JSONL exactly once and ask, for every compaction/summary and
compaction/prune event, whether each seq in its data.shadowedSeqs is STILL
present in the log as a complete event after the compaction happened.

"Recoverable" means the original event at the shadowed seq is still present in
the raw log, i.e. its content could be re-surfaced verbatim. We test this
empirically from the log itself (we do NOT assume DSH rewrites the raw log).

n_chars is the character length of the shadowed event's stored JSON line — the
amount of content that would be re-surfaced verbatim. It is 0 when the seq is
absent from the log (not recoverable).

Session logs are NEVER printed by this script: only aggregates, event-type
names, and seq integers are emitted.
"""

import argparse
import csv
import glob
import json
import os

import zstandard


def read_lines(path):
    """Yield each non-blank stored JSON line from a zstd JSONL file."""
    with zstandard.open(path, "rb") as fh:
        data = fh.read()
    for ln in data.splitlines():
        s = ln.strip()
        if s:
            yield s


def resolve_log(uuid):
    """Return the raw log path for a session uuid, or None if not found.

    Prefers a session.v3.* file over any other in the session dir, matching the
    convention used by compaction_census.py.
    """
    pattern = os.path.join(
        os.path.expanduser("~/.dsh/sessions/*"),
        "session-%s" % uuid,
    )
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not dirs:
        # fall back: any dir whose basename contains the uuid
        dirs = [
            d for d in glob.glob(os.path.join(os.path.expanduser("~/.dsh/sessions/*"), "*"))
            if uuid in os.path.basename(d) and os.path.isdir(d)
        ]
    if not dirs:
        return None
    d = dirs[0]
    cands = []
    for f in os.listdir(d):
        if f.startswith("session") and f.endswith(".jsonl.zstd"):
            cands.append(os.path.join(d, f))
    if not cands:
        return None
    v3 = [c for c in cands if "session.v3." in os.path.basename(c)]
    return v3[0] if v3 else sorted(cands)[0]


def probe_session(path):
    """Stream one log; return (rows, n_sessions_loaded_ok).

    rows: list of dicts with shadowed_seq, compaction_seq, event_type,
    recoverable, n_chars (uuid filled in by caller).
    """
    seq_to_type = {}     # seq -> top-level type (first occurrence wins)
    seq_to_nchars = {}   # seq -> char length of the stored JSON line
    compactions = []     # (compaction_seq, [shadowed seqs]) in file order

    for line in read_lines(path):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        seq = ev.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            seq_to_type.setdefault(seq, ev.get("type"))
            seq_to_nchars.setdefault(seq, len(line))
        et = ev.get("type")
        if et in ("compaction/summary", "compaction/prune"):
            data = ev.get("data")
            if isinstance(data, dict) and "shadowedSeqs" in data:
                ss = data["shadowedSeqs"]
                if isinstance(ss, list):
                    compactions.append((seq, [x for x in ss if isinstance(x, int)
                                              and not isinstance(x, bool)]))

    rows = []
    for cseq, shadowed in compactions:
        for s in shadowed:
            present = s in seq_to_type
            rows.append({
                "compaction_seq": cseq,
                "shadowed_seq": s,
                "event_type": seq_to_type.get(s, "unknown"),
                "recoverable": present,
                "n_chars": seq_to_nchars.get(s, 0) if present else 0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="data/corpus_profile.csv")
    ap.add_argument("--output", default="data/recall_probe.csv")
    args = ap.parse_args()

    with open(args.profile, newline="") as fh:
        viable = [r["uuid"] for r in csv.DictReader(fh) if int(r["n_paths"]) >= 3]

    all_rows = []
    loaded = 0
    for uuid in viable:
        path = resolve_log(uuid)
        if path is None:
            continue
        try:
            rows = probe_session(path)
        except Exception:
            continue
        loaded += 1
        for r in rows:
            r = {"uuid": uuid, **r}
            all_rows.append(r)

    # write CSV
    cols = ["uuid", "compaction_seq", "shadowed_seq", "event_type",
            "recoverable", "n_chars"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})

    # --- report (aggregates only, no log content) ---
    total = len(all_rows)
    rec = [r for r in all_rows if r["recoverable"]]
    notrec = [r for r in all_rows if not r["recoverable"]]

    print("sessions loaded: %d of %d viable" % (loaded, len(viable)))
    print("total shadowed seqs examined: %d" % total)
    print("recoverable: %d (%.1f%%)" % (len(rec), 100.0 * len(rec) / total if total else 0.0))
    print("not recoverable: %d (%.1f%%)" % (len(notrec), 100.0 * len(notrec) / total if total else 0.0))
    if notrec:
        from collections import Counter
        hist = Counter(r["event_type"] for r in notrec)
        print("  not-recoverable by event_type:")
        for t, n in sorted(hist.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print("    %s: %d" % (t, n))
        top5 = sorted(notrec, key=lambda r: (-r["n_chars"], str(r["event_type"])))[:5]
        print("  5 largest not-recoverable by n_chars (event_type, n_chars):")
        for r in top5:
            print("    (%s, %d)" % (r["event_type"], r["n_chars"]))
    else:
        print("  5 largest not-recoverable by n_chars: (none)")


if __name__ == "__main__":
    main()
