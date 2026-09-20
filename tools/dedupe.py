#!/usr/bin/env python3
"""Deduplicate the dsh session corpus across roots by session UUID.

READ-ONLY with respect to the roots: only path/size metadata is read
(os.walk, os.path.getsize). No session log is ever opened for its
contents. The only file written is data/session_index.csv.

The session UUID is the 36-char id in a path component of the form
'session-<uuid>' (also matched when embedded in a longer component,
e.g. 'dsh-session-<uuid>'). Components that are a bare '<uuid>' are
used as a fallback (older layouts used plain uuid dir names).
"""
import csv
import os
import re
from collections import OrderedDict

ROOTS = [
    os.path.expanduser("~/.dsh/sessions"),
    os.path.expanduser("~/.dsh.pre_v016_20260917_164138"),
    os.path.expanduser("~/.dsh.backup_pre_v015_20260912"),
    os.path.expanduser("~/.dsh.backup_rc7"),
    os.path.expanduser(
        "~/Downloads/dsh-session-session-f90c794a-f384-4cac-b5d0-a9b9146f1d11"
    ),
]
LIVE = ROOTS[0]

UUID_RE = re.compile(
    r"session-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
BARE_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def extract_uuid(path):
    parts = path.split(os.sep)
    for part in parts:
        m = UUID_RE.search(part)
        if m:
            return m.group(1)
    for part in parts:
        if BARE_UUID_RE.fullmatch(part):
            return part
    return None


def is_session_file(name):
    return name.startswith("session") and "jsonl" in name


def generation(fname):
    return "v3" if "v3" in fname else "v1"


def main():
    # uuid -> list of (root_index, root, path, size)
    sessions = {}
    unassigned = []
    for ri, root in enumerate(ROOTS):
        if not os.path.isdir(root):
            print(f"WARN: root missing: {root}")
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not is_session_file(fn):
                    continue
                path = os.path.join(dirpath, fn)
                uid = extract_uuid(path)
                if uid is None:
                    unassigned.append(path)
                    continue
                size = os.path.getsize(path)
                sessions.setdefault(uid, []).append((ri, root, path, size))

    rows = []
    for uid in sorted(sessions):
        copies = sessions[uid]
        seen = [root for root in ROOTS if any(c[1] == root for c in copies)]
        # largest copy wins; ties -> earlier root, then lexicographic path
        chosen = max(copies, key=lambda c: (c[3], -c[0], c[2]))
        fname = os.path.basename(chosen[2])
        rows.append(
            [uid, len(copies), ";".join(seen), chosen[2], chosen[3], generation(fname)]
        )

    out = os.path.expanduser("~/local-llm-lab/data/session_index.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["uuid", "n_copies", "roots_seen", "chosen_path", "chosen_bytes",
             "filename_generation"]
        )
        w.writerows(rows)

    # ---- stats for the report ----
    total = len(rows)
    multi = sum(1 for r in rows if len(r[2].split(";")) > 1)
    not_in_live = [r for r in rows if LIVE not in r[2].split(";")]

    per_root = OrderedDict((root, 0) for root in ROOTS)
    for r in rows:
        seen = r[2].split(";")
        for root in ROOTS:
            if root in seen:
                per_root[root] += 1

    print(f"total_unique_uuids: {total}")
    print(f"in_more_than_one_root: {multi}")
    print(f"not_in_live_sessions: {len(not_in_live)}")
    for r in not_in_live:
        short = ";".join(os.path.basename(rt) if rt != ROOTS[0] else "LIVE"
                         for rt in r[2].split(";"))
        print(f"  not-in-live: {r[0]}  roots=[{short}]")
    print("unique_uuids_per_root:")
    for root, n in per_root.items():
        print(f"  {root}: {n}")
    if unassigned:
        print(f"WARN: {len(unassigned)} session files had no extractable UUID:")
        for p in unassigned[:20]:
            print(f"  {p}")
    print(f"csv_rows_written: {total} -> {out}")


if __name__ == "__main__":
    main()
