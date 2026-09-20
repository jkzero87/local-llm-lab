#!/usr/bin/env python3
"""Post-compaction re-read detector.

Walks ~/.dsh/sessions for files named session*.jsonl(.zstd), keeps the files
containing at least one compaction/summary event, and for each session uuid
(the path segment 'session-<uuid>') keeps the file with the most compactions.
No CSV input, no joins: the uuid comes from the path segment alone.

Per session, in seq order:
  1. map seq -> set of targets for each tool/call
  2. for each compaction/summary, the union of targets of tool/calls whose
     seq is in data.shadowedSeqs = the "dropped targets"
  3. scan forward from that compaction: a later tool/call requesting a
     dropped target is a re-read. Each dropped target is scored TWICE:
     - was_reread_path:     an exact path match in a later tool/call
     - was_reread_basename: a basename match in a later tool/call
     seqs_until_reread = (seq of the earliest re-read of either kind)
                         - compaction seq; empty when there is no re-read.

Outputs (paths relative to the current directory):
  data/reread_events.csv    uuid,compaction_seq,target,dropped_at_seq,
                            was_reread_path,was_reread_basename,seqs_until_reread
  data/reread_coverage.csv  uuid,compaction_seq,n_shadowed_seqs,
                            n_shadowed_toolcalls,n_dropped_targets

dropped_at_seq is the largest shadowed tool/call seq that produced the target.

Files are streamed line by line (zstd via `zstd -dc`); a file is never held
in memory and no file contents are printed.
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# Tools whose file_path argument is a target.
FILE_TOOLS = {
    "read": "file_path",
    "write": "file_path",
    "edit": "file_path",
    "read_image": "file_path",
}

# Bash tokens ending in one of these extensions count as file targets even
# when they contain no slash.
CODE_EXTS = {
    "py", "js", "mjs", "cjs", "ts", "jsx", "tsx", "go", "rs", "rb",
    "sh", "bash", "zsh", "fish", "c", "h", "cpp", "hpp", "cc", "hh",
    "java", "kt", "kts", "swift", "sql", "yml", "yaml", "toml", "json",
    "jsonl", "md", "txt", "html", "htm", "css", "scss", "vue", "svelte",
    "php", "pl", "lua", "r", "m", "mm", "proto", "graphql", "wasm",
    "csv", "tsv", "xml",
}

# Leading shell redirection, e.g. "2>", "10>>", "<"
REDIR_RE = re.compile(r"^\d*[<>]+")


def open_lines(path):
    """Yield lines from a .jsonl or .jsonl.zstd file, streaming."""
    if str(path).endswith(".zstd"):
        proc = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def parse_envelope(line):
    line = line.strip()
    if not line:
        return None
    try:
        env = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return env if isinstance(env, dict) else None


def extract_targets(name, arguments):
    """Return the set of target file paths for one tool/call."""
    if name in FILE_TOOLS:
        fp = arguments.get(FILE_TOOLS[name])
        if isinstance(fp, str) and fp:
            return {fp}
        return set()
    if name == "bash":
        cmd = arguments.get("command")
        if not isinstance(cmd, str):
            return set()
        targets = set()
        for tok in cmd.split():
            t = REDIR_RE.sub("", tok)
            if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
                t = t[1:-1]
            if not t or t.startswith("-"):
                continue
            if t == "/dev/null":
                continue
            if "/" in t:
                targets.add(t)
            elif "." in t and t.rsplit(".", 1)[-1] in CODE_EXTS:
                targets.add(t)
        return targets
    return set()


def uuid_from_path(path):
    """Session uuid from the 'session-<uuid>' path segment (parent dir first,
    then the filename stem as a fallback)."""
    for seg in (path.parent.name, path.name):
        if seg.startswith("session-") and seg[len("session-"):]:
            return seg[len("session-"):]
    return None


def count_compactions(path):
    """Stream a file and count compaction/summary events."""
    n = 0
    for line in open_lines(path):
        env = parse_envelope(line)
        if env and env.get("type") == "compaction/summary":
            n += 1
    return n


def collect_events(path):
    """Stream a file; return (calls, comps):
    calls = seq -> set of targets (tool/calls only),
    comps = [(seq, shadowedSeqs)] for each compaction/summary."""
    calls = {}
    comps = []
    for line in open_lines(path):
        env = parse_envelope(line)
        if env is None:
            continue
        etype = env.get("type")
        seq = env.get("seq")
        if not isinstance(seq, int):
            continue
        data = env.get("data")
        if etype == "tool/call":
            if not isinstance(data, dict):
                continue
            args = data.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tg = extract_targets(data.get("name"), args)
            if tg:
                calls.setdefault(seq, set()).update(tg)
        elif etype == "compaction/summary":
            ss = data.get("shadowedSeqs") if isinstance(data, dict) else None
            if not isinstance(ss, list):
                ss = []
            ss = [s for s in ss if isinstance(s, int) and not isinstance(s, bool)]
            comps.append((seq, ss))
    return calls, comps


def main():
    ap = argparse.ArgumentParser(description="Post-compaction re-read detector")
    ap.add_argument("--sessions", default="~/.dsh/sessions",
                    help="sessions root (default: ~/.dsh/sessions)")
    ap.add_argument("--outdir", default="data",
                    help="output directory (default: data)")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.sessions))
    outdir = Path(args.outdir)

    # 1. Candidate files: session*.jsonl* under the sessions root.
    candidates = []
    for p in sorted(root.rglob("session*.jsonl*")):
        if p.is_file():
            uuid = uuid_from_path(p)
            if uuid:
                candidates.append((uuid, p))

    # 2. Keep files with >= 1 compaction; per uuid keep the most compactions.
    best = {}
    for uuid, p in candidates:
        n = count_compactions(p)
        if n < 1:
            continue
        cur = best.get(uuid)
        if cur is None or n > cur[0]:
            best[uuid] = (n, p)

    outdir.mkdir(parents=True, exist_ok=True)
    events_path = outdir / "reread_events.csv"
    coverage_path = outdir / "reread_coverage.csv"

    event_rows = []
    coverage_rows = []
    total_compactions = 0
    total_shadowed_seqs = 0
    total_shadowed_toolcalls = 0
    total_dropped_targets = 0
    reread_path = 0
    reread_basename = 0
    seqs_until_vals = []
    sessions_with_reread = set()

    for uuid in sorted(best):
        _, p = best[uuid]
        calls, comps = collect_events(p)
        comps.sort(key=lambda c: c[0])
        call_seqs = sorted(calls)

        session_had_reread = False
        for cseq, shadowed in comps:
            total_compactions += 1
            shadowed_set = set(shadowed)
            n_shadowed_toolcalls = sum(1 for s in shadowed if s in calls)
            total_shadowed_seqs += len(shadowed)
            total_shadowed_toolcalls += n_shadowed_toolcalls

            # Dropped targets: union of targets of shadowed tool/calls,
            # remembering the (largest) seq of the call that produced each.
            dropped = {}
            for s in shadowed_set:
                for t in calls.get(s, ()):
                    if t not in dropped or s > dropped[t]:
                        dropped[t] = s

            # Forward scan: first later tool/call seq at which each exact
            # target / basename reappears.
            later_exact = {}
            later_base = {}
            for s2 in call_seqs:
                if s2 <= cseq:
                    continue
                for t in calls[s2]:
                    if t not in later_exact:
                        later_exact[t] = s2
                    b = os.path.basename(t)
                    if b and b not in later_base:
                        later_base[b] = s2

            total_dropped_targets += len(dropped)
            for target in sorted(dropped):
                b = os.path.basename(target)
                fp = later_exact.get(target)
                fb = later_base.get(b) if b else None
                wp = fp is not None
                wb = fb is not None
                seqs_until = ""
                if wp or wb:
                    seqs_until = min(x for x in (fp, fb) if x is not None) - cseq
                    seqs_until_vals.append(seqs_until)
                    session_had_reread = True
                if wp:
                    reread_path += 1
                if wb:
                    reread_basename += 1
                event_rows.append(
                    (uuid, cseq, target, dropped[target], wp, wb, seqs_until)
                )

            coverage_rows.append(
                (uuid, cseq, len(shadowed), n_shadowed_toolcalls, len(dropped))
            )

        if session_had_reread:
            sessions_with_reread.add(uuid)

    with open(events_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uuid", "compaction_seq", "target", "dropped_at_seq",
                    "was_reread_path", "was_reread_basename", "seqs_until_reread"])
        w.writerows(event_rows)
    with open(coverage_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uuid", "compaction_seq", "n_shadowed_seqs",
                    "n_shadowed_toolcalls", "n_dropped_targets"])
        w.writerows(coverage_rows)

    pct_path = 100.0 * reread_path / total_dropped_targets if total_dropped_targets else 0.0
    pct_base = 100.0 * reread_basename / total_dropped_targets if total_dropped_targets else 0.0
    median = statistics.median(seqs_until_vals) if seqs_until_vals else None

    print(f"sessions with a compaction:  {len(best)}")
    print(f"total compactions:           {total_compactions}")
    print(f"shadowed seqs:               {total_shadowed_seqs} "
          f"({total_shadowed_toolcalls} were tool/calls)")
    print(f"dropped targets:             {total_dropped_targets}")
    print(f"re-reads (exact path):       {reread_path} ({pct_path:.1f}% of dropped targets)")
    print(f"re-reads (basename):         {reread_basename} ({pct_base:.1f}% of dropped targets)")
    print(f"sessions with >=1 re-read:   {len(sessions_with_reread)}")
    print(f"median seqs_until_reread:    {median}"
          if median is not None else "median seqs_until_reread:    n/a (no re-reads)")
    print(f"wrote {events_path} ({len(event_rows)} rows)")
    print(f"wrote {coverage_path} ({len(coverage_rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
