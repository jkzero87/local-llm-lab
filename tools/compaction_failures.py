#!/usr/bin/env python3
"""Aggregate compaction failures and checkpoint sizes across ~/.dsh/sessions.

READ-ONLY: only reads session files; never prints session content.
Outputs:
  data/compaction_failures.csv  (uuid, n_start, n_summary, n_end_err, first_event_ts)
  stdout: aggregate report only (totals, error strings, top-5 failing sessions,
          all-error session count, checkpoint output-token stats).
"""

import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter

SESSIONS_DIR = os.path.expanduser("~/.dsh/sessions")
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "compaction_failures.csv")


def iter_lines(path):
    """Yield decoded lines from a plain or zstd-compressed session file."""
    if path.endswith(".zstd"):
        # Use the system zstd binary: verified working here. The zstandard
        # 0.25.0 C backend raises io.UnsupportedOperation when its
        # stream_reader is iterated under Python 3.14, and its decompress()
        # only returns the first frame of a multi-frame file.
        with subprocess.Popen(
            ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                yield line
        if proc.returncode != 0:
            raise OSError(f"zstd -dc exited with {proc.returncode}")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line


def session_uuid(path):
    """Extract the session UUID from the file path (session-<uuid>/... or <uuid> filename)."""
    m = re.search(r"session-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", path)
    if m:
        return m.group(1)
    base = os.path.basename(path)
    for suffix in (".jsonl.zstd", ".jsonl"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = re.sub(r"^session(-v\d+)?-", "", base)
    return base


def analyze_session(path):
    """Return (uuid, n_start, n_summary, n_end_err, first_event_ts, error_counts, output_tokens)."""
    uuid = session_uuid(path)
    n_start = n_summary = n_end_err = 0
    first_event_ts = None
    errors = Counter()
    output_tokens = []
    for raw in iter_lines(path):
        line = raw.strip() if isinstance(raw, (bytes, bytearray)) else raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        etype = obj.get("type")
        if etype == "session":
            if first_event_ts is None:
                first_event_ts = obj.get("createdAt")
            continue
        if first_event_ts is None and isinstance(obj.get("time"), (int, float)):
            first_event_ts = obj["time"]
        data = obj.get("data") or {}
        if etype == "compaction/start":
            n_start += 1
        elif etype == "compaction/summary":
            n_summary += 1
            usage = data.get("usage") or {}
            ot = usage.get("outputTokens")
            if isinstance(ot, (int, float)):
                output_tokens.append(ot)
        elif etype == "compaction/end":
            err = data.get("error")
            if err is not None:
                n_end_err += 1
                errors[str(err)] += 1
    return uuid, n_start, n_summary, n_end_err, first_event_ts, errors, output_tokens


def percentile(values, pct):
    """Linear-interpolation percentile (same convention as numpy's default)."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def main():
    files = sorted(
        p
        for p in glob.glob(os.path.join(SESSIONS_DIR, "**", "session*.jsonl*"), recursive=True)
        if os.path.isfile(p) and (p.endswith(".jsonl") or p.endswith(".jsonl.zstd"))
    )

    rows = []
    error_totals = Counter()
    all_output_tokens = []
    for path in files:
        try:
            uuid, n_start, n_summary, n_end_err, ts, errors, toks = analyze_session(path)
        except OSError as exc:
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        rows.append((uuid, n_start, n_summary, n_end_err, ts))
        error_totals.update(errors)
        all_output_tokens.extend(toks)

    rows.sort(key=lambda r: (r[0]))

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uuid", "n_start", "n_summary", "n_end_err", "first_event_ts"])
        for row in rows:
            w.writerow(row)

    # ---- aggregate report (no session content) ----
    total_starts = sum(r[1] for r in rows)
    total_summaries = sum(r[2] for r in rows)
    total_errors = sum(r[3] for r in rows)
    failure_rate = total_errors / total_starts if total_starts else 0.0

    print("=== compaction failure / checkpoint report ===")
    print(f"sessions scanned:            {len(rows)}")
    print(f"compaction/start total:      {total_starts}")
    print(f"compaction/summary total:    {total_summaries}")
    print(f"compaction/end w/ error:     {total_errors}")
    print(f"failure rate (errors/starts): {failure_rate:.4f} ({failure_rate * 100:.2f}%)")
    print()
    print("--- distinct error strings (most common first) ---")
    for err, cnt in error_totals.most_common():
        print(f"{cnt:5d}  {err}")
    print()
    print("--- 5 sessions with most failures ---")
    top5 = sorted(rows, key=lambda r: (r[3], r[1]), reverse=True)[:5]
    for uuid, n_start, n_summary, n_end_err, _ in top5:
        fr = n_end_err / n_start if n_start else float("nan")
        print(f"{uuid}  failures={n_end_err}  starts={n_start}  failure_rate={fr:.4f}")
    print()
    all_err_only = sum(1 for r in rows if r[3] > 0 and r[2] == 0)
    print(f"sessions with n_end_err>0 and n_summary==0: {all_err_only}")
    print()
    print("--- successful checkpoint output tokens ---")
    if all_output_tokens:
        within = sum(1 for t in all_output_tokens if abs(t - 8192) <= 200)
        print(f"count:   {len(all_output_tokens)}")
        print(f"min:     {min(all_output_tokens)}")
        print(f"median:  {statistics.median(all_output_tokens)}")
        print(f"p90:     {percentile(all_output_tokens, 90)}")
        print(f"max:     {max(all_output_tokens)}")
        print(f"within 200 of 8192:      {within}")
    else:
        print("no successful checkpoints found")
    print()
    print(f"csv written: {CSV_PATH}")


if __name__ == "__main__":
    main()
