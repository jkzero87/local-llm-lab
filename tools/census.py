#!/usr/bin/env python3
"""Build a census of dsh session logs.

Read-only with respect to ~/.dsh: this script only opens session files for
reading (via a streaming zstd decompressor) and writes a single CSV into the
workspace. It never holds a whole file in memory and never prints decompressed
session content.
"""

import csv
import glob
import json
import os

import zstandard as zstd

HOME = os.path.expanduser("~")
SESSIONS_GLOB = os.path.join(HOME, ".dsh", "sessions", "*", "*", "session*.jsonl.zstd")
OUT_CSV = os.path.join(HOME, "local-llm-lab", "data", "session_census.csv")
CHUNK = 1 << 20  # 1 MiB read chunk
PHRASE = b"incomplete checkpoint"

COLUMNS = [
    "path",
    "filename_generation",
    "first_event_ts",
    "last_event_ts",
    "n_events",
    "n_compaction_summary",
    "n_tool_call",
    "n_tool_result",
    "sum_summary_output_tokens",
    "max_summary_output_tokens",
    "n_truncated_checkpoints",
]


def iter_lines(path, chunk=CHUNK):
    """Yield newline-terminated line chunks from a zstd-compressed file.

    Reads fixed-size chunks from the streaming decompressor and buffers only a
    trailing partial line, so memory stays bounded regardless of file size.
    """
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        reader = dctx.stream_reader(f, closefd=True)
        buf = b""
        while True:
            data = reader.read(chunk)
            if not data:
                break
            buf += data
            idx = buf.rfind(b"\n")
            if idx == -1:
                continue  # no complete line yet
            head, buf = buf[: idx + 1], buf[idx + 1 :]
            for line in head.split(b"\n"):
                yield line
        if buf:
            yield buf


def event_ts(ev):
    """Return the timestamp field of an event, or None.

    Regular events carry `time` (epoch ms); the leading `session` header event
    carries `createdAt` (epoch ms) instead.
    """
    for key in ("time", "createdAt"):
        val = ev.get(key)
        if isinstance(val, (int, float)):
            return val
    return None


def filename_generation(path):
    base = os.path.basename(path)
    return "v3" if "v3" in base else "v1"


def census_file(path):
    row = {
        "path": path,
        "filename_generation": filename_generation(path),
        "first_event_ts": None,
        "last_event_ts": None,
        "n_events": 0,
        "n_compaction_summary": 0,
        "n_tool_call": 0,
        "n_tool_result": 0,
        "sum_summary_output_tokens": 0,
        "max_summary_output_tokens": None,
        "n_truncated_checkpoints": 0,
    }

    for raw in iter_lines(path):
        if not raw.strip():
            continue
        # Per-event record count (every non-empty line is one event record).
        row["n_events"] += 1
        if PHRASE in raw:
            row["n_truncated_checkpoints"] += 1

        try:
            ev = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue  # unparseable event: no ts / type / usage
        if not isinstance(ev, dict):
            continue

        ts = event_ts(ev)
        if ts is not None:
            if row["first_event_ts"] is None:
                row["first_event_ts"] = ts
            row["last_event_ts"] = ts

        etype = ev.get("type")
        if etype == "tool/call":
            row["n_tool_call"] += 1
        elif etype == "tool/result":
            row["n_tool_result"] += 1
        elif etype == "compaction/summary":
            row["n_compaction_summary"] += 1
            data = ev.get("data")
            if isinstance(data, dict):
                usage = data.get("usage")
                if isinstance(usage, dict):
                    ot = usage.get("outputTokens")
                    if isinstance(ot, (int, float)):
                        row["sum_summary_output_tokens"] += ot
                        if row["max_summary_output_tokens"] is None or ot > row["max_summary_output_tokens"]:
                            row["max_summary_output_tokens"] = ot

    if row["max_summary_output_tokens"] is None:
        row["max_summary_output_tokens"] = 0
    return row


def main():
    files = sorted(glob.glob(SESSIONS_GLOB))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    with open(OUT_CSV, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for path in files:
            row = census_file(path)
            rows.append(row)
            writer.writerow(row)

    total_rows = len(rows)
    sum_compaction = sum(r["n_compaction_summary"] for r in rows)
    files_with_compaction = sum(1 for r in rows if r["n_compaction_summary"] >= 1)
    top5 = sorted(rows, key=lambda r: (r["n_compaction_summary"], r["path"]), reverse=True)[:5]

    print(f"total rows written: {total_rows}")
    print(f"sum of n_compaction_summary: {sum_compaction}")
    print(f"files with n_compaction_summary >= 1: {files_with_compaction}")
    print("top 5 files by n_compaction_summary:")
    for r in top5:
        print(f"  {r['n_compaction_summary']:>4}  first_event_ts={r['first_event_ts']}  {r['path']}")


if __name__ == "__main__":
    main()
