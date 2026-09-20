#!/usr/bin/env python3
"""Build a per-event exact token-count cache for DSH sessions.

Reads every session file under ~/.dsh/sessions READ-ONLY. For each
assistant/message, tool/result, and user/message event, the event's textual
payload (exactly the parts meter_check.py counts: dict content parts with
text/reasoning/tool-call/tool-result payloads) is POSTed to llama-server
http://localhost:8092/tokenize and the TOKEN COUNT ONLY is recorded in
data/token_counts.csv with columns uuid,seq,type,n_tokens,n_tokens_reasoning.
For assistant/message events, n_tokens_reasoning is the count of ONLY the
reasoning parts concatenated (blank for non-assistant rows).

Tokens and text are never written anywhere. Resumable: rows whose n_tokens
is set -- and, for assistant/message rows, whose n_tokens_reasoning is also
set -- are skipped on re-runs (reasoning backfill does not re-tokenize the
full text). On success the cache file is rewritten in full with the 5-column
header; on error it stops and reports, keeping appended partial rows so the
re-run resumes. On any endpoint error (HTTP failure, timeout/stall,
unexpected response shape) the script stops and reports -- no retries.

Usage: python3 tools/tokenize_cache.py
"""

import csv
import json
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import zstandard as zstd

ROOT = Path.home() / ".dsh" / "sessions"
OUT = Path(__file__).resolve().parent.parent / "data" / "token_counts.csv"
URL = "http://localhost:8092/tokenize"
TIMEOUT = 30          # seconds per request; a stall is an error and stops us
WORKERS = 4
SURFACE_TYPES = ("assistant/message", "tool/result", "user/message")


def read_lines(path):
    if path.name.endswith(".zstd"):
        with open(path, "rb") as f:
            data = zstd.ZstdDecompressor().stream_reader(f).read()
    else:
        data = path.read_bytes()
    for line in data.splitlines():
        if line.strip():
            yield line


def content_text(c):
    """Textual payload of a content value (mirrors meter_check.content_len)."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = ""
        for x in c:
            if isinstance(x, str):
                out += x
            elif isinstance(x, dict):
                t = x.get("text")
                if isinstance(t, str):
                    out += t
        return out
    if isinstance(c, dict):
        t = c.get("text")
        return t if isinstance(t, str) else ""
    return ""


def part_text(p):
    """Textual payload of one content part (mirrors meter_check.part_text_len)."""
    if not isinstance(p, dict):
        return ""
    t = p.get("type")
    if t in ("text", "reasoning"):
        v = p.get("text")
        return v if isinstance(v, str) else ""
    if t == "tool-call":
        s = str(p.get("name") or "")
        a = p.get("arguments")
        if isinstance(a, str):
            s += a
        elif a is not None:
            s += json.dumps(a, ensure_ascii=False, separators=(",", ":"))
        return s
    if t == "tool-result":
        return content_text(p.get("content"))
    return ""


def event_text(e):
    """Full textual payload of one surface event (concatenated parts)."""
    d = e.get("data") or {}
    if e["type"] == "user/message":
        content = d.get("content") or []
    else:
        content = (d.get("message") or {}).get("content") or []
    return "".join(part_text(p) for p in content if isinstance(p, dict))


def reasoning_text(e):
    """Concatenated text of ONLY the reasoning parts (assistant events)."""
    d = e.get("data") or {}
    content = (d.get("message") or {}).get("content") or []
    return "".join(
        p.get("text")
        for p in content
        if isinstance(p, dict)
        and p.get("type") == "reasoning"
        and isinstance(p.get("text"), str)
    )


def pick_files():
    """One file per session dir; dedupe plain/v3 pairs (prefer v3, then size)."""
    by_dir = defaultdict(list)
    for f in ROOT.rglob("session*.jsonl*"):
        if f.is_file():
            by_dir[f.parent].append(f)
    picks = []
    for d, fl in sorted(by_dir.items()):
        if len(fl) == 1:
            picks.append(fl[0])
        else:
            v3 = [x for x in fl if "v3" in x.name]
            picks.append(max(v3 or fl, key=lambda p: p.stat().st_size))
    return picks


def load_events(path):
    events = []
    for i, line in enumerate(read_lines(path)):
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(e, dict) or e.get("type") is None:
            continue
        if e.get("seq") is None:
            e["seq"] = i + 1
        events.append(e)
    events.sort(key=lambda e: e["seq"])
    return events


def load_cached():
    """(uuid, seq) -> [n_tokens, n_tokens_reasoning, type]; blanks -> None.

    Tolerates both the old 4-column and the new 5-column rows; if the same
    (uuid, seq) appears twice the later row wins.
    """
    cache = {}
    if OUT.exists():
        with open(OUT, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 4 or not row[0] or row[0] == "uuid":
                    continue
                n = int(row[3]) if row[3] else None
                r = int(row[4]) if len(row) > 4 and row[4] else None
                cache[(row[0], row[1])] = [n, r, row[2]]
    return cache


def is_done(cache, uuid, seq, typ):
    """A row is done when n_tokens is set AND, for assistant/message rows,
    n_tokens_reasoning is also set."""
    row = cache.get((uuid, seq))
    if not row or row[0] is None:
        return False
    return typ != "assistant/message" or row[1] is not None


def tokenize(text):
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read().decode())
    t = d.get("tokens")
    if not isinstance(t, list):
        raise ValueError("unexpected tokenize response shape")
    return len(t)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(
                ["uuid", "seq", "type", "n_tokens", "n_tokens_reasoning"]
            )

    cache = load_cached()
    jobs = []
    for p in pick_files():
        uuid = p.parent.name
        if uuid.startswith("session-"):
            uuid = uuid[len("session-"):]
        for e in load_events(p):
            if e["type"] not in SURFACE_TYPES:
                continue
            seq = str(e["seq"])
            if is_done(cache, uuid, seq, e["type"]):
                continue
            jobs.append(
                (
                    uuid,
                    seq,
                    e["type"],
                    event_text(e),
                    reasoning_text(e) if e["type"] == "assistant/message" else "",
                )
            )

    state = {"n": 0}
    lock = threading.Lock()

    def work(job):
        uuid, seq, typ, text, rtext = job
        row = cache.get((uuid, seq)) or [None, None, typ]
        n = row[0]
        if n is None:
            n = 0 if not text else tokenize(text)
        r = row[1]
        if r is None and typ == "assistant/message":
            r = 0 if not rtext else tokenize(rtext)
        return (uuid, seq, typ, n, r if r is not None else "")

    try:
        with open(OUT, "a", newline="") as f:
            w = csv.writer(f)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for res in ex.map(work, jobs):
                    with lock:
                        w.writerow(list(res))
                        f.flush()
                        state["n"] += 1
                        if state["n"] % 2000 == 0:
                            print(f"progress: {state['n']} events tokenized")
    except Exception as e:  # HTTPError / URLError / timeout / bad shape
        print(
            f"ERROR: tokenize failed after {state['n']} events this run: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # success: rewrite the cache uniformly with the 5-column header
    cache = load_cached()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "seq", "type", "n_tokens", "n_tokens_reasoning"])
        for (u, s), row in cache.items():
            w.writerow(
                [u, s, row[2], row[0], row[1] if row[1] is not None else ""]
            )

    n_reasoning = sum(
        1
        for row in cache.values()
        if row[2] == "assistant/message" and row[1] is not None
    )
    print(
        f"done: {state['n']} events tokenized this run; "
        f"{len(cache)} events in cache "
        f"({n_reasoning} assistant with reasoning counts)"
    )


if __name__ == "__main__":
    main()
