#!/usr/bin/env python3
"""Token usage and API-equivalent cost for dsh session logs.

Reads ~/.dsh/sessions/*/*/session*.jsonl.zstd (matches both the old
session.jsonl.zstd and the v3 session.v3.jsonl.zstd filenames).

Usage:
    python3 tools/session_cost.py --since 2026-09-18 --trm 3151.73
    python3 tools/session_cost.py --since 2026-08-19 --workspace jira --trm 3151.73

--trm is required on purpose: the COP/USD rate is stated by the caller,
never guessed by the script.

Per-session table: tokens only. Local models are gguf paths and do not
map onto any API tier, so the totals section prices the summed tokens at
each of the eight tiers, in USD and COP. The spread across tiers is the
point.
"""
import argparse
import glob
import json
import os
import sys
import time
import zstandard

# USD per million tokens: input / cacheRead / output.
# Rates as published on the vendor pricing pages, checked 2026-09-17.
PRICES = {
    "GPT-6 Astra":              {"in": 10.00, "cre": 1.00,  "out": 50.00,
                                 "url": "https://openai.com/api/pricing/", "checked": "2026-09-17"},
    "Claude Opus 5":            {"in": 5.00,  "cre": 0.50,  "out": 25.00,
                                 "url": "https://www.anthropic.com/pricing", "checked": "2026-09-17"},
    "GPT-5.6 Terra":            {"in": 2.00,  "cre": 0.20,  "out": 12.00,
                                 "url": "https://openai.com/api/pricing/", "checked": "2026-09-17"},
    "Claude Sonnet 5":          {"in": 2.00,  "cre": 0.20,  "out": 10.00,
                                 "url": "https://www.anthropic.com/pricing", "checked": "2026-09-17"},
    "Claude Haiku 4.5":         {"in": 1.00,  "cre": 0.10,  "out": 5.00,
                                 "url": "https://www.anthropic.com/pricing", "checked": "2026-09-17"},
    "GPT-5.6 Luna":             {"in": 0.20,  "cre": 0.02,  "out": 1.20,
                                 "url": "https://openai.com/api/pricing/", "checked": "2026-09-17"},
    "DeepSeek V4.1 Flash peak":     {"in": 0.30, "cre": 0.006, "out": 1.20,
                                 "url": "https://api-docs.deepseek.com/quick_start/pricing", "checked": "2026-09-17"},
    "DeepSeek V4.1 Flash off-peak": {"in": 0.15, "cre": 0.003, "out": 0.60,
                                 "url": "https://api-docs.deepseek.com/quick_start/pricing", "checked": "2026-09-17"},
}


def load_events(path):
    with zstandard.open(path, "rb") as fh:
        return [json.loads(l) for l in fh.read().splitlines() if l.strip()]


def summarize(events):
    meta = events[0]
    s = {
        "id": meta.get("id", "?"),
        "workspace": meta.get("cwd", "?"),
        "created": meta.get("createdAt"),
        "input": 0, "output": 0, "cache": 0,
        "turns": 0, "tools": 0,
        "tmin": None, "tmax": None,
        "comp_starts": {}, "comp_ends": {},
    }
    for e in events:
        t = e.get("time")
        if t:
            s["tmin"] = t if s["tmin"] is None else min(s["tmin"], t)
            s["tmax"] = t if s["tmax"] is None else max(s["tmax"], t)
        typ = e.get("type")
        data = e.get("data") or {}
        if typ == "turn/start":
            s["turns"] += 1
        elif typ == "tool/call":
            s["tools"] += 1
        # usage: old format assistant/chunk (data.chunk.type == "usage"),
        # v3 assistant/message (data.usage), compaction/summary (data.usage).
        # Verified zero overlap between the three sources.
        usage = None
        if typ == "assistant/chunk":
            chunk = data.get("chunk") or {}
            if chunk.get("type") == "usage":
                usage = chunk.get("usage")
        elif typ in ("assistant/message", "compaction/summary"):
            usage = data.get("usage")
        if usage:
            s["input"] += usage.get("inputTokens", 0)
            s["output"] += usage.get("outputTokens", 0)
            s["cache"] += usage.get("cacheReadTokens", 0)
        elif typ == "compaction/start":
            s["comp_starts"][data.get("compactionId")] = t
        elif typ == "compaction/end":
            s["comp_ends"][data.get("compactionId")] = t
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="YYYY-MM-DD, include sessions created on/after this UTC date")
    ap.add_argument("--workspace", help="substring to filter session cwd")
    ap.add_argument("--trm", required=True, type=float, help="COP per USD, stated by the caller")
    ap.add_argument("--until", help="YYYY-MM-DD, exclude sessions created on/after this UTC date")
    ap.add_argument("--output", help="write the summary to this file instead of stdout")
    args = ap.parse_args()
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d")) * 1000  # UTC midnight ms
    until = time.mktime(time.strptime(args.until, "%Y-%m-%d")) * 1000 if args.until else None
    out = open(args.output, "w") if args.output else sys.stdout

    base = os.path.expanduser("~/.dsh/sessions")
    by_dir = {}
    for f in glob.glob(os.path.join(base, "*", "*", "session*.jsonl.zstd")):
        d = os.path.dirname(f)
        name = os.path.basename(f)
        # one row per session: if a dir holds both the old and the v3
        # file, the v3 file re-records the same session; keep the v3.
        if "session.v3." in name:
            by_dir[d] = f
        elif d not in by_dir:
            by_dir[d] = f

    rows = []
    for f in sorted(by_dir.values()):
        events = load_events(f)
        s = summarize(events)
        if s["created"] is None or s["created"] < since:
            continue
        if until is not None and s["created"] >= until:
            continue
        if args.workspace and args.workspace not in s["workspace"]:
            continue
        rows.append(s)
    rows.sort(key=lambda s: s["created"])

    if not rows:
        print(f"no sessions created on/after {args.since}"
              + (f" before {args.until}" if args.until else "")
              + (f" with cwd containing {args.workspace!r}" if args.workspace else ""), file=out)
        return

    print(f"{'session':<46} {'created (UTC)':<19} {'mins':>6} {'turns':>5} {'tools':>6} "
          f"{'input':>9} {'output':>8} {'cached':>9} {'comp':>4} {'comp%':>6}", file=out)
    tot = {"input": 0, "output": 0, "cache": 0, "turns": 0, "tools": 0, "mins": 0.0,
           "compms": 0.0}
    for s in rows:
        wall = (s["tmax"] - s["tmin"]) / 60000 if s["tmin"] and s["tmax"] else 0.0
        comp_ms = sum(e - st for k, st in s["comp_starts"].items()
                      for e in [s["comp_ends"].get(k)] if e and e >= st)
        share = 100.0 * comp_ms / (wall * 60000) if wall > 0 else 0.0
        created = time.strftime("%Y-%m-%d %H:%M", time.gmtime(s["created"] / 1000))
        ws = os.path.basename(s["workspace"].rstrip("/")) if s["workspace"] != "?" else "?"
        print(f"{s['id'][:44]:<46} {created:<19} {wall:>6.1f} {s['turns']:>5} {s['tools']:>6} "
              f"{s['input']:>9} {s['output']:>8} {s['cache']:>9} {len(s['comp_starts']):>4} {share:>6.1f}",
              file=out)
        for k in ("input", "output", "cache", "turns", "tools"):
            tot[k] += s[k]
        tot["mins"] += wall
        tot["compms"] += comp_ms

    print(file=out)
    print(f"totals  {len(rows)} sessions  {tot['mins']:.1f} wall-clock minutes  "
          f"{tot['turns']} turns  {tot['tools']} tool calls", file=out)
    print(f"        input {tot['input']:,}  output {tot['output']:,}  cacheRead {tot['cache']:,}",
          file=out)
    comp_share = 100.0 * tot["compms"] / (tot["mins"] * 60000) if tot["mins"] > 0 else 0.0
    print(f"        compaction {tot['compms'] / 60000:.1f} min  "
          f"({comp_share:.1f}% of window wall clock)", file=out)

    print(file=out)
    print(f"API-equivalent cost of the same totals, USD per million tokens "
          f"(input / cacheRead / output), checked 2026-09-17; COP at {args.trm:.2f} USD/COP:",
          file=out)
    print(f"{'tier':<28} {'in $':>10} {'cache $':>10} {'out $':>10} {'USD':>12} {'COP':>16}",
          file=out)
    for tier, p in PRICES.items():
        usd = (tot["input"] * p["in"] + tot["cache"] * p["cre"] + tot["output"] * p["out"]) / 1e6
        print(f"{tier:<28} {tot['input'] * p['in'] / 1e6:>10.2f} {tot['cache'] * p['cre'] / 1e6:>10.2f} "
              f"{tot['output'] * p['out'] / 1e6:>10.2f} {usd:>12,.2f} {usd * args.trm:>16,.0f}",
              file=out)
    for tier, p in PRICES.items():
        print(f"  # {tier}: {p['url']} (checked {p['checked']})", file=out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
