#!/usr/bin/env python3
"""Probe the tool-call event schema of a dsh session JSONL stream on stdin.

Prints aggregated schema facts only:
  1. tool/call events: distinct tool names with counts
  2. per tool name: key path where arguments live + distinct argument key names
  3. compaction/summary events: top-level key names + shadowedSeqs value type
  4. the key name that links tool/result back to its tool/call (id matched by
     digest inside this process; raw values are never printed)
"""
import sys
import json
import hashlib
from collections import Counter, defaultdict

NAME_TIERS = (("tool_name", "toolName"), ("name", "tool"))
ARGS_TIERS = ("arguments", "args", "input", "parameters", "payload")
ID_KEYS = ("id", "call_id", "callId", "tool_call_id", "toolCallId",
           "request_id", "requestId", "correlation_id", "correlationId",
           "seq", "seq_id", "seqId")


def norm(t):
    return str(t).lower().replace(".", "/").replace("_", "/")


def payload_of(obj):
    """Event payload: the line itself, or a dict under 'event'/'data' if that
    is where the 'type' field actually lives."""
    if isinstance(obj, dict):
        for k in ("event", "data"):
            v = obj.get(k)
            if isinstance(v, dict) and "type" in v:
                return v
    return obj


def type_of(payload):
    if isinstance(payload, dict):
        for k in ("type", "kind"):
            v = payload.get(k)
            if isinstance(v, str):
                return v
    return None


def find_keys(obj, wanted, max_depth=8):
    found = []

    def rec(o, path, depth):
        if depth > max_depth:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                p = path + (k,)
                if k in wanted:
                    found.append(p)
                rec(v, p, depth + 1)
        elif isinstance(o, list):
            for v in o:
                rec(v, path, depth + 1)

    rec(obj, (), 0)
    found.sort(key=lambda p: (len(p), p))
    return found


def get(obj, path):
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def pstr(path):
    return ".".join(path) if path else "(event itself)"


def tool_name_of(p):
    for tier in NAME_TIERS:
        for path in find_keys(p, tier):
            v = get(p, path)
            if isinstance(v, str):
                return pstr(path), v
    return None, None


def args_of(p):
    for key in ARGS_TIERS:
        paths = find_keys(p, (key,))
        if paths:
            path = paths[0]  # shallowest
            return path, get(p, path)
    return None, None


def parse_args_value(v):
    if isinstance(v, dict):
        return v, "dict"
    if isinstance(v, list):
        return v, "list"
    if isinstance(v, str):
        try:
            d = json.loads(v)
        except ValueError:
            return None, "unparsable string"
        if isinstance(d, (dict, list)):
            return d, "json string -> " + type(d).__name__
    return None, "other"


def digest(value):
    return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()


def main():
    bad = 0
    type_counts = Counter()
    top_key_counts = Counter()
    calls, results, compactions = [], [], []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        payload = payload_of(obj)
        if not isinstance(payload, dict):
            continue
        top_key_counts.update(payload.keys())
        t = type_of(payload)
        if t is not None:
            type_counts[t] += 1
        nt = norm(t) if t is not None else None
        if nt == "tool/call":
            calls.append(payload)
        elif nt == "tool/result":
            results.append(payload)
        elif nt == "compaction/summary":
            compactions.append(payload)

    # ---------- 1 ----------
    print("== 1. tool/call events ==")
    name_counts = Counter()
    name_path_counts = Counter()
    if calls:
        for p in calls:
            loc, name = tool_name_of(p)
            if name is None:
                name_counts["<no name found>"] += 1
            else:
                name_path_counts[loc] += 1
                name_counts[name] += 1
        print("event count: %d (unparseable lines skipped: %d)" % (len(calls), bad))
        print("tool name key path(s): %s" % dict(name_path_counts))
        for name, n in name_counts.most_common():
            print("  %s: %d" % (name, n))
    else:
        print("no events matched type tool/call; observed type values: %s"
              % (dict(type_counts) or "{}"))
        print("top-level event keys observed: %s" % (dict(top_key_counts) or "{}"))

    # ---------- 2 ----------
    print()
    print("== 2. arguments location and argument keys per tool ==")
    if calls:
        arg_locs = defaultdict(Counter)   # name -> Counter("path (kind)")
        arg_keys = defaultdict(Counter)   # name -> Counter(arg key name)
        for p in calls:
            _, name = tool_name_of(p)
            path, raw = args_of(p)
            if path is None:
                arg_locs[name]["<no args key found>"] += 1
                continue
            value, kind = parse_args_value(raw)
            arg_locs[name]["%s (%s)" % (pstr(path), kind)] += 1
            if isinstance(value, dict):
                arg_keys[name].update(value.keys())
        for name in name_counts:
            print("tool: %s" % name)
            for loc, n in arg_locs[name].most_common():
                print("  arguments at: %s  [%d events]" % (loc, n))
            keys = sorted(arg_keys[name])
            print("  argument keys: %s" % (keys or "<none>"))
    else:
        print("n/a (no tool/call events)")

    # ---------- 3 ----------
    print()
    print("== 3. compaction/summary events ==")
    if compactions:
        top = Counter()
        for p in compactions:
            top.update(p.keys())
        print("event count: %d" % len(compactions))
        print("top-level keys (key: event count): %s" % dict(top))
        seq_paths = Counter()
        value_types = Counter()
        elem_types = Counter()
        for p in compactions:
            for path in find_keys(p, ("shadowedSeqs",)):
                v = get(p, path)
                seq_paths[pstr(path)] += 1
                if isinstance(v, list):
                    value_types["list"] += 1
                    for e in v:
                        elem_types[type(e).__name__] += 1
                else:
                    value_types[type(v).__name__] += 1
        if seq_paths:
            for loc, n in seq_paths.most_common():
                print("shadowedSeqs at: %s  [%d occurrences]" % (loc, n))
            print("value type: %s" % dict(value_types))
            print("list element types (total elements): %s" % dict(elem_types))
        else:
            print("shadowedSeqs: not found in these events")
    else:
        print("no events matched type compaction/summary")

    # ---------- 4 ----------
    print()
    print("== 4. tool/result -> tool/call link ==")
    if calls and results:
        def id_sets(events):
            sets = defaultdict(set)
            presence = Counter()
            for p in events:
                for path in find_keys(p, ID_KEYS):
                    v = get(p, path)
                    if isinstance(v, (str, int)) and not isinstance(v, bool):
                        sets[pstr(path)].add(digest(v))
                        presence[pstr(path)] += 1
            return sets, presence
        csets, cpres = id_sets(calls)
        rsets, rpres = id_sets(results)
        pairs = []
        for cp, cset in csets.items():
            for rp, rset in rsets.items():
                inter = len(cset & rset)
                if inter:
                    pairs.append((inter, cp, rp, len(rset)))
        pairs.sort(reverse=True)
        print("tool/call events: %d; tool/result events: %d" % (len(calls), len(results)))
        if pairs:
            for inter, cp, rp, rtotal in pairs[:10]:
                print("tool/call[%s] == tool/result[%s]: %d/%d result ids found in call ids"
                      % (cp, rp, inter, rtotal))
        else:
            print("no id-key overlap found between tool/call and tool/result")
            print("  id keys seen in calls: %s" % dict(cpres))
            print("  id keys seen in results: %s" % dict(rpres))
    else:
        print("n/a (needs both tool/call and tool/result events)")


if __name__ == "__main__":
    main()
