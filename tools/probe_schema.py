#!/usr/bin/env python3
"""Probe dsh session log schema: prints only key names and type values, no content."""
import json
import sys
import zstandard as zstd

files = sys.argv[1:]


def iter_lines(path, chunk=1 << 20):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        r = dctx.stream_reader(f, closefd=True)
        buf = b""
        while True:
            data = r.read(chunk)
            if not data:
                break
            buf += data
            # keep a trailing partial line
            idx = buf.rfind(b"\n")
            if idx == -1:
                continue
            lines, buf = buf[: idx + 1], buf[idx + 1 :]
            for ln in lines.split(b"\n"):
                yield ln
        if buf:
            yield buf


for p in files:
    print(f"=== {p}")
    types = {}
    first = None
    for raw in iter_lines(p):
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        t = ev.get("type")
        types[t] = types.get(t, 0) + 1
        if first is None:
            first = ev
            print("first-event keys:", sorted(ev.keys()))
            if "usage" in ev:
                print("  usage keys:", sorted(ev["usage"].keys()))
            # show nested structure of first 3 values that are dicts, key names only
            for k, v in ev.items():
                if isinstance(v, dict) and k not in ("usage",):
                    print(f"  nested '{k}' keys:", sorted(v.keys()))
        if sum(types.values()) > 4000:
            break
    print("event type counts (sampled up to 4000):", types)
    # ts field candidates from first event
    if first:
        for k in first.keys():
            if "ts" in k.lower() or "time" in k.lower() or "date" in k.lower():
                print("ts candidate:", k, "=", first[k])
