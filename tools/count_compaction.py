import json, sys
from collections import Counter
events, trunc = Counter(), Counter()
bad = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except Exception:
        bad += 1
        continue
    t = None
    for k in ("type", "event", "kind", "name"):
        v = o.get(k)
        if isinstance(v, str):
            t = v
            break
    if t and t.startswith("compaction"):
        events[t] += 1
        s = json.dumps(o)
        if '"truncated":true' in s or '"truncated": true' in s or '"truncated":1' in s:
            trunc[t] += 1
print("events:", dict(events))
print("truncated:", dict(trunc))
print("unparsed lines:", bad)
