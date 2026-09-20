#!/usr/bin/env python3
"""Validate an exact-token context meter against real DSH sessions.

READ-ONLY over ~/.dsh/sessions. Aggregate-only output; no session content
is ever printed (event keys and counts only).

Meter model per assistant/message step (in seq order):
  observed_prompt = usage.inputTokens + usage.cacheReadTokens (0 if absent)
  base            = observed_prompt of the session's first such step
  surface tokens  = exact token counts (llama-server /tokenize, cached in
                    data/token_counts.csv by tools/tokenize_cache.py) of every
                    non-shadowed surface event
                    (assistant/message, tool/result, user/message) with seq
                    strictly before the step;
                    shadowed = union of shadowedSeqs (or shadowedRange) from
                    compaction/summary and compaction/prune events with
                    seq <= step seq
                    NOTE: no separate summary-text term. dsh commits each
                    checkpoint as a user/message event in the surface, so
                    surface tokens already count it; adding the
                    compaction/summary `summary` text on top would
                    double-count every checkpoint.
  surface_est     = surface tokens (exact, summed)
  metered         = base + surface_est
  reserved        = latest known header.config.maxTokens (request/header)
                    with seq <= step seq (0 if never seen)
  headroom        = contextWindow - metered - reserved, where contextWindow
                    is the latest request/context value with seq <= step seq

"clean" flag: a step is "clean" when its session has no compaction event
with seq before the step, so its surface = every prior event (nothing
shadowed, no checkpoint yet); used only for the sanity line.

Writes data/meter_check.csv with columns:
  uuid,step,observed_prompt,base,surface_est,metered,reserved,headroom
(step = top-level seq of the assistant/message event; headroom blank when
contextWindow is unknown for that step.)
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import zstandard as zstd

ROOT = Path.home() / ".dsh" / "sessions"
OUT = Path(__file__).resolve().parent.parent / "data" / "meter_check.csv"
TOK = Path(__file__).resolve().parent.parent / "data" / "token_counts.csv"

SURFACE_TYPES = ("assistant/message", "tool/result", "user/message")


def read_lines(path: Path):
    if path.name.endswith(".zstd"):
        with open(path, "rb") as f:
            data = zstd.ZstdDecompressor().stream_reader(f).read()
    else:
        data = path.read_bytes()
    return [l for l in data.decode("utf-8", "replace").splitlines() if l.strip()]


def content_len(c):
    """Character length of a tool-result `content` value (str / list / dict)."""
    if c is None:
        return 0
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        n = 0
        for x in c:
            if isinstance(x, str):
                n += len(x)
            elif isinstance(x, dict) and isinstance(x.get("text"), str):
                n += len(x["text"])
        return n
    if isinstance(c, dict) and isinstance(c.get("text"), str):
        return len(c["text"])
    return 0


def part_text_len(p):
    """Character length of the textual payload carried by one content part."""
    if not isinstance(p, dict):
        return 0
    t = p.get("type")
    if t in ("text", "reasoning"):
        v = p.get("text")
        return len(v) if isinstance(v, str) else 0
    if t == "tool-call":
        n = len(str(p.get("name") or ""))
        a = p.get("arguments")
        if isinstance(a, str):
            n += len(a)
        elif a is not None:
            n += len(json.dumps(a, ensure_ascii=False, separators=(",", ":")))
        return n
    if t == "tool-result":
        return content_len(p.get("content"))
    return 0


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
            e["seq"] = i + 1  # keep ordering sane for the rare seq-less line
        events.append(e)
    events.sort(key=lambda e: e["seq"])
    return events


def load_token_counts():
    """(uuid, seq-as-str) -> exact token count, from data/token_counts.csv."""
    counts = {}
    if TOK.exists():
        with open(TOK, newline="") as f:
            rd = csv.reader(f)
            next(rd, None)
            for row in rd:
                if len(row) >= 4 and row[3]:
                    counts[(row[0], row[1])] = int(row[3])
    return counts


def load_reasoning_counts():
    """(uuid, seq-as-str) -> reasoning-only token count (assistant rows)."""
    counts = {}
    if TOK.exists():
        with open(TOK, newline="") as f:
            rd = csv.reader(f)
            next(rd, None)
            for row in rd:
                if len(row) >= 5 and row[4]:
                    counts[(row[0], row[1])] = int(row[4])
    return counts


def sticky_lookup(entries, seq):
    """Latest value at entry seq <= seq; entries = [(seq, value_or_None)]."""
    val = None
    for s, v in entries:
        if s > seq:
            break
        if v is not None:
            val = v
    return val


def process_session(path, tok):
    uuid = path.parent.name
    if uuid.startswith("session-"):
        uuid = uuid[len("session-"):]
    events = load_events(path)

    # shadowing ops (summary + prune), ordered by seq
    comps = []
    for e in events:
        if e["type"] not in ("compaction/summary", "compaction/prune"):
            continue
        d = e.get("data") or {}
        seqs = d.get("shadowedSeqs")
        if not seqs:
            r = d.get("shadowedRange") or {}
            s, en = r.get("start"), r.get("end")
            if s is not None and en is not None:
                seqs = list(range(s, en + 1))
        comps.append((e["seq"], set(seqs or [])))
    first_comp = comps[0][0] if comps else None

    # surface events with exact token counts
    surface = []
    missing = 0
    for e in events:
        t = e["type"]
        if t not in SURFACE_TYPES:
            continue
        n = tok.get((uuid, str(e["seq"])))
        if n is None:
            n = 0
            missing += 1
        surface.append((e["seq"], n))

    headers, ctxs, turn_end = [], [], {}
    for e in events:
        t = e["type"]
        if t == "request/header":
            d = e.get("data") or {}
            cfg = (d.get("header") or {}).get("config") or {}
            headers.append((e["seq"], cfg.get("maxTokens")))
        elif t == "request/context":
            d = e.get("data") or {}
            if d.get("contextWindow") is not None:
                ctxs.append((e["seq"], d["contextWindow"]))
        elif t == "turn/end":
            d = e.get("data") or {}
            turn = d.get("turn")
            if turn is not None:
                reason = d.get("reason")
                kind = reason.get("kind") if isinstance(reason, dict) else reason
                turn_end[turn] = (e["seq"], kind)

    # measured steps: assistant/message with usage
    steps = []
    for e in events:
        if e["type"] != "assistant/message":
            continue
        u = (e.get("data") or {}).get("usage")
        if not isinstance(u, dict) or "inputTokens" not in u:
            continue
        observed = u.get("inputTokens", 0) + (u.get("cacheReadTokens") or 0)
        steps.append((e["seq"], observed, (e.get("data") or {}).get("turn")))
    if not steps:
        return [], uuid, 0

    base = steps[0][1]
    rows = []
    for seq, observed, turn in steps:
        # shadow set as of this step (cumulative: compactions persist)
        shadowed = set()
        for cs, cseqs in comps:
            if cs > seq:
                break
            shadowed |= cseqs
        tokens = sum(t for s, t in surface if s < seq and s not in shadowed)
        reserved = sticky_lookup(headers, seq) or 0
        ctx = sticky_lookup(ctxs, seq)
        maxtok = None
        te = turn_end.get(turn)
        if te and te[0] >= seq and te[1] == "max-tokens":
            maxtok = True
        rows.append(
            {
                "uuid": uuid,
                "step": seq,
                "observed": observed,
                "base": base,
                "tokens": tokens,
                "clean": first_comp is None or seq < first_comp,
                "y": observed - base,
                "reserved": reserved,
                "ctx": ctx,
                "_maxtok": maxtok,
            }
        )
    return rows, uuid, missing


def pct(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def main():
    tok = load_token_counts()
    if not tok:
        print(
            f"error: no token counts in {TOK}; "
            f"run tools/tokenize_cache.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    picks = pick_files()
    all_rows = []
    missing = 0
    for p in picks:
        rows, _, m = process_session(p, tok)
        all_rows.extend(rows)
        missing += m

    # --- clean steps only (no prior compaction in session), for the sanity line ---
    clean = [r for r in all_rows if r["clean"]]

    # finalize rows with exact token counts
    for r in all_rows:
        r["surface_est"] = r["tokens"]
        r["metered"] = r["base"] + r["surface_est"]
        r["error"] = r["metered"] - r["observed"]
        r["headroom"] = (
            r["ctx"] - r["metered"] - r["reserved"]
            if r["ctx"] is not None
            else None
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["uuid", "step", "observed_prompt", "base", "surface_est",
             "metered", "reserved", "headroom"]
        )
        for r in all_rows:
            w.writerow(
                [
                    r["uuid"],
                    r["step"],
                    r["observed"],
                    r["base"],
                    f'{r["surface_est"]:.1f}',
                    f'{r["metered"]:.1f}',
                    r["reserved"],
                    "" if r["headroom"] is None else r["headroom"],
                ]
            )

    cal_errors = [r["error"] for r in clean]
    steps = len(all_rows)
    sessions = len({r["uuid"] for r in all_rows})
    errors = [r["error"] for r in all_rows]
    within = sum(
        1 for r in all_rows if abs(r["error"]) <= 0.10 * r["observed"]
    )
    big = [r for r in all_rows if r["observed"] >= 16000]
    within_big = sum(
        1 for r in big if abs(r["error"]) <= 0.10 * r["observed"]
    )
    neg = [r for r in all_rows if r["headroom"] is not None and r["headroom"] < 0]
    neg_ctx_known = sum(1 for r in all_rows if r["headroom"] is not None)
    maxtok = sum(1 for r in neg if r["_maxtok"])

    print(f"[meter_check] {OUT}")
    print(
        f"token cache: {len(tok)} rows in {TOK.name}; "
        f"{missing} surface events not in cache (counted as 0)"
    )
    print(
        f"1. steps measured: {steps}; sessions covered: {sessions} "
        f"({len(picks)} files read, one plain/v3 pair deduped)"
    )
    print(
        f"2. error (metered - observed_prompt): mean={statistics.fmean(errors):.0f} "
        f"median={statistics.median(errors):.0f} p10={pct(errors, 10):.0f} "
        f"p90={pct(errors, 90):.0f} (linear-interpolation percentiles)"
    )
    print(
        f"3. within 10% of observed (|error| <= 0.10*observed): "
        f"{within}/{steps} = {within / steps:.3f}"
    )
    print(
        f"   of which observed_prompt >= 16000: "
        f"{within_big}/{len(big)} = "
        f"{(within_big / len(big)) if big else float('nan'):.3f}"
    )
    print(
        f"4. headroom < 0: {len(neg)} "
        f"(of {neg_ctx_known} steps with known contextWindow)"
    )
    print(f"5. of those, followed by turn/end max-tokens: {maxtok}")
    how_neg = sorted(-r["headroom"] for r in neg)
    if how_neg:
        print(
            f"6. how negative (-headroom of the {len(how_neg)} negative steps): "
            f"median={pct(how_neg, 50):.0f} p90={pct(how_neg, 90):.0f}"
        )
    else:
        print("6. how negative: no negative-headroom steps")
    buckets = [
        ("<8000", lambda r: r["observed"] < 8000),
        ("8000-15999", lambda r: 8000 <= r["observed"] < 16000),
        ("16000-23999", lambda r: 16000 <= r["observed"] < 24000),
        (">=24000", lambda r: r["observed"] >= 24000),
    ]
    print("7. by observed_prompt bucket:")
    for label, pred in buckets:
        bs = [r for r in all_rows if pred(r)]
        if not bs:
            print(f"   {label}: n=0")
            continue
        be = [r["error"] for r in bs]
        bw = sum(1 for r in bs if abs(r["error"]) <= 0.10 * r["observed"])
        print(
            f"   {label}: n={len(bs)} within10={bw / len(bs):.3f} "
            f"err mean={statistics.fmean(be):+.0f} "
            f"med={statistics.median(be):+.0f} "
            f"p10={pct(be, 10):+.0f} p90={pct(be, 90):+.0f} "
            f"mean-obs={statistics.fmean(r['observed'] for r in bs):.0f}"
        )
    sanity = statistics.median(cal_errors)
    print(
        f"sanity: clean-prefix median error = {sanity:+.0f} "
        f"(limit +/-500: {'PASS' if abs(sanity) <= 500 else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
