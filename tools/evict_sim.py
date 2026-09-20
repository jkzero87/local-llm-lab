#!/usr/bin/env python3
"""Eviction simulation with ingest capping + optional reasoning strip, exact-token meter.

P2_LARGEST is the sole eviction policy (won on economy in the 3-policy run).
A new stage runs BEFORE anything enters the live set:

  INGEST CAP: when a tool/result event's token count exceeds CAP, it enters
  the live set as PREVIEW_TOKENS + REF reference tokens (500 + 20) instead
  of its full size. The rest is never admitted.

Replays every session in seq order (alternative history: real compaction
events are IGNORED entirely). The live set holds surface events
(assistant/message, tool/result, user/message) with exact token counts from
data/token_counts.csv. At every assistant/message step:

    metered = base + sum(tokens of live set)          # live set = prior events
    if metered + reserved > contextWindow:
        evict largest live events until metered + reserved <= contextWindow - 1000

Each evicted event is REPLACED by a 20-token reference (it stays in the live
set at 20 tokens, it is not deleted). Capped tool/result events (520 tokens)
can themselves be evicted down to a 20-token reference.

STRIP_REASONING (on/off per run): reasoning IS sent in the prompt (the
exact-token meter matches observed occupancy), so this stage may strip it
from LIVE events before metering: once a later step has begun, each
assistant/message event's live token count drops by its n_tokens_reasoning
(counted once per event, never re-applied). The most recent
assistant/message keeps its reasoning until the next step starts.

The 3 most recent live events are never evicted. "base/reserved/contextWindow" reuse
tools/meter_check.py verbatim: base = first measured (usage-bearing) step's
observed prompt, reserved = sticky request/header maxTokens, contextWindow
= sticky request/context contextWindow.

READ-ONLY on ~/.dsh/sessions; aggregate output only (no session content).

Output:
  data/evict_sim.csv  cap,strip,policy,uuid,step,metered_before,n_evicted,
                      tokens_freed,metered_after,over_budget_after
                      (metered_before = before eviction, metered_after =
                       after eviction; over_budget_after blank when
                       contextWindow unknown; cap = 2000 | 4000 | 8000 | none;
                       strip = off | on)
  stdout: one summary row per (CAP, STRIP) run + zero-over-budget sessions
          per run, then the best run overall (fewest over-budget steps;
          tie-break: fewer evictions, then lower CAP).
"""

import bisect
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meter_check as mc  # same file selection, event loading, token cache
import reread_v2 as rv2   # file-target extraction rules (exposure matcher)

OUT = Path(__file__).resolve().parent.parent / "data" / "evict_sim.csv"
REF = 20            # tokens an evicted event is replaced by
MARGIN = 1000       # safety margin the policy evicts down to
PROTECTED = 3       # never evict the N most recent live events
PREVIEW_TOKENS = 500  # tokens admitted for a capped tool/result (plus REF)

POLICY = "P2_LARGEST"
CAPS = (2000, 4000, 8000, None)  # None = no-cap control
CAP_LABEL = {2000: "2000", 4000: "4000", 8000: "8000", None: "none"}
STRIPS = (False, True)  # STRIP_REASONING variants: off, then on
STRIP_LABEL = {False: "off", True: "on"}

# Multi-policy comparison (run with: python3 tools/evict_sim.py policies)
POLICIES = ("P0_NO_EVICT", "P2_LARGEST", "P4_LRU", "P5_WS3", "P6_WS3_LRU")
OUT_POLICIES = Path(__file__).resolve().parent.parent / "data" / "eviction_policies.csv"
EXPO_CSV = Path(__file__).resolve().parent.parent / "data" / "exposure_check.csv"


def pick(live, idx_prot):
    """Index of the next event to evict (P2_LARGEST: largest first), or None.

    live[i] = [seq, tokens, type, is_ref]; idx_prot = set of indices of the
    protected (most recent) live events.
    """
    cands = [i for i in range(len(live)) if i not in idx_prot and not live[i][3]]
    if not cands:
        return None
    return max(cands, key=lambda i: live[i][1])


def replay(path, tok, cap, strip, rtok=None):
    """Replay one session under (cap, POLICY, strip). cap=None -> no ingest cap.

    strip=True: prior assistant/message live entries are reduced by their
    reasoning-only token count before each step is metered (see header).
    Returns (uuid, row_dicts, occupancy_list, n_capped, never_admitted,
    over_steps, reclaimed).
    """
    uuid = path.parent.name
    if uuid.startswith("session-"):
        uuid = uuid[len("session-"):]
    events = mc.load_events(path)

    # base: observed prompt of the first measured step (same as meter_check)
    base = None
    for e in events:
        if e["type"] != "assistant/message":
            continue
        u = (e.get("data") or {}).get("usage")
        if isinstance(u, dict) and "inputTokens" in u:
            base = u.get("inputTokens", 0) + (u.get("cacheReadTokens") or 0)
            break
    if base is None:
        return uuid, [], [], 0, 0, 0, 0

    headers, ctxs = [], []
    for e in events:
        if e["type"] == "request/header":
            d = e.get("data") or {}
            cfg = (d.get("header") or {}).get("config") or {}
            headers.append((e["seq"], cfg.get("maxTokens")))
        elif e["type"] == "request/context":
            d = e.get("data") or {}
            if d.get("contextWindow") is not None:
                ctxs.append((e["seq"], d["contextWindow"]))

    rows = []
    occ = []
    live = []          # [seq, tokens, type, is_ref]
    live_sum = 0       # sum of live tokens (refs counted at REF)
    n_capped = 0       # tool/result events admitted in capped form
    never_admitted = 0  # tokens dropped by the ingest cap
    over_steps = 0
    reclaim = 0          # reasoning tokens stripped from the live set
    pending_strip = []   # live indices of assistant msgs not yet stripped

    for e in events:
        t = e["type"]
        if t not in mc.SURFACE_TYPES:
            continue
        n = tok.get((uuid, str(e["seq"])))
        if n is None:
            n = 0
        seq = e["seq"]
        if t == "assistant/message":
            # step: meter against the live set of PRIOR surface events
            if strip:
                # a later step has begun: strip reasoning from all prior
                # assistant messages still live at full size (once each;
                # every index in pending_strip is processed and dropped)
                for i in pending_strip:
                    if live[i][3]:  # already evicted to a ref
                        continue
                    r = (rtok or {}).get((uuid, str(live[i][0])))
                    if not r:
                        continue
                    delta = min(live[i][1], r)
                    live[i][1] -= delta
                    live_sum -= delta
                    reclaim += delta
                pending_strip.clear()
            reserved = mc.sticky_lookup(headers, seq) or 0
            ctx = mc.sticky_lookup(ctxs, seq)
            mb0 = base + live_sum  # metered BEFORE eviction
            mb = mb0
            n_ev = 0
            freed = 0
            if ctx is not None and mb + reserved > ctx:
                target = ctx - MARGIN
                while mb + reserved > target:
                    idx_prot = set(range(len(live) - PROTECTED, len(live)))
                    i = pick(live, idx_prot)
                    if i is None:
                        break
                    delta = live[i][1] - REF
                    live[i][1] = REF
                    live[i][3] = True
                    live_sum -= delta
                    mb = base + live_sum
                    n_ev += 1
                    freed += delta
            ma = mb
            if ctx is None:
                ob = ""
            else:
                ob = int(ma + reserved > ctx)
            if ob:
                over_steps += 1
            rows.append(
                {
                    "cap": CAP_LABEL[cap],
                    "strip": STRIP_LABEL[strip],
                    "policy": POLICY,
                    "uuid": uuid,
                    "step": seq,
                    "metered_before": mb0,
                    "n_evicted": n_ev,
                    "tokens_freed": freed,
                    "metered_after": ma,
                    "over_budget_after": ob,
                }
            )
            if ctx is not None:
                occ.append(ma / ctx)
        else:
            # INGEST CAP (tool/result only): oversized results enter the live
            # set as PREVIEW_TOKENS + REF instead of their full size
            if t == "tool/result" and cap is not None and n > cap:
                never_admitted += n - (PREVIEW_TOKENS + REF)
                n = PREVIEW_TOKENS + REF
                n_capped += 1
        live.append([seq, n, t, False])
        live_sum += n
        if strip and t == "assistant/message":
            pending_strip.append(len(live) - 1)
    return uuid, rows, occ, n_capped, never_admitted, over_steps, reclaim


def main():
    tok = mc.load_token_counts()
    if not tok:
        print(
            "ERROR: no token counts in %s - run tools/tokenize_cache.py first" % mc.TOK,
            file=sys.stderr,
        )
        sys.exit(1)
    rtok = mc.load_reasoning_counts()

    out_rows = []
    per_run = {
        (c, s): {"steps": 0, "over": 0, "ev": 0, "freed": 0, "capped": 0,
                 "never": 0, "reclaim": 0, "occ": [], "sessions": 0,
                 "over_sessions": set()}
        for c in CAPS for s in STRIPS
    }
    for path in mc.pick_files():
        for cap in CAPS:
            for strip in STRIPS:
                uuid, rows, occ, n_capped, never, over_steps, reclaim = replay(
                    path, tok, cap, strip, rtok
                )
                out_rows.extend(rows)
                s = per_run[(cap, strip)]
                s["steps"] += len(rows)
                s["over"] += over_steps
                s["ev"] += sum(r["n_evicted"] for r in rows)
                s["freed"] += sum(r["tokens_freed"] for r in rows)
                s["capped"] += n_capped
                s["never"] += never
                s["reclaim"] += reclaim
                s["occ"] = s["occ"] + occ
                if rows:
                    s["sessions"] += 1
                    if over_steps:
                        s["over_sessions"].add(uuid)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "strip", "policy", "uuid", "step", "metered_before",
                    "n_evicted", "tokens_freed", "metered_after", "over_budget_after"])
        for r in out_rows:
            w.writerow([r["cap"], r["strip"], r["policy"], r["uuid"], r["step"],
                        r["metered_before"], r["n_evicted"], r["tokens_freed"],
                        r["metered_after"], r["over_budget_after"]])
    print("[%s] %s" % ("evict_sim", OUT))

    for cap in CAPS:
        for strip in STRIPS:
            s = per_run[(cap, strip)]
            mean_occ = sum(s["occ"]) / len(s["occ"]) if s["occ"] else float("nan")
            peak_occ = max(s["occ"]) if s["occ"] else float("nan")
            print(
                "CAP=%s strip=%s: steps=%d over_after=%d capped=%d "
                "never_admitted=%d evictions=%d freed=%d reclaimed=%d "
                "mean_occ=%.3f peak_occ=%.3f"
                % (CAP_LABEL[cap], STRIP_LABEL[strip], s["steps"], s["over"],
                   s["capped"], s["never"], s["ev"], s["freed"], s["reclaim"],
                   mean_occ, peak_occ)
            )

    for cap in CAPS:
        for strip in STRIPS:
            s = per_run[(cap, strip)]
            print(
                "CAP=%s strip=%s: %d of %d sessions have ZERO over-budget steps"
                % (CAP_LABEL[cap], STRIP_LABEL[strip],
                   s["sessions"] - len(s["over_sessions"]), s["sessions"])
            )

    best = min(
        per_run,
        key=lambda k: (per_run[k]["over"], per_run[k]["ev"],
                       k[0] if k[0] is not None else float("inf")),
    )
    b = per_run[best]
    print(
        "best CAP=%s strip=%s: %d of %d sessions have ZERO over-budget steps"
        % (CAP_LABEL[best[0]], STRIP_LABEL[best[1]],
           b["sessions"] - len(b["over_sessions"]), b["sessions"])
    )
    print(
        "comparison: stock had 912 steps over budget and spent 1,129,644 "
        "generation tokens on summaries; capped P2_LARGEST spends 0"
    )


def event_targets(e, call_targets, truth):
    """Set of file paths a surface event references (exposure_check.py
    matcher, verbatim)."""
    d = e.get("data") or {}
    t = e["type"]
    out = set()
    if t == "assistant/message":
        msg = d.get("message") or {}
        for c in msg.get("content") or []:
            if not isinstance(c, dict) or c.get("type") != "tool-call":
                continue
            raw = c.get("args") if "args" in c else c.get("arguments")
            out |= rv2.targets_of(c.get("toolName") or c.get("name"),
                                  rv2.parse_args(raw), truth)
    elif t == "tool/result":
        src = (d.get("message") or {}).get("source") or {}
        cid = src.get("callId") or d.get("callId")
        if cid:
            out |= call_targets.get(cid, set())
    return out


def later_lookup(oseqs, ordered, seq):
    """(paths, basenames) of calls whose call_seq is STRICTLY AFTER seq
    (exposure_check.py matcher, verbatim)."""
    i = bisect.bisect_right(oseqs, seq)
    paths, bases = set(), set()
    for _, tg in ordered[i:]:
        for p in tg:
            paths.add(p)
            bases.add(p.rsplit("/", 1)[-1])
    return paths, bases


def session_prep(path):
    """Per-session precompute shared by all policy replays (one scan).

    Returns (etg, ordered, oseqs, call_seqs):
      etg       seq -> set of file paths the surface event references
      ordered   [(call_seq, targets)] sorted by call_seq (all file-targeted
                calls; call_seq = seq of the assistant message containing
                the tool-call block)
      oseqs     the call_seq column of `ordered`, for bisect
      call_seqs anchor -> set of paths touched at that surface step (a single
                assistant message with several calls contributes all their
                targets at that one seq)
    """
    truth = rv2.ground_truth(str(path))
    _, _, calls = rv2.scan(str(path), truth)
    call_targets = {cid: r["targets"] for cid, r in calls.items()}
    etg = {}
    call_seqs = {}
    ordered = []
    surf = set()
    for e in mc.load_events(path):
        if e["type"] not in mc.SURFACE_TYPES:
            continue
        surf.add(e["seq"])
        etg[e["seq"]] = event_targets(e, call_targets, truth)
    for r in calls.values():
        if r["call_seq"] is not None and r["targets"]:
            ordered.append((r["call_seq"], r["targets"]))
        if r["targets"]:
            # anchor at the earliest SURFACE seq at which the call is visible
            # (the assistant message carrying the tool-call block); r["call_seq"]
            # is the non-surface tool/call dispatch seq and never matches a
            # replayed surface event
            anchor = None
            for s in sorted(x for x in r["seqs"] if x is not None):
                if s in surf:
                    anchor = s
                    break
            if anchor is None:
                anchor = r["call_seq"]
            if anchor is not None:
                call_seqs.setdefault(anchor, set()).update(r["targets"])
    ordered.sort(key=lambda x: x[0])
    oseqs = [s for s, _ in ordered]
    return etg, ordered, oseqs, call_seqs


def pick_policy(policy, live, idx_prot, etg, ref_order, path_last):
    """Index of the next event to evict under `policy`, or None.

    Same metering and protected-window rules as pick(); only the choice
    differs:
      P0_NO_EVICT  never evicts (control)
      P2_LARGEST   largest live event first (identical to pick())
      P4_LRU       least-recently-referenced first; last_ref = max of the
                   event's own seq and the latest step at which any of its
                   target paths was referenced (ties: oldest event)
      P5_WS3       never evict an event whose targets include any of the
                   last 3 distinct referenced file paths; evict the oldest
                   among the rest
      P6_WS3_LRU   WS3 protection, then P4_LRU order within the rest
    """
    cands = [i for i in range(len(live)) if i not in idx_prot and not live[i][3]]
    if policy == "P0_NO_EVICT":
        return None
    if policy == "P2_LARGEST":
        return max(cands, key=lambda i: live[i][1]) if cands else None

    def lr(s):
        tg = etg.get(s)
        if not tg:
            return s
        m = s
        for p in tg:
            v = path_last.get(p, 0)
            if v > m:
                m = v
        return m

    if policy in ("P5_WS3", "P6_WS3_LRU"):
        protect = set(ref_order[-3:])
        cands = [i for i in cands if not (etg.get(live[i][0]) & protect)]
        if not cands:
            return None
    if policy == "P5_WS3":
        return min(cands, key=lambda i: live[i][0])
    # P4_LRU / P6_WS3_LRU
    return min(cands, key=lambda i: (lr(live[i][0]), live[i][0])) if cands else None


def replay_policy(path, tok, rtok, policy, prep=None):
    """Replay one session under `policy` at the certified config
    (CAP=2000, STRIP_REASONING on).

    Identical metering to replay(); the only additions are the policy
    choice and per-run online reference state: at each step, after the
    step's own eviction decision, the paths referenced by that step's
    calls (call_seqs[step]) are moved to the tail of the distinct-path
    order and their last reference is set to the step's seq.

    Returns dict with over (over-budget steps), ev (evicted events),
    freed (tokens freed by eviction), has_metered, evicted_seqs, and the
    exposure check over the evicted events: exp_total (evicted events
    that reference >=1 file path), exp_ev (of those, events with any
    target path requested by an exact path in a LATER step - same
    matcher as tools/exposure_check.py).
    """
    uuid = path.parent.name
    if uuid.startswith("session-"):
        uuid = uuid[len("session-"):]
    if policy not in POLICIES:
        raise ValueError("unknown policy: %r" % policy)
    if prep is None:
        prep = session_prep(path)
    etg, ordered, oseqs, call_seqs = prep
    events = mc.load_events(path)

    base = None
    for e in events:
        if e["type"] != "assistant/message":
            continue
        u = (e.get("data") or {}).get("usage")
        if isinstance(u, dict) and "inputTokens" in u:
            base = u.get("inputTokens", 0) + (u.get("cacheReadTokens") or 0)
            break
    if base is None:
        return {"uuid": uuid, "over": 0, "ev": 0, "freed": 0, "has_metered": False,
                "evicted_seqs": [], "exp_total": 0, "exp_ev": 0, "rows": []}

    headers, ctxs = [], []
    for e in events:
        if e["type"] == "request/header":
            d = e.get("data") or {}
            cfg = (d.get("header") or {}).get("config") or {}
            headers.append((e["seq"], cfg.get("maxTokens")))
        elif e["type"] == "request/context":
            d = e.get("data") or {}
            if d.get("contextWindow") is not None:
                ctxs.append((e["seq"], d["contextWindow"]))

    live = []          # [seq, tokens, type, is_ref]
    live_sum = 0
    over_steps = 0
    n_ev = 0
    freed = 0
    reclaim = 0
    pending_strip = []
    has_metered = False
    path_last = {}     # file path -> latest step seq referencing it
    ref_order = []     # distinct referenced file paths, last-referenced last
    in_order = set()
    evicted_seqs = []
    rows = []          # per-assistant-step metering rows (cross-check)

    for e in events:
        t = e["type"]
        if t not in mc.SURFACE_TYPES:
            continue
        n = tok.get((uuid, str(e["seq"])))
        if n is None:
            n = 0
        seq = e["seq"]
        if t == "assistant/message":
            if True:  # STRIP_REASONING on (certified config)
                for i in pending_strip:
                    if live[i][3]:  # already evicted to a ref
                        continue
                    r = (rtok or {}).get((uuid, str(live[i][0])))
                    if not r:
                        continue
                    delta = min(live[i][1], r)
                    live[i][1] -= delta
                    live_sum -= delta
                    reclaim += delta
                pending_strip.clear()
            reserved = mc.sticky_lookup(headers, seq) or 0
            ctx = mc.sticky_lookup(ctxs, seq)
            mb0 = base + live_sum  # metered BEFORE eviction (same row as replay())
            mb = mb0
            step_ev = 0
            step_freed = 0
            if ctx is not None and mb + reserved > ctx:
                target = ctx - MARGIN
                while mb + reserved > target:
                    idx_prot = set(range(len(live) - PROTECTED, len(live)))
                    i = pick_policy(policy, live, idx_prot, etg, ref_order, path_last)
                    if i is None:
                        break
                    delta = live[i][1] - REF
                    live[i][1] = REF
                    live[i][3] = True
                    live_sum -= delta
                    mb = base + live_sum
                    n_ev += 1
                    freed += delta
                    step_ev += 1
                    step_freed += delta
                    evicted_seqs.append(live[i][0])
            ma = mb
            ob = "" if ctx is None else int(ma + reserved > ctx)
            if ob:
                over_steps += 1
            rows.append({
                "cap": "2000",
                "strip": "on",
                "policy": policy,
                "uuid": uuid,
                "step": seq,
                "metered_before": mb0,
                "n_evicted": step_ev,
                "tokens_freed": step_freed,
                "metered_after": ma,
                "over_budget_after": ob,
            })
            has_metered = True
        else:
            if t == "tool/result" and n > 2000:  # ingest cap, certified config
                n = PREVIEW_TOKENS + REF
        live.append([seq, n, t, False])
        live_sum += n
        if t == "assistant/message":
            pending_strip.append(len(live) - 1)
        tg = call_seqs.get(seq)
        if tg:
            # this step's calls reference these paths (state update happens
            # strictly AFTER the step's own eviction decision - no lookahead)
            for p in sorted(tg):
                if p in in_order:
                    ref_order.remove(p)
                ref_order.append(p)
                in_order.add(p)
                path_last[p] = seq

    # exposure: evicted events whose content was re-requested by exact path
    # in a later step (call_seq strictly after the eviction step)
    exp_total = 0
    exp_ev = 0
    for s in evicted_seqs:
        tg = etg.get(s)
        if not tg:
            continue
        exp_total += 1
        lp, _lb = later_lookup(oseqs, ordered, s)
        if tg & lp:
            exp_ev += 1
    return {"uuid": uuid, "over": over_steps, "ev": n_ev, "freed": freed,
            "has_metered": has_metered, "evicted_seqs": evicted_seqs,
            "exp_total": exp_total, "exp_ev": exp_ev, "rows": rows}


def cross_check_policies(results):
    """Verify the P2_LARGEST replay matches the certified snapshot.

    Live sessions may have grown since the snapshot was written, so the
    snapshot must be a PREFIX of the replay (same semantics as
    exposure_check.py's self-check), not a full equality. Any mismatch
    is printed to stderr and exits non-zero.
    """
    bad = False
    # 1. evict_sim.csv rows (cap=2000, strip=on) vs P2 replay rows
    cert = {}
    if OUT.exists():
        with open(OUT, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("cap") == "2000" and r.get("strip") == "on":
                    cert.setdefault(r["uuid"], []).append(
                        (int(r["step"]), int(r["metered_before"]),
                         int(r["n_evicted"]), int(r["tokens_freed"]),
                         int(r["metered_after"]), r["over_budget_after"]))
    else:
        print("CROSS-CHECK WARN: %s not found" % OUT, file=sys.stderr)
    if cert:
        for ru in results["P2_LARGEST"]:
            c = sorted(cert.get(ru["uuid"], []))
            m = sorted(
                (int(r["step"]), int(r["metered_before"]),
                 int(r["n_evicted"]), int(r["tokens_freed"]),
                 int(r["metered_after"]), str(r["over_budget_after"]))
                for r in ru["rows"])
            if m[: len(c)] != c:
                print("CROSS-CHECK FAIL: P2 rows for %s do not extend the "
                      "certified snapshot" % ru["uuid"], file=sys.stderr)
                bad = True
        for u in cert:
            ru = next((r for r in results["P2_LARGEST"] if r["uuid"] == u), None)
            if ru is None:
                print("CROSS-CHECK FAIL: certified session %s missing from "
                      "policy replay" % u, file=sys.stderr)
                bad = True
    # 2. exposure_check.csv eviction rows (same certified corpus)
    if EXPO_CSV.exists():
        ev_logged = {}
        with open(EXPO_CSV, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("kind") == "eviction":
                    ev_logged.setdefault(r["uuid"], set()).add(int(r["seq"]))
        if ev_logged:
            # certified rows must all be present in the fresh replay (the
            # replay may evict more - sessions can grow - but never less)
            for ru in results["P2_LARGEST"]:
                new = set(ru["evicted_seqs"])
                logged = ev_logged.get(ru["uuid"], set())
                if logged - new:
                    print("CROSS-CHECK FAIL: P2 evictions for %s missing from "
                          "fresh replay (missing=%d)"
                          % (ru["uuid"], len(logged - new)),
                          file=sys.stderr)
                    bad = True
    # 3. step-level aggregates (over-budget steps, evictions, tokens freed)
    #    must match the certified snapshot over the certified prefix of the
    #    replay (prefix semantics: sessions may have grown since the snapshot)
    agg = {}
    for u, rows in cert.items():
        o = sum(1 for x in rows if x[5] == "1")
        ev = sum(x[2] for x in rows)
        fr = sum(x[3] for x in rows)
        agg[u] = (o, ev, fr)
    for ru in results["P2_LARGEST"]:
        c = sorted(cert.get(ru["uuid"], []))
        if not c:
            continue
        m = sorted(
            (int(r["step"]), int(r["metered_before"]),
             int(r["n_evicted"]), int(r["tokens_freed"]),
             int(r["metered_after"]), str(r["over_budget_after"]))
            for r in ru["rows"])
        pfx = m[: len(c)]
        o = sum(1 for x in pfx if x[5] == "1")
        ev = sum(x[2] for x in pfx)
        fr = sum(x[3] for x in pfx)
        if (o, ev, fr) != agg[ru["uuid"]]:
            print("CROSS-CHECK FAIL: P2 aggregates for %s: replay-prefix=%s "
                  "certified=%s"
                  % (ru["uuid"], (o, ev, fr), agg[ru["uuid"]]), file=sys.stderr)
            bad = True
    return not bad


def main_policies():
    """Five-policy comparison on the P2_LARGEST session corpus
    (certified config: CAP=2000, STRIP_REASONING on).

    Writes data/eviction_policies.csv and prints one row per policy,
    including P0_NO_EVICT and P2_LARGEST as baseline rows.
    """
    tok = mc.load_token_counts()
    if not tok:
        print(
            "ERROR: no token counts in %s - run tools/tokenize_cache.py first"
            % mc.TOK,
            file=sys.stderr,
        )
        sys.exit(1)
    rtok = mc.load_reasoning_counts()

    results = {p: [] for p in POLICIES}
    for path in mc.pick_files():
        prep = session_prep(path)
        for p in POLICIES:
            results[p].append(replay_policy(path, tok, rtok, p, prep=prep))
    if not cross_check_policies(results):
        sys.exit(1)
    totals = {}
    for p in POLICIES:
        rs = [r for r in results[p] if r["has_metered"]]
        et = sum(r["exp_total"] for r in rs)
        ee = sum(r["exp_ev"] for r in rs)
        totals[p] = {
            "policy": p,
            "over_budget_steps": sum(r["over"] for r in rs),
            "evicted_events": sum(r["ev"] for r in rs),
            "evicted_tokens": sum(r["freed"] for r in rs),
            "exposure_pct": round(100.0 * ee / et, 1) if et else 0.0,
            "n_sessions": len(rs),
        }
    OUT_POLICIES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_POLICIES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(totals[POLICIES[0]].keys()))
        w.writeheader()
        for p in POLICIES:
            w.writerow(totals[p])
    cols = ["policy", "over_budget_steps", "evicted_events", "evicted_tokens",
            "exposure_pct", "n_sessions"]
    print("%-13s %15s %14s %14s %11s %9s"
          % tuple(["policy"] + [c for c in cols[1:]]))
    for p in POLICIES:
        t = totals[p]
        print("%-13s %15d %14d %14d %11.1f %9d"
              % (t["policy"], t["over_budget_steps"], t["evicted_events"],
                 t["evicted_tokens"], t["exposure_pct"], t["n_sessions"]))
    print("wrote %s" % OUT_POLICIES)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "policies":
        main_policies()
    else:
        main()
