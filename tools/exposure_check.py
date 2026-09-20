#!/usr/bin/env python3
"""Exposure check for the CERTIFIED config: CAP=2000, P2_LARGEST, STRIP_REASONING on.

Replays every session with tools/evict_sim.py's replay() logic verbatim
(same base derivation, sticky lookups, protected last 3, margin 1000,
20-token refs, 500+20 capped admission), but with logging hooks at the
three mutation sites:

  * EVICTION      - a live event is replaced by its 20-token reference:
                    tokens_removed = size - 20 (full content loss).
  * INGEST_CAP    - a tool/result larger than CAP is admitted at 520:
                    tokens_removed = size - 520 (never admitted, partial
                    loss of everything beyond the 500-token preview).
  * STRIP_RECLAIM - reasoning tokens stripped from a live assistant message
                    once the next step starts (tokens_removed = stripped).

Each logged event also carries its FILE TARGETS extracted with
tools/reread_v2.py's rules (PATH_TOOLS/bash-token extraction + ground-truth
pre-pass), so we can split exposure into file-touching vs non-file and
cross-check recoverability.

Self-check: the replay must reproduce data/evict_sim.csv row-for-row for
cap=2000, strip=on (step, metered_before, n_evicted, tokens_freed,
metered_after, over_budget_after on every step of every session). Row-for-row
equality of the metered state at every step is the proof that the replay is
the certified run, so the logged exposure events are exactly the certified
evictions and caps (and the capped / never-admitted / reclaimed aggregates
are the certified ones, even though the CSV stores no column for them).

Re-read cross-check (item 2): a dropped file target is "recoverable" if the
same path is requested again later in the same session (reread_v2 semantics,
call_seq strictly after the exposure event). Exact path match = lower bound,
basename match = upper bound.

READ-ONLY on ~/.dsh/sessions; aggregate output only (no session content).

Output:
  data/exposure_check.csv  kind,uuid,seq,event_type,tokens_removed,
                           n_targets,file_targets,is_file_touching
                           (kind = eviction | ingest_cap | strip_reclaim;
                            file_targets '|'-joined, paths only)
  stdout: 5-item report + self-check verdict.
"""

import bisect
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meter_check as mc   # file selection, event loading, token caches
import reread_v2 as rv2    # file-target extraction rules (verbatim reuse)
import evict_sim as es     # pick() and the certified constants

CERT_CAP = 2000
CERT_STRIP = True
REF = es.REF
MARGIN = es.MARGIN
PROTECTED = es.PROTECTED
PREVIEW_TOKENS = es.PREVIEW_TOKENS
CAP_ADMIT = PREVIEW_TOKENS + REF  # 520

OUT = Path(__file__).resolve().parent.parent / "data" / "exposure_check.csv"
CERT_CSV = Path(__file__).resolve().parent.parent / "data" / "evict_sim.csv"


def event_targets(e, call_targets, truth):
    """File targets an event refers to (reread_v2 extraction rules)."""
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


def replay_logged(path, tok, rtok):
    """Verbatim copy of evict_sim.replay() for (CAP=2000, strip=True) with
    logging hooks. Returns (uuid, rows, occ, n_capped, never_admitted,
    over_steps, reclaimed, log) where log is a list of
    (kind, seq, event_type, tokens_removed, file_targets-set).
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
        return uuid, [], [], 0, 0, 0, 0, []

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

    # file targets per surface event (reread_v2 rules, ground-truth pre-pass)
    truth = rv2.ground_truth(str(path))
    _, _, calls = rv2.scan(str(path), truth)
    call_targets = {cid: r["targets"] for cid, r in calls.items()}
    etg = {}
    for e in events:
        if e["type"] in mc.SURFACE_TYPES:
            etg[e["seq"]] = event_targets(e, call_targets, truth)

    # later-call index for the re-read cross-check (reread_v2 semantics)
    ordered = sorted(
        ((r["call_seq"], r["targets"]) for r in calls.values()
         if r["call_seq"] is not None and r["targets"]),
        key=lambda x: x[0],
    )
    oseqs = [s for s, _ in ordered]

    rows = []
    occ = []
    live = []          # [seq, tokens, type, is_ref]
    live_sum = 0       # sum of live tokens (refs counted at REF)
    n_capped = 0       # tool/result events admitted in capped form
    never_admitted = 0  # tokens dropped by the ingest cap
    over_steps = 0
    reclaim = 0          # reasoning tokens stripped from the live set
    pending_strip = []   # live indices of assistant msgs not yet stripped
    log = []             # (kind, seq, event_type, tokens_removed, targets)

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
            if CERT_STRIP:
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
                    log.append(("strip_reclaim", live[i][0], "assistant/message",
                                delta, etg.get(live[i][0], set())))
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
                    i = es.pick(live, idx_prot)
                    if i is None:
                        break
                    delta = live[i][1] - REF
                    live[i][1] = REF
                    live[i][3] = True
                    live_sum -= delta
                    mb = base + live_sum
                    n_ev += 1
                    freed += delta
                    log.append(("eviction", live[i][0], live[i][2], delta,
                                etg.get(live[i][0], set())))
            ma = mb
            if ctx is None:
                ob = ""
            else:
                ob = int(ma + reserved > ctx)
            if ob:
                over_steps += 1
            rows.append(
                {
                    "cap": es.CAP_LABEL[CERT_CAP],
                    "strip": es.STRIP_LABEL[CERT_STRIP],
                    "policy": es.POLICY,
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
            if t == "tool/result" and CERT_CAP is not None and n > CERT_CAP:
                never_admitted += n - CAP_ADMIT
                log.append(("ingest_cap", seq, "tool/result",
                            n - CAP_ADMIT, etg.get(seq, set())))
                n = CAP_ADMIT
                n_capped += 1
        live.append([seq, n, t, False])
        live_sum += n
        if CERT_STRIP and t == "assistant/message":
            pending_strip.append(len(live) - 1)
    return uuid, rows, occ, n_capped, never_admitted, over_steps, reclaim, log


def later_lookup(oseqs, ordered, seq):
    """Sets of exact paths / basenames requested in calls AFTER `seq`."""
    i = bisect.bisect_right(oseqs, seq)
    paths, bases = set(), set()
    for _, targets in ordered[i:]:
        for p in targets:
            paths.add(p)
            bases.add(p.rsplit("/", 1)[-1])
    return paths, bases


def load_certified():
    """certified rows from data/evict_sim.csv (cap=2000, strip=on)."""
    per = defaultdict(list)
    agg = {"steps": 0, "ev": 0, "freed": 0, "over": 0, "sessions": set()}
    with open(CERT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["cap"] != "2000" or r["strip"] != "on":
                continue
            ob = "" if r["over_budget_after"] == "" else int(r["over_budget_after"])
            key = (r["uuid"], int(r["step"]), int(r["metered_before"]),
                   int(r["n_evicted"]), int(r["tokens_freed"]),
                   int(r["metered_after"]), ob)
            per[r["uuid"]].append(key)
            agg["steps"] += 1
            agg["ev"] += int(r["n_evicted"])
            agg["freed"] += int(r["tokens_freed"])
            agg["over"] += 1 if r["over_budget_after"] == "1" else 0
            agg["sessions"].add(r["uuid"])
    return per, agg


def main():
    tok = mc.load_token_counts()
    rtok = mc.load_reasoning_counts()
    files = mc.pick_files()

    log_rows = []       # (kind, uuid, seq, event_type, tokens_removed, targets)
    per_session = {}    # uuid -> per-kind (events, tokens, file_events, file_tokens)
    rer_pairs = 0       # (exposure event, target) pairs among evictions+caps
    rer_exact = 0
    rer_base = 0

    replay_rows = defaultdict(list)
    agg = {"n_capped": 0, "never": 0, "reclaim": 0, "over": 0, "steps": 0}
    sessions_replayed = 0

    for path in files:
        uuid, rows, occ, n_capped, never, over_steps, reclaimed, log = \
            replay_logged(path, tok, rtok)
        sessions_replayed += 1
        for r in rows:
            replay_rows[uuid].append(
                (uuid, r["step"], r["metered_before"], r["n_evicted"],
                 r["tokens_freed"], r["metered_after"],
                 r["over_budget_after"]))
        agg["n_capped"] += n_capped
        agg["never"] += never
        agg["reclaim"] += reclaimed
        agg["over"] += over_steps
        agg["steps"] += len(rows)

        # re-read index for this session
        truth = rv2.ground_truth(str(path))
        _, _, calls = rv2.scan(str(path), truth)
        ordered = sorted(
            ((r["call_seq"], r["targets"]) for r in calls.values()
             if r["call_seq"] is not None and r["targets"]),
            key=lambda x: x[0],
        )
        oseqs = [s for s, _ in ordered]

        for kind, seq, etype, delta, targets in log:
            is_ft = 1 if targets else 0
            log_rows.append((kind, uuid, seq, etype, delta, sorted(targets),
                             is_ft))
            s = per_session.setdefault(
                uuid, defaultdict(lambda: [0, 0, 0, 0]))  # ev, tok, f_ev, f_tok
            s[kind][0] += 1
            s[kind][1] += delta
            if is_ft:
                s[kind][2] += 1
                s[kind][3] += delta
            if kind in ("eviction", "ingest_cap") and targets:
                later_paths, later_bases = later_lookup(oseqs, ordered, seq)
                for p in targets:
                    rer_pairs += 1
                    if p in later_paths:
                        rer_exact += 1
                    if p.rsplit("/", 1)[-1] in later_bases:
                        rer_base += 1

    # ---------- self-check against data/evict_sim.csv ----------
    # The certified CSV is a snapshot: live sessions may have grown since.
    # Verify per session that the certified rows are a step-for-step PREFIX
    # of the replayed rows (identical values, then only new rows allowed).
    cert, cert_agg = load_certified()
    bad = 0
    grown = []
    for uuid in sorted(set(cert) | set(replay_rows)):
        c = sorted(cert.get(uuid, []), key=lambda t: t[1])
        m = sorted(replay_rows.get(uuid, []), key=lambda t: t[1])
        if m[:len(c)] != c:
            bad += 1
            if bad <= 3:
                print("SELF-CHECK MISMATCH %s: certified=%d replayed=%d"
                      % (uuid, len(c), len(m)), file=sys.stderr)
        elif len(m) > len(c):
            grown.append((uuid, len(m) - len(c)))
    # aggregates: certified must be a lower bound (growth only adds rows)
    rep_ev = sum(x[3] for v in replay_rows.values() for x in v)
    rep_freed = sum(x[4] for v in replay_rows.values() for x in v)
    steps_ok = agg["steps"] >= cert_agg["steps"]
    ev_ok = rep_ev >= cert_agg["ev"]
    ev_ok = cert_agg["ev"] == sum(
        x[3] for v in replay_rows.values() for x in v)
    freed_ok = rep_freed >= cert_agg["freed"]
    over_ok = agg["over"] >= cert_agg["over"]
    rows_ok = bad == 0
    self_ok = rows_ok and steps_ok and ev_ok and freed_ok and over_ok

    # ---------- write aggregate CSV ----------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "uuid", "seq", "event_type", "tokens_removed",
                    "n_targets", "file_targets", "is_file_touching"])
        for kind, uuid, seq, etype, delta, targets, is_ft in log_rows:
            w.writerow([kind, uuid, seq, etype, delta, len(targets),
                        "|".join(targets), is_ft])

    # ---------- report ----------
    def kind_stats(kinds):
        ev_n = sum(1 for r in log_rows if r[0] in kinds)
        ev_t = sum(r[4] for r in log_rows if r[0] in kinds)
        f_n = sum(1 for r in log_rows if r[0] in kinds and r[6])
        f_t = sum(r[4] for r in log_rows if r[0] in kinds and r[6])
        return ev_n, ev_t, f_n, f_t

    ev_n, ev_t, ev_fn, ev_ft = kind_stats(("eviction",))
    cap_n, cap_t, cap_fn, cap_ft = kind_stats(("ingest_cap",))
    st_n, st_t, st_fn, st_ft = kind_stats(("strip_reclaim",))
    all_n = ev_n + cap_n + st_n
    all_t = ev_t + cap_t + st_t
    all_fn = ev_fn + cap_fn + st_fn
    all_ft = ev_ft + cap_ft + st_ft

    exposed_sessions = {u for u, s in per_session.items()
                        if any(s[k][0] for k in
                               ("eviction", "ingest_cap", "strip_reclaim"))}
    sess_any_ev = sum(1 for u in exposed_sessions
                      if per_session[u]["eviction"][0]
                      or per_session[u]["ingest_cap"][0])

    print("=" * 72)
    print("EXPOSURE CHECK - certified config CAP=2000, P2_LARGEST, "
          "STRIP_REASONING on")
    print("=" * 72)
    print("sessions replayed: %d (files selected by meter_check.pick_files)"
          % sessions_replayed)
    print()
    print("SELF-CHECK vs data/evict_sim.csv (cap=2000, strip=on, snapshot):")
    print("  certified rows are a step-for-step PREFIX of replay: %s"
          % ("PASS" if rows_ok else "FAIL"), end="")
    print(" (%d certified rows, %d mismatching sessions)"
          % (cert_agg["steps"], bad))
    if grown:
        tot_growth = sum(n for _, n in grown)
        print("  sessions grown since certification: %d (+%d rows total)"
              % (len(grown), tot_growth))
        for uuid, n in grown[:5]:
            print("    %s +%-4d rows" % (uuid, n))
    print("  steps    %8d  certified %d  %s" % (agg["steps"], cert_agg["steps"],
                                                "PASS" if steps_ok else "FAIL"))
    print("  evictions %8d  certified %d  %s" % (
        rep_ev, cert_agg["ev"], "PASS" if ev_ok else "FAIL"))
    print("  freed    %10d  certified %d  %s" % (
        rep_freed, cert_agg["freed"], "PASS" if freed_ok else "FAIL"))
    print("  over-budget steps %4d  certified %d  %s" % (
        agg["over"], cert_agg["over"], "PASS" if over_ok else "FAIL"))
    print("  sessions       %4d  certified %d" % (
        len(set(replay_rows)), len(cert_agg["sessions"])))
    print("  => certified snapshot reproduced as exact prefix: %s"
          % ("YES" if self_ok else "NO - report below is UNRELIABLE"))
    print()
    print("1) EXPOSURE COUNTS (real content removed from live context)")
    print("   evictions    %6d events   %12s tokens  (full loss -> 20-token ref)"
          % (ev_n, format(ev_t, ",")))
    print("      file-touching  %6d events   %12s tokens" % (ev_fn, format(ev_ft, ",")))
    print("      non-file       %6d events   %12s tokens" % (ev_n - ev_fn, format(ev_t - ev_ft, ",")))
    print("   ingest caps  %6d events   %12s tokens  (never admitted; 500-preview kept)"
          % (cap_n, format(cap_t, ",")))
    print("      file-touching  %6d events   %12s tokens" % (cap_fn, format(cap_ft, ",")))
    print("      non-file       %6d events   %12s tokens" % (cap_n - cap_fn, format(cap_t - cap_ft, ",")))
    print("   strip-reclaim %6d events   %12s tokens  (reasoning stripped post-hoc)"
          % (st_n, format(st_t, ",")))
    print("      file-touching  %6d events   %12s tokens" % (st_fn, format(st_ft, ",")))
    print("      non-file       %6d events   %12s tokens" % (st_n - st_fn, format(st_t - st_ft, ",")))
    print("   TOTAL        %6d events   %12s tokens" % (all_n, format(all_t, ",")))
    print()
    print("2) RE-READ CROSS-CHECK (of eviction+cap file targets, reread_v2 rules)")
    print("   dropped target pairs (event, path): %d" % rer_pairs)
    print("   re-read later by EXACT path (lower bound): %d  (%.1f%%)"
          % (rer_exact, 100.0 * rer_exact / rer_pairs if rer_pairs else 0.0))
    print("   re-read later by BASENAME (upper bound): %d  (%.1f%%)"
          % (rer_base, 100.0 * rer_base / rer_pairs if rer_pairs else 0.0))
    print()
    print("3) STRIP-RECLAIM vs FILE EXPOSURE")
    print("   strip-reclaimed tokens: %s total" % format(st_t, ","))
    print("     from file-touching assistant messages: %s (%.1f%%)"
          % (format(st_ft, ","), 100.0 * st_ft / st_t if st_t else 0.0))
    print("     from non-file assistant messages:      %s (%.1f%%)"
          % (format(st_t - st_ft, ","), 100.0 * (st_t - st_ft) / st_t if st_t else 0.0))
    print()
    print("4) PER-SESSION EXPOSURE")
    print("   sessions with any exposure event: %d / %d (%.1f%%)"
          % (len(exposed_sessions), sessions_replayed,
             100.0 * len(exposed_sessions) / sessions_replayed
             if sessions_replayed else 0.0))
    print("   sessions with an eviction or ingest cap: %d" % sess_any_ev)
    print("   file-touching share of ALL exposure tokens: %.1f%% (%s of %s)"
          % (100.0 * all_ft / all_t if all_t else 0.0,
             format(all_ft, ","), format(all_t, ",")))
    print("   file-touching share of eviction+cap tokens: %.1f%%"
          % (100.0 * (ev_ft + cap_ft) / (ev_t + cap_t)
             if (ev_t + cap_t) else 0.0))
    print()
    print("5) SUMMARY")
    print("   certified run removed %s tokens from live context: "
          "evictions %s + never-admitted (caps) %s + strip-reclaim %s"
          % (format(all_t, ","), format(ev_t, ","), format(cap_t, ","),
             format(st_t, ",")))
    print("   of which file-touching: %s tokens (%.1f%%) - recoverable in "
          "principle by re-reading the target" % (format(all_ft, ","),
                                                   100.0 * all_ft / all_t
                                                   if all_t else 0.0))
    print("   %d of %d dropped eviction/cap file targets were requested again "
          "later in the session (exact path; basename: %d)"
          % (rer_exact, rer_pairs, rer_base))
    print("   aggregate detail: %s" % OUT)


if __name__ == "__main__":
    main()
