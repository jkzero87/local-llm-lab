#!/usr/bin/env python3
"""Profile WHICH targets get re-read after compaction.

Reads data/reread_events.csv (written by reread_v2.py) - the CSV only, no
session files re-parsed - and prints four sections:
  1. by mechanism (summary / prune / combined)
  2. concentration (few hot files per session, or spread out)
  3. repeat offenders: (session, target) pairs dropped AND re-read more than
     once in one session - the signature of a compactor looping on a file
  4. top re-read basenames; full list written to data/reread_hot_targets.csv

Run from the repo root:  python3 tools/reread_profile.py
"""
import csv, os, collections, sys

SRC = os.path.join(os.getcwd(), "data", "reread_events.csv")
OUT = os.path.join(os.getcwd(), "data", "reread_hot_targets.csv")


def median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2] if vals else None


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} - run reread_v2.py first")
    rows = list(csv.DictReader(open(SRC)))
    if not rows:
        sys.exit("no rows")

    def exact(r):
        return r.get("was_reread_path") == "1"

    def basehit(r):
        return r.get("was_reread_basename") == "1"

    print(f"rows: {len(rows)}")

    # ---- 1. by mechanism ----------------------------------------------------
    print(f"\n1. BY MECHANISM")
    print(f"   {'mech':<10} {'dropped':>8} {'exact':>14} {'basename':>14}"
          f" {'median seqs':>13} {'sessions>=1':>13}")
    for label, sub in (
        ("summary", [r for r in rows if r["mechanism"] == "summary"]),
        ("prune", [r for r in rows if r["mechanism"] == "prune"]),
        ("combined", rows),
    ):
        n = len(sub)
        ex = [r for r in sub if exact(r)]
        bs = [r for r in sub if basehit(r)]
        med = median(int(r["seqs_until_reread"]) for r in ex)
        sess = len({r["uuid"] for r in ex})
        print(f"   {label:<10} {n:>8} {len(ex):>8} ({len(ex)/n:.1%})"
              f" {len(bs):>8} ({len(bs)/n:.1%}) {med:>13} {sess:>13}")

    # ---- 2. concentration, per session ---------------------------------------
    per_sess = collections.defaultdict(list)
    for r in rows:
        if exact(r):
            per_sess[r["uuid"]].append(r["target"])
    fracs = []
    for uid, tgts in per_sess.items():
        c = collections.Counter(tgts)
        top3 = sum(n for _, n in c.most_common(3))
        fracs.append(top3 / len(tgts))
    fracs.sort()
    print(f"\n2. CONCENTRATION")
    print(f"   sessions with a re-read: {len(fracs)}")
    if fracs:
        print(f"   median share of a session's re-reads from its top 3 targets: "
              f"{fracs[len(fracs) // 2]:.0%}")
        print(f"   sessions where top 3 = 100% of re-reads: "
              f"{sum(1 for f in fracs if f >= 0.999)}")

    # ---- 3. repeat offenders --------------------------------------------------
    pair = collections.Counter((r["uuid"], r["target"]) for r in rows if exact(r))
    repeats = [(k, n) for k, n in pair.items() if n > 1]
    print(f"\n3. REPEAT OFFENDERS (same target dropped+re-read >1x in one session)")
    print(f"   distinct (session, target) pairs re-read at all : {len(pair)}")
    if pair:
        print(f"   of those, hit more than once                    : {len(repeats)}"
              f" ({len(repeats)/len(pair):.0%})")
    if repeats:
        worst = sorted(repeats, key=lambda kv: -kv[1])[:8]
        print(f"   worst:")
        for (uid, tgt), n in worst:
            print(f"     {n:>3}x  {uid[:8]}  {os.path.basename(tgt)}")
        print(f"   total redundant drops (n-1 summed): "
              f"{sum(n - 1 for _, n in repeats)}")

    # ---- 4. hot basenames ------------------------------------------------------
    base = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        b = os.path.basename(r["target"])
        base[b][0] += 1
        if exact(r):
            base[b][1] += 1
    print(f"\n4. TOP 15 RE-READ BASENAMES")
    print(f"   {'file':<34} {'times_dropped':>14} {'times_reread':>14}")
    for b, (d, h) in sorted(base.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"   {b[:33]:<34} {d:>14} {h:>14}")

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basename", "times_dropped", "times_reread"])
        for b, (d, h) in sorted(base.items(), key=lambda kv: -kv[1][1]):
            w.writerow([b, d, h])
    print(f"\nwrote {OUT} ({len(base)} rows)")


if __name__ == "__main__":
    main()
