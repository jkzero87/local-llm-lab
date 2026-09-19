#!/usr/bin/env python3
"""Census of compaction failures across ~/.dsh session logs.

Reads only compaction/* events; writes a summary plus the raw JSON lines of
every failed compaction/end to the output file. No other log content is read
or written.

Usage:
    python3 tools/compaction_census.py --output logs/compaction_failures_census.log
"""
import argparse
import glob
import json
import os
import time
import zstandard


def load_events(path):
    with zstandard.open(path, "rb") as fh:
        data = fh.read()
    out = []
    for ln in data.splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def utc(ms):
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000.0))


def classify(err):
    e = str(err).lower()
    if "truncated at the token cap" in e:
        return "truncated_at_cap"
    if "no text summary" in e:
        return "no_text_summary"
    if "aborted" in e:
        return "aborted"
    if "terminated" in e:
        return "terminated"
    if "context size" in e or "context-length" in e or "context_length" in e:
        return "context_size"
    if "timed out" in e or "timeout" in e:
        return "timeout"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="path for the census report")
    ap.add_argument("--fork-date",
                    help="split sessions by creation date (YYYY-MM-DD); "
                         "sessions created on or after the date are post-fork")
    args = ap.parse_args()

    dsh = os.path.expanduser("~/.dsh")
    dirs = {}
    for p in glob.glob(os.path.join(dsh, "sessions", "*", "*")):
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.startswith("session") and f.endswith(".jsonl.zstd"):
                dirs.setdefault(p, []).append(os.path.join(p, f))

    errors = []          # (sid, t, cls, msg)
    per_session = {}     # sid -> dict
    trunc_class_days = {}

    for p, cands in sorted(dirs.items()):
        # session id is the directory name; a v3 file wins over v0 in a dir
        sid = os.path.basename(p)
        v3 = [c for c in cands if "session.v3." in os.path.basename(c)]
        path = v3[0] if v3 else sorted(cands)[0]
        events = load_events(path)
        created = None
        stats = {"created": None, "starts": 0, "success": 0, "summary": 0, "errors": 0,
                 "cls": {}}
        for ev in events:
            et = ev.get("type")
            if et == "session":
                created = ev.get("createdAt")
                stats["created"] = created
            elif et == "compaction/start":
                stats["starts"] += 1
            elif et == "compaction/summary":
                stats["summary"] += 1
            elif et == "compaction/end":
                data = ev.get("data", {})
                err = data.get("error")
                t = ev.get("time", 0)
                if err:
                    cls = classify(err)
                    stats["errors"] += 1
                    stats["cls"][cls] = stats["cls"].get(cls, 0) + 1
                    errors.append((sid, t, cls, err, path))
                    if created is not None:
                        day = utc(created)
                        trunc_class_days.setdefault(day, {}).setdefault(cls, 0)
                        trunc_class_days[day][cls] += 1
                else:
                    stats["success"] += 1
        per_session[sid] = stats

    # report
    with open(args.output, "w") as out:
        out.write(f"census  {len(per_session)} sessions  "
                  f"{sum(s['starts'] for s in per_session.values())} compaction starts  "
                  f"{sum(s['success'] for s in per_session.values())} successful ends  "
                  f"{len(errors)} failed ends\n")
        out.write("\n-- failed ends by class --\n")
        cls_totals = {}
        for _, _, cls, _, _ in errors:
            cls_totals[cls] = cls_totals.get(cls, 0) + 1
        for c in sorted(cls_totals, key=cls_totals.get, reverse=True):
            out.write(f"  {c:20s} {cls_totals[c]:4d}\n")
        out.write("\n-- failed ends by session creation date (all classes) --\n")
        for day in sorted(trunc_class_days):
            row = trunc_class_days[day]
            total = sum(row.values())
            detail = "  ".join(f"{c}={n}" for c, n in sorted(row.items()))
            out.write(f"  {day}  {total:4d}  {detail}\n")
        if args.fork_date:
            out.write(f"\n-- pre-fork / post-fork split (fork date {args.fork_date}) --\n")
            split = {}
            for sid, s in per_session.items():
                day = utc(s["created"]) if s["created"] else None
                era = "post-fork" if (day and day >= args.fork_date) else "pre-fork"
                d = split.setdefault(era, {"sessions": 0, "starts": 0,
                                           "success": 0, "errors": 0, "cls": {}})
                d["sessions"] += 1
                d["starts"] += s["starts"]
                d["success"] += s["success"]
                d["errors"] += s["errors"]
                for c, n in s["cls"].items():
                    d["cls"][c] = d["cls"].get(c, 0) + n
            for era in ("pre-fork", "post-fork"):
                d = split.get(era, {"sessions": 0, "starts": 0, "success": 0,
                                    "errors": 0, "cls": {}})
                out.write(f"  {era:10s} {d['sessions']:4d} sessions  "
                          f"{d['starts']:4d} compaction starts  "
                          f"{d['success']:4d} successful ends  "
                          f"{d['errors']:4d} failed ends\n")
            out.write("  failed ends by class:\n")
            for era in ("pre-fork", "post-fork"):
                d = split.get(era, {"cls": {}})
                detail = "  ".join(f"{c}={n}" for c, n in sorted(d["cls"].items())) or "none"
                out.write(f"    {era:9s} {detail}\n")
        out.write("\n-- sessions with >=10 compaction starts --\n")
        for sid in sorted(per_session, key=lambda s: per_session[s]["starts"],
                          reverse=True):
            s = per_session[sid]
            if s["starts"] >= 10:
                created = utc(s["created"]) if s["created"] else "?"
                errdetail = " ".join(f"{c}={n}" for c, n in sorted(s["cls"].items()))
                out.write(f"  {sid[:18]:20s} {created}  starts={s['starts']:3d} "
                          f"ok={s['success']:3d} summary={s['summary']:3d} "
                          f"errors={s['errors']:3d}  {errdetail}\n")
        out.write("\n-- raw failed compaction/end lines --\n")
        for sid, t, cls, err, path in sorted(errors, key=lambda e: e[1]):
            out.write(f"{sid[:18]}  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t/1000.0))}  "
                      f"{cls:20s}  {err}\n")
        out.write(f"\nsource: {len(dirs)} session dirs under ~/.dsh/sessions/\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
