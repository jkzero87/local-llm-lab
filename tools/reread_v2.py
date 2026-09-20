#!/usr/bin/env python3
"""
Post-compaction re-read detector (v3 - callId linked, summary + prune) for dsh session logs.

For each compaction/summary event, collects the file targets of the tool/calls
it shadowed, then scans forward for the same target being requested again.
A re-read means the compactor dropped something the agent still needed.

No CSV inputs, no uuid join: it walks the session tree, keeps files that
contain at least one compaction, and dedupes by session uuid afterwards.

Usage:
    python3 reread.py                      # defaults to ~/.dsh/sessions
    python3 reread.py /path/to/sessions
Outputs (into ./data/ relative to cwd):
    reread_events.csv     one row per dropped target
    reread_coverage.csv   one row per compaction
"""
import csv, io, json, os, re, subprocess, sys
from collections import defaultdict

SESS_ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.dsh/sessions")
OUT_DIR = os.path.join(os.getcwd(), "data")

PATH_TOOLS = {"read", "write", "edit", "read_image", "str_replace_editor", "view"}
GROUND_TOOLS = {"read", "write", "edit", "read_image"}
EXTS = (".py", ".md", ".txt", ".json", ".yml", ".yaml", ".csv", ".tsv", ".js",
        ".ts", ".jsx", ".tsx", ".log", ".sh", ".bash", ".zsh", ".jsonl",
        ".zstd", ".gz", ".zip", ".toml", ".cfg", ".ini", ".conf", ".html",
        ".htm", ".xml", ".rels", ".sql", ".rs", ".go", ".c", ".h", ".cpp",
        ".hpp", ".java", ".rb", ".php", ".css", ".scss", ".ipynb", ".docx",
        ".xlsx", ".pptx", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".gguf",
        ".env", ".lock", ".patch", ".diff")
SKIP_TOKENS = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/", "//"}
TOKEN_RE = re.compile(r"""[^\s'"|;&<>()]+""")


def uuid_of(path):
    m = re.search(r"session-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", path)
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


def open_session(path):
    """Yield decoded lines from a .zstd or plain .jsonl session file."""
    if path.endswith(".zstd"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        for line in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace"):
            yield line
        p.wait()
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line


def norm_file(tok):
    """Normalize a structured tool file_path argument. Always accepted."""
    tok = tok.strip().strip("'\"`,)")
    if not tok or tok in SKIP_TOKENS:
        return None
    if tok.startswith("~"):
        tok = os.path.expanduser(tok)
    return os.path.normpath(tok)


def bash_token(tok, truth):
    """Acceptance for a raw bash token.

    A token is a file target only if it contains no ':', does not start with
    '-' or 'http', does not end with '/', AND (it ends with an allowlisted
    extension OR it is in the session ground-truth set).
    """
    tok = tok.strip().strip("'\"`,)")
    if not tok or tok in SKIP_TOKENS:
        return None
    if ":" in tok or tok.startswith("-") or tok.startswith("http"):
        return None
    if tok.endswith("/"):
        return None
    if not (tok.endswith(EXTS) or tok in truth):
        return None
    if tok.startswith("~"):
        tok = os.path.expanduser(tok)
    return os.path.normpath(tok)


def targets_of(name, args, truth=frozenset()):
    """Return the set of file targets a tool call refers to."""
    out = set()
    if not isinstance(args, dict):
        return out
    if name in PATH_TOOLS:
        fp = args.get("file_path") or args.get("path")
        if isinstance(fp, str):
            n = norm_file(fp)
            if n:
                out.add(n)
        return out
    if name == "bash":
        cmd = args.get("command")
        if isinstance(cmd, str):
            for tok in TOKEN_RE.findall(cmd):
                n = bash_token(tok, truth)
                if n:
                    out.add(n)
    return out


def parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def ground_truth(path):
    """Pre-pass over the session: collect every file_path/path argument from
    read/write/edit/read_image calls, from both tool/call events and
    tool-call content blocks inside assistant/message. Those structured
    arguments are real files."""
    truth = set()

    def collect(name, args):
        if name not in GROUND_TOOLS or not isinstance(args, dict):
            return
        for key in ("file_path", "path"):
            fp = args.get(key)
            if isinstance(fp, str):
                n = norm_file(fp)
                if n:
                    truth.add(n)

    for line in open_session(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        d = e.get("data") or {}
        if t == "assistant/message":
            for c in (d.get("message") or {}).get("content", []):
                if c.get("type") != "tool-call":
                    continue
                collect(c.get("toolName") or c.get("name"),
                        parse_args(c.get("args") if "args" in c else c.get("arguments")))
        elif t == "tool/call":
            collect(d.get("name"), parse_args(d.get("arguments")))
    return truth


def scan(path, truth=frozenset()):
    """Main pass (after the ground-truth pre-pass). Links each tool call to
    EVERY seq it is visible at.

    The surface does not contain `tool/call` events: the call lives inside the
    assistant/message as a `tool-call` content block, and its output arrives as
    a `tool/result`. compaction shadows those, never the tool/call event. So a
    call counts as shadowed if its assistant/message seq, its tool/call seq, or
    its tool/result seq appears in shadowedSeqs.
    """
    calls = {}          # callId -> {"seqs": set, "targets": set, "call_seq": int}
    compactions = []    # (seq, [shadowed seqs])
    n_events = 0
    for line in open_session(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        n_events += 1
        t = e.get("type")
        seq = e.get("seq")
        d = e.get("data") or {}

        if t == "assistant/message":
            for c in (d.get("message") or {}).get("content", []):
                if c.get("type") != "tool-call":
                    continue
                cid = c.get("toolCallId") or c.get("id")
                if not cid:
                    continue
                rec = calls.setdefault(cid, {"seqs": set(), "targets": set(), "call_seq": seq})
                rec["seqs"].add(seq)
                args = parse_args(c.get("args") if "args" in c else c.get("arguments"))
                rec["targets"] |= targets_of(c.get("toolName") or c.get("name"), args, truth)

        elif t == "tool/call":
            cid = d.get("callId")
            if not cid:
                continue
            rec = calls.setdefault(cid, {"seqs": set(), "targets": set(), "call_seq": seq})
            rec["seqs"].add(seq)
            rec["call_seq"] = seq
            rec["targets"] |= targets_of(d.get("name"), parse_args(d.get("arguments")), truth)

        elif t == "tool/result":
            src = (d.get("message") or {}).get("source") or {}
            cid = src.get("callId") or d.get("callId")
            if not cid:
                continue
            rec = calls.setdefault(cid, {"seqs": set(), "targets": set(), "call_seq": seq})
            rec["seqs"].add(seq)

        elif t in ("compaction/summary", "compaction/prune"):
            sh = d.get("shadowedSeqs") or []
            if sh:
                compactions.append((seq, [s for s in sh if isinstance(s, int)],
                                    t.split("/")[1]))

    return n_events, compactions, calls


def analyse(path):
    truth = ground_truth(path)
    n_events, compactions, calls = scan(path, truth)
    if not compactions:
        return None
    uid = uuid_of(path)
    ordered = sorted(((r["call_seq"], r["targets"]) for r in calls.values()
                      if r["call_seq"] is not None and r["targets"]),
                     key=lambda x: x[0])
    ev_rows, cov_rows = [], []

    for cseq, shadowed, mech in compactions:
        sh = set(shadowed)
        shadowed_calls = [r for r in calls.values() if r["seqs"] & sh]
        dropped = {}
        for r in shadowed_calls:
            for tgt in r["targets"]:
                prev = dropped.get(tgt)
                if prev is None or r["call_seq"] < prev:
                    dropped[tgt] = r["call_seq"]
        later_paths, later_bases = {}, {}
        for s, tg in ordered:
            if s <= cseq:
                continue
            for t in tg:
                later_paths.setdefault(t, s)
                later_bases.setdefault(os.path.basename(t), s)
        for tgt, src_seq in sorted(dropped.items()):
            hit_p = later_paths.get(tgt)
            hit_b = later_bases.get(os.path.basename(tgt))
            ev_rows.append(dict(
                uuid=uid, mechanism=mech, compaction_seq=cseq, target=tgt,
                dropped_at_seq=src_seq,
                was_reread_path=int(hit_p is not None),
                was_reread_basename=int(hit_b is not None),
                seqs_until_reread=(hit_p - cseq) if hit_p is not None else ""))
        cov_rows.append(dict(
            uuid=uid, mechanism=mech, compaction_seq=cseq, n_shadowed_seqs=len(sh),
            n_shadowed_toolcalls=len(shadowed_calls), n_dropped_targets=len(dropped)))
    return dict(uuid=uid, path=path, n_events=n_events,
                n_compactions=len(compactions), n_calls=len(calls),
                ev=ev_rows, cov=cov_rows)


def main():
    files = []
    for root, _, names in os.walk(SESS_ROOT):
        for n in names:
            if n.startswith("session") and (".jsonl" in n):
                files.append(os.path.join(root, n))
    print(f"session files found: {len(files)}", flush=True)

    best = {}
    for i, f in enumerate(sorted(files), 1):
        try:
            r = analyse(f)
        except Exception as ex:
            print(f"  !! {os.path.basename(os.path.dirname(f))}: {ex}", flush=True)
            continue
        if not r:
            continue
        prev = best.get(r["uuid"])
        if prev is None or (r["n_compactions"], r["n_events"]) > (prev["n_compactions"], prev["n_events"]):
            best[r["uuid"]] = r
        if i % 25 == 0:
            print(f"  scanned {i}/{len(files)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    ev = [r for s in best.values() for r in s["ev"]]
    cov = [r for s in best.values() for r in s["cov"]]

    for name, rows in (("reread_events.csv", ev), ("reread_coverage.csv", cov)):
        p = os.path.join(OUT_DIR, name)
        with open(p, "w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")

    def block(label, evs, covs):
        tot_sh = sum(c["n_shadowed_seqs"] for c in covs)
        tot_sh_calls = sum(c["n_shadowed_toolcalls"] for c in covs)
        n = len(evs)
        rr_p = sum(r["was_reread_path"] for r in evs)
        rr_b = sum(r["was_reread_basename"] for r in evs)
        hits = len({r["uuid"] for r in evs if r["was_reread_path"]})
        dists = sorted(r["seqs_until_reread"] for r in evs if r["was_reread_path"])
        med = dists[len(dists) // 2] if dists else None
        print(f"\n--- {label} ---")
        print(f"  events                   : {len(covs)}")
        print(f"  shadowed seqs            : {tot_sh}")
        print(f"  shadowed calls resolved  : {tot_sh_calls}")
        print(f"  dropped file targets     : {n}")
        if n:
            print(f"  re-read, exact path      : {rr_p} ({rr_p/n:.1%})  [lower bound]")
            print(f"  re-read, basename match  : {rr_b} ({rr_b/n:.1%})  [upper bound]")
            print(f"  sessions with a re-read  : {hits}")
            print(f"  median seqs to re-read   : {med}")

    print("\n=== RESULT ===")
    print(f"sessions with shadowing : {len(best)}")
    block("compaction/summary", [r for r in ev if r["mechanism"] == "summary"],
          [c for c in cov if c["mechanism"] == "summary"])
    block("compaction/prune", [r for r in ev if r["mechanism"] == "prune"],
          [c for c in cov if c["mechanism"] == "prune"])
    block("COMBINED", ev, cov)


if __name__ == "__main__":
    main()
