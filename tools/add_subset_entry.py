#!/usr/bin/env python3
"""Patch tools/evict_sim.py with an opt-in fixed-reserve (8192) subset regime.

Adds:
  - constants PROFILE_CSV / OUT_POLICIES_SUBSET / CERT_SUBSET / MIN_N_PATHS
  - subset_uuids()            (n_paths >= MIN_N_PATHS in data/corpus_profile.csv)
  - main_policies_subset()    (five-policy sweep on that subset; writes
                               data/eviction_policies_subset.csv and a NEW
                               regime snapshot data/cert_reserve8192_subset.csv;
                               the old certified-snapshot cross-check is
                               deliberately NOT run here)
  - dispatch: `python3 tools/evict_sim.py policies-subset`

The default `policies` and bare entry points are untouched. Idempotent:
running twice is a no-op the second time.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "evict_sim.py"

ANCHOR_CONST = (
    'EXPO_CSV = Path(__file__).resolve().parent.parent / "data" / "exposure_check.csv"\n'
)

BLOCK_CONST = '''
# Fixed-reserve (8192) subset regime. Opt-in only: `policies-subset`.
# The default `policies` / bare entry points keep the full corpus and their
# existing cross-checks against data/evict_sim.csv / data/exposure_check.csv.
PROFILE_CSV = Path(__file__).resolve().parent.parent / "data" / "corpus_profile.csv"
OUT_POLICIES_SUBSET = Path(__file__).resolve().parent.parent / "data" / "eviction_policies_subset.csv"
CERT_SUBSET = Path(__file__).resolve().parent.parent / "data" / "cert_reserve8192_subset.csv"
MIN_N_PATHS = 3


def subset_uuids():
    """Session UUIDs with n_paths >= MIN_N_PATHS in data/corpus_profile.csv."""
    out = set()
    with open(PROFILE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n_paths"]) >= MIN_N_PATHS:
                out.add(r["uuid"])
    return out


def _file_uuid(path):
    u = path.parent.name
    if u.startswith("session-"):
        u = u[len("session-"):]
    return u


def main_policies_subset():
    """Five-policy sweep under the fixed-reserve (8192) regime, restricted
    to the viable subset (n_paths >= MIN_N_PATHS in data/corpus_profile.csv).

    Writes data/eviction_policies_subset.csv (per-policy totals) and a NEW
    regime snapshot data/cert_reserve8192_subset.csv (P2_LARGEST per-step
    rows, same column layout as data/evict_sim.csv).

    cross_check_policies() is deliberately NOT called for this entry point:
    the existing certified snapshot was built under the old per-session
    reserve and the full corpus, so it is expected to disagree with this
    regime. Gating it off here only; the old entry points keep it.
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
    allowed = subset_uuids()
    if not allowed:
        print("ERROR: no viable sessions (n_paths >= %d) in %s"
              % (MIN_N_PATHS, PROFILE_CSV), file=sys.stderr)
        sys.exit(1)
    results = {p: [] for p in POLICIES}
    for path in mc.pick_files():
        if _file_uuid(path) not in allowed:
            continue
        prep = session_prep(path)
        for p in POLICIES:
            results[p].append(replay_policy(path, tok, rtok, p, prep=prep))
    # NOTE: cross_check_policies() intentionally gated off - see docstring.
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
    OUT_POLICIES_SUBSET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_POLICIES_SUBSET, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(totals[POLICIES[0]].keys()))
        w.writeheader()
        for p in POLICIES:
            w.writerow(totals[p])
    with open(CERT_SUBSET, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "strip", "policy", "uuid", "step", "metered_before",
                    "n_evicted", "tokens_freed", "metered_after",
                    "over_budget_after"])
        for r in results["P2_LARGEST"]:
            for row in r["rows"]:
                w.writerow([row["cap"], row["strip"], row["policy"],
                            row["uuid"], row["step"], row["metered_before"],
                            row["n_evicted"], row["tokens_freed"],
                            row["metered_after"], row["over_budget_after"]])
    cols = ["policy", "over_budget_steps", "evicted_events", "evicted_tokens",
            "exposure_pct", "n_sessions"]
    print("%-13s %15s %14s %14s %11s %9s"
          % tuple(["policy"] + [c for c in cols[1:]]))
    for p in POLICIES:
        t = totals[p]
        print("%-13s %15d %14d %14d %11.1f %9d"
              % (t["policy"], t["over_budget_steps"], t["evicted_events"],
                 t["evicted_tokens"], t["exposure_pct"], t["n_sessions"]))
    print("wrote %s" % OUT_POLICIES_SUBSET)
    print("wrote %s" % CERT_SUBSET)
    print("cross_check vs old certified snapshot: GATED OFF for this entry point")
'''

ANCHOR_DISPATCH = (
    "if __name__ == \"__main__\":\n"
    "    if len(sys.argv) > 1 and sys.argv[1] == \"policies\":\n"
    "        main_policies()\n"
    "    else:\n"
    "        main()\n"
)

DISPATCH_NEW = (
    "if __name__ == \"__main__\":\n"
    "    if len(sys.argv) > 1 and sys.argv[1] == \"policies\":\n"
    "        main_policies()\n"
    "    elif len(sys.argv) > 1 and sys.argv[1] == \"policies-subset\":\n"
    "        main_policies_subset()\n"
    "    else:\n"
    "        main()\n"
)


def main():
    src = TARGET.read_text()
    if "def main_policies_subset" in src:
        print("already applied; nothing to do")
        return
    assert src.count(ANCHOR_CONST) == 1, "constant anchor not unique"
    assert src.count(ANCHOR_DISPATCH) == 1, "dispatch anchor not unique"
    src = src.replace(ANCHOR_CONST, ANCHOR_CONST + BLOCK_CONST)
    src = src.replace(ANCHOR_DISPATCH, DISPATCH_NEW)
    TARGET.write_text(src)
    print("patched %s: added policies-subset entry point" % TARGET)


if __name__ == "__main__":
    main()
