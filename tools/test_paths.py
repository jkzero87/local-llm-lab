#!/usr/bin/env python3
"""Tests for the bash-token acceptance function in reread_v2.py.

Run before touching the corpus:
    python3 tools/test_paths.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reread_v2 import bash_token

CASES = [
    # (expected, token)
    ("REJECT", "Mozilla/5.0"),
    ("REJECT", "AppleWebKit/537.36"),
    ("REJECT", "/m:r"),
    ("REJECT", "/m:oMath"),
    ("REJECT", "/home/jkzero/dsh-scratch"),
    ("REJECT", "_audit"),
    ("REJECT", "/usr/bin/python"),
    ("REJECT", "python"),
    ("REJECT", "end"),
    ("ACCEPT", "/home/j/proj/spec.md"),
    ("ACCEPT", "./old.py"),
    ("ACCEPT", "word/document.xml"),
    ("ACCEPT", "session.jsonl.zstd"),
    ("ACCEPT", "/p/a.docx"),
]


def main():
    truth = frozenset()
    failures = 0
    for expect, tok in CASES:
        got = bash_token(tok, truth)
        ok = (got is not None) if expect == "ACCEPT" else (got is None)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {expect:6s}  {tok!r} -> {got!r}")
        if not ok:
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
