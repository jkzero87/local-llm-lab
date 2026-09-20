#!/usr/bin/env python3
"""Simulate reference-eviction against real compactions.

Read-only scan of ~/.dsh/sessions (session*.jsonl and *.jsonl.zstd).
For every compaction/summary event, compare the actual summary-retention
cost against a hypothetical reference-eviction that keeps ~20 tokens per
shadowed span instead of the generated summary.

Aggregate-only output; no session content is printed.
"""

import csv
import json
from pathlib import Path

SESSIONS_DIR = Path.home() / ".dsh" / "sessions"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "eviction_sim.csv"
REF_TOKENS_PER_SPAN = 20


def read_lines(path: Path):
    """Yield decoded text lines from a plain or zstd-compressed file."""
    try:
        if path.name.endswith(".zstd"):
            import zstandard

            with path.open("rb") as f:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(f) as r:
                    yield r.read()
        else:
            with path.open("rb") as f:
                for raw in f:
                    yield raw
    except ImportError:
        # Fall back to zstd binary for zstd files.
        if path.name.endswith(".zstd"):
            import subprocess

            p = subprocess.run(
                ["zstdcat", str(path)], stdout=subprocess.PIPE, check=False
            )
            if p.returncode == 0:
                yield p.stdout
        else:
            with path.open("rb") as f:
                for raw in f:
                    yield raw


def iter_jsonl(path: Path):
    buf = b""
    for chunk in read_lines(path):
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    if buf.strip():
        try:
            yield json.loads(buf)
        except json.JSONDecodeError:
            pass


def summary_text(summary) -> str:
    """Flatten the checkpoint summary (list of parts or plain string)."""
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        return "".join(
            part.get("text", "")
            for part in summary
            if isinstance(part, dict)
        )
    return ""


def main():
    files = sorted(
        p
        for p in SESSIONS_DIR.rglob("session*.jsonl*")
        if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.zstd"))
    )

    rows = []
    seen = set()  # (compactionId, seq) dedupe across plain/v3 duplicates
    skipped = 0

    for path in files:
        for event in iter_jsonl(path):
            if not isinstance(event, dict) or event.get("type") != "compaction/summary":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            shadowed = data.get("shadowedSeqs")
            if not isinstance(shadowed, list):
                skipped += 1
                continue
            uuid = data.get("compactionId", "")
            seq = event.get("seq")
            key = (uuid, seq)
            if key in seen:
                continue
            seen.add(key)

            shadowed_tokens = data.get("shadowedTokenCount")
            if not isinstance(shadowed_tokens, (int, float)):
                skipped += 1
                continue
            usage = data.get("usage") or {}
            out_tokens = usage.get("outputTokens")
            if not isinstance(out_tokens, (int, float)):
                out_tokens = 0

            n_spans = len(shadowed)
            text = summary_text(data.get("summary"))
            retained_actual = len(text) / 4.0
            ref_cost = n_spans * REF_TOKENS_PER_SPAN

            rows.append(
                {
                    "uuid": uuid,
                    "compaction_seq": seq,
                    "n_spans": n_spans,
                    "shadowedTokenCount": int(shadowed_tokens),
                    "retained_actual": round(retained_actual, 1),
                    "gen_cost": int(out_tokens),
                    "net_actual": int(shadowed_tokens) - round(retained_actual),
                    "net_reference": int(shadowed_tokens) - ref_cost,
                }
            )

    rows.sort(key=lambda r: (r["uuid"], r["compaction_seq"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uuid",
                "compaction_seq",
                "n_spans",
                "shadowedTokenCount",
                "retained_actual",
                "gen_cost",
                "net_actual",
                "net_reference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    if n == 0:
        print("no compaction/summary events found")
        return
    t_shadow = sum(r["shadowedTokenCount"] for r in rows)
    t_ret = sum(r["retained_actual"] for r in rows)
    t_gen = sum(r["gen_cost"] for r in rows)
    t_net_actual = sum(r["net_actual"] for r in rows)
    t_net_ref = sum(r["net_reference"] for r in rows)
    t_ref_cost = sum(r["n_spans"] * REF_TOKENS_PER_SPAN for r in rows)
    nonneg = [r for r in rows if r["net_actual"] <= 0]

    print(f"compactions simulated: {n}")
    print(f"  files scanned: {len(files)}")
    print(f"  rows skipped (missing fields): {skipped}")
    print("totals:")
    print(f"  shadowedTokenCount: {t_shadow}")
    print(f"  retained_actual:    {t_ret:.1f}")
    print(f"  gen_cost:           {t_gen}")
    print(f"  net_actual:         {t_net_actual}")
    print(f"  net_reference:      {t_net_ref}")
    print("means per compaction:")
    print(f"  net_actual:    {t_net_actual / n:.1f}")
    print(f"  net_reference: {t_net_ref / n:.1f}")
    print("compression ratio (shadowedTokenCount / retained cost):")
    print(f"  actual summary:   {t_shadow / t_ret:.2f}x" if t_ret else "  actual summary: n/a")
    print(f"  reference:        {t_shadow / t_ref_cost:.2f}x" if t_ref_cost else "  reference: n/a")
    print(f"compactions with net_actual <= 0: {len(nonneg)}")
    print(f"generation tokens reference-eviction would not have spent: {t_gen}")
    print(f"csv: {OUT_CSV}")


if __name__ == "__main__":
    main()
