# Strata IQ3_S: does context alone flip the marker answer? (pre-registered 2026-10-06, before any run)

## Facts before this test
- Marker: `~/bench/marcador_17500_v1.json`, frozen (17.5k tokens, `max_tokens` 4000). Correct answer: **6**
  (count of `def result_score` in the prompt).
- Strata v0.1.39, Qwen3.8-Flash-Next GSQ IQ3_S, `~/strata/strata-iq3_s.json`.
- At `--max-context 65536`: answered 5 in 4 of 11 runs (2026-10-05).
- At `--max-context 32768`: answered 6 in 2 of 2 runs (2026-10-05). Two runs are not enough.

## Plan
- 10 runs of the frozen marker at `--max-context 32768`.
- Config: a copy of today's `strata-iq3_s.json` (+ its `.shared-settings.json`, reasoning_effort
  medium) with exactly two changes: `--max-context 65536` -> `32768`, and the `log` file path (so the
  production log is not overwritten). Everything else identical: pack, `--kv int8`,
  `--kv-resident 32768`, `--spec 4`, MTP, `--resident-experts`, `--vram-reserve-mib 959`,
  `STRATA_PREFILL_MMQ=0`. `strata-iq3_s.json` itself is not modified.
- Swap off, as `strata_up` does. dsh is not started (no other client).
- Strata is restarted before every run, so no run can reuse a previous run's prompt from the
  prompt cache; each run's `cache_n` is recorded to confirm it is 0.
- Per run, logged to `ctx32768_runs.md`: answer, prefill t/s, gen t/s, wall time (plus cache_n,
  completion tokens, finish reason).
- A run that errors or ends without an answer (`finish_reason` length / empty) is reported as such
  and counts as **not correct**.

## Decision rule (fixed now)
- **10/10 correct** at 32768: recommend switching Strata (`--max-context`) and the dsh
  `contextWindow` to 32768, and re-aligning the compaction trigger to 32768.
- **9/10 or fewer** correct: the wrong answers are not caused by context alone. Stop and investigate;
  no config change is recommended.
- The decision is a recommendation only; strata-iq3_s.json and the dsh config are changed by the repo owner, not
  by this test.

- 2026-10-06 16:09: not run; replaced by the closing comparison below (notes/2026-10-06-strata-vs-27b-rule.md).
