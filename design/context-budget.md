# Context budget: making overflow an impossible state

Design investigation only; no plugin code. Sources: installed `dsh-compaction-basic`
(+ fork `~/dsh-compaction-select`), `dsh-agent-loop`, `dsh-session`, `dsh-tool-cordis`,
`dsh-token-meter` — lib and README on disk.

## What exists today

- The session store (`dsh-session`) is an append-only log. Model-visible history is a
  derived **surface**: an ordered projection of `system/message`, `user/message`,
  `assistant/message`, `tool/result`. A `replace` op (`{startSeq, endSeq}`) shadows a
  range: originals stay in the log, excluded from future inputs; `deriveEventMessage`
  re-derives any original by seq. Node 0 (while a `system/message`) is protected from
  shadowing. `replaceGeneration` bumps per rewrite and invalidates prefix/KV reuse from
  the first shadowed message.
- The agent loop exposes three waterfalls that matter here: `agent/pre-step` (serial; a
  listener may replace the messages entering the step or `{kind:'reject'}` it — runs once
  per turn; retries reuse the rendered assembly), `agent/request` (routing only:
  provider/model/reasoningEffort — never messages), and `agent/request-error` (a
  listener may return `{kind:'retry'}`; otherwise the failure is terminal).
- Tool pipeline (`dsh-tool-cordis`): `tools/pre-execute` (allow/deny/ask) →
  `tools/execute` → `tools/post-execute` (may inspect and **replace** the result before
  it is logged) → `tools/result` (frozen, observe-only).
- A token meter exists (`ctx.tokenMeter`): prices the system prompt (surface node), tool
  schemas (from the last `request/header` event), and every surface node — the full
  routed envelope.
- `dsh-compaction-basic` (installed; tool-result pruner mounted) already uses the meter
  in a serial `agent/pre-step` listener: above `thresholdTokens =
  floor(contextWindow × thresholdRatio)` (default 0.8) it summarizes the oldest
  head-anchored balanced range into a `replace`-shadows checkpoint (one extra LLM
  request), retaining a priced tail (default 16%). On provider-confirmed
  `CONTEXT_WINDOW_EXCEEDED` it re-runs from `agent/request-error`, bypassing the
  threshold, and retries only if `replaceGeneration` advanced.

## The four answers

**1. The compactor.** Selected history items are **shadowed, not deleted**: the summary
commits as a replacement event recording `shadowedSeqs`; originals remain in the
append-only log, resolvable by seq, and back the human transcript. Only the model-visible
projection loses them — original content is *retained in the store, discarded from model
input*. The fork differs only in the summary prompt (60-line cap, replace-prior-
checkpoint) and in dropping the node-0 system-head protection; shadowing is identical.

**2. The interception points.** All three exist. (a) Tool result before history:
`tools/post-execute` — replacement is allowed pre-log. (b) Request before send:
`agent/pre-step` — may replace the entering messages or reject the step; there is
**no** post-`buildRequest` envelope-modification cascade — system prompt and tool schemas
cannot be modified at send time. (c) Failed request: `agent/request-error`.

**3. The overflow path.** `compactionRetries` (default 1) = extra summarization attempts
*within one* pressure compaction while the envelope stays above the soft threshold.
`maxOverflowRetries` (default 1) = retries after a confirmed overflow. On overflow,
recovery compacts first and retries **only if `replaceGeneration` advanced** — it never
resends the same request. But the pre-step check is advisory: if compaction fails or
exhausts attempts, the listener logs a warning and calls `next()` — the over-size
request is sent anyway.

**4. What the store supports.** Yes. A surface `replace` may put *any* message in place
of a shadowed range — including one that is purely a resolvable reference (seq/path/
handle) to the shadowed originals. The model already enforces complete shadowed-node
coverage at append, node-0 protection, tool-pair balance, and whole-history
`foldSurface` validation. "Replace a history item with a reference" is expressible today;
the compactor simply chooses summary text as its replacement content.

## What this permits — the single mechanism

The residual failure class (indivisible-unit and envelope-only overflow) exists only
because the pre-step check is a **soft threshold with a warn-and-continue fallback**. The
single mechanism:

**Make `agent/pre-step` an authoritative admission gate on the hard cap.** Before the
step enters, meter the exact envelope (system + tools + surface + entering messages,
plus a reserved output budget) and require it to fit `contextWindow`. If it does not,
shrink by committing surface replacements — pruner first, then head-anchored shadowing,
node-0-protected, reference- or summary-bearing — until it fits; if no surface operation
can make it fit, **reject the step** with a precise diagnostic (which unit, its price,
what is required) instead of sending.

One change, one invariant: *a request is sent only if its metered envelope fits the
window; otherwise the surface is shrunk or the step is refused.* Everything the meter
sees can no longer overflow; `agent/request-error` demotes from primary recovery to
defense in depth. It is the only mechanism that fits the surface: the shrinker
(compactor + pruner), the replacement primitive (surface `replace` + shadowing,
reference-capable), and the envelope meter all exist — only the binding check and the
refuse-branch are missing.

### What could not be verified
- **Meter accuracy**: pricing is heuristic (char-ratio + block overhead); provider-side
  counting may differ. "Cannot occur" holds for the metered envelope, not the provider's
  exact count; a margin inside the cap plus the surviving error path covers the residual.
- **Provider error fidelity**: the gate's residual and recovery both key on
  `CONTEXT_WINDOW_EXCEEDED`; unadapted routes phrasing overflow differently are
  unclassified.
- **Envelope-only overflow** (system prompt + tools alone ≥ window): the gate can only
  refuse — no surface operation touches system/tools; a config error surfaced at step
  time, not a provider 400.
- **Wire equivalence**: the meter prices the session's last `request/header` + surface;
  whether that equals byte-for-byte what `buildRequest` serializes was not verified.

## Measured case for reversible eviction
(2026-09-20)

### 1. The chain, as four findings in order

1. **Eviction is necessary.** On the 24-session viable subset with a fixed
   8192-token reserve, `P0_NO_EVICT` leaves 1046 over-budget steps;
   `P2_LARGEST` takes it to 0 using 1668 evictions. `P4_LRU` also reaches 0
   but needs 2671 evictions. `P5_WS3` and `P6_WS3_LRU` leave 178 and 179.
2. **Eviction cannot be made accurate.** exposure_pct is 86.8 / 87.5 / 88.2 /
   88.1 for P2 / P4 / P5 / P6 — invariant across four structurally different
   policies, on sessions selected for file reuse. A keep/drop classifier
   therefore has a ceiling near 13%.
3. **Therefore eviction must be reversible rather than accurate.**
4. **Reversal is free.** A probe over the same 24 sessions found 3271 of 3271
   shadowed seqs recoverable (100%), so no extra persistence is needed.

### 2. Corpus and its limits

203 sessions total; only 24 reference 3 or more distinct file paths, 6
reference 8 or more; the median session references none. Exposure measured
over the full corpus is therefore meaningless, which invalidates the earlier
80.5% figure recorded for `P2_LARGEST`.

### 3. Instrument caveats, stated plainly

- per-session `maxTokens` in the corpus is heterogeneous (0 / 8192 / 11264 /
  16384), so `over_budget_steps` is only comparable under a fixed reserve.
  The 8192 constant is a choice, not a measurement.
- `P4`/`P5`/`P6` compute recency in seq order, but surface order differs: the
  latest checkpoint occupies the surface position of what it replaced.
  Measured on one session, 9 of 300 pairs inverted (3%), and the displaced
  element is the checkpoint itself. `P0` and `P2` consult only size and are
  unaffected.

### 4. Why summary-based compaction deadlocks

Across ~536 compaction attempts in 45 sessions that ever compacted: 414 ok,
64 "summarization truncated at the token cap" across 22 sessions, 13
"summarization produced no text summary content" across 8. The summarizer is
itself a generation call, so it needs output headroom to create output
headroom; when the window is full it truncates, the checkpoint is rejected,
nothing is shadowed, and the next identical request fails the same way.
Reference-eviction has zero generation cost and cannot fail this way.

### 5. One field observation, quoted exactly

```
400: request (32822 tokens) exceeds the available context size (32768 tokens)
```

This is the advisory pre-step check sending an over-size request after a
compaction failure — code path plus a field occurrence.
