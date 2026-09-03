# local-llm-lab

Measurement notes from running a 27B model locally on a single RTX 5060 Ti (16 GB)
with llama.cpp, and from configuring an agent harness on top of it.

Hardware: RTX 5060 Ti 16GB, Ryzen 5 5600G, 64 GB RAM, Ubuntu.
Model: Qwen3.8-27B UD-IQ4_XS (14 GB), llama.cpp build 10751.

Each entry follows the same shape: what I believed, what I measured, what the
measurement changed. Raw log lines backing every number are in `logs/`.

---

## 1. A conclusion I had to reverse

In August I closed multi-token prediction (MTP) as unusable on this card:
it would not coexist with file reading, and the context ceiling was 8192 tokens.

That verdict was measured on an older quantization and an older build. After
replacing both, I retested rather than trusting the note:

| | August | Now |
|---|---|---|
| Context ceiling | 8192 | 32768 |
| 17.5k-token prefill | died | 861 t/s, survived |
| Generation | ~24 t/s baseline | 57-59 t/s |

The ceiling was never architectural. The newer quantization freed 1306 MiB of
weights, and that was the whole difference.

**Lesson: a measured verdict expires when its inputs change.**

## 2. A prediction that held to two decimal places

MTP raised the recurrent-state (RS) buffer from 149.62 MiB to 598.50 MiB —
exactly 4.00x. The model uses Gated DeltaNet layers whose recurrent state must be
checkpointed per draft token so rejected drafts can roll back, so I predicted the
buffer scales with `--spec-draft-n-max + 1`, not with context length.

Prediction for `n-max 1`: 299.24 MiB. Measured: **299.25 MiB**.

This is not documented anywhere I could find, and it turns speculation depth into
a directly calculable VRAM cost.

## 3. A lever closed by arithmetic

Can MTP and vision run together? Reducing draft depth and context freed 571 MiB;
the vision projector needed 885 MiB of weights plus a 248 MiB compute buffer
reserved only when the projector is loaded.

Result: **it fit with 6 MiB to spare** — which on this card means it does not fit.
Every failure here comes from memory requested at use time, not load time. I did
not send it a single request.

Closed on numbers, not preference.

## 4. The agent was not the problem; the configuration was

The local agent answered "who is the president of Colombia" correctly by searching
the web. Asked "which is the biggest rock band in Colombian history," it did not
search, answered from memory, and fabricated both the band members and four bands
that do not exist.

Root cause, found by auditing the harness config: **nothing in the system prompt
decides when to search.** The only guidance is a tool description reading
"discover current information on the web." The model self-maps: a question that
sounds temporal triggers search; one that sounds timeless does not.

Fix — two sentences in `~/.dsh/AGENTS.md` (in `configs/`):

- verify named entities with search before asserting them
- when corrected, re-audit the whole answer, not just the flagged part

Same question afterward: searched first, correct members, sourced, zero fabrications.

A second gap surfaced the same way — the harness never tells the model today's
date, so it reasoned from its training cutoff and misread correct sources as
confusing. Fixed by regenerating the date into the instructions file at launch.

## 5. Open: `launch timed out`

The server has crashed twice under long agentic workloads with
`CUDA error: the launch timed out and was terminated` — the GPU watchdog killing a
kernel, not an out-of-memory condition.

Suspect: CUDA graphs batching many kernels into one launch. But the machine
reports `Display Active: Disabled`, so the classic X11 display watchdog is not
armed — which undercuts the simplest explanation. Third-party reports on the same
GPU architecture (sm_120) describe non-deterministic llama.cpp hangs under
sustained inference where disabling CUDA graphs did not help, pointing at GSP
firmware instead. Cause remains unidentified. With
`GGML_CUDA_DISABLE_GRAPHS=1` the server survived a workload that had killed it
twice.

**This is n=1 and I am not calling it fixed.** The failure is intermittent; an
earlier instance took 7000+ tokens to appear while today's came at 747. Persisted
as a systemd drop-in so a restarted service inherits it, and left open pending
more runs.

---

## Method

- Loading a config certifies nothing. This card has loaded cleanly with 468 and
  526 MiB free and died on the first real request. Certification means a real
  workload.
- Near-perfect speculative acceptance can mean the model is copying text already
  in context rather than drafting. Vary the prompt.
- What an agent reports about the disk is not true until a command confirms it.
- Register what will be measured before measuring, not the threshold for accepting it.
