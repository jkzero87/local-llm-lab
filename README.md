# local-llm-lab

Measurement notes from running local models on a single RTX 5060 Ti (16 GB):
a dense 27B and a large MoE (Flash-Next) with its experts in system RAM, on
llama.cpp and on Strata, and from configuring an agent harness (dsh) on top.

**Hardware.** RTX 5060 Ti 16 GB (compute capability 12.0, driver 595.91.07),
Ryzen 5 5600G, 61 GB RAM visible to the OS, DDR4-3200 dual channel, PCIe 3.0 x8
(fixed by the A520 board + 5600G), Ubuntu.

**Models on disk now** (`~/models`):

| Model | Files | Served by |
|---|---|---|
| Qwen3.8-27B GSQ RCO IQ3_S + MTP head | 11.29 GiB (+ 0.86 GiB mmproj) | llama.cpp, `:8092` |
| Qwen3.8-Flash-Next GSQ RCO IQ3_S (MoE) | 51.05 + 26.82 GiB (+ 0.85 GiB mmproj) | Strata, `:8080` |
| Qwen3.5-4B MTP UD-Q4_K_XL | 2.79 GiB | llama.cpp on CPU, `:8093` (camera project) |

The 27B UD-IQ4_XS and the Flash-Next UD-IQ4_XS measured in entries 1-10 are no
longer on disk.

**Builds.** llama.cpp build 10751 (`3466812d1`) for the 27B and the 4B; build
10679 (`50f068fff`, a branch build) for the Flash-Next llama.cpp runs in
entries 6-9; Strata v0.1.39 (`6f32ec0`) for Flash-Next today.

Each entry follows the same shape: what I believed, what I measured, what the
measurement changed. Raw log lines backing every number are in `logs/`; the
launch scripts and configs are in `configs/` (see [How the stack is launched](#how-the-stack-is-launched)).

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

## 6. A MoE with its experts in RAM: prefill is set by `-ub`, and the prompt cache beats every flag

*Flash-Next UD-IQ4_XS, llama.cpp build 10679, `-c 32768`, `--fit on`, the 17.5k-token
marker prompt. Log: `logs/2026-08-28_flashnext_ub_sweep_and_prompt_cache.log`.*

I expected the usual levers (threads, KV type, speculative decoding) to matter.
The one that mattered was the micro-batch size:

| `-ub` | prefill (t/s) | generation (t/s) |
|---|---|---|
| 256 | 49.23 | 14.73 |
| 2048 | 142.04 | 13.07 |
| 4096 | 179.50 | 12.86 |
| 8192 | 176.80 | 12.43 |

The experts live in system RAM and are copied to the GPU for every micro-batch,
over PCIe 3.0 x8. A bigger micro-batch amortizes each copy over more tokens,
up to 4096; past that it is flat. Generation stays near 13 t/s in every
configuration: that is the DDR4 bandwidth floor, and no flag moves it.

The bigger lever was the prompt cache. Sending the same 17,543-token prompt
twice: the first request took 139.26 s in total; the second evaluated **4**
prompt tokens and took 55.66 s, all of it generation.

**Lesson: on this machine a stable prompt prefix between turns is worth more
than any server flag.**

## 7. Context is paid in host RAM, not in VRAM

*Same model and build, `--fit on`, 17.5k marker. Logs:
`logs/2026-08-30_flashnext_context_and_ub512.log`,
`logs/2026-08-29_flashnext_32k_memory_snapshots.log`.*

I treated a larger context as a VRAM question. With `--fit` it is not: fit
holds VRAM near its target and pushes whatever does not fit to the CPU side.
At 65536 it projected 67,018 MiB of device memory against 15,585 MiB free and
moved the MoE tensors to system memory; in both logs that print it, about 87 GiB
of the model is CPU-mapped. As context grew, generation fell:

| context | `-ub` | prefill (t/s) | generation (t/s) |
|---|---|---|---|
| 32768 | 4096 | 179.50 | 12.86 |
| 65536 | 4096 | 153.66 | 11.24 |
| 98304 | 512 | 65.79 | 11.44 |
| 131072 | not logged | 133.61 | 5.61 |

At 32768, `llama-server`'s own swap use (VmSwap) was 0 kB after load and 177-780 MB after one marker run,
with 1-2 GB of the 7 GB swap in use: the host side was already under pressure
at the smallest context. I have no swap readings for the larger contexts, so
the curve above is speed only.

## 8. `-ub 512` freed compute buffer, and `--fit` spent it on resident layers

*Same log as entry 7.*

The compute buffer scales with the micro-batch. At `-ub 4096` (context 65536)
the CUDA0 compute buffer was 4,528.56 MiB and the CUDA0 model buffer
7,151.29 MiB. At `-ub 512` (context 98304) the compute buffer was 1,188.48 MiB
and `--fit` used the room for weights: CUDA0 model buffer 9,907.54 MiB, even
with the KV cache growing from 816 to 1,224 MiB.

Generation barely moved (11.24 → 11.44 t/s) and prefill fell from 153.66 to
65.79 t/s (entry 6). The two runs also differ in context, so this is not a
clean A/B. **More resident weights did not buy generation speed here;
the prefill cost was real.**

## 9. ngram speculation on a MoE with experts in RAM: a regression, and one crash

*Flash-Next UD-IQ4_XS, build 10679. Logs: `logs/2026-08-29_ngram_bench.log`,
`logs/2026-08-28_ngram_real_session.log`.*

Speculation that drafts from text already in context is free in VRAM, so I
expected it to help or do nothing. In a real dsh session it hurt. Two sessions
on 2026-08-28, both `-c 32768`, both with a ~9.2k-token prefill:

| session | ngram drafting | prefill | generation in that task |
|---|---|---|---|
| `fn_vision1024_1817` | none (no draft lines) | 9,237 tokens at 140.38 t/s | 11.50 t/s |
| `fn_prod_1830` | on (a later task logs `draft acceptance = 0.29688`, 38 / 128) | 9,198 tokens at 84.17 t/s | 10.87 t/s |

The two sessions are not identical: the first also had the vision projector
loaded, and neither log records `-ub`. The direction matches the marker bench
(-ub 4096, KV q8_0, six runs A-F): the runs that show draft lines generated at
11.77 and 11.68 t/s (acceptance 0.437 and 0.267) against 12.35-13.15 t/s for the
others. Run B died with `CUDA error: the launch timed out and was terminated`.
The per-run speculation flags were passed on the command line and are not in
any log, so I cannot tie that crash to a specific draft length.

**Closed: no ngram speculation for this model on this machine.**

## 10. Reasoning effort on a real agentic task

*Flash-Next UD-IQ4_XS, `-c 98304`, dsh. Log: `logs/2026-08-30_reasoning_effort_dsh_sessions.log`.*

I assumed lower effort means a faster session. The test was the same dsh
prompt ("audit my dsh installation and write a report") on 2026-08-30, one
clean session per server start.

On llama.cpp, dsh's own effort field does not reach the model: the pi-ai adapter
writes no `chat_template_kwargs`, so only the `llama-server` command line sets
the effort (my Aug 30 finding; that check is not in these logs). A session's
effort is therefore whatever its server was started with, and these three
server logs were written at verbosity 3. They record neither the command line
nor the rendered template ("Reasoning effort is set to ..."). **No arm's effort
is verified.** The `low` in one log's file name is a name, not a measurement,
and the `low` that dsh sent in `d3abf75a` never reached the model.

| server log | dsh session | effort | turn time | output tokens | outcome |
|---|---|---|---|---|---|
| `flashnext_preserve_1001` | `960c2dbf` | not verified | 13.2 min | 5,108 | completed |
| `flashnext_low_1043` | `fe040eb3` | not verified | aborted after 5.6 min | 0 | `stream idle timeout after 300000ms`; the server logged 3 cancelled tasks |
| `flashnext_low_1043` | `e6911e39` (retry) | not verified | 28.9 min | 15,754 | completed |
| `flashnext_c98304_0911` | `d3abf75a` | not verified (dsh sent `low`, which is not applied) | 19.3 min | 6,796 | completed |

What the logs do show: the same prompt on the same model and context took
13.2 to 28.9 minutes per completed session, with 5,108 to 15,754 output tokens.
That is a wide spread, and I cannot attribute it to effort. **Open: the
low-vs-medium comparison has to be rerun with the server's effort flag
recorded (`-lv 4`, or the argv saved next to the log) before it says anything.**

Footnote: on one small logic puzzle (Strata IQ3_S, temperature 0;
`logs/effort_proxy` data in the same log file) low used 653 completion tokens vs
806 for medium. A single short prompt does not settle effort for agentic work.

## 11. Strata: Flash-Next at three times the prefill, and one answer that changes with context

*Strata v0.1.39, Qwen3.8-Flash-Next, the same 17.5k marker (correct answer: 6).
Log: `logs/2026-10-05_strata_flashnext.log`. Every number below names its context.*

Strata keeps a cache of the hot experts on the GPU and streams the rest from RAM.

**The sm_120 garbled-prompt bug.** On RTX 50-series cards (sm_120) Strata's MMQ
prompt path made UD-IQ4_XS prompts read as garbage; the workaround is
`STRATA_PREFILL_MMQ=0` (upstream issue
[Niko1221/Strata#968](https://github.com/Niko1221/Strata/issues/968)). My
config sets it (`configs/strata/strata-iq3_s.json`), and every number below
was taken with it. I did not log a run without it.

**UD-IQ4_XS vs GSQ IQ3_S:**

| quant | context | swap | prefill (t/s) | generation (t/s) | wall (s) | answer |
|---|---|---|---|---|---|---|
| UD-IQ4_XS | 65536 | on (VmSwap 1.2 → 1.6 GB) | 286.8 | 19.0 | 95.9 | 6 |
| UD-IQ4_XS | 65536 | on (VmSwap 1.2 → 1.7 GB) | 335.1 | 19.8 | 86.7 | 6 |
| UD-IQ4_XS | 65536 | off (0 kB) | 383.2 | 18.1 | 81.2 | 6 |
| GSQ IQ3_S | 32768 | off | 699.6 | 41.1 | 38.4 | 6 |
| GSQ IQ3_S, no vision | 32768 | off | 699.8 | 39.8 | 38.8 | 6 |

Against llama.cpp on the same machine (entry 6: 179.50 / 12.86 t/s at 32768)
this is about 3.9x the prefill and 3.2x the generation, with a different quant.

**The oomd kill.** The first IQ3_S start did not survive: at 16:22:40 on
2026-10-05 `systemd-oomd` killed 17 processes in the desktop-app scope the start
had been launched from (journal lines in the log). Setup then flagged that IQ3_S "needs
62 GB RAM, you have 61", and I re-ran it in its low-RAM mode: the GPU holds
about 20% of the experts and the rest are read once from a 46.84 GiB packed copy.
Since then `strata_up` turns swap off before starting, and runs Strata and dsh
as separate `systemd-run --user` units so an OOM kill takes only one cgroup.

**Pointing dsh at it.** The dsh web profile (`configs/dsh-profile-web/cordis.patch.yml`)
moved the `flashnext-local` provider from `:8093` to `:8080`, set the context
window to 65536, dropped image input, and made Flash-Next IQ3_S at medium effort
the default model instead of the 27B.

**Open problem: the answer changes with context.** `strata-iq3_s.json` runs at
**65536** today. At 32768 IQ3_S answered 6 in both runs. At 65536 it answered
**5** in the first run and in 3 of 10 alternating medium/high runs (runs 1 and 5
medium, run 4 high), with prefill 687.7-709.3 t/s and generation 37.1-44.2 t/s.
A wrong count from the same prompt and the same weights, depending only on the
context setting, is not something I can explain yet. Until I can, a 32768 run
is the reference for this marker.

## 12. The 27B moved to GSQ IQ3_S + MTP: the ceiling went from 32768 to 73728

*Qwen3.8-27B GSQ RCO IQ3_S with MTP, llama.cpp build 10751, 17.5k marker.
Log: `logs/2026-09-25_27b_gsq_context_ladder.log`.*

Entry 1 ended with UD-IQ4_XS: ceiling 32768, 861 t/s prefill, 57-59 t/s
generation. The GSQ IQ3_S file is 11.29 GiB, and its CUDA0 model buffer is
11,159.69 MiB:

| context | survived | min VRAM free (MiB) | prefill (t/s) | generation (t/s) | drafts accepted | answer |
|---|---|---|---|---|---|---|
| 32768 | yes | 2100 | 827.9 | 59.45 | 299 / 306 | 6 |
| 49152 | yes | 1378 | 822.4 | 55.70 | 522 / 582 | 6 |
| 57344 | yes | 1016 | 820.4 | 57.32 | 438 / 465 | 6 |
| 65536 | yes | 656 | 820.4 | 57.44 | 287 / 306 | 6 |
| 73728 | yes | 294 | 820.4 | 57.27 | 425 / 453 | 6 |
| 81920 | no (out of memory) | 112 | - | - | - | - |

Prefill is about 4% lower than UD-IQ4_XS; generation is unchanged. The memory
saved buys context: production runs at 65536 (`configs/launch/manifiestate`)
with 656 MiB of VRAM to spare on this marker, and CUDA graphs disabled (entry 5).

## 13. Open: does the 27B write faster than Flash-Next in practice?

*Log: `logs/2026-10_real_session_generation_speed.log`.*

Per-token generation in real dsh sessions, counting only tasks of 200 tokens or more:

- 27B GSQ on llama.cpp: 13 server logs from 2026-09-25 to 2026-10-02, with per-log
  medians from 43.88 to 58.55 t/s.
- Flash-Next IQ3_S on Strata (65536 context): one session on 2026-10-05,
  19 tasks, median 35.2 t/s (range 30.0-42.2).

On these numbers the 27B writes faster per token. But these are different
tasks, and per-token speed is not time to a finished answer: a model that
needs fewer tokens or fewer tool rounds can still finish first. I have not run
the same task on both models; that comparison is the next measurement.

---

## How the stack is launched

The two stacks share one 16 GB GPU and cannot run together: each one fills
the card. Each start script refuses to run while the other stack holds its port.

**27B stack: `manifiestate` / `chao`** (`configs/launch/`)
- `manifiestate` starts three services, each detached in its own session
  (`setsid nohup`) so closing the terminal or Ctrl+C cannot kill them:
  `imgproxy` on `:8091` (converts webp images to PNG for llama-server), the 27B
  GSQ on `:8092` (llama.cpp, MTP, 65536 context, CUDA graphs off), and dsh web
  on `:3080`. It records each PID and then follows the model log.
  It refuses to start while Strata is on `:8080`.
- `chao` stops dsh, the model and imgproxy by those PIDs (or whatever owns their
  ports), waits until each process is really gone, and prints GPU memory in use.
- `manifiestate_v` is an older variant with the vision projector at 49152.

**Flash-Next stack: `strata_up` / `strata_down`**
- `strata_up` turns swap off (Strata is tuned for no swap), then starts Strata on
  `:8080` and dsh web on `:3080` as two transient `systemd-run --user` units,
  `strata-iq3s` and `dsh-web`. Each unit has its own cgroup, so an OOM kill or a
  closed terminal takes down only that unit. There are no unit files: the units
  exist only while they run. It refuses to start while the 27B is on `:8092`.
- `strata_down` stops both units (systemd waits for each cgroup to empty) and
  turns swap back on.
- Strata's own config is `configs/strata/strata-iq3_s.json` (65536 context,
  `STRATA_PREFILL_MMQ=0`). `strata-unsloth-ud-iq4_xs.HISTORICAL.json` is kept
  for reference; its model is deleted.

**Port `:8093` is shared by two things:** `flashnext.sh` (the old llama.cpp
Flash-Next launcher from entries 6-9) and the CPU-only 4B model of the camera
project. Only one of them can run at a time.

---

## Method

- Loading a config certifies nothing. This card has loaded cleanly with 468 and
  526 MiB free and died on the first real request. Certification means a real
  workload.
- Near-perfect speculative acceptance can mean the model is copying text already
  in context rather than drafting. Vary the prompt.
- What an agent reports about the disk is not true until a command confirms it.
- Register what will be measured before measuring, not the threshold for accepting it.
