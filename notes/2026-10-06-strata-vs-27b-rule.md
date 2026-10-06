# 27B (llama.cpp) vs Flash-Next IQ3_S (Strata): one same-test comparison (pre-registered 2026-10-06, before any run)

Closes open question 13 in the README. Written and pushed before the first shot.

## Test
- Prompt: the frozen 17.5k-token marker (`~/bench/marcador_17500_v1.json`, `max_tokens` 4000, no
  temperature set). Correct answer: **6** (count of `def result_score` in the prompt).
- Client: one `curl` POST to `/v1/chat/completions`, non-streaming; wall time measured around it.
- **5 shots per arm.** The server is restarted before every shot so no shot can reuse a previous
  one's prompt from the prompt cache; each shot's `cache_n` is recorded to prove it (expected 0).
- Nothing else uses the GPU: the camera project's 4B on `:8093` (CPU) is stopped first; no dsh.

## Arms
1. **27B**: Qwen3.8-27B GSQ RCO IQ3_S + MTP on llama.cpp build 10751, with the exact server
   parameters of `configs/launch/manifiestate`: `GGML_CUDA_DISABLE_GRAPHS=1`,
   `--spec-type draft-mtp --spec-draft-n-max 3 -ngl 99 -fa on -c 65536 -ub 256 -ctk q8_0 -ctv q8_0
   -ctkd q8_0 -ctvd q8_0 -t 6 --parallel 1 --reasoning-effort medium --port 8092`. Server only
   (no imgproxy, no dsh). Swap may be on. Run first.
2. **Strata**: Flash-Next GSQ IQ3_S on Strata v0.1.39 with `strata-iq3_s.json` exactly as in
   production (`--max-context 65536`, shared settings `reasoning_effort: medium`), started like
   `strata_up` (transient user unit), swap off. Run second.

## Recorded
- Per shot: answer, correct (yes/no), prefill t/s, generation t/s, wall time (s), `cache_n`,
  completion tokens, finish reason.
- Per shot (after the request): RAM used, the server's VmSwap, free VRAM.
- A shot that errors, times out (900 s) or ends without an answer counts as **not correct**, and its
  wall time is the time until it failed.

## Claim and rule (fixed now)
Claim: **"the 27B is the better choice on this machine."**
- **Supported** if 27B correct count >= Strata correct count **and** 27B median wall time <=
  Strata median wall time.
- Otherwise the claim is not supported, and the README reports what the numbers say, whatever
  they are.
