# configs

Copies of the files that launch and configure the local stack, as they were on
2026-10-06. Home paths are written as `~`; API keys and tokens are not stored in
any of them (`apiKeyEnv` names an environment variable, not a key). See
"How the stack is launched" in the top-level README.

| Path | What it is |
|---|---|
| `launch/manifiestate`, `launch/chao` | Start / stop the 27B stack: imgproxy `:8091`, 27B GSQ on llama.cpp `:8092`, dsh web `:3080` |
| `launch/manifiestate_v` | Older 27B variant with the vision projector (calls `start_27b_gsq_vision.sh`, not copied) |
| `launch/strata_up`, `launch/strata_down` | Start / stop the Flash-Next stack: Strata `:8080` and dsh web `:3080` as the transient `systemd-run --user` units `strata-iq3s` and `dsh-web` (no unit files exist; these scripts create them) |
| `launch/flashnext.sh` | Old llama.cpp launcher for Flash-Next on `:8093` (entries 6-9); its model path points at the GSQ IQ3_S now |
| `strata/strata-iq3_s.json`, `strata/run-iq3_s.sh`, `strata/strata-iq3_s.shared-settings.json` | Strata config in use: Flash-Next GSQ IQ3_S, 65536 context, `STRATA_PREFILL_MMQ=0` |
| `strata/strata-unsloth-ud-iq4_xs.HISTORICAL.json` | **Historical.** Strata config for Flash-Next UD-IQ4_XS (entry 11); that model has been deleted |
| `dsh-profile-web/cordis.patch.yml` | dsh web profile patch: providers, models, default model and effort |
| `settings.yaml`, `AGENTS.md`, `AGENTS.md.rules`, `agent-presets/` | Earlier dsh configuration (entry 4) |
