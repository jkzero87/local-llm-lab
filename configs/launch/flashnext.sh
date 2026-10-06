#!/usr/bin/env bash
MODEL=$(ls $(find ~/models/Qwen3.8-Flash-Next-GSQ/IQ3_S -name "*00001-of-*.gguf" | head -1) | head -1)
echo "Modelo: $MODEL"
~/llama.cpp-qwen4exp/build/bin/llama-server \
  -m "$MODEL" \
  --alias flashnext --host 127.0.0.1 --port 8093 \
  -c 32768 -fa on -ctk f16 -ctv f16 \
  -b 1024 -ub 256 -t 6 -np 1 \
  --jinja --fit on --fit-target 1536 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --metrics 2>&1 | tee ~/logs/flashnext_$(date +%H%M).log
