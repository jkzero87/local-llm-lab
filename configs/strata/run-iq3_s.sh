#!/bin/sh
cd "~/strata"
exec "~/strata/.venv/bin/python" "~/strata/serve/server.py" "--engine" "strata" "--config" "~/strata/strata-iq3_s.json" "--port" "8080"
