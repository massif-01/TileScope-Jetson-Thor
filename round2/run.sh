#!/usr/bin/env bash
set -eu
# run.sh TASK NEW_OUTPUT_DIR [native|tile]
task=$1
out=$2
mkdir "$out"
out=$(cd "$out" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
export PATH="/usr/local/cuda-13.3/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.3
export CUDA_TILE_CACHE_DIR=off CUDA_TILE_TEMP_DIR="$out/compiler" CUDA_TILE_DUMP_BYTECODE="$out/compiler"
mkdir "$CUDA_TILE_TEMP_DIR"
py=${PROBE_PYTHON:-/opt/tilescope/venv/bin/python}
model=${CHECKPOINT:-/opt/models/Qwen3.8-27B-NVFP4}
args=(--task "$task" --checkpoint "$model")
if [[ $# -gt 2 ]]; then args+=(--only "$3"); fi
printf '%q ' "$py" "$here/probe.py" "${args[@]}" > "$out/command"
date -u +%FT%TZ > "$out/start.utc"
"$py" "$here/monitor.py" > "$out/telemetry.jsonl" 2> "$out/monitor.stderr" &
observer=$!
trap 'kill "$observer" 2>/dev/null || true; wait "$observer" 2>/dev/null || true' EXIT
status=0
timeout 360 "$py" -u "$here/probe.py" "${args[@]}" > "$out/stdout" 2> "$out/stderr" || status=$?
date -u +%FT%TZ > "$out/end.utc"
printf '%s\n' "$status" > "$out/exit"
exit "$status"
