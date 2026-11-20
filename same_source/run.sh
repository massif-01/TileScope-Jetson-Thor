#!/usr/bin/env bash
# Caller serializes access to the same device lock as the operator CLI.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$1; site=$2; py=$3
mkdir "$out"; out=$(cd "$out" && pwd)
cp "$here/probe.py" "$out/kernel-source.py"
site=$(cd "$site" && pwd)
export PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$out/cache" TRITON_TILEIR_RUNTIME_FALLBACK=0
export TRITON_DEFAULT_BACKEND=nvidia ENABLE_TILE=0
printf '%s\n' "$py $here/probe.py $out" > "$out/command"
date -u +%FT%TZ > "$out/start.utc"
"$py" "$here/../round2/monitor.py" > "$out/telemetry.jsonl" 2> "$out/monitor.stderr" &
mon=$!
trap 'kill "$mon" 2>/dev/null || :; wait "$mon" 2>/dev/null || :' EXIT
status=0
timeout 180 "$py" -u "$here/probe.py" "$out" > "$out/stdout" 2> "$out/stderr" || status=$?
echo "$status" > "$out/exit"
date -u +%FT%TZ > "$out/end.utc"
exit "$status"
