#!/usr/bin/env bash
# Preserve the exact command, stdout, stderr and process status in a fresh directory.
set -eu
dir=$1
shift
mkdir "$dir"
{
    printf '%q' "$1"
    for arg in "${@:2}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
} > "$dir/command.sh"
pwd > "$dir/cwd"
date -u +%FT%TZ > "$dir/start.utc"
status=0
"$@" > "$dir/stdout" 2> "$dir/stderr" || status=$?
printf '%s\n' "$status" > "$dir/exit"
date -u +%FT%TZ > "$dir/end.utc"
exit "$status"
