#!/usr/bin/env bash
# Isolate the published CPython 3.12 ARM64 wheel; keep the shared deployment environment intact.
set -eu
dest=$1
py=${2:-/opt/tilescope/venv/bin/python}
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
if test -e "$dest/site"; then echo "Choose a fresh isolated destination" >&2; exit 2; fi
mkdir -p "$dest/wheel"
wheel='triton_tileir-3.7.1+tileir13.4.0-cp312-cp312-linux_aarch64.whl'
remote='https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.7.1-tileir-13.4.0'
cd "$dest/wheel"
curl -fsSL --connect-timeout 10 --max-time 540 "$remote/${wheel/+/%2B}" -o "$wheel"
printf '%s  %s\n' ac671ebaa5d633f96e2e107e18a2df89e096bdd54e718e852bc97df8920ae002 "$wheel" | sha256sum -c -
"$py" -m pip install --no-cache-dir --no-index --no-deps --target "$dest/site" "./$wheel"
