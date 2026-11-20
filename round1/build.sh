#!/usr/bin/env bash
# Build the unchanged upstream BF16 target and its shared-input bridge.
set -eu
: "${GEMM:?point to the pinned thor-gemm-bench checkout}"
: "${CUTLASS:?point to a clean CUTLASS checkout}"
: "${CUDA:?point to the chosen CUDA Toolkit}"
: "${RUN:?point to a fresh absolute run directory}"
case ${1:-} in
  smoke|bridge) stage=$1 ;;
  *) echo "Usage: build.sh smoke|bridge" >&2; exit 2 ;;
esac

here="$(cd "$(dirname "$0")" && pwd)"
build="$RUN/build"
mkdir -p "$build"
cd "$build"

gemm_commit=$(git -C "$GEMM" rev-parse HEAD)
cutlass_commit=$(git -C "$CUTLASS" rev-parse HEAD)
test "$gemm_commit" = 0ca5f0d5b5270cea24e3540faf76a5e647b56c60
test -z "$(git -C "$GEMM" status --porcelain)"
test -z "$(git -C "$CUTLASS" status --porcelain)"
printf 'thor-gemm-bench %s\nCUTLASS %s\n' "$gemm_commit" "$cutlass_commit"
"$CUDA/bin/nvcc" --version

if [[ $stage == smoke ]]; then
    mkdir "$build/smoke-nvcc"
    ln -s "$GEMM/benchmarks" benchmarks
    ln -s "$GEMM/include" include
    # Copy env.mk so an upstream env.local.mk cannot affect this build.
    cp "$GEMM/env.mk" env.mk
    "$here/record.sh" "$RUN/build-smoke" timeout 600 make -f "$GEMM/Makefile" bench_bf16_cutlass \
        "NVCC=$CUDA/bin/nvcc -v --keep --keep-dir $build/smoke-nvcc" \
        "CUTLASS_DIR=$CUTLASS" "CUTLASS_COMMIT=$cutlass_commit"
    exit 0
fi

# Match the upstream target's architecture, language standard and optimization.
# Use the shared CUDA runtime for interop with the Python process.
mkdir "$build/bridge-nvcc"
"$here/record.sh" "$RUN/build-bridge" timeout 600 "$CUDA/bin/nvcc" \
    -v --keep --keep-dir "$build/bridge-nvcc" -shared -Xcompiler=-fPIC --cudart=shared \
    -std=c++17 -O3 -arch=sm_110a --expt-relaxed-constexpr --expt-extended-lambda \
    "-DCUTLASS_COMMIT=\"$cutlass_commit\"" \
    -I "$GEMM/benchmarks" -I "$GEMM/include" -I "$CUTLASS/include" \
    -I "$CUTLASS/tools/util/include" -I "$CUTLASS/examples/common" \
    "$here/cutlass_bridge.cu" -o "$build/libtilescope_cutlass.so"

"$here/record.sh" "$RUN/cutlass-elf" "$CUDA/bin/cuobjdump" --list-elf "$build/libtilescope_cutlass.so"
