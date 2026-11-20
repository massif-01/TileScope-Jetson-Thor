// Wrapper for thor-gemm-bench/benchmarks/bench_bf16_cutlass.cu.
// The included upstream file retains NVIDIA's copyright and BSD-3-Clause notice.
#include <cstdio>

#define main tilescope_upstream_main
#include "bench_bf16_cutlass.cu"
#undef main

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "The upstream BF16 CUTLASS kernel was not compiled"
#endif

struct PreparedGemm {
  Gemm gemm;
  void* workspace = nullptr;
};

static int cutlass_error(const char* where, cutlass::Status status) {
  std::fprintf(stderr, "%s: %s\n", where, cutlassGetStatusString(status));
  return 1;
}

extern "C" int tilescope_prepare(
    int m, int n, int k, const ElementA* a, const ElementB* b, const float* c,
    float* d, cudaStream_t stream, PreparedGemm** out) {
  *out = nullptr;
  Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a, cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}),
       b, cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1})},
      {{1.f, 0.f}, c, cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}),
       d, cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1})}};
  args.scheduler.max_swizzle_size = 0;

  auto status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cutlass_error("Gemm::can_implement", status);
  }

  auto* prepared = new PreparedGemm;
  const size_t workspace_bytes = Gemm::get_workspace_size(args);
  if (workspace_bytes != 0) {
    auto error = cudaMalloc(&prepared->workspace, workspace_bytes);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc workspace: %s\n", cudaGetErrorString(error));
      delete prepared;
      return 1;
    }
  }

  status = prepared->gemm.initialize(args, prepared->workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    cudaFree(prepared->workspace);
    delete prepared;
    return cutlass_error("Gemm::initialize", status);
  }
  *out = prepared;
  return 0;
}

extern "C" int tilescope_launch(PreparedGemm* prepared, cudaStream_t stream) {
  auto status = prepared->gemm.run(stream);
  return status == cutlass::Status::kSuccess ? 0 : cutlass_error("Gemm::run", status);
}

extern "C" int tilescope_destroy(PreparedGemm* prepared) {
  auto status = prepared->workspace ? cudaFree(prepared->workspace) : cudaSuccess;
  delete prepared;
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaFree workspace: %s\n", cudaGetErrorString(status));
    return 1;
  }
  return 0;
}

extern "C" int tilescope_versions(int* driver, int* runtime) {
  auto status = cudaDriverGetVersion(driver);
  if (status == cudaSuccess) {
    status = cudaRuntimeGetVersion(runtime);
  }
  if (status != cudaSuccess) {
    std::fprintf(stderr, "query CUDA versions: %s\n", cudaGetErrorString(status));
    return 1;
  }
  return 0;
}
