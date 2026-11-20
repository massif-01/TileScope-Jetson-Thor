#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <vector>

using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

// Python hashes these exact input buffers after process exit and removes the scratch bytes.
static void snapshot(const char *name, const void *device, size_t bytes) {
  std::vector<unsigned char> host(bytes);
  CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
  std::ofstream f(std::string("input-") + name + ".bin", std::ios::binary);
  f.write(reinterpret_cast<const char *>(host.data()), bytes);
  if (!f) { std::perror("input snapshot"); std::exit(3); }
}

template <typename Tensor>
void snapshot_tensor(const char *name, Tensor &t) {
  snapshot(name, t.device_data(), (t.size() * cutlass::sizeof_bits<typename Tensor::Element>::value + 7) / 8);
}

template <typename GemmType, typename Arguments, typename Validator>
int measure(GemmType &gemm, Arguments &args, const Options &opt, Validator validate) {
  int runtime = 0, driver = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  if (props.major != 11 || props.minor != 0) { std::cerr << "device must be sm110\n"; return 4; }
  std::printf("{\"phase\":\"identity\",\"device\":\"%s\",\"sm\":110,\"sms\":%d,"
              "\"driver_api\":%d,\"cuda_runtime\":%d,\"mnk\":[%d,%d,%d],"
              "\"seed\":%llu,\"warmup\":%d,\"samples\":%d,\"format\":\"%s\"}\n", props.name,
              props.multiProcessorCount, driver, runtime, opt.m, opt.n, opt.k,
              (unsigned long long) opt.seed, opt.warmup, opt.iterations, CAL_FORMAT);
  auto start = Clock::now();
  CUTLASS_CHECK(gemm.can_implement(args));
  cutlass::device_memory::allocation<uint8_t> workspace(GemmType::get_workspace_size(args));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUTLASS_CHECK(gemm.initialize(args, workspace.get(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  double init = milliseconds(start);
  start = Clock::now();
  CUTLASS_CHECK(gemm.run(stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  double first = milliseconds(start);
  auto check = [&](const char *stage) {
    bool ok = validate();
    std::printf("{\"phase\":\"correctness\",\"stage\":\"%s\",\"passed\":%s}\n", stage, ok ? "true" : "false");
    if (!ok) { std::fflush(stdout); std::exit(5); }
  };
  check("direct");
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  start = Clock::now();
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUTLASS_CHECK(gemm.run(stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  double capture = milliseconds(start);
  CUDA_CHECK(cudaGraphLaunch(exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  check("graph");
  for (int i = 0; i < opt.warmup; ++i) { CUDA_CHECK(cudaGraphLaunch(exec, stream)); }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::printf("{\"phase\":\"setup\",\"initialize_ms\":%.9g,\"first_ms\":%.9g,\"capture_ms\":%.9g,\"operations_per_graph\":1}\n",
              init, first, capture);
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < opt.iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(a, stream));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    std::printf("{\"phase\":\"sample\",\"i\":%d,\"us\":%.12g}\n", i, double(ms)*1000.0);
  }
  check("post");
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
