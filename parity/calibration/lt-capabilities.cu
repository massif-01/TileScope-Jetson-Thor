// API-support observations corresponding to thor-gemm-bench/probes/probe_nvfp4.cu
// at 0ca5f0d5b5270cea24e3540faf76a5e647b56c60. No timed or correctness claim.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdio>
#include "partner/helper.h"

struct Case {
  const char *name;
  cublasComputeType_t compute;
  cudaDataType_t scalar, output;
  cublasLtMatmulMatrixScale_t block;
};

int main() {
  constexpr int n = 1024;
  constexpr size_t workspace_bytes = 64 * 1024 * 1024;
  const auto v16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  const auto v32 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  Case cases[] = {
    {"C32F outF VEC16", CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, v16},
    {"C32F outF VEC32", CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, v32},
    {"C16F outH VEC16", CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_16F, v16},
    {"C16F outH VEC32", CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_16F, v32},
    {"C32F outH VEC16", CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, v16},
    {"C32F outBF VEC16", CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, v16},
    {"C16F outBF VEC16", CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_16BF, v16},
    {"C32F_F16 outF VEC16", CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F, CUDA_R_32F, v16},
    {"C32F_FBF outF VEC16", CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F, CUDA_R_32F, v16},
    {"C16F outF VEC16", CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_32F, v16}};
  cublasLtHandle_t handle;
  if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
  void *a, *b, *d, *scale, *workspace;
  CUDA_CHECK(cudaMalloc(&a,n*n/2)); CUDA_CHECK(cudaMalloc(&b,n*n/2));
  CUDA_CHECK(cudaMalloc(&d,n*n*4)); CUDA_CHECK(cudaMalloc(&scale,n*n/16));
  CUDA_CHECK(cudaMalloc(&workspace,workspace_bytes));
  CUDA_CHECK(cudaMemset(a,0,n*n/2)); CUDA_CHECK(cudaMemset(b,0,n*n/2));
  std::printf("{\"version\":%zu,\"mnk\":[1024,1024,1024],\"workspace_bytes\":%zu}\n",cublasLtGetVersion(),workspace_bytes);
  for (const auto &c: cases) {
    cudaMemset(scale,c.block==v16 ? 0x38 : 0x7f,n*n/16);
    cublasLtMatmulDesc_t op=nullptr;
    cublasLtMatrixLayout_t ab=nullptr,cd=nullptr;
    cublasLtMatmulPreference_t pref=nullptr;
    cublasStatus_t st=cublasLtMatmulDescCreate(&op,c.compute,c.scalar);
    const char *stage="descriptor";
    auto step=[&](const char *name, auto call) {
      if(st==CUBLAS_STATUS_SUCCESS) { stage=name; st=call(); }
    };
    step("A matrix layout",[&]{return cublasLtMatrixLayoutCreate(&ab,CUDA_R_4F_E2M1,n,n,n);});
    step("output matrix layout",[&]{return cublasLtMatrixLayoutCreate(&cd,c.output,n,n,n);});
    step("A scale mode",[&]{return cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&c.block,sizeof(c.block));});
    step("B scale mode",[&]{return cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&c.block,sizeof(c.block));});
    step("A scale pointer",[&]{return cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scale,sizeof(scale));});
    step("B scale pointer",[&]{return cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scale,sizeof(scale));});
    step("preference",[&]{return cublasLtMatmulPreferenceCreate(&pref);});
    step("workspace limit",[&]{return cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));});
    int heur=-1, count=0, direct=-1, sync=-1;
    if (st==CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulHeuristicResult_t res{};
      heur=cublasLtMatmulAlgoGetHeuristic(handle,op,ab,ab,cd,cd,pref,1,&res,&count);
      float f1=1, f0=0; __half h1=__float2half(1), h0=__float2half(0);
      const void *one=c.scalar==CUDA_R_16F ? (void*)&h1 : (void*)&f1;
      const void *zero=c.scalar==CUDA_R_16F ? (void*)&h0 : (void*)&f0;
      direct=cublasLtMatmul(handle,op,one,a,ab,b,ab,zero,d,cd,d,cd,nullptr,workspace,workspace_bytes,0);
      if (direct==CUBLAS_STATUS_SUCCESS) sync=cudaDeviceSynchronize();
    }
    std::printf("{\"case\":\"%s\",\"setup_status\":%d,\"last_setup_stage\":\"%s\",\"heuristic_status\":%d,"
                "\"returned_algorithms\":%d,\"null_algorithm_status\":%d,\"launch_sync_status\":%d}\n",
                c.name,int(st),stage,heur,count,direct,sync);
    if(pref) cublasLtMatmulPreferenceDestroy(pref);
    if(cd) cublasLtMatrixLayoutDestroy(cd);
    if(ab) cublasLtMatrixLayoutDestroy(ab);
    if(op) cublasLtMatmulDescDestroy(op);
  }
  cublasLtDestroy(handle);
  CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(scale)); CUDA_CHECK(cudaFree(workspace));
  return 0;
}
