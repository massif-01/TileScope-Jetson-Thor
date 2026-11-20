// Includes the frozen partner kernel configuration and independent CUTLASS reference.
#define main partner_main
#include CAL_PARTNER
#undef main
#ifndef CUTLASS_ARCH_MMA_SM100_SUPPORTED
#error Required SM100 tensor core path is unavailable
#endif
#include "measure.hpp"

int main(int argc, const char **argv) {
  Options opt;
  opt.parse(argc, argv);
  initialize(opt);
#if CAL_PACKED
  snapshot_tensor("A", block_A);
  snapshot_tensor("B", block_B);
  snapshot_tensor("SFA", block_SFA);
  snapshot_tensor("SFB", block_SFB);
  snapshot_tensor("normalization", block_Normconst);
#else
  snapshot("A", block_A.get(), block_A.size()*sizeof(ElementA));
  snapshot("B", block_B.get(), block_B.size()*sizeof(ElementB));
#endif
  Gemm gemm;
  auto args = args_from_options(opt);
  return measure(gemm, args, opt, [&] { return verify(opt); });
}
