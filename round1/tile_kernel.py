"""Fixed-shape cuTile BF16 GEMM, with the CUTLASS baseline's physical layout."""

import cuda.tile as ct


@ct.kernel
def bf16_gemm(a, bt, dt):
    row = ct.bid(0)
    col = ct.bid(1)
    acc = ct.full((64, 64), 0, dtype=ct.float32)
    for step in range(a.shape[1] // 32):
        lhs = ct.load(a, index=(row, step), shape=(64, 32), padding_mode=ct.PaddingMode.ZERO)
        rhs = ct.load(bt, index=(col, step), shape=(64, 32), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(lhs, ct.permute(rhs, (1, 0)), acc)
    ct.store(dt, index=(col, row), tile=ct.permute(acc, (1, 0)))
